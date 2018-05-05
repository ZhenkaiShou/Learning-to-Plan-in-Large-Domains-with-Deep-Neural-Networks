import copy
import numpy as np
import os
import queue
import tensorflow as tf
import threading
from constants import *
from gamestate import *
from model import *
from util import extract_inputs, normalize_probability, get_state_sequence, reshape_feature_sequence, transform

class TreeNode(object):
  def __init__(self, parent_edge, p, v):
    self.parent_edge = parent_edge
    self.children_edge = [Edge(self, x) for x in p]
    self.v = v[0]
    self.lock = threading.Lock()

  def add_dirichlet_noise(self, epsilon):
    # Add Dirichlet noise to the root node to encourage exploration.
    noise = np.random.dirichlet([DIR_NOISE for _ in self.children_edge], 1)[0]
    for i in range(len(self.children_edge)):
      self.children_edge[i].p = (1 - epsilon) * self.children_edge[i].p + epsilon * noise[i]

  def get_tree_policy(self):
    # Return the index of the action with highest (Q+U) value.
    n_sum = np.sum([x.n for x in self.children_edge])
    q_plus_u = [x.get_q_plus_u(n_sum) for x in self.children_edge]
    action = np.argmax(q_plus_u)
    
    return action

class Edge(object):
  def __init__(self, parent_node, p):
    self.parent_node = parent_node
    self.children_node = None
    self.w = 0
    self.n = 0
    self.q = 0
    self.p = p
    self.lock = threading.Lock()
  
  def get_q_plus_u(self, n_sum):
    # Return the (Q+U) value of the edge.
    return self.q + C_PUCT * self.p * np.sqrt(n_sum + 1) / (self.n + 1)
  
  def expand(self, p, v):
    # Expand the next node.
    self.children_node = TreeNode(self, p, v)
  
  def add_virtual_loss(self):
    # Add virtual loss to encourage exploration.
    self.w -= VIRTUAL_LOSS
    self.n += VIRTUAL_LOSS
    self.q = self.w / self.n
    
  def backup(self, v):
    # Backup the edge and remove the virtual loss.
    self.w += VIRTUAL_LOSS + v 
    self.n += -VIRTUAL_LOSS + 1
    self.q = self.w / self.n
  
class MCTS(object):
  def __init__(self, root_state, player, randomness_level, advanced_evaluation = ADVANCED_EVALUATION, num_simulations = NUM_SIMULATIONS, search_threads = SEARCH_THREADS, parent_dir = ""):
    length = len(root_state)
    self.root_state = copy.deepcopy(root_state)
    self.root_node = [None for _ in range(length)]
    self.step = np.zeros(length, np.int32)
    self.blocked = np.zeros(length, np.int32)
    self.player = player
    self.randomness_level = randomness_level
    self.advanced_evaluation = advanced_evaluation
    self.num_simulations = num_simulations
    self.search_threads = search_threads
    self.parent_dir = parent_dir
  
  def tree_search(self):
    length = len(self.root_state)
    # Use multiple threads to speed up searching.
    semaphore = [threading.Semaphore(self.num_simulations) for _ in range(length)]
    q_main = queue.Queue()
    q_forward = queue.Queue()
    q_feedback = [[queue.Queue() for _ in range(self.search_threads)] for _ in range(length)]
    thread_evaluator = threading.Thread(target = self.evaluator_thread, args = (q_main, q_forward, q_feedback))
    thread_searcher = [[threading.Thread(target = self.searcher_thread, args = (i, j, semaphore[i], q_forward, q_feedback[i][j])) for j in range(self.search_threads)] for i in range(length)]
    # Start evaluator thread.
    thread_evaluator.start()
    
    # Evaluate the root node if it does not exist.
    root_node_none = np.where([x == None for x in self.root_node])[0]
    if len(root_node_none) > 0:
      q_forward.put(([self.root_state[x] for x in root_node_none]))
      (p, v) = q_main.get()
      # Initialize the root nodes.
      for i in range(len(root_node_none)):
        self.root_node[root_node_none[i]] = TreeNode(None, p[i], v[i])
    else:
      q_forward.put((None))
      (empty) = q_main.get()
    
    # Add Dirichlet noise to the root node.
    for x in self.root_node:
      if self.randomness_level == "high":
        x.add_dirichlet_noise(EPSILON_NOISE[0])
      elif self.randomness_level == "low":
        x.add_dirichlet_noise(EPSILON_NOISE[1])
    
    # Then starts the searcher threads.
    for i in range(length):
      for j in range(self.search_threads):
        thread_searcher[i][j].start()
    # Join all threads.
    for i in range(length):
      for j in range(self.search_threads):
        thread_searcher[i][j].join()
    thread_evaluator.join()
    
    # Output the probability distribution pi.
    n_list = [[y.n for y in x.children_edge] for x in self.root_node]
    pi = [[] for _ in range(length)]
    for i in range(length):
      pi[i] = n_list[i] / np.sum([x for x in n_list[i]])
    
    return pi
  
  def searcher_thread(self, i, j, semaphore, q_forward, q_feedback):
    while semaphore.acquire(False):
      state = copy.deepcopy(self.root_state[i])
      node = self.root_node[i]
      terminate = False
      # This is the selecting phase, which terminates 
      # when a new node is added to the tree or when it reaches a terminal node.
      while not terminate:
        node.lock.acquire()
        # Select the move with hightest (Q+U) value.
        action = node.get_tree_policy()
        # Add virtual loss to prevent other threads from exploring the same edge.
        edge = node.children_edge[action]
        edge.add_virtual_loss()
        node.lock.release()
        # Update state.
        state.select_move(state.legal_moves[action])
        # Check whether the next node is already in the tree.
        if edge.children_node == None:
          # Try to acquire the lock if the next node is not in the tree.
          if edge.lock.acquire(False):
            # Check again to avoid special case that the next node is already in the tree after the first check.
            if edge.children_node == None:
              # The node is indeed not in the tree. Send Computation request.
              q_forward.put((i, j, "Eval", state))
              # And wait for the result.
              (p, v) = q_feedback.get()
              # Now expand the node.
              edge.expand(p, v)
              terminate = True
              # Finally release the lock.
              edge.lock.release()
            else:
              # The node is already in the tree. Release the lock.              
              edge.lock.release()
          else:
            # Unable to get access. Send blocked package to notify the evaluator.
            q_forward.put((i, j, "Blocked", None))
            # And wait until the lock is released.
            edge.lock.acquire()
            edge.lock.release()
            q_forward.put((i, j, "Unblocked", None))
        
        if not terminate:
          # Only check terminal node when no new node is expanded.
          if len(edge.children_node.children_edge) == 0:
            # Special case for terminal node.
            q_forward.put((i, j, "Terminal", None))
            terminate = True
        # Update node.
        node = edge.children_node
      
      # This is the backup phase, which backs up the V value from leaf node until the root node.
      v = node.v
      while node.parent_edge != None:
        v = -v
        node.parent_edge.backup(v)
        node = node.parent_edge.parent_node
  
  def evaluator_thread(self, q_main, q_forward, q_feedback):
    length = len(self.root_state)
    index_list = []
    state_list = []
    signal = np.zeros(length, np.int16)
    
    if USE_GPU:
      # Additional configurations for running on GPU.
      os.environ["CUDA_VISIBLE_DEVICES"] = VISIBLE_DEVICE_MCTS
      gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = MEMORY_MCTS)
      gpu_config = tf.ConfigProto(gpu_options = gpu_options)
    else:
      gpu_config = None
    
    # Build models to evaluate p and v.
    model = Model(enhance = False, imitate = self.advanced_evaluation, optimize = 0)
    with tf.Session(config = gpu_config) as sess:
      saver = tf.train.Saver()
      # Restore the player.
      if self.player == BEST_PLAYER:
        file_name = "BestPlayer"
      else:
        file_name = "Player_" + format(self.player, "05d")
      saver.restore(sess, self.parent_dir + PLAYER_DIR + file_name)
      
      # Read the first package from the main thread to evaluate root nodes.
      (state) = q_forward.get()
      if state != None:
        transformed_state = copy.deepcopy(state)
        transformed_state = [transform(x, np.random.choice(SYMMETRY)) for x in transformed_state]
        inputs = extract_inputs(transformed_state)
        if self.advanced_evaluation:
          p, v = sess.run([model.p_hat, model.v_hat], feed_dict = {model.Inputs: inputs})
        else:
          p, v = sess.run([model.p, model.v], feed_dict = {model.Inputs: inputs})
        legal_moves = [x.legal_moves for x in transformed_state]
        p = normalize_probability(p, legal_moves)
        # The output p is a list of vectors, each of which indicates the probability of each possible move.
        # The output v is a column vector of shape (len(state), 1), which predicts the return of the game.
        q_main.put((p, v))
      else:
        q_main.put((None))
      
      while np.any(self.step < self.num_simulations):
        # Ready to read data.
        (i, j, str, state) = q_forward.get()
        if str == "Eval":
          # Normal package that contains a state to be evaluated.
          index_list.append([i, j])
          state_list.append(state)
          signal[i] += 1
        elif str == "Blocked":
          # Special case that the thread is blocked.
          self.blocked[i] += 1
        elif str == "Unblocked":
          # Remove the block signal.
          self.blocked[i] -= 1
        elif str == "Terminal":
          # Special case that the thread reaches a terminal node.
          self.step[i] += 1
        
        if np.all(np.logical_or(self.blocked+signal == self.search_threads, self.step+self.blocked+signal == self.num_simulations)):
          if len(state_list) > 0:
            # Evaluate all states in the list when all threads have send packages or when the search ends.
            transformed_state = copy.deepcopy(state_list)
            transformed_state = [transform(x, np.random.choice(SYMMETRY)) for x in transformed_state]
            inputs = extract_inputs(transformed_state)
            if self.advanced_evaluation:
              p, v = sess.run([model.p_hat, model.v_hat], feed_dict = {model.Inputs: inputs})
            else:
              p, v = sess.run([model.p, model.v], feed_dict = {model.Inputs: inputs})
            legal_moves = [x.legal_moves for x in transformed_state]
            p = normalize_probability(p, legal_moves)
            # Return p and v to the corresponding thread.
            for k in range(len(state_list)):
              self.step[index_list[k][0]] += 1
              q_feedback[index_list[k][0]][index_list[k][1]].put((p[k], v[k]))
            index_list = []
            state_list = []
          signal = np.zeros(length, np.int16)
    tf.contrib.keras.backend.clear_session()
  
  def get_action_sequence(self):
    # Get the action sequence of the most frequently visited states.
    length = len(self.root_state)
    sequence = [[] for i in range(length)]
    for i in range(length):
      node = self.root_node[i]
      state = copy.deepcopy(self.root_state[i])
      # Loop until the leaf node or the terminal node.
      while node != None and len(node.children_edge) > 0:
        # Select the most frequently visited action.
        n_list = [x.n for x in node.children_edge]
        arg_max = np.argwhere(n_list == np.amax(n_list))
        pi = np.zeros(len(n_list))
        pi[arg_max] = 1 / len(arg_max)
        index_i = np.random.choice(len(n_list), p = pi)
        index = state.legal_moves[index_i]
        # Update node and state.
        node = node.children_edge[index_i].children_node
        state.select_move(index)
        # Add the action to the sequence.
        if node != None:
          sequence[i].append(index)
    
    return sequence
  
  def get_subtree(self, i, index_i):
    # Cut the tree from the root node i and only preserve the subtree of the index_i-th action.
    index = self.root_state[i].legal_moves[index_i]
    self.root_state[i].select_move(index)
    if self.root_node[i] != None:
      # Cut the tree if the root node exists.
      if self.root_node[i].children_edge[index_i].children_node != None:
        # And if the target children node also exists.
        self.root_node[i] = self.root_node[i].children_edge[index_i].children_node
        self.root_node[i].parent_edge.parent_node = None
        self.root_node[i].parent_edge = None
      else:
        # Delete the root node if the target children node does not exist.
        self.root_node[i] = None
    self.step[i] = 0
    self.blocked[i] = 0
  
  def remove_node(self, remove_list):
    # Remove nodes.
    self.root_state = np.delete(self.root_state, remove_list, 0).tolist()
    self.root_node = np.delete(self.root_node, remove_list, 0).tolist()
    self.step = np.delete(self.step, remove_list, 0)
    self.blocked = np.delete(self.blocked, remove_list, 0)