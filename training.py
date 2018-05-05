import copy
import csv
import glob
import multiprocessing
import numpy as np
import os
import os.path
import time
import tqdm
import pickle
from constants import *
from mcts import *
from model import *
from util import extract_inputs, transform, get_state_sequence, get_child_state, reshape_feature_sequence, normalize_probability

def training(p):
  player = p
  # Create folders.
  if not os.path.isdir(PLAYER_DIR):
    os.makedirs(PLAYER_DIR)
  if not os.path.isdir(DATA_DIR):
    os.makedirs(DATA_DIR)
  if not os.path.isdir(TEMPDATA_DIR):
    os.makedirs(TEMPDATA_DIR)
  if not os.path.isdir(RECORD_DIR):
    os.makedirs(RECORD_DIR)
  # Create empty General Statistics file.
  if not os.path.isfile(RECORD_DIR + "General Statistics.csv"):
    with open(RECORD_DIR + "General Statistics.csv", "w") as f:
      fieldnames = ["Player", "Best Player", "Vs. Best Player", "Value Loss 1", "Policy Loss1", "Total Loss", "Value Loss 2", "Policy Loss 2", "Feature Loss", "Value Loss 3", "Policy Loss 3"]
      writer = csv.DictWriter(f, fieldnames = fieldnames, lineterminator = "\n")
      writer.writeheader()
  
  while True:
    # Checkpoint for the current best player.
    if player == 0:
      # Set the initial player as the current best player.
      initialize_player()
    else:
      # Hold a tournament between the current best player and the latest player.
      tournament(player)
    # Generate selfplay games from the current best player.
    selfplay(player)
    # Optimize the current neural network.
    optimize_player(player)
    # Delete old selfplay games.
    delete_data()
    player += 1

def initialize_player():
  # Use multiprocessing to avoid hanging of tensorflow.
  with multiprocessing.Pool(1) as p:
    p.starmap(initialize_player_process, [[]])
  p.join()

def initialize_player_process():
  import tensorflow as tf
  # Select the initial player as the current best player.
  model = Model(enhance = True, imitate = True, optimize = -1)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Save the initial player.
    saver = tf.train.Saver()
    file_name = "BestPlayer"
    saver.save(sess, PLAYER_DIR + file_name)
    file_name = "Player_" + format(0, "05d")
    saver.save(sess, PLAYER_DIR + file_name)
  tf.contrib.keras.backend.clear_session()
  
  # Update General Statistics file.
  with open(RECORD_DIR + "General Statistics.csv", "a") as f:
    fieldnames = ["Player", "Best Player", "Vs. Best Player", "Value Loss 1", "Policy Loss1", "Total Loss", "Value Loss 2", "Policy Loss 2", "Feature Loss", "Value Loss 3", "Policy Loss 3"]
    writer = csv.DictWriter(f, fieldnames = fieldnames, lineterminator = "\n")
    writer.writerow({"Player": 0, "Best Player": 0, "Vs. Best Player": 0.5})
  print("Initial Player 0 generated.")

def tournament(player):
  # Arrange number of playing black and white for each process.
  base = TOURNAMENT_GAMES // NUM_PROCESSING
  reminder = TOURNAMENT_GAMES % NUM_PROCESSING
  arg_list = [None for _ in range(NUM_PROCESSING)]
  for i in range(NUM_PROCESSING):
    num_games = base + 1 if i < reminder else base
    if num_games % 2 == 0:
      num_blacks = num_games // 2
      num_whites = num_games // 2
    else:
      num_blacks = num_games // 2 + 1 if i % 2 == 0 else num_games // 2
      num_whites = num_games // 2 if i % 2 == 0 else num_games // 2 + 1
    arg_list[i] = [player, num_blacks, num_whites]
  # Start multiprocessing.
  with multiprocessing.Pool(NUM_PROCESSING) as p:
    total_win = np.sum(p.starmap(tournament_process, arg_list))
  # Compute the winning rate.
  winning_rate = total_win / TOURNAMENT_GAMES
  # Select the best player.
  with multiprocessing.Pool(1) as p:
    result = p.starmap(select_best_player_process, [[player, winning_rate]])[0]
  print(result)
  
def tournament_process(player, num_blacks, num_whites):
  # The latest player challenges the current best player.
  total_win = 0
  step = 0
  pbar = tqdm.tqdm(total = num_blacks + num_whites, ncols = 60, unit = "game")
  np.random.seed()
  
  while step < num_blacks + num_whites:
    if step < num_blacks:
      # Play black for the first half of the games.
      play_black = True
      current_player = player
      length = np.minimum(num_blacks - step, GAME_THREADS)
    else:
      # Play white for the second half of the games.
      play_black = False
      current_player = BEST_PLAYER
      length = np.minimum(num_blacks + num_whites - step, GAME_THREADS)
    mcts_latest = MCTS([GameState() for _ in range(length)], player, "low")
    mcts_best = MCTS([GameState() for _ in range(length)], BEST_PLAYER, "low")
    
    while len(mcts_latest.root_state) > 0:
      remove_list = []
      mcts = mcts_latest if current_player == player else mcts_best
      # Run the Monte-Carlo Tree Search for the current player.
      pi = mcts.tree_search()
      for i in range(len(mcts_latest.root_state)):
        # Choose the best action from output policy.
        arg_max = np.argwhere(pi[i] == np.amax(pi[i]))
        pi[i] = np.zeros_like(pi[i])
        pi[i][arg_max] = 1 / len(arg_max)
        index_i = np.random.choice(len(mcts.root_state[i].legal_moves), p = pi[i])
        # Update the root node and root state for both players.
        mcts_latest.get_subtree(i, index_i)
        mcts_best.get_subtree(i, index_i)
        
        if mcts_latest.root_state[i].game_ends:
          # Score the tournament: +1 for win, +0.5 for tie, +0 for loss.
          if play_black:
            if mcts_latest.root_state[i].score[0] == 1:
              total_win += 1
            elif mcts_latest.root_state[i].score[0] == 0:
              total_win += 0.5
          else:
            if mcts_latest.root_state[i].score[1] == 1:
              total_win += 1
            elif mcts_latest.root_state[i].score[1] == 0:
              total_win += 0.5
          # Add the terminal node to the remove list.
          remove_list.append(i)          
          step += 1
      if len(remove_list) > 0:
        # Remove terminal nodes.
        mcts_latest.remove_node(remove_list)
        mcts_best.remove_node(remove_list)
      # Update the current player.
      current_player = BEST_PLAYER if current_player == player else player
    pbar.update(length)
  pbar.close()
  
  return total_win

def select_best_player_process(player, winning_rate):
  import tensorflow as tf
  # The latest player becomes the best player if it wins at least 55% of the tournament games.
  if winning_rate >= WIN_CRITERIA:
    model = Model(enhance = True, imitate = True, optimize = -1)
    with tf.Session() as sess:
      # Restore the latest player.
      saver = tf.train.Saver()
      file_name = "Player_" + format(player, "05d")
      saver.restore(sess, PLAYER_DIR + file_name)
      # Then save it as the current best player.
      file_name = "BestPlayer"
      saver.save(sess, PLAYER_DIR + file_name)
    tf.contrib.keras.backend.clear_session()
    message = "Tournament finished. Player " + str(player) + " won in the tournament with " + format(winning_rate * 100, ".2f") + "% winning rate."
  else:
    message = "Tournament finished. Player " + str(player) + " failed in the tournament with " + format(winning_rate * 100, ".2f") + "% winning rate."
  
  if winning_rate >= WIN_CRITERIA:
    best_player = player
  else:
    # Obtain the best player from record file.
    with open(RECORD_DIR + "General Statistics.csv", "r") as f:
      data = list(csv.reader(f))[-1]
      best_player = data[1]
    
  # Update General Statistics file.
  with open(RECORD_DIR + "General Statistics.csv", "a") as f:
    fieldnames = ["Player", "Best Player", "Vs. Best Player", "Value Loss 1", "Policy Loss1", "Total Loss", "Value Loss 2", "Policy Loss 2", "Feature Loss", "Value Loss 3", "Policy Loss 3"]
    writer = csv.DictWriter(f, fieldnames = fieldnames, lineterminator = "\n")
    writer.writerow({"Player": player, "Best Player": best_player, "Vs. Best Player": winning_rate})
  
  return message

def selfplay(player):
  t0 = time.time()
  # Arrange number of games for each process.
  offset = 0
  base = SELFPLAY_GAMES // NUM_PROCESSING
  reminder = SELFPLAY_GAMES % NUM_PROCESSING
  arg_list = [None for _ in range(NUM_PROCESSING)]
  for i in range(NUM_PROCESSING):
    num_games = base + 1 if i < reminder else base
    arg_list[i] = [player, offset, num_games]
    offset += num_games
  # Start multiprocessing.
  with multiprocessing.Pool(NUM_PROCESSING) as p:
    p.starmap(selfplay_process, arg_list)
  p.join()
  diff = time.time() - t0
  print("Self play finished. Generated " + str(SELFPLAY_GAMES) + " games in " + str(diff) + " seconds.")

def selfplay_process(player, offset, num_games):
  # Generate selfplay games from the current best player.
  data = []
  step = 0
  pbar = tqdm.tqdm(total = num_games, ncols = 60, unit = "game")
  np.random.seed()
  
  while step < num_games:
    length = np.minimum(num_games - step, GAME_THREADS)
    data_list = [[] for _ in range(length)]
    mcts = MCTS([GameState() for _ in range(length)], BEST_PLAYER, "high")
    
    while len(mcts.root_state) > 0:
      remove_list = []
      # Run the Monte-Carlo Tree Search.
      pi = mcts.tree_search()
      # Get the predicted action sequence.
      a = mcts.get_action_sequence()
      
      for i in range(len(mcts.root_state)):
        # Add state, policy and action sequence to the data list.
        data_list[i].append([copy.deepcopy(mcts.root_state[i]), pi[i], None, a[i]])
        if mcts.root_state[i].step >= EXPLORE_STEPS:
          # Policy is greedy for later steps.
          arg_max = np.argwhere(pi[i] == np.amax(pi[i]))
          pi[i] = np.zeros_like(pi[i])
          pi[i][arg_max] = 1 / len(arg_max)
        # Randomly choose action from output policy.
        index_i = np.random.choice(len(mcts.root_state[i].legal_moves), p = pi[i])
        # Update the root node and root state.
        mcts.get_subtree(i, index_i)
        
        if mcts.root_state[i].game_ends:
          # Add score to the data list.
          score = mcts.root_state[i].score
          for j in range(len(data_list[i])):
            data_list[i][j][2] = score[0] if data_list[i][j][0].is_black else score[1]
          # Save the data [state, pi, z, a] of a finished game to file.
          file_name = "TrainingData_" + format(SELFPLAY_GAMES * player + offset + step, "07d") + ".pkl"
          with open(DATA_DIR + file_name, "wb") as fp:
            pickle.dump(data_list[i], fp)
          # Add the terminal node to remove list.
          remove_list.append(i)
          step += 1
      if(len(remove_list) > 0):
        # Remove terminal nodes.
        mcts.remove_node(remove_list)
        data_list = np.delete(data_list, remove_list, 0).tolist()
    pbar.update(length)
  pbar.close()

def optimize_player(player):
  # Prepare training data.
  base = CHECKPOINT_INTERVAL // NUM_PROCESSING
  reminder = CHECKPOINT_INTERVAL % NUM_PROCESSING
  arg_list = [None for _ in range(NUM_PROCESSING)]
  for i in range(NUM_PROCESSING):
    num_minibatches = base + 1 if i < reminder else base
    arg_list[i] = [player, num_minibatches, i]
  # Start multiprocessing.
  with multiprocessing.Pool(NUM_PROCESSING) as p:
    p.starmap(prepare_training_data_process, arg_list)
  p.join()
  # Use multiprocessing to avoid hanging of tensorflow.
  with multiprocessing.Pool(1) as p:
    p.starmap(optimize_player_process, [[player]])
  p.join()
  
  # Prepare evaluation data.
  base = EVALUATION_MINIBATCH // NUM_PROCESSING
  reminder = EVALUATION_MINIBATCH % NUM_PROCESSING
  arg_list = [None for _ in range(NUM_PROCESSING)]
  for i in range(NUM_PROCESSING):
    num_minibatches = base + 1 if i < reminder else base
    arg_list[i] = [player, num_minibatches, i]
  # Start multiprocessing.
  with multiprocessing.Pool(NUM_PROCESSING) as p:
    p.starmap(prepare_training_data_process, arg_list)
  p.join()
  # Use multiprocessing to avoid hanging of tensorflow.
  with multiprocessing.Pool(1) as p:
    p.starmap(evaluate_player_process, [[player]])
  p.join()

def prepare_training_data_process(player, num_minibatches, id):
  state = [[None for _ in range(MINIBATCH_SIZE)] for _ in range(num_minibatches)]
  pi = [[None for _ in range(MINIBATCH_SIZE)] for _ in range(num_minibatches)]
  z = [[None for _ in range(MINIBATCH_SIZE)] for _ in range(num_minibatches)]
  a = [[None for _ in range(MINIBATCH_SIZE)] for _ in range(num_minibatches)]
  inputs = [None for _ in range(num_minibatches)]
  state_sequence = [None for _ in range(num_minibatches)]
  sequence_length = [None for _ in range(num_minibatches)]
  sequence_index = [None for _ in range(num_minibatches)]
  inputs_sequence = [None for _ in range(num_minibatches)]
  np.random.seed()
  
  for i in range(num_minibatches):
    # Randomly select training data from the most recent games.
    length = np.minimum(SELFPLAY_GAMES * (player+1), RECENT_GAMES)
    game_id = np.random.randint(length, size = MINIBATCH_SIZE) + SELFPLAY_GAMES * (player+1) - length
    for j in range(MINIBATCH_SIZE):
      file_name = "TrainingData_" + format(game_id[j], "07d") + ".pkl"
      with open(DATA_DIR + file_name, "rb") as fp:
        data = pickle.load(fp)
        k = np.random.randint(len(data))
        state[i][j] = data[k][0]
        pi[i][j] = data[k][1]
        z[i][j] = [data[k][2]]
        a[i][j] = data[k][3]
    
    # Transform the action sequence into state sequence.
    state_sequence[i], sequence_length[i], sequence_index[i] = get_state_sequence(state[i], a[i])
    
    index = 0
    for j in range(MINIBATCH_SIZE):
      # Randomly transform the state.
      n = np.random.choice(SYMMETRY)
      state[i][j] = transform(state[i][j], n)
      # Update the probability distribution.
      pi_temp = np.zeros(BOARD_SIZE_X*BOARD_SIZE_Y+1, np.float32)
      pi_temp[state[i][j].legal_moves] = pi[i][j]
      pi[i][j] = pi_temp
      # Update the state sequence.
      state_sequence[i][index:index+sequence_length[i][j]] = [transform(state_sequence[i][index+k], n) for k in range(sequence_length[i][j])]
      index += sequence_length[i][j]
    # Extract inputs from state.
    inputs[i] = extract_inputs(state[i])
    inputs_sequence[i] = extract_inputs(state_sequence[i])
  
  # Save the training data to file.
  data = [inputs, pi, z, sequence_length, sequence_index, inputs_sequence]
  file_name = "TempData_" + format(id, "04d") + ".pkl"
  with open(TEMPDATA_DIR + file_name, "wb") as fp:
    pickle.dump(data, fp)

def optimize_player_process(player):
  import tensorflow as tf
  if USE_GPU:
    # Additional configurations for running on GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = VISIBLE_DEVICE_OPTM
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = MEMORY_OPTM)
    gpu_config = tf.ConfigProto(gpu_options = gpu_options)
  else:
    gpu_config = None
  
  # Load training data from files.
  inputs = []
  pi = []
  z = []
  sequence_length = []
  sequence_index = []
  inputs_sequence = []
  for i in range(NUM_PROCESSING):
    file_name = "TempData_" + format(i, "04d") + ".pkl"
    with open(TEMPDATA_DIR + file_name, "rb") as fp:
      data = pickle.load(fp)
    inputs += data[0]
    pi += data[1]
    z += data[2]
    sequence_length += data[3]
    sequence_index += data[4]
    inputs_sequence += data[5]
  
  # Set learning rate according to the training progress.
  if player < LEARNING_RATE_RANGE[0]:
    lr = LEARNING_RATE[0]
  elif player < LEARNING_RATE_RANGE[1]:
    lr = LEARNING_RATE[1]
  else:
    lr = LEARNING_RATE[2]  
  
  # Build model for optimization.
  model_forward = Model(enhance = False, imitate = False, optimize = 0)
  model = Model(enhance = True, imitate = True, optimize = 1)
  with tf.Session(config = gpu_config) as sess:
    # Restore the latest player.
    saver = tf.train.Saver()
    file_name = "Player_" + format(player, "05d")
    saver.restore(sess, PLAYER_DIR + file_name)
    
    # Stage 1: Alpha Zero loss.
    for i in tqdm.tqdm(range(CHECKPOINT_INTERVAL), ncols = 60, unit = "update"):
      # Run the optimization.
      feed_dict = {model.Inputs: inputs[i], model.Pi: pi[i], model.Z: z[i], model.LR: lr}
      sess.run(model.train_op1, feed_dict = feed_dict)

    # Compute the feature sequence.
    feature_sequence = [None for _ in range(CHECKPOINT_INTERVAL)]
    for i in tqdm.tqdm(range(CHECKPOINT_INTERVAL), ncols = 60, unit = "minibatch"):
      feature_sequence[i] = sess.run(model_forward.x, feed_dict = {model_forward.Inputs: inputs_sequence[i]})
      feature_sequence[i] = reshape_feature_sequence(feature_sequence[i], sequence_length[i], sequence_index[i])
    
    # Save the current player.
    saver = tf.train.Saver()
    file_name = "Player_" + format(player+1, "05d")
    saver.save(sess, PLAYER_DIR + file_name)
  tf.contrib.keras.backend.clear_session()
  
  # Update model.
  model_forward = Model(enhance = False, imitate = False, optimize = 0)
  model = Model(enhance = True, imitate = True, optimize = 2)
  with tf.Session(config = gpu_config) as sess:
    # Restore the latest player.
    saver = tf.train.Saver()
    file_name = "Player_" + format(player+1, "05d")
    saver.restore(sess, PLAYER_DIR + file_name)
  
    # Stage 2: LSTM enhancement.
    for i in tqdm.tqdm(range(CHECKPOINT_INTERVAL), ncols = 60, unit = "update"):
      # Run the optimization.
      feed_dict = {model.Inputs: inputs[i], model.Feature_Sequence: feature_sequence[i], model.Sequence_Length: sequence_length[i], model.Pi: pi[i], model.Z: z[i], model.LR: lr}
      sess.run(model.train_op2, feed_dict = feed_dict)
    
    # Save the current player.
    saver = tf.train.Saver()
    file_name = "Player_" + format(player+1, "05d")
    saver.save(sess, PLAYER_DIR + file_name)
  tf.contrib.keras.backend.clear_session()
  
  # Update model.
  model_forward = Model(enhance = False, imitate = False, optimize = 0)
  model = Model(enhance = True, imitate = True, optimize = 3)
  with tf.Session(config = gpu_config) as sess:
    # Restore the latest player.
    saver = tf.train.Saver()
    file_name = "Player_" + format(player+1, "05d")
    saver.restore(sess, PLAYER_DIR + file_name)
    
    # Stage 3: Feature imitation. 
    for i in tqdm.tqdm(range(CHECKPOINT_INTERVAL), ncols = 60, unit = "update"):
      # Run the optimization.
      feed_dict = {model.Inputs: inputs[i], model.Feature_Sequence: feature_sequence[i], model.Sequence_Length: sequence_length[i], model.Pi: pi[i], model.Z: z[i], model.LR: lr}
      sess.run(model.train_op34, feed_dict = feed_dict)
    
    # Save the current player.
    saver = tf.train.Saver()
    file_name = "Player_" + format(player+1, "05d")
    saver.save(sess, PLAYER_DIR + file_name)
  tf.contrib.keras.backend.clear_session()

def evaluate_player_process(player):
  import tensorflow as tf
  if USE_GPU:
    # Additional configurations for running on GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = VISIBLE_DEVICE_OPTM
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = MEMORY_OPTM)
    gpu_config = tf.ConfigProto(gpu_options = gpu_options)
  else:
    gpu_config = None
  
  # Load training data from files.
  inputs = []
  pi = []
  z = []
  sequence_length = []
  sequence_index = []
  inputs_sequence = []
  for i in range(NUM_PROCESSING):
    file_name = "TempData_" + format(i, "04d") + ".pkl"
    with open(TEMPDATA_DIR + file_name, "rb") as fp:
      data = pickle.load(fp)
    inputs += data[0]
    pi += data[1]
    z += data[2]
    sequence_length += data[3]
    sequence_index += data[4]
    inputs_sequence += data[5]
  
  # Build model for evaluation.
  model_forward = Model(enhance = False, imitate = False, optimize = 0)
  model = Model(enhance = True, imitate = True, optimize = -1)
  with tf.Session(config = gpu_config) as sess:
    # Restore the latest player.
    saver = tf.train.Saver()
    file_name = "Player_" + format(player+1, "05d")
    saver.restore(sess, PLAYER_DIR + file_name)
    
    # Evaluate the losses.
    avg_value_loss1 = 0
    avg_policy_loss1 = 0
    avg_total_loss = 0
    avg_value_loss2 = 0
    avg_policy_loss2 = 0
    avg_feature_loss = 0
    avg_value_loss3 = 0
    avg_policy_loss3 = 0
    
    feature_sequence = [None for _ in range(EVALUATION_MINIBATCH)]
    for i in tqdm.tqdm(range(EVALUATION_MINIBATCH), ncols = 60, unit = "evaluation"):
      # Run the evaluation.
      feature_sequence[i] = sess.run(model_forward.x, feed_dict = {model_forward.Inputs: inputs_sequence[i]})
      feature_sequence[i] = reshape_feature_sequence(feature_sequence[i], sequence_length[i], sequence_index[i])
      feed_dict = {model.Inputs: inputs[i], model.Feature_Sequence: feature_sequence[i], model.Sequence_Length: sequence_length[i], model.Pi: pi[i], model.Z: z[i]}
      target = [model.value_loss1, model.policy_loss1, model.loss1, model.value_loss2, model.policy_loss2, model.feature_loss, model.value_loss3, model.policy_loss3]
      value_loss1, policy_loss1, total_loss, value_loss2, policy_loss2, feature_loss, value_loss3, policy_loss3 = sess.run(target, feed_dict = feed_dict)
      
      # Update average loss.
      avg_value_loss1 += value_loss1
      avg_policy_loss1 += policy_loss1
      avg_total_loss += total_loss
      avg_value_loss2 += value_loss2
      avg_policy_loss2 += policy_loss2
      avg_feature_loss += feature_loss
      avg_value_loss3 += value_loss3
      avg_policy_loss3 += policy_loss3
    
    # Compute average loss.
    avg_value_loss1 /= EVALUATION_MINIBATCH
    avg_policy_loss1 /= EVALUATION_MINIBATCH
    avg_total_loss /= EVALUATION_MINIBATCH
    avg_value_loss2 /= EVALUATION_MINIBATCH
    avg_policy_loss2 /= EVALUATION_MINIBATCH
    avg_feature_loss /= EVALUATION_MINIBATCH
    avg_value_loss3 /= EVALUATION_MINIBATCH
    avg_policy_loss3 /= EVALUATION_MINIBATCH
  tf.contrib.keras.backend.clear_session()
  
  # Update General Statistics file.
  with open(RECORD_DIR + "General Statistics.csv", "r") as f:
    data = list(csv.reader(f))
    best_player = data[-1][1]
    winning_rate = data[-1][2]
  with open(RECORD_DIR + "General Statistics.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerows(data[:-1])
  with open(RECORD_DIR + "General Statistics.csv", "a") as f:
    fieldnames = ["Player", "Best Player", "Vs. Best Player", "Value Loss 1", "Policy Loss 1", "Total Loss", "Value Loss 2", "Policy Loss 2", "Feature Loss", "Value Loss 3", "Policy Loss 3"]
    writer = csv.DictWriter(f, fieldnames = fieldnames, lineterminator = "\n")
    content = {"Player": player, "Best Player": best_player, "Vs. Best Player": winning_rate}
    content.update({"Value Loss 1": format(avg_value_loss1, ".8f"), "Policy Loss 1": format(avg_policy_loss1, ".8f"), "Total Loss": format(avg_total_loss, ".8f")})
    content.update({"Value Loss 2": format(avg_value_loss2, ".8f"), "Policy Loss 2": format(avg_policy_loss2, ".8f")})
    content.update({"Feature Loss": format(avg_feature_loss, ".8f"), "Value Loss 3": format(avg_value_loss3, ".8f"), "Policy Loss 3": format(avg_policy_loss3, ".8f")})
    writer.writerow(content)
  
  print("Basic Loss = [" + format(avg_value_loss1, ".8f") + ", " + format(avg_policy_loss1, ".8f") + ", " + format(avg_total_loss, ".8f") + "].")
  print("Enhanced Loss = [" + format(avg_value_loss2, ".8f") + ", " + format(avg_policy_loss2, ".8f") + "].")
  print("Imitated Loss = [" + format(avg_feature_loss, ".8f") + ", " + format(avg_value_loss3, ".8f") + ", " + format(avg_policy_loss3, ".8f") + "].")

def delete_data():
  # Delete old selfplay games.
  filename = "TrainingData_*.pkl"
  pattern = DATA_DIR + filename
  files = list(sorted(glob.glob(pattern)))
  if len(files) > RECENT_GAMES:
    for i in range(len(files) - RECENT_GAMES):
      os.remove(files[i])