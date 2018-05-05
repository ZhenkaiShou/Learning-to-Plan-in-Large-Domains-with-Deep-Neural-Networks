import copy
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import os.path
import pickle
import sys
import tensorflow as tf
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets
from constants import *
from gamestate import *
from mcts import *
from model import *
from util import extract_inputs, transform, get_state_sequence, reshape_feature_sequence, index_to_pos, pos_to_index

class GameWindow(QtWidgets.QMainWindow):
  def __init__(self):
    super(GameWindow, self).__init__()
    self.player_black = HUMAN_PLAYER
    self.player_white = HUMAN_PLAYER
    self.evaluator = BEST_PLAYER
    self.hint = False
    self.start = False
    self.state = None
    self.move_history = None
    self.state_history = None
    
    # Basic configurations.
    self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    self.update_title()
    self.setFixedSize(500, 600)
    
    # Add menubar.
    self.menu = QtWidgets.QMenu("Menu", self)
    # New game action.
    self.new_game_action = QtWidgets.QAction("New Game", self.menu)
    self.new_game_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_N)
    self.new_game_action.triggered.connect(self.new_game_handler)
    # Edit players action.
    self.players_action = QtWidgets.QAction("Players", self.menu)
    self.players_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_P)
    self.players_action.triggered.connect(self.players_handler)
    # Quit action.
    self.quit_action = QtWidgets.QAction("Quit", self.menu)
    self.quit_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
    self.quit_action.triggered.connect(self.quit_handler)
    # Add actions to the menubar.
    self.menu.addAction(self.new_game_action)
    self.menu.addAction(self.players_action)
    self.menu.addAction(self.quit_action)
    self.menuBar().addMenu(self.menu)
    
    # Add optionbar.
    self.option = QtWidgets.QMenu("Options", self)
    # Undo action.
    self.undo_action = QtWidgets.QAction("Undo", self.option)
    self.undo_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_Z)
    self.undo_action.triggered.connect(self.undo_handler)
    # Hint action.
    self.hint_action = QtWidgets.QAction("Hint", self.option)
    self.hint_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_H)
    self.hint_action.triggered.connect(self.hint_handler)
    # Start action.
    self.start_action = QtWidgets.QAction("Start", self.option)
    self.start_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_S)
    self.start_action.triggered.connect(self.start_handler)
    # Evaluate action.
    self.eval_action = QtWidgets.QAction("Evaluate", self.option)
    self.eval_action.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_E)
    self.eval_action.triggered.connect(self.eval_handler)
    # Add actions to the optionbar.
    self.option.addAction(self.undo_action)
    self.option.addAction(self.hint_action)
    self.option.addAction(self.start_action)
    self.option.addAction(self.eval_action)
    self.menuBar().addMenu(self.option)
    
    # Add main widget.
    self.main_widget = QtWidgets.QWidget(self)
    
    # Add buttonbar.
    self.button_widget = QtWidgets.QWidget(self.main_widget)
    # Undo button.
    self.undo_button = QtWidgets.QPushButton("Undo (Z)", self.button_widget)
    self.undo_button.setFixedSize(100, 30)
    self.undo_button.setToolTip("Undo the last move.")
    self.undo_button.clicked.connect(self.undo_handler)
    # Hint button.
    self.hint_button = QtWidgets.QPushButton("Hint (H)", self.button_widget)
    self.hint_button.setFixedSize(100, 30)
    self.hint_button.setToolTip("Show legal moves.")
    self.hint_button.clicked.connect(self.hint_handler)
    # Start button.
    self.start_button = QtWidgets.QPushButton("Start (S)", self.button_widget)
    self.start_button.setFixedSize(100, 30)
    self.start_button.setToolTip("Start the bot player(s).")
    self.start_button.clicked.connect(self.start_handler)
    # Evaluate button.
    self.eval_button = QtWidgets.QPushButton("Evaluate (E)", self.button_widget)
    self.eval_button.setFixedSize(100, 30)
    self.eval_button.setToolTip("Evaluate the current state.")
    self.eval_button.clicked.connect(self.eval_handler)
    # Add buttons to the buttonbar.
    button_layout = QtWidgets.QHBoxLayout(self.button_widget)
    button_layout.addWidget(self.undo_button)
    button_layout.addWidget(self.hint_button)
    button_layout.addWidget(self.start_button)
    button_layout.addWidget(self.eval_button)

    # Create game canvas.
    self.game_canvas = GameCanvas(self.main_widget, self)

    # Add contents.
    main_layout = QtWidgets.QVBoxLayout(self.main_widget)
    main_layout.addWidget(self.button_widget)
    main_layout.addWidget(self.game_canvas)
    main_layout.setContentsMargins(0, 0, 0, 0)
    
    # Add pass button.
    self.pass_button = QtWidgets.QPushButton("Pass", self.main_widget)
    self.pass_button.setGeometry(225, 540, 50, 30)
    self.pass_button.clicked.connect(self.pass_button_handler)
    
    # Update buttons.
    self.update_buttons()
    
    self.main_widget.setFocus()
    self.setCentralWidget(self.main_widget)
  
  def new_game_handler(self):
    # Update and store state.
    self.state = GameState()
    self.move_history = []
    self.state_history = []
    self.state_history.append(copy.deepcopy(self.state))
    if self.player_black != HUMAN_PLAYER:
      self.mcts_black = MCTS([self.state], self.player_black, None)
    else:
      self.mcts_black = None
    if self.player_white != HUMAN_PLAYER:
      self.mcts_white = MCTS([self.state], self.player_white, None)
    else:
      self.mcts_white = None
    
    # Update game board.
    self.game_canvas.draw_gameboard()
    
    # Update title.
    self.update_title()
    
    # Update buttons.
    if self.start:
      self.start_action.trigger()
    self.update_buttons()
  
  def players_handler(self):
    edit_players_dialog = EditPlayersDialog([self.player_black, self.player_white, self.evaluator])
    if edit_players_dialog.exec_() == QtWidgets.QDialog.Accepted:
      value = edit_players_dialog.get_value()
      player_black_old = self.player_black
      player_white_old = self.player_white
      self.player_black = value[0]
      self.player_white = value[1]
      self.evaluator = value[2]
      if self.state != None:
        # Update MCTS states.
        if self.player_black != HUMAN_PLAYER:
          if self.player_black != player_black_old:
            self.mcts_black = MCTS([self.state], self.player_black, None)
        else:
          self.mcts_black = None
        if self.player_white != HUMAN_PLAYER:
          if self.player_white != player_white_old:
            self.mcts_white = MCTS([self.state], self.player_white, None)
        else:
          self.mcts_white = None
        
        # Update title.
        self.update_title()
        
        # Update buttons.
        if self.player_black != player_black_old or self.player_white != player_white_old:
          if self.start:
            self.start_action.trigger()
        self.update_buttons()
  
  def quit_handler(self):
    self.close()
  
  def pass_button_handler(self):
    # Update and store state.
    index = pos_to_index(PASS)
    index_i = np.argwhere(np.equal(self.state.legal_moves, index))[0][0]
    self.move_history.append(index)
    self.state_history.append(copy.deepcopy(self.state.select_move(index)))
    if self.mcts_black != None:
      self.mcts_black.get_subtree(0, index_i)
    if self.mcts_white != None:
      self.mcts_white.get_subtree(0, index_i)
    
    # Update game board.
    self.game_canvas.draw_gameboard()
    
    # Update buttons.
    self.update_buttons()
    
    # Run bots.
    if not self.state.game_ends and self.start:
      if self.state.is_black:
        if self.player_black != HUMAN_PLAYER:
          t = threading.Thread(target = self.run_bot, args = (BLACK,))
          t.start()
      else:
        if self.player_white != HUMAN_PLAYER:
          t = threading.Thread(target = self.run_bot, args = (WHITE,))
          t.start()
  
  def undo_handler(self):
    # Remove and restore state.
    self.move_history.pop()
    self.state_history.pop()
    self.state = copy.deepcopy(self.state_history[-1])
    if self.mcts_black != None:
      self.mcts_black = MCTS([self.state], self.player_black, None)
    if self.mcts_white != None:
      self.mcts_white = MCTS([self.state], self.player_white, None)
    
    # Update game board.
    self.game_canvas.draw_gameboard()
    
    # Update buttons.
    if self.start:
      self.start_action.trigger()
    self.update_buttons()
  
  def hint_handler(self):
    if not self.hint:
      self.hint = True
      self.hint_action.setText("Hide")
      self.hint_button.setText("Hide (H)")
      self.hint_button.setToolTip("Hide legal moves.")
    else:
      self.hint = False
      self.hint_action.setText("Hint")
      self.hint_button.setText("Hint (H)")
      self.hint_button.setToolTip("Show legal moves.")
    if self.state is not None:
      self.game_canvas.draw_gameboard()
  
  def start_handler(self):
    # Update buttons.
    if not self.start:
      self.start = True
      self.start_action.setText("Stop")
      self.start_button.setText("Stop (S)")
      self.start_button.setToolTip("Stop the bot player(s).")
    else:
      self.start = False
      self.start_action.setText("Start")
      self.start_button.setText("Start (S)")
      self.start_button.setToolTip("Start the bot player(s).")
    self.update_buttons()
    
    # Run bots.
    if not self.state.game_ends and self.start:
      if self.state.is_black:
        if self.player_black != HUMAN_PLAYER:
          t = threading.Thread(target = self.run_bot, args = (BLACK,))
          t.start()
      else:
        if self.player_white != HUMAN_PLAYER:
          t = threading.Thread(target = self.run_bot, args = (WHITE,))
          t.start()
  
  def eval_handler(self):
    # Evaluate the probability distribution.
    mcts = MCTS([self.state], self.evaluator, None)
    pi = mcts.tree_search()[0]
    # Get the predicted action sequence.
    a = mcts.get_action_sequence()
      
    # Evaluate the value.
    model = Model(enhance = True, imitate = True, optimize = 0)
    with tf.Session() as sess:
      saver = tf.train.Saver()
      # Restore the player.
      if self.evaluator == BEST_PLAYER:
        file_name = "BestPlayer"
      else:
        file_name = "Player_" + format(self.evaluator, "05d")
      saver.restore(sess, PLAYER_DIR + file_name)
      
      # Transform the action sequence into state sequence.
      state_sequence, sequence_length, sequence_index = get_state_sequence([self.state], a)
      
      # Randomly transform the state.
      n = np.random.choice(SYMMETRY)
      transformed_state = copy.deepcopy(self.state)
      transformed_state = transform(transformed_state, n)
      # Update the state sequence.
      state_sequence = [transform(x, n) for x in state_sequence]
      # Extract inputs from state.
      inputs = extract_inputs([transformed_state])
      inputs_sequence = extract_inputs(state_sequence)
      
      # Compute the feature sequence.
      feature_sequence = sess.run(model.x, feed_dict = {model.Inputs: inputs_sequence})
      feature_sequence = reshape_feature_sequence(feature_sequence, sequence_length, sequence_index)
      
      # Compute the value and policy.
      feed_dict = {model.Inputs: inputs, model.Feature_Sequence: feature_sequence, model.Sequence_Length: sequence_length}
      target = [model.v, model.p, model.v_prime, model.p_prime, model.v_hat, model.p_hat]
      v, p, v_prime, p_prime, v_hat, p_hat = sess.run(target, feed_dict = feed_dict)
      v = v[0, 0]
      p = normalize_probability(p, [transformed_state.legal_moves])
      p = p[0]
      v_prime = v_prime[0, 0]
      p_prime = normalize_probability(p_prime, [transformed_state.legal_moves])
      p_prime = p_prime[0]
      v_hat = v_hat[0, 0]
      p_hat = normalize_probability(p_hat, [transformed_state.legal_moves])
      p_hat = p_hat[0]
      
      print("V     = " + str(v))
      print("V'    = " + str(v_prime))
      print("V_hat = " + str(v_hat))
      print("P     = " + str(p))
      print("P'    = " + str(p_prime))
      print("P_hat = " + str(p_hat))
      print("Pi    = " + str(pi))
      print("===================")
    tf.contrib.keras.backend.clear_session()
    self.game_canvas.draw_gameboard(pi, v)
  
  def update_title(self):
    if self.state == None:
      self.setWindowTitle("Othello")
    else:
      if self.player_black == HUMAN_PLAYER:
        black_name = "Human"
      elif self.player_black == BEST_PLAYER:
        black_name = "Best Bot"
      else:
        black_name = "Bot " + str(self.player_black)
      if self.player_white == HUMAN_PLAYER:
        white_name = "Human"
      elif self.player_white == BEST_PLAYER:
        white_name = "Best Bot"
      else:
        white_name = "Bot " + str(self.player_white)
      self.setWindowTitle("Othello " + black_name + "[B] vs. " + white_name + "[W]")
  
  def update_buttons(self):
    # Pass button.
    if self.state != None and (not self.start or self.start and (self.player_black == HUMAN_PLAYER if self.state.is_black else self.player_white == HUMAN_PLAYER)):
      if not self.state.game_ends and self.state.legal_moves[0] == pos_to_index(PASS):
        self.pass_button.setEnabled(True)
      else:
        self.pass_button.setEnabled(False)
    else:
      self.pass_button.setEnabled(False)
    
    # Undo button.
    if self.state != None and self.state.step > 0:
      self.undo_action.setEnabled(True)
      self.undo_button.setEnabled(True)
    else:
      self.undo_action.setEnabled(False)
      self.undo_button.setEnabled(False)
    
    # Hint button.
    if self.state != None:
      self.hint_action.setEnabled(True)
      self.hint_button.setEnabled(True)
    else:
      self.hint_action.setEnabled(False)
      self.hint_button.setEnabled(False)
    
    # Evaluate button.
    if self.state != None and not self.state.game_ends:
      self.eval_action.setEnabled(True)
      self.eval_button.setEnabled(True)
    else:
      self.eval_action.setEnabled(False)
      self.eval_button.setEnabled(False)
    
    # Start button.
    if self.state != None and not self.state.game_ends and (self.player_black != HUMAN_PLAYER or self.player_white != HUMAN_PLAYER):
      self.start_action.setEnabled(True)
      self.start_button.setEnabled(True)
    else:
      if self.start:
        self.start_action.trigger()
      self.start_action.setEnabled(False)
      self.start_button.setEnabled(False)
  
  def run_bot(self, color):
    mcts = self.mcts_black if color == BLACK else self.mcts_white
    # Run the Monte-Carlo Tree Search for the current player.
    pi = mcts.tree_search()[0]
    
    arg_max = np.argwhere(pi == np.amax(pi))
    pi = np.zeros_like(pi)
    pi[arg_max] = 1 / len(arg_max)
    index_i = np.random.choice(len(self.state.legal_moves), p = pi)
    index = self.state.legal_moves[index_i]
    
    # Update and store states.
    self.move_history.append(index)
    self.state_history.append(copy.deepcopy(self.state.select_move(index)))          
    if self.mcts_black != None:
      self.mcts_black.get_subtree(0, index_i)
    if self.mcts_white != None:
      self.mcts_white.get_subtree(0, index_i)
    
    # Update game board.
    self.game_canvas.draw_gameboard()
    
    # Update buttons.
    self.update_buttons()
    
    # Run bots.
    if not self.state.game_ends and self.start:
      if self.state.is_black:
        if self.player_black != HUMAN_PLAYER:
          t = threading.Thread(target = self.run_bot, args = (BLACK,))
          t.start()
      else:
        if self.player_white != HUMAN_PLAYER:
          t = threading.Thread(target = self.run_bot, args = (WHITE,))
          t.start()
  
class GameCanvas(FigureCanvas):
  def __init__(self, parent, window):
    self.window = window
    fig = Figure(figsize = (5, 5), dpi = 100, facecolor = (1, 1, 0.8))
    # Draw the empty game board.
    self.ax = fig.add_subplot(111)
    self.ax.set_position([0.1, 0.1, 0.8, 0.8])
    self.ax.set_facecolor("none")
    self.ax.set_xticks(range(BOARD_SIZE_X + 1))
    self.ax.set_yticks(range(BOARD_SIZE_Y + 1))
    self.ax.grid(color = "k", linestyle = "-", linewidth = 1)
    self.ax.xaxis.set_tick_params(bottom = "off", top = "off", labelbottom = "off")
    self.ax.yaxis.set_tick_params(left = "off", right = "off", labelleft = "off")
    self.ax.invert_yaxis()
    
    super(GameCanvas, self).__init__(fig)
    self.mpl_connect("button_press_event", self.onclick)
    self.setParent(parent)
    FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
    FigureCanvas.updateGeometry(self)
    
  def draw_gameboard(self, pi = None, v = None):
    self.ax.cla()
    # Draw the empty game board.
    self.ax.set_xticks(range(BOARD_SIZE_X + 1))
    self.ax.set_yticks(range(BOARD_SIZE_Y + 1))
    self.ax.grid(color = "k", linestyle = "-", linewidth = 1)
    self.ax.xaxis.set_tick_params(bottom = "off", top = "off", labelbottom = "off")
    self.ax.yaxis.set_tick_params(left = "off", right = "off", labelleft = "off")
    self.ax.invert_yaxis()
    
    # Create black stones and white stones.
    black_stone = mpatches.Circle((0, 0), 0.45, facecolor = (0, 0, 0, 1), edgecolor = (0, 0, 0, 1), linewidth = 1, clip_on = False, zorder = 10)
    white_stone = mpatches.Circle((0, 0), 0.45, facecolor = (0.9, 0.9, 0.9, 1), edgecolor = (0, 0, 0, 1), linewidth = 1, clip_on = False, zorder = 10)
    black_stone_shallow = mpatches.Circle((0, 0), 0.45, facecolor = (0, 0, 0, 0.5), edgecolor = (0, 0, 0, 0.5), linestyle = "--", linewidth = 1, clip_on = False, zorder = 10)
    white_stone_shallow = mpatches.Circle((0, 0), 0.45, facecolor = (0.9, 0.9, 0.9, 0.5), edgecolor = (0, 0, 0, 0.5), linestyle = "--", linewidth = 1, clip_on = False, zorder = 10)
    black_stone_highlight = mpatches.Circle((0, 0), 0.25, facecolor = (0, 0, 0, 1), edgecolor = (0.9, 0.9, 0.9, 1), fill = False, linewidth = 2, clip_on = False, zorder = 20)
    white_stone_highlight = mpatches.Circle((0, 0), 0.25, facecolor = (0, 0, 0, 1), edgecolor = (0, 0, 0, 1), fill = False, linewidth = 2, clip_on = False, zorder = 20)
    
    # Get positions of black stones and white stones.
    blacks = np.argwhere(np.equal(self.window.state.board, BLACK))
    whites = np.argwhere(np.equal(self.window.state.board, WHITE))
    
    # Add stones to the game board.
    for x in blacks:
      if x[0] != PASS[0]:
        black_copy = copy.copy(black_stone)
        black_copy.center = np.array([x[1], x[0]]) + 0.5
        self.ax.add_patch(black_copy)
        if not self.window.state.is_black and len(self.window.move_history) > 0:
          if x[0] == index_to_pos(self.window.move_history[-1])[0] and x[1] == index_to_pos(self.window.move_history[-1])[1]:
            # Highlight the last black stone.
            black_copy = copy.copy(black_stone_highlight)
            black_copy.center = np.array([x[1], x[0]]) + 0.5
            self.ax.add_patch(black_copy)
    
    for x in whites:
      if x[0] != PASS[0]:
        white_copy = copy.copy(white_stone)
        white_copy.center = np.array([x[1], x[0]]) + 0.5
        self.ax.add_patch(white_copy)
        if self.window.state.is_black and len(self.window.move_history) > 0:
          if x[0] == index_to_pos(self.window.move_history[-1])[0] and x[1] == index_to_pos(self.window.move_history[-1])[1]:
            # Highlight the last white stone.
            white_copy = copy.copy(white_stone_highlight)
            white_copy.center = np.array([x[1], x[0]]) + 0.5
            self.ax.add_patch(white_copy)
    
    if self.window.hint or pi is not None:
      for i in range(len(self.window.state.legal_moves)):
        pos = np.array(index_to_pos(self.window.state.legal_moves[i]))
        if pos[0] != PASS[0]:
          if self.window.state.is_black:
            # Show legal moves.
            black_copy = copy.copy(black_stone_shallow)
            black_copy.center = np.array([pos[1], pos[0]]) + 0.5
            self.ax.add_patch(black_copy)
            if pi is not None:
              # Add predictions.
              self.ax.text(pos[1]+0.5, pos[0]+0.5, str(int(pi[i]*100)), horizontalalignment="center", verticalalignment="center", fontsize=12)
          else:
            white_copy = copy.copy(white_stone_shallow)
            white_copy.center = np.array([pos[1], pos[0]]) + 0.5
            self.ax.add_patch(white_copy)
            if pi is not None:
              self.ax.text(pos[1]+0.5, pos[0]+0.5, str(int(pi[i]*100)), horizontalalignment="center", verticalalignment="center", fontsize=12)
    
    # Add description.
    step_desc = "Step " + format(self.window.state.step, "02d") + ", "
    if self.window.state.game_ends:
      if self.window.state.score[0] == 1:
        color_desc = "Black wins, "
      elif self.window.state.score[0] == 0:
        color_desc = "Draw, "
      else:
        color_desc = "White wins, "
    else:
      if self.window.state.is_black:
        color_desc = "Black, "
      else:
        color_desc = "White, "
    if v != None:
      stone_desc = "B: " + format(len(blacks), "02d") + ", W: " + format(len(whites), "02d") + ", "
      value_desc = "VN = " + format((v + 1) / 2 * 100, ".2f") + "%. "
      self.ax.set_title(step_desc + color_desc + stone_desc + value_desc)
    else:
      stone_desc = "B: " + format(len(blacks), "02d") + ", W: " + format(len(whites), "02d") + "."
      self.ax.set_title(step_desc + color_desc + stone_desc)
    
    self.draw()
  
  def onclick(self, event):
    if event.inaxes == self.ax:
      if not self.window.start or self.window.start and (self.window.player_black == HUMAN_PLAYER if self.window.state.is_black else self.window.player_white == HUMAN_PLAYER):
        if self.window.state != None:
          index = pos_to_index([int(event.ydata), int(event.xdata)])
          if index in self.window.state.legal_moves:
            # Update and store states.
            index_i = np.argwhere(np.equal(self.window.state.legal_moves, index))[0][0]
            self.window.move_history.append(index)
            self.window.state_history.append(copy.deepcopy(self.window.state.select_move(index)))          
            if self.window.mcts_black != None:
              self.window.mcts_black.get_subtree(0, index_i)
            if self.window.mcts_white != None:
              self.window.mcts_white.get_subtree(0, index_i)
            
            #get_activations(self.window.state)
            
            # Update game board.
            self.draw_gameboard()
            
            # Update buttons.
            self.window.update_buttons()
            
            # Run bots.
            if not self.window.state.game_ends and self.window.start:
              if self.window.state.is_black:
                if self.window.player_black != HUMAN_PLAYER:
                  t = threading.Thread(target = self.window.run_bot, args = (BLACK,))
                  t.start()
              else:
                if self.window.player_white != HUMAN_PLAYER:
                  t = threading.Thread(target = self.window.run_bot, args = (WHITE,))
                  t.start()

class EditPlayersDialog(QtWidgets.QDialog):
  def __init__(self, value):
    super(EditPlayersDialog, self).__init__()
    self.setFixedSize(450, 200)
    self.setWindowTitle("Players")
    
    # Add widgets.
    self.player_black_widget = EditPlayerWidget("Black:", value[0], True)
    self.player_white_widget = EditPlayerWidget("White:", value[1], True)
    self.evaluator_widget = EditPlayerWidget("Evaluator:", value[2], False)
    layout = QtWidgets.QVBoxLayout(self)
    layout.addWidget(self.player_black_widget)
    layout.addWidget(self.player_white_widget)
    layout.addWidget(self.evaluator_widget)
    
    # Add button box.
    buttonbox = QtWidgets.QDialogButtonBox(self)
    buttonbox.addButton(QtWidgets.QDialogButtonBox.Ok)
    buttonbox.addButton(QtWidgets.QDialogButtonBox.Cancel)
    buttonbox.setOrientation(QtCore.Qt.Horizontal)
    buttonbox.accepted.connect(self.accept)
    buttonbox.rejected.connect(self.reject)
    layout.addWidget(buttonbox)
  
  def get_value(self):
    return [self.player_black_widget.value, self.player_white_widget.value, self.evaluator_widget.value]

class EditPlayerWidget(QtWidgets.QWidget):
  def __init__(self, tag, value, enable_human_player):
    super(EditPlayerWidget, self).__init__()
    self.value = value
    
    # Add radio buttons.
    layout = QtWidgets.QHBoxLayout(self)
    self.label = QtWidgets.QLabel(tag)
    self.radio_humanplayer = QtWidgets.QRadioButton("Human")
    self.radio_humanplayer.toggled.connect(self.radio_humanplayer_handler)
    if not enable_human_player:
      self.radio_humanplayer.setEnabled(False)
      self.radio_humanplayer.setToolTip("Only for bots.")
    else:
      self.radio_humanplayer.setToolTip("Select human player.")
    self.radio_bestplayer = QtWidgets.QRadioButton("Best Bot")
    self.radio_bestplayer.toggled.connect(self.radio_bestplayer_handler)
    self.radio_bestplayer.setToolTip("Select the best bot player.")
    self.radio_customplayer = QtWidgets.QRadioButton("Custom Bot")
    self.radio_customplayer.toggled.connect(self.radio_customplayer_handler)
    self.radio_customplayer.setToolTip("Select a specific bot. Fill in the bot id (0~99999).")
    self.customplayer_edit = QtWidgets.QLineEdit()
    self.customplayer_edit.setFixedWidth(50)
    self.customplayer_edit.setValidator(QtGui.QIntValidator(0, 99999))
    self.customplayer_edit.textChanged.connect(self.text_changed_handler)
    self.customplayer_edit.setEnabled(False)
    layout.addWidget(self.label)
    layout.addWidget(self.radio_humanplayer)
    layout.addWidget(self.radio_bestplayer)
    layout.addWidget(self.radio_customplayer)
    layout.addWidget(self.customplayer_edit)
    
    # Set initial value.
    if self.value == HUMAN_PLAYER:
      self.radio_humanplayer.setChecked(True)
    elif self.value == BEST_PLAYER:
      self.radio_bestplayer.setChecked(True)
    else:
      self.radio_customplayer.setChecked(True)
      self.customplayer_edit.setText(str(self.value))
  
  def radio_humanplayer_handler(self):
    if self.radio_humanplayer.isChecked():
      self.value = HUMAN_PLAYER
  
  def radio_bestplayer_handler(self):
    if self.radio_bestplayer.isChecked():
      self.value = BEST_PLAYER
  
  def radio_customplayer_handler(self):
    if self.radio_customplayer.isChecked():
      self.customplayer_edit.setEnabled(True)
    else:
      self.customplayer_edit.setEnabled(False)
      
  def text_changed_handler(self):
    if self.customplayer_edit.text() != "":
      self.value = int(self.customplayer_edit.text())
    else:
      self.value = 0

if __name__ == "__main__":
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
  qApp = QtWidgets.QApplication(sys.argv)
  gamewindow = GameWindow()
  gamewindow.show()
  sys.exit(qApp.exec_())