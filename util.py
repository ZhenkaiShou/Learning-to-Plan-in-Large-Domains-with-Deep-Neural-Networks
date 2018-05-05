import copy
import numpy as np
from constants import *

def extract_inputs(state):
  # Extract inputs from game.
  length = len(state)
  board = [x.board for x in state]
  is_black = [x.is_black for x in state]
  
  # Input feature contains 4 channels: the current player channel, the opponent channel, "if black plays" channel, and "if white plays" channel.
  player = [get_channel(board[i], is_black[i], True) for i in range(length)]
  player = np.reshape(player, [-1, BOARD_SIZE_Y, BOARD_SIZE_X, 1])
  opponent = [get_channel(board[i], is_black[i], False) for i in range(length)]
  opponent = np.reshape(opponent, [-1, BOARD_SIZE_Y, BOARD_SIZE_X, 1])
  black_plays = [get_action_channel(is_black[i], True) for i in range(length)]
  white_plays = [get_action_channel(is_black[i], False) for i in range(length)]
  inputs = np.concatenate((player, opponent, black_plays, white_plays), 3)
  inputs.astype(np.float32)
  
  return inputs

def get_channel(board, is_black, is_player):
  channel = np.zeros((BOARD_SIZE_Y, BOARD_SIZE_X))
  if is_player:
    # Return the player channel.
    channel[np.where(np.equal(board, get_color(is_black, True)))] = 1
  else:
    # Return the opponent channel.
    channel[np.where(np.equal(board, get_color(is_black, False)))] = 1
  
  return channel

def get_color(is_black, is_player):
  if is_player:
    # Return the color of the player.
    color = BLACK if is_black else WHITE
  else:
    # Return the color of the opponent.
    color = WHITE if is_black else BLACK
  
  return color

def get_action_channel(is_black, is_black_plays_channel):
  if is_black_plays_channel:
    # Return the "if black plays" channel.
    channel = np.ones((BOARD_SIZE_Y, BOARD_SIZE_X, 1)) if is_black else np.zeros((BOARD_SIZE_Y, BOARD_SIZE_X, 1))
  else:
    # Return the "if white plays" channel.
    channel = np.zeros((BOARD_SIZE_Y, BOARD_SIZE_X, 1)) if is_black else np.ones((BOARD_SIZE_Y, BOARD_SIZE_X, 1))
  
  return channel

def normalize_probability(p_in, legal_moves):
  # Normalize the probability to the legal move set.
  p_out = [p_in[i, legal_moves[i]] for i in range(len(legal_moves))]
  p_out = [np.divide(x, np.sum(x)) for x in p_out]
  
  return p_out

def get_state_sequence(state, a):
  # Transform the action sequence into state sequence.
  length = len(state)
  state_sequence = []
  sequence_length = np.zeros(length, dtype = np.int32)
  sequence_index = np.zeros(length - 1, dtype = np.int32)
  index = 0
  for i in range(length):
    s = copy.deepcopy(state[i])
    sequence = [copy.deepcopy(s)] + [copy.deepcopy(s.select_move(x)) for x in a[i]]
    sequence.reverse()
    sequence_length[i] = len(sequence)
    if i < length - 1:
      index += len(sequence)
      sequence_index[i] = index
    state_sequence.extend(sequence)
  
  return state_sequence, sequence_length, sequence_index

def get_child_state(state):
  # Get child state.
  length = len(state)
  legal_moves = [x.legal_moves for x in state]
  child_state = [copy.deepcopy(state[i]).select_move(x) for i in range(length) for x in legal_moves[i]]
  total_length = len(child_state)
  index = [np.array(legal_moves[i]) + (BOARD_SIZE_Y*BOARD_SIZE_X+1) * i for i in range(length)]
  index = np.array([y for x in index for y in x])
  child_index = total_length * np.ones([(BOARD_SIZE_Y*BOARD_SIZE_X+1) * length])
  child_index[index] = range(total_length)

  return child_state, child_index
  
def reshape_feature_sequence(feature_sequence, sequence_length, sequence_index):
  # Split and pad the feature sequence into an array.
  feature_sequence = np.split(feature_sequence, sequence_index, 0)
  padding_length = np.max(sequence_length) - sequence_length
  feature_sequence = [np.pad(feature_sequence[i], [[0, padding_length[i]], [0, 0], [0, 0], [0, 0]], "constant") for i in range(len(sequence_length))]
  feature_sequence = np.asarray(feature_sequence)
  
  return feature_sequence
  
def transform(state, i):
  # Transform the game board and legal moves.
  # i = 0: original
  # i = 1: rotate 90
  # i = 2: rotate 180
  # i = 3: rotate 270
  # i = 4: flip in x direction
  # i = 5: flip in y direction
  # i = 6: transpose
  # i = 7: transpose over the secondary diagonal
  if i == 0:
    pass
  else:
    length = len(state.legal_moves)
    position = [index_to_pos(state.legal_moves[j]) for j in range(length)]
    if i == 1:
      state.board = np.rot90(state.board, 1)
      position = [[BOARD_SIZE_X-position[j][1]-1, position[j][0]] if position[j][0] != PASS[0] else position[j] for j in range(length)]
    elif i == 2:
      state.board = np.rot90(state.board, 2)
      position = [[BOARD_SIZE_Y-position[j][0]-1, BOARD_SIZE_X-position[j][1]-1] if position[j][0] != PASS[0] else position[j] for j in range(length)]
    elif i == 3:
      state.board = np.rot90(state.board, 3)
      position = [[position[j][1], BOARD_SIZE_Y-position[j][0]-1] if position[j][0] != PASS[0] else position[j] for j in range(length)]
    elif i == 4:
      state.board = np.fliplr(state.board)
      position = [[position[j][0], BOARD_SIZE_X-position[j][1]-1] if position[j][0] != PASS[0] else position[j] for j in range(length)]
    elif i == 5:
      state.board = np.flipud(state.board)
      position = [[BOARD_SIZE_Y-position[j][0]-1, position[j][1]] if position[j][0] != PASS[0] else position[j] for j in range(length)]
    elif i == 6:
      state.board = np.transpose(state.board, [1, 0])
      position = [[position[j][1], position[j][0]] if position[j][0] != PASS[0] else position[j] for j in range(length)]
    elif i == 7:
      state.board = np.rot90(np.flipud(state.board), 1)
      position = [[BOARD_SIZE_X-position[j][1]-1, BOARD_SIZE_Y-position[j][0]-1] if position[j][0] != PASS[0] else position[j] for j in range(length)]
    state.legal_moves = [pos_to_index(position[j]) for j in range(length)]

  return state

def pos_to_index(position):
  # Return the index of a position.
  return position[0] * BOARD_SIZE_X + position[1] if position[0] != PASS[0] else BOARD_SIZE_X * BOARD_SIZE_Y

def index_to_pos(index):
  # Return the position of an index.
  return [index//BOARD_SIZE_X, index%BOARD_SIZE_X] if index != BOARD_SIZE_X * BOARD_SIZE_Y else PASS