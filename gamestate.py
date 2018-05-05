import numpy as np
from constants import *
from util import pos_to_index, index_to_pos

class GameState(object):
  def __init__(self):
    self.board = np.zeros((BOARD_SIZE_Y, BOARD_SIZE_X), np.int16)
    # Initialize center stones.
    self.board[BOARD_SIZE_Y//2-1, BOARD_SIZE_X//2-1] = WHITE
    self.board[BOARD_SIZE_Y//2, BOARD_SIZE_X//2] = WHITE
    self.board[BOARD_SIZE_Y//2-1, BOARD_SIZE_X//2] = BLACK
    self.board[BOARD_SIZE_Y//2, BOARD_SIZE_X//2-1] = BLACK
    
    self.step = 0
    self.is_black = True
    self.legal_moves = []
    self.both_pass = 0
    self.game_ends = False
    self.score = None
    
    # Update legal moves.
    self.update_legal_moves()
  
  def update_legal_moves(self):
    legal_moves = []
    # Select free spaces as initial candidates.
    candidates = np.argwhere(np.equal(self.board, EMPTY))
    # Check legability for each candidate.
    for x in candidates:
      for i in [1, 2, 3, 4, 6, 7, 8, 9]:
        if self.check_in_direction(x, i, False):
          legal_moves.append(pos_to_index(x))
          break
    # Add pass move if and only if no other legal moves exist.
    if len(legal_moves) == 0:
      legal_moves.append(pos_to_index(PASS))
    self.legal_moves = legal_moves
  
  def select_move(self, index):
    if index != pos_to_index(PASS):
      # Case for non-pass move.
      position = index_to_pos(index)
      for i in [1, 2, 3, 4, 6, 7, 8, 9]:
        self.check_in_direction(position, i, True)
      self.both_pass = 0
    else:
      # Case for pass move.
      self.both_pass += 1
      if self.both_pass == 2:
        # The game ends if both players pass.
        self.game_ends = True
        self.scoring()
        self.legal_moves = []
    
    # Update states.
    self.step += 1
    self.is_black = False if self.is_black else True
    
    # Update legal moves for next state if the game does not end.
    if not self.game_ends:
      self.update_legal_moves()
    
    return self
  
  def check_in_direction(self, position, direction, is_play):
    # is_play = False: check the capturablility of a position in a specific direction.
    # is_play = True: update the game board along a specific direction when playing at this position.
    (y, x) = position
    capturable = False
    j = 0
    # Map direction index to position shift.
    if direction == 1:
      dx = -1
      dy = 1
      distance = np.minimum(BOARD_SIZE_Y-y-1, x)
    elif direction == 2:
      dx = 0
      dy = 1
      distance = BOARD_SIZE_Y-y-1
    elif direction == 3:
      dx = 1
      dy = 1
      distance = np.minimum(BOARD_SIZE_Y-y-1, BOARD_SIZE_X-x-1)
    elif direction == 4:
      dx = -1
      dy = 0
      distance = x
    elif direction == 6:
      dx = 1
      dy = 0
      distance = BOARD_SIZE_X-x-1
    elif direction == 7:
      dx = -1
      dy = -1
      distance = np.minimum(y, x)
    elif direction == 8:
      dx = 0
      dy = -1
      distance = y
    else:
      dx = 1
      dy = -1
      distance = np.minimum(y, BOARD_SIZE_X-x-1)
    
    if self.is_black:
      # Logical part for black.
      for i in range(1, distance+1):
        value = self.board[y+i*dy, x+i*dx]
        if i == 1:
          if value != WHITE:
            break
        else:
          if value == EMPTY:
            break
          elif value == BLACK:
            capturable = True
            j = i
            break
    else:
      # Logical part for white.
      for i in range(1, distance+1):
        value = self.board[y+i*dy, x+i*dx]
        if i == 1:
          if value != BLACK:
            break
        else:
          if value == EMPTY:
            break
          elif value == WHITE:
            capturable = True
            j = i
            break
    
    if is_play:
      # Update the game board if a stone is played at this position.
      if self.is_black:
        for i in range(j):
          self.board[y+i*dy, x+i*dx] = BLACK
      else:
        for i in range(j):
          self.board[y+i*dy, x+i*dx] = WHITE
    else:
      # Return capturable if we want to check the legability of this position.
      return capturable
  
  def scoring(self):
    # Return a vector [black_score, white_score].
    # Winning a game = 1 score,
    # losing a game = -1 score,
    # drawing a game = 0 score.
    blacks = np.count_nonzero(self.board == BLACK)
    whites = np.count_nonzero(self.board == WHITE)
    if blacks > whites:
      self.score = [1, -1]
    elif blacks == whites:
      self.score = [0, 0]
    else:
      self.score = [-1, 1]