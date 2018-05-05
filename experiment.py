import csv
import multiprocessing
import numpy as np
import os
import tqdm
from constants import *
from mcts import *

def gameplay_performance(player, iterations, groups, tournament_games = TOURNAMENT_GAMES, simulations_baseline = NUM_SIMULATIONS, simulations_challenger = NUM_SIMULATIONS):
  # Create empty Game-Play Performance file.
  if not os.path.isfile(RECORD_DIR + "Game-Play Performance.csv"):
    with open(RECORD_DIR + "Game-Play Performance.csv", "w") as f:
      fieldnames = ["Player", "Number of Simulations"] + ["Winning Rate " + str(i+1) for i in range(iterations)] + ["Average Winning Rate"]
      writer = csv.DictWriter(f, fieldnames = fieldnames, lineterminator = "\n")
      writer.writeheader()
  
  for i in range(groups):
    winning_rate = np.zeros(iterations)
    for j in range(iterations):
      # Hold a tournament between the challenger and the baseline player.
      winning_rate[j] = gameplay_tournament(player, tournament_games, simulations_baseline, int((i+1)/groups*simulations_challenger))
      print("Challenger " + str(player) + " with " + str(int((i+1)/groups*simulations_challenger)) + " simulations wins " + format(winning_rate[j] * 100, ".2f") + "% of the tournament games.")
    avg_winning_rate = np.mean(winning_rate)
    # Update Game-Play Performance file.
    with open(RECORD_DIR + "Game-Play Performance.csv", "a") as f:
      fieldnames = ["Player", "Number of Simulations"] + ["Winning Rate " + str(j+1) for j in range(iterations)] + ["Average Winning Rate"]
      writer = csv.DictWriter(f, fieldnames = fieldnames, lineterminator = "\n")
      contents = {"Player": player, "Number of Simulations": int((i+1)/groups*simulations_challenger)}
      for j in range(iterations):
        contents.update({"Winning Rate " + str(j+1): winning_rate[j]})
      contents.update({"Average Winning Rate": avg_winning_rate})
      writer.writerow(contents)

def gameplay_tournament(player, tournament_games, simulations_baseline, simulations_challenger):
  # Arrange number of playing black and white for each process.
  base = tournament_games // NUM_PROCESSING
  reminder = tournament_games % NUM_PROCESSING
  arg_list = [None for _ in range(NUM_PROCESSING)]
  for i in range(NUM_PROCESSING):
    num_games = base + 1 if i < reminder else base
    if num_games % 2 == 0:
      num_blacks = num_games // 2
      num_whites = num_games // 2
    else:
      num_blacks = num_games // 2 + 1 if i % 2 == 0 else num_games // 2
      num_whites = num_games // 2 if i % 2 == 0 else num_games // 2 + 1
    arg_list[i] = [player, num_blacks, num_whites, simulations_baseline, simulations_challenger]
  # Start multiprocessing.
  with multiprocessing.Pool(NUM_PROCESSING) as p:
    total_win = np.sum(p.starmap(gameplay_tournament_process, arg_list))
  winning_rate = total_win / tournament_games
  
  return winning_rate

def gameplay_tournament_process(player, num_blacks, num_whites, simulations_baseline, simulations_challenger):
  # The challenger plays against the baseline player.
  total_win = 0
  step = 0
  pbar = tqdm.tqdm(total = num_blacks + num_whites, ncols = 60, unit = "game")
  np.random.seed()
  
  while step < num_blacks + num_whites:
    if step < num_blacks:
      # Play black for the first half of the games.
      play_black = True
      challenger_to_play = True
      length = np.minimum(num_blacks - step, GAME_THREADS)
    else:
      # Play white for the second half of the games.
      play_black = False
      challenger_to_play = False
      length = np.minimum(num_blacks + num_whites - step, GAME_THREADS)
    mcts_challenger = MCTS([GameState() for _ in range(length)], player, "low", advanced_evaluation = True, num_simulations = simulations_challenger, search_threads = 1)
    mcts_baseline = MCTS([GameState() for _ in range(length)], player, "low", advanced_evaluation = False, num_simulations = simulations_baseline, search_threads = 1)
    
    while len(mcts_challenger.root_state) > 0:
      remove_list = []
      mcts = mcts_challenger if challenger_to_play else mcts_baseline
      # Run the Monte-Carlo Tree Search for the current player.
      pi = mcts.tree_search()
      for i in range(len(mcts_challenger.root_state)):
        # Choose the best action from output policy.
        arg_max = np.argwhere(pi[i] == np.amax(pi[i]))
        pi[i] = np.zeros_like(pi[i])
        pi[i][arg_max] = 1 / len(arg_max)
        index_i = np.random.choice(len(mcts.root_state[i].legal_moves), p = pi[i])
        # Update the root node and root state for both players.
        mcts_challenger.get_subtree(i, index_i)
        mcts_baseline.get_subtree(i, index_i)
        
        if mcts_challenger.root_state[i].game_ends:
          # Score the tournament: +1 for win, +0.5 for tie, +0 for loss.
          if play_black:
            if mcts_challenger.root_state[i].score[0] == 1:
              total_win += 1
            elif mcts_challenger.root_state[i].score[0] == 0:
              total_win += 0.5
          else:
            if mcts_challenger.root_state[i].score[1] == 1:
              total_win += 1
            elif mcts_challenger.root_state[i].score[1] == 0:
              total_win += 0.5
          # Add the terminal node to the remove list.
          remove_list.append(i)          
          step += 1
      if len(remove_list) > 0:
        # Remove terminal nodes.
        mcts_challenger.remove_node(remove_list)
        mcts_baseline.remove_node(remove_list)
      # Update the current player.
      challenger_to_play = not challenger_to_play
    pbar.update(length)
  pbar.close()
  
  return total_win

def comparison_with_alpha_zero(player, iterations, advanced_evaluation_alpha_zero, advanced_evaluation_standard, dir, tournament_games = TOURNAMENT_GAMES, num_simulations = NUM_SIMULATIONS):
  # Create empty Comparison with Alpha Zero file.
  if not os.path.isfile(RECORD_DIR + "Comparison with Alpha Zero.csv"):
    with open(RECORD_DIR + "Comparison with Alpha Zero.csv", "w") as f:
      fieldnames = ["Player", "Alpha Zero Using Future Knowledge", "Standard Model Using Future Knowledge"] + ["Winning Rate " + str(i+1) for i in range(iterations)] + ["Average Winning Rate"]
      writer = csv.DictWriter(f, fieldnames = fieldnames, lineterminator = "\n")
      writer.writeheader()
  
  winning_rate = np.zeros(iterations)
  for i in range(iterations):
    # Hold a tournament between the standard model and the Alpha Zero variant.
    winning_rate[i] = alpha_zero_tournament(player, advanced_evaluation_alpha_zero, advanced_evaluation_standard, dir, tournament_games, num_simulations)
    print("Standard player " + str(player) + " wins " + format(winning_rate[i] * 100, ".2f") + "% of the tournament games.")
  avg_winning_rate = np.mean(winning_rate)
  # Update Comparison with Alpha Zero file.
  with open(RECORD_DIR + "Comparison with Alpha Zero.csv", "a") as f:
    fieldnames = ["Player", "Alpha Zero Using Future Knowledge", "Standard Model Using Future Knowledge"] + ["Winning Rate " + str(i+1) for i in range(iterations)] + ["Average Winning Rate"]
    writer = csv.DictWriter(f, fieldnames = fieldnames, lineterminator = "\n")
    contents = {"Player": player, "Alpha Zero Using Future Knowledge": 1 if advanced_evaluation_alpha_zero else 0, "Standard Model Using Future Knowledge": 1 if advanced_evaluation_standard else 0}
    for i in range(iterations):
      contents.update({"Winning Rate " + str(i+1): winning_rate[i]})
    contents.update({"Average Winning Rate": avg_winning_rate})
    writer.writerow(contents)

def alpha_zero_tournament(player, advanced_evaluation_alpha_zero, advanced_evaluation_standard, dir, tournament_games, num_simulations):
  # Arrange number of playing black and white for each process.
  base = tournament_games // NUM_PROCESSING
  reminder = tournament_games % NUM_PROCESSING
  arg_list = [None for _ in range(NUM_PROCESSING)]
  for i in range(NUM_PROCESSING):
    num_games = base + 1 if i < reminder else base
    if num_games % 2 == 0:
      num_blacks = num_games // 2
      num_whites = num_games // 2
    else:
      num_blacks = num_games // 2 + 1 if i % 2 == 0 else num_games // 2
      num_whites = num_games // 2 if i % 2 == 0 else num_games // 2 + 1
    arg_list[i] = [player, num_blacks, num_whites, advanced_evaluation_alpha_zero, advanced_evaluation_standard, dir, num_simulations]
  # Start multiprocessing.
  with multiprocessing.Pool(NUM_PROCESSING) as p:
    total_win = np.sum(p.starmap(alpha_zero_tournament_process, arg_list))
  winning_rate = total_win / tournament_games
  
  return winning_rate

def alpha_zero_tournament_process(player, num_blacks, num_whites, advanced_evaluation_alpha_zero, advanced_evaluation_standard, dir, num_simulations):
  # The standard model plays against the Alpha Zero variant.
  total_win = 0
  step = 0
  pbar = tqdm.tqdm(total = num_blacks + num_whites, ncols = 60, unit = "game")
  np.random.seed()
  
  while step < num_blacks + num_whites:
    if step < num_blacks:
      # Play black for the first half of the games.
      play_black = True
      standard_to_play = True
      length = np.minimum(num_blacks - step, GAME_THREADS)
    else:
      # Play white for the second half of the games.
      play_black = False
      standard_to_play = False
      length = np.minimum(num_blacks + num_whites - step, GAME_THREADS)
    mcts_standard = MCTS([GameState() for _ in range(length)], player, "low", advanced_evaluation = advanced_evaluation_standard, num_simulations = num_simulations)
    mcts_alphazero = MCTS([GameState() for _ in range(length)], player, "low", advanced_evaluation = advanced_evaluation_alpha_zero, num_simulations = num_simulations, parent_dir = dir)
    
    while len(mcts_standard.root_state) > 0:
      remove_list = []
      mcts = mcts_standard if standard_to_play else mcts_alphazero
      # Run the Monte-Carlo Tree Search for the current player.
      pi = mcts.tree_search()
      for i in range(len(mcts_standard.root_state)):
        # Choose the best action from output policy.
        arg_max = np.argwhere(pi[i] == np.amax(pi[i]))
        pi[i] = np.zeros_like(pi[i])
        pi[i][arg_max] = 1 / len(arg_max)
        index_i = np.random.choice(len(mcts.root_state[i].legal_moves), p = pi[i])
        # Update the root node and root state for both players.
        mcts_standard.get_subtree(i, index_i)
        mcts_alphazero.get_subtree(i, index_i)
        
        if mcts_standard.root_state[i].game_ends:
          # Score the tournament: +1 for win, +0.5 for tie, +0 for loss.
          if play_black:
            if mcts_standard.root_state[i].score[0] == 1:
              total_win += 1
            elif mcts_standard.root_state[i].score[0] == 0:
              total_win += 0.5
          else:
            if mcts_standard.root_state[i].score[1] == 1:
              total_win += 1
            elif mcts_standard.root_state[i].score[1] == 0:
              total_win += 0.5
          # Add the terminal node to the remove list.
          remove_list.append(i)          
          step += 1
      if len(remove_list) > 0:
        # Remove terminal nodes.
        mcts_standard.remove_node(remove_list)
        mcts_alphazero.remove_node(remove_list)
      # Update the current player.
      standard_to_play = not standard_to_play
    pbar.update(length)
  pbar.close()
  
  return total_win