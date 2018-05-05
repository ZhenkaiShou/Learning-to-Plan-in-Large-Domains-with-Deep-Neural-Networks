# Data
BOARD_SIZE_X           = 8                   # Size of the game board (19)
BOARD_SIZE_Y           = 8                   # Size of the game board (19)

# Game Constants
BLACK                  = 1                   # Constant for black stones
WHITE                  = -1                  # Constant for white stones
EMPTY                  = 0                   # Constant for empty position
PASS                   = [-1, -1]            # Constant for a pass move

# Player Constants
BEST_PLAYER            = -1                  # Constant for the current best player
HUMAN_PLAYER           = -10                 # Constant for a human player

# Symmetry
# 0: original
# 1: rotate 90
# 2: rotate 180
# 3: rotate 270
# 4: flip in x direction
# 5: flip in y direction
# 6: transpose
# 7: transpose over the secondary diagonal
SYMMETRY               = [0,1,2,3,4,5,6,7]   # Allowed symmetries

# Model
NUM_FEATURES           = 4                   # Number of feature planes (17)
NUM_FILTERS            = 64                  # Number of filters in a convolutional layer (256)
NUM_RESBLOCKS          = 9                   # Number of residual blocks (19 or 39)
NUM_FILTERS_P          = 2                   # Number of filters in the convolutional layer of the policy head (2)
NUM_FILTERS_V          = 1                   # Number of filters in the convolutional layer of the value head (1)
NUM_FC_V               = 64                  # Number of units in the fully connected layer of the value head (256)
NUM_RESBLOCKS_IMIT     = 3                   # Number of residual layers in imitation block

# MCTS
ADVANCED_EVALUATION    = True                # Whether to use calibrated evaluation during MCTS
SEARCH_THREADS         = 2                   # Number of threads for tree search (8)
NUM_SIMULATIONS        = 100                 # Number of simulations per tree search (1600)
EPSILON_NOISE          = [0.25, 0.05]        # The factor of Dirichlet noise to the root node for [selfplay, tournament] (0.25)
DIR_NOISE              = 0.03                # The parameter of Dirichlet distribution (0.03)
VIRTUAL_LOSS           = 3                   # Virtual losses that encourages exploration (3)
EXPLORE_STEPS          = 6                   # Number of steps that encourages exploration (30)
C_PUCT                 = 1.5                 # Exploration constant in the tree policy (5)

# Training
GAME_THREADS           = 50                  # Number of games sent to MCTS in parallel
SELFPLAY_GAMES         = 2500                # Number of games to be generated per selfplay (25000)
RECENT_GAMES           = 25000               # Number of the most recent games for training (500000)
MINIBATCH_SIZE         = 256                 # Size of minibatch for one update (1024)
CHECKPOINT_INTERVAL    = 500                 # Interval for each check of the current best agent (1000)
EVALUATION_MINIBATCH   = 200                 # Number of minibatches for evaluation
TOURNAMENT_GAMES       = 400                 # Number of games to be played during a tournament (400)
WIN_CRITERIA           = 0.55                # Winning rate criteria for the challenger (0.55)

# General
NUM_PROCESSING         = 1                   # Number of multiprocessing
USE_GPU                = False               # Whether GPU is used during training
VISIBLE_DEVICE_MCTS    = ""                  # The index of device observable to tensorflow during tree search
VISIBLE_DEVICE_OPTM    = ""                  # The index of device observable to tensorflow during optimization
MEMORY_MCTS            = 0.1                 # The fraction of GPU consumption during tree search
MEMORY_OPTM            = 0.9                 # The fraction of GPU consumption during optimization

# Optimization
LEARNING_RATE          = [1e-2, 1e-3, 1e-4]  # Learning rate for optimization ([1e-2, 1e-3, 1e-4])
LEARNING_RATE_RANGE    = [100, 200]          # Learning rate annealing range ([400, 600])
MOMENTUM_OPT           = 0.9                 # Momentum parameter for optimization (0.9)
C_L2                   = 1e-4                # L2 regularisation parameter for the loss function (1e-4)
C_FEATURE              = 10                  # Scaling factor for feature loss

# Saving
PLAYER_DIR             = "./Saves/Players/"
DATA_DIR               = "./Saves/TrainingData/"
TEMPDATA_DIR           = "./Saves/TempData/"
RECORD_DIR             = "./Saves/Record/"