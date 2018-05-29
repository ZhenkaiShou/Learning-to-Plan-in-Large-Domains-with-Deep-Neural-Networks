# Deep-Networks-that-Learn-to-Plan
Repository for master thesis "Learning to Plan in Large Domains with Deep Neural Networks".

## About this Thesis

### Abstract
In the domain of artificial intelligence, effective and efficient planning is one key factor to developing an adaptive agent which can solve tasks in complex environments. However, traditional planning algorithms only work properly in small domains. Learning to plan, which requires an agent to apply the knowledge learned from past experience to planning, can scale planning to large domains. Recent advances in deep learning widen the access to better learning techniques. Combining traditional planning algorithms with modern learning techniques in a proper way enables an agent to extract useful knowledge and thus show good performance in large domains.

This thesis aims to explore learning to plan in large domains with deep neural networks. The main contributions of this thesis include: (1) a literature survey on learning to plan; (2) proposing a new network architecture that learns from planning, combining this network with a planner, implementing and testing this idea in the game Othello.

### Neural Networks that Learn from Planning
The [neural network](./figures/Neural_Network_Architecture.pdf) can evaluate a certain state by using not only the current state itself but also the planning result starting from that state. The network learns feature from the current state and gives a basic estimation by using this feature. It also learns contextual feature from the planning result and corrects its basic estimation by using this contextual feature. 



## About this Repository
This repository contains python codes for the implementation and experiments of this thesis, training an artificial player to play Othello from scratch.

### Prerequisites
* Python >= 3.5
* TensorFlow >= 1.4.0
* TQDM
* PyQt5

### Running the Codes
Install all the above packages and download all files in this repository.

To train the artificial Othello player, run the training.py file:
```
python3 training.py
```

To play the game with user interface, run the playgame.py file:
```
python3 playgame.py
```
### Hyperparameters
All the hyperparameters can be tuned in the constants.py file.

If you want to disable the random transition function (for data augmentation), set SYMMETRY to:
```
SYMMETRY               = [0]                 # Allowed symmetries
```

If you want to train the player using GPU, turn on the GPU option:
```
USE_GPU                = True                # Whether GPU is used during training
```

If you are using a server and want to better utilize the server resource, set NUM_PROCESSING to a proper value, e.g. 25:
```
NUM_PROCESSING         = 25                  # Number of multiprocessing
```

If you turn on the GPU option, be sure to tell the program which GPU TensorFlow can use and how much it can comsume for each process, e.g. use GPU No.0 and each process consumes 2% GPU during tree search:
```
VISIBLE_DEVICE_MCTS    = "0"                 # The index of device observable to tensorflow during tree search
VISIBLE_DEVICE_OPTM    = "0"                 # The index of device observable to tensorflow during optimization
MEMORY_MCTS            = 0.02                # The fraction of GPU consumption during tree search
MEMORY_OPTM            = 0.9                 # The fraction of GPU consumption during optimization
```
