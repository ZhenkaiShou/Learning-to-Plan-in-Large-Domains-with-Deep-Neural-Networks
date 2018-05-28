# Deep-Networks-that-Learn-to-Plan
Repository for master thesis "Learning to Plan in Large Domains with Deep Neural Networks".

## Thesie Abstract
In the domain of artificial intelligence, effective and efficient planning is one key factor to developing an adaptive agent which can solve tasks in complex environments. However, traditional planning algorithms only work properly in small domains. Learning to plan, which requires an agent to apply the knowledge learned from past experience to planning, can scale planning to large domains. Recent advances in deep learning widen the access to better learning techniques. Combining traditional planning algorithms with modern learning techniques in a proper way enables an agent to extract useful knowledge and thus show good performance in large domains.

This thesis aims to explore learning to plan in large domains with deep neural networks. The main contributions of this thesis include: (1) a literature survey on learning to plan; (2) proposing a new network architecture that learns from planning, combining this network with a planner, implementing and testing this idea in the game Othello.

## About this Repository
This repository contains python codes for the implementation and experiments of this thesis, training an artificial player to play Othello from scratch. The main framework (training, basic network architecture) follows Alpha Zero. But we

## Prerequisites
Python >= 3.5
TensorFlow >= 1.4.0
TQDM
PyQt5

## Running the Codes
Install all the above packages and download all files in this repository.
To train the artificial Othello player, run the training.py file:
'''
python3 training.py
'''

To play the game with user interface, run the playgame.py file:
'''
python3 playgame.py
'''
