

# Investigating the Performance and Reliability, of the Q-Learning Algorithm in Various Unknown 2 Dimensional Grid Environments

This Jupyter notebook provides a platform to evaluate the performance and reliability of the Q-learning algorithm in various 2-dimensional environments using the OpenAI gymnasium library. The related paper was published at the Proceedings of the 11th RSI International Conference on Robotics and Mechatronics (ICRoM 2023), held in Tehran, Iran from December 19-21, 2023. You can find the paper and the presentation PowerPoint related to the conference in this repository.

In addition to Q-learning, this Jupyter notebook also includes an implementation of a value iteration algorithm from scratch, which is accessible to users. This repository is a great opportunity for beginners to experiment with value iteration, Q-learning, and OpenAI gymnasium.

## Acknowledgements

Make sure to cite the paper by Amirhossein Nourian et al. if you use this code for your research

## Description

This repository contains the following items:

--[OpenAI frozen lake environment](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)  from Gymnasium.
--Implementation of the Value Iteration algorithm.
--Implementation of the Q-Learning algorithm.
--Visualization of Q-Learning Algorithm results.
--Published paper related to the code.
--Presentation related to this code in ICROM conference.

## Setup 

To run the script you'll need the following dependencies:

- [OpenAI Gym](https://gymnasium.farama.org/index.html) 
- [Numpy](https://numpy.org/)  
- [Mathplotlib](https://matplotlib.org/)  

which should all be available through Pip.

No additional setup is needed, so simply clone the repo:
```sh
git clone https://github.com/amirhnourian/Open_AI_Frozenlake.git
cd Open_AI_Frozenlake
```  

## Usage 

he input data consists of the FrozenLake maps and the hyperparameters for the Q-learning algorithm. The output data is the Q-table after solving Q-learning. You can test the policy with the provided function. Lastly, there is a code available for finding the correct number of policies in a set of given episodes with a sampling coefficient.

To use the code, you need to run each cell in the correct order and follow the instructions provided in the Jupyter file. I recommend that you first read the related paper to familiarize yourself with the code and its purpose. If you have any further questions, please let me know.















