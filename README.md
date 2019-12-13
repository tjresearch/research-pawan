# Diversity Based Hierarchical Reinforcement Learning (DHRL)

### Background:
Hierarchical Reinforcement Learning (HRL) has proven itself to be an effective method for reducing sample complexity of reinforcement learning algorithms by partitioning the state space into regions in which a particular policy called an option is effective. The predominant implementation has been with a two level system where the bottom level contains a variety of options, and the top level has one policy which chooses between the options.

### Motivation:
Scaling this seemingly recursive design to three levels or more introduces more complications, but has the potential benefit of being able to learn long term policies or strategies and also being able to perform multiple tasks without drops in performance after learning a new task. One major issue with HRL and especially with hierarchies with three or more layers is how options should be discovered. Formaly, an option is a policy π_h which has a set of initiation states, and a set of termination states. These are not easy to learn and many algorithms involve hand desiging options. 

### Project Overview
This project explores how options for  HRL can be formed by adapting the [Diversity Is All You Need (DIAYN)](https://arxiv.org/abs/1802.06070) algorithm to a n level hierarchy. DIAYN is an option discovery method which uses entropy maximization and mutual information bewteen options and the states they visit to force options which are all equally effective and as diverse as possible. Since these options were discovered without the task being known, they can also be re-used for multiple tasks.
My goal with this project is to adapt DIAYN and hopefully show that it is able to learn a hierarchy which can solve several ATARI ram environments, each faster than the previous one.

### Requirements (Linux)
1. Python 3.6
2. Tensorflow 2.0 GPU and its dependencies
3. OpenAI Gym and its dependencies
4. MuJoCo python

### Run instructions (Work in Progress)
Open a terminal within the repository.
pip install -r requirements.txt
Train command:
Test pre-trained model:
### Description of each folder:
#### Presentations:
Contains presentations I made over the year
#### Journals:
Contains weekly progress reports
#### Resources/Models:
This is where you will find repository code for DIAYN, HAC, and OpenAI's soft-actor-critic implementation
#### Resources/scratchwork:
Contains miscelanious models I attempted to implement such as Double DQN and Soft-actor-critic

### Replays of some solved environments(Work in Progress)
