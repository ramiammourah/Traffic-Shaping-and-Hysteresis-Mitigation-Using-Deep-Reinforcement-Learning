# Traffic-Shaping-and-Hysteresis-Mitigation-Using-Deep-Reinforcement-Learning
## Here is a summary of the different elements of the implementation provided by OpenAI's ChatGPT:
## 1] CustomEnvironment Class:
This code defines a custom environment in Python. The environment is a traffic simulation and it has four agents, which are controlled by reinforcement learning algorithms. The environment initializes the variables, sets up the action and observation spaces, and resets the simulation to its initial state.

The traffic simulation is based on the IDM model, and it uses pickle files to load initial traffic conditions, such as vehicle locations, speeds, and accelerations. The environment removes some vehicles to have a fixed number of 16 vehicles and updates the IDs of RL agents. The RL agents are initialized with observations based on their relative positions and speeds to the vehicles ahead and behind them.

The environment's action space is discrete with three possible actions, and the observation space is a box with low and high values of [0,0] and [100,100], respectively, representing the vehicles' leader and follower space headway. The environment has a timestep variable that tracks the number of iterations.

The environment also has variables for the flow, density, spacing, average speed, and rewards. It has functions to calculate the flow, density, and average speed based on the vehicle locations and speeds. The reward function is not defined in the code. The environment has variables to track RL agents' accelerations and their terminations and truncations, which are set to False in the reset function.

## 2] CentralizedDQNAgent Class:
This code defines a centralized Deep Q-Network (DQN) agent using Keras in Python. The agent is designed to work in an environment where multiple agents operate in the same state space and can take actions simultaneously.

The agent uses a neural network with four hidden layers, each with a different number of neurons (512, 256, 128, and 64), and an output layer with the same number of neurons as the number of possible actions. The model is trained using the mean squared error loss and the Adam optimizer with a learning rate of 0.01. The agent stores a deque of past experiences, with a maximum length of 50000.

During training, the agent samples a batch of experiences from the deque and uses them to update the neural network's weights. The agent also updates a target network's weights periodically to stabilize the training process. The agent's exploration rate, epsilon, decreases over time from an initial value of 1.0 to a minimum value of 0.01 using a linear decay schedule.

The agent has the following methods:

__init__(self, state_size, action_size): initializes the agent's parameters, including the neural network model and target network model.
_build_model(self): builds the neural network model.
remember(self, observations, actions, rewards, new_observations): stores the agent's experiences in the deque.
act(self, obs): chooses an action for each agent based on the exploration rate and the neural network's output. Returns a dictionary of actions for each agent.
act_optimally(self, obs): chooses an action for each agent based on the neural network's output. Returns a dictionary of actions for each agent.
replay(self, batch_size): samples a batch of experiences from the deque and uses them to update the neural network's weights. Also updates the target network's weights periodically and decreases the exploration rate.
load(self, name): loads the model's weights from a file with the given name.
save(self, name, name2): saves the model's weights to two files with the given names (one for the model and one for the target model).

## 3] Training Execution:
This code seems to be training a centralized deep Q-learning agent in a custom environment using the OpenAI Gym interface. The environment is reset at the start of each episode and the agent chooses actions based on its policy. The environment is stepped through using the chosen actions, and the resulting observations, rewards, and termination/truncation flags are recorded. The agent remembers the experience and updates its policy using random experiences from its memory. The training loop continues until either the environment terminates or is truncated. Finally, the trained agent is saved to a file.

## Additional Information:
The utils.py file contains the implementation of the Intelligent Drive Car Following Model which is called into the main code. It also contains a few other helper functions.

