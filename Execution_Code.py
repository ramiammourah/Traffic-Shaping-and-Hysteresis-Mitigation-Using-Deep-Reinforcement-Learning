# Import Required Libraries
from gym.spaces import Discrete, Box
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from copy import copy
import random
from utils import moving_average, delete_multiple_element, move_step
from matplotlib import pyplot as plt
import gc

class CustomEnvironment():
    def __init__(self):
        
        # Possible Useful Initializations
        self.timestep = None
        self.location = None
        self.Density = None
        self.Flow = None
        self.speed = None
        self.acceleration = None
        self.spacing = None
        self.AverageSpeed0 = None
        self.FlowRL = []
        self.AverageSpeed = []
        self.list_of_RL_indices = []
        self.Vehicle_Trajectories = {}
        self.possible_agents = ['RL0', 'RL4', 'RL8', 'RL12']
        self.RL0_acceleration = None
        self.RL4_acceleration = None
        self.RL8_acceleration = None
        self.RL12_acceleration = None
        self.RL_acceleration = None
        self.reward_compiler = None
        self.episode_reward = []
        self.DensityRL = []
        self.loop_length = 333
        self.terminations = False
        self.truncations =False
        self.agents = None

        
        # Action Space
        self.action_space = Discrete(3)
        
        # Observation Space
        self.low = np.array([0, 0], dtype=np.float32)
        self.high = np.array([100, 100], dtype=np.float32)
        self.observation_space = Box(self.low, self.high, dtype=np.float32)
        
        
        
    def reset(self, seed=None, return_info=False, options=None):
        
        # Reset Terminations and Truncations
        self.terminations = False
        self.truncations = False
        
        self.reward_compiler = 0
        
        del self.agents
        self.agents = copy(self.possible_agents)
        
        self.timestep = 0

        # Initializing First Vehicle Locations
        with open("Location.pkl", "rb") as fp: 
            self.location = [3940.161998761294,3955.7716848321998,3971.1893730761994,3986.509359095753,4001.8429835962565,4017.204271619366,4032.4384397120707,4047.3048417557275, 4061.7058521401696,
                                    4075.8584993438203,4090.2339320802075,4105.288843969282,4121.073831249855,4136.909742046702,4151.726050380981,4165.121330380292,4177.836441437975,4191.227555610173,
                                    4206.444327814801,4223.480833061669,4240.809552391871,4257.262180636416,4262.830362454598] 
              
        with open("Density.pkl", "rb") as fp:
            self.Density = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

        with open("Flow.pkl", "rb") as fp:
            self.Flow = [0, 0.02945, 0.05963794, 0.08822865, 0.11372298, 0.13472832, 0.1527702 , 0.16865735, 0.1803432 , 0.18682734, 0.18927784, 0.18897887, 0.18678924, 0.18344084, 0.179423,
                        0.17509278, 0.17064917, 0.16616738, 0.16164567, 0.15709085, 0.15254243, 0.147978  , 0.14309774, 0.13884889]
            
        with open("Speed.pkl", "rb") as fp: 
            self.speed = [6.45896532, 6.33011539, 6.244123  , 6.22574128, 6.2378426, 6.18778395, 5.99933241, 5.71376824, 5.50101869, 5.53559054,
                                5.85813977, 6.30341164, 6.49222139, 6.06286615, 5.20339439, 4.60366477, 4.744602  , 5.59994243, 6.69184811, 7.22526372,
                                6.96521873, 6.62712716, 6.03690826]
            
        with open("Acceleration.pkl", "rb") as fp: 
            self.acceleration = [-0.08287979, -0.06528455, -0.02731665, -0.00077029, -0.02161973, -0.09495193, -0.1667023 , -0.15324791, -0.02805578,  0.14435064,
                                        0.25210762,  0.17527099, -0.14016924, -0.47137829, -0.43209506, -0.03599888,  0.39881309,  0.61708524,  0.46384711,  0.04060762,
                                        -0.14017239, -0.08552199,  0]
            
        with open("Spacing.pkl", "rb") as fp:   
            self.spacing = [11.60968607, 11.41768824, 11.31998602, 11.3336245 , 11.36128802, 11.23416809, 10.86640204, 10.40101038, 10.1526472 , 10.37543274, 11.05491189,
                                    11.78498728, 11.8359108 , 10.81630833,  9.39528, 8.71511106,  9.39111417, 11.2167722 , 13.03650525, 13.32871933, 12.45262824, 11.89981812]
            
        with open("AverageSpeed.pkl", "rb") as fp:   
            self.AverageSpeed0 = [29.45, 29.81897112, 29.40955163, 28.43074568, 26.94566371, 25.46169918, 24.09390667, 22.54289981, 20.7585933 , 18.92778357, 17.17989735,
                                         15.56576984, 14.11083411, 12.81592877, 11.67285197, 10.66557304,  9.7745519 ,  8.98031525,  8.26793942,  7.6271213 , 7.04657126,  6.5044429 ,  6.03690826]
            
        # Reset Flow List After Introducing RL Agents
        self.FlowRL = []
        
        # Reset Density
        self.DensityRL = []
        
        self.AverageSpeed = []
        self.AverageSpeed.append(np.mean(self.speed))
        
        # Removing 7 Vehicles to have 16 Remaining
        list_of_indices_to_delete = [0, 3 , 7, 10, 14, 17, 20]
        delete_multiple_element(self.location, list_of_indices_to_delete)
        delete_multiple_element(self.speed, list_of_indices_to_delete)
        delete_multiple_element(self.acceleration, list_of_indices_to_delete)
        
        # Updated RL Vehicle IDs after Removal
        self.list_of_RL_indices = [0, 4, 8, 12]
        
        # Defining and Reseting each RL Agent Observations
        TH0_F = (self.location[0] - self.location[15]-self.loop_length-4) / (self.speed[15])
        TH0_L = (self.location[1] - self.location[0]-4) / (self.speed[0])

        TH4_F = (self.location[4] - self.location[3]-4) / (self.speed[3])
        TH4_L = (self.location[5] - self.location[4]-4) / (self.speed[4])

        TH8_F = (self.location[8] - self.location[7]-4) / (self.speed[7])
        TH8_L = (self.location[9] - self.location[8]-4) / (self.speed[8])

        TH12_F = (self.location[12] - self.location[11]-4) / (self.speed[11])
        TH12_L = (self.location[13] - self.location[12]-4) / (self.speed[12])

        observations = {'RL0': (TH0_F, TH0_L),
                        'RL4': (TH4_F, TH4_L),
                        'RL8': (TH8_F, TH8_L),
                        'RL12': (TH12_F, TH12_L)
                       }

        return observations

    def step(self, actions):
#         print('')
#         print('Time Step = ', self.timestep)

        # Logging Average Speed
        if np.mod(self.timestep,10) == 0:
            self.AverageSpeed.append(np.mean(self.speed))
            
        # Logging Trajectories (self.Vehicle_Trajectories = {[], [], ...})
#         for i in range(len(self.location)):
#             if np.mod(i,4)==0:
#                 self.Vehicle_Trajectories['RL'+str(i)].append(self.location[i])
#             else:
#                 self.Vehicle_Trajectories['IDM'+str(i)].append(self.location[i])
        
        # Logging Flow and Density Values
        if np.mod(self.timestep, 10) == 0:
            self.DensityRL.append(len(self.location))
            self.FlowRL.append(len(self.location) * np.average(self.speed) / 1000)
        
        # Execute Actions
        RL0_action = actions['RL0']
        RL4_action = actions['RL4']
        RL8_action = actions['RL8']
        RL12_action = actions['RL12']
        
        # Actions Legend: (0: decelerate with -1m/s2, 1: do nothing with 0 m/s2, 2: accelerate with 1 m/s2)
        if RL0_action == 0:
            self.RL0_acceleration = -1
        elif RL0_action == 1:
            self.RL0_acceleration = 0
        elif RL0_action == 2:
            self.RL0_acceleration = 1
        
        if RL4_action == 0:
            self.RL4_acceleration = -1
        elif RL4_action == 1:
            self.RL4_acceleration = 0
        elif RL4_action == 2:
            self.RL4_acceleration = 1
            
        if RL8_action == 0:
            self.RL8_acceleration = -1
        elif RL8_action == 1:
            self.RL8_acceleration = 0
        elif RL8_action == 2:
            self.RL8_acceleration = 1
            
        if RL12_action == 0:
            self.RL12_acceleration = -1
        elif RL12_action == 1:
            self.RL12_acceleration = 0
        elif RL12_action == 2:
            self.RL12_acceleration = 1
        # else:
            # print("IMPOSSIBLE ACTION")
        
        self.RL_acceleration = [self.RL0_acceleration, self.RL4_acceleration, 
                                self.RL8_acceleration, self.RL12_acceleration]
        
        # Execute Actions (1 step of Movement of RL Agents and IDM Vehicles)
        self.location, self.speed, self.acceleration = move_step(self.location, self.speed, self.acceleration, self.RL_acceleration, self.list_of_RL_indices)
                
        # Check Termination Conditions
        rewards = {'RL0':self.RL0_acceleration*10, 'RL4': self.RL4_acceleration*10, 
                                'RL8': self.RL8_acceleration*10, 'RL12': self.RL12_acceleration*10}
        
        
        # Defining the Collision Detection Check (We only care about longitudinal collision since this is a loop)
        for i in self.agents:
            c = copy(int(i[2:]))
            if c != 0:
                if (self.location[c] - 4) <= self.location[c-1] or self.location[c] >= (self.location[c+1] - 4):
                    print('VEHICLE NUMBER ', i, ' COLLIDED')
#                     collision_check[i] = True
                    rewards[i] = -1000
#                     terminations = {a: True for a in self.agents}
                    self.terminations = True
                    self.agents = []
                    
            elif c == 0:
                if (self.location[c] - 4) <= (self.location[15]-self.loop_length) or self.location[c] >= (self.location[c+1]-4):
                    print('VEHICLE NUMBER ', c, ' COLLIDED')
#                     collision_check[i] = True
                    rewards[i] = -1000
#                     terminations = {a: True for a in self.agents}
                    self.terminations = True
                    self.agents = []
            del c

        # Check truncation conditions (overwrites termination conditions)
        if self.timestep > 100:
            self.truncations = True
            self.agents = []
        self.timestep += 1
        
        self.reward_compiler += sum(rewards.values())/4
        
        # print('rewards = ', rewards)
                   
        # Get Observations
        # Defining and Reseting each RL Agent Observations
        TH0_F = (self.location[0] - self.location[15]-self.loop_length-4) / (self.speed[15])
        TH0_L = (self.location[1] - self.location[0]-4) / (self.speed[0])

        TH4_F = (self.location[4] - self.location[3]-4) / (self.speed[3])
        TH4_L = (self.location[5] - self.location[4]-4) / (self.speed[4])

        TH8_F = (self.location[8] - self.location[7]-4) / (self.speed[7])
        TH8_L = (self.location[9] - self.location[8]-4) / (self.speed[8])

        TH12_F = (self.location[12] - self.location[11]-4) / (self.speed[11])
        TH12_L = (self.location[13] - self.location[12]-4) / (self.speed[12])

        observations = {'RL0': (TH0_F, TH0_L),
                        'RL4': (TH4_F, TH4_L),
                        'RL8': (TH8_F, TH8_L),
                        'RL12': (TH12_F, TH12_L)
                       }

        infos = {"RL0": {}, "RL4": {}, "RL8": {}, "RL12": {}, 'timestep':self.timestep}
        
        return observations, rewards, terminations, truncations, infos

    
    def render(self):
        pass
    

class CentralizedDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.gamma = 0.999    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.totalsteps = 110
        self.totalepisodes = 1000
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network_frequency = 25
        self.timestep = 0
        

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(512, input_dim=self.state_size, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.01))
        return model
    
    def remember(self, observations, actions, rewards, new_observations):
        for agt_id in env.possible_agents:
            state = observations[agt_id]
            action = actions[agt_id]
            reward = rewards[agt_id]
            new_state = new_observations[agt_id]
            self.memory.append((state, action, reward, new_state))

    def act(self, obs):
        actions = {}
        for agt_id in env.possible_agents:
            agent_state = obs[agt_id]
            agent_state = np.reshape(agent_state, (1,2))
            if np.random.rand() <= self.epsilon:
                actions[agt_id] = random.randrange(self.action_size)
            else:
                act_values = self.model.predict(agent_state)
                actions[agt_id] = np.argmax(act_values[0])
        self.timestep += 1
        return actions # Return the tuple of actions
    
    def act_optimally(self, obs):
        actions = {}
        for agt_id in env.possible_agents:
            agent_state = obs[agt_id]
            agent_state = np.reshape(agent_state, (1,2))
            act_values = self.model.predict(agent_state)
            actions[agt_id] = np.argmax(act_values[0])
        self.timestep += 1
        return actions # Return the tuple of actions

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:
            next_state = np.reshape(next_state, (1,2))
            state = np.reshape(state, (1,2))
            target = reward

            if not env.terminations:
                target = (reward + self.gamma * np.amax(self.target_model.predict(next_state)))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon -= (1/(self.totalsteps * self.totalepisodes))
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min
        # update target network every update_target_network_frequency steps
#         print('Network Time Step = ', timestep)
        if self.timestep % self.update_target_network_frequency == 0:
            self.target_model.set_weights(self.model.get_weights())

    def load(self, name):
        self.model.load_weights(name)
        self.target_model.load_weights(name)

    def save(self, name, name2):
        self.model.save_weights(name)
        self.target_model.save_weights(name2)

# Initialize the environment and agent
import time
start_time = time.time()

env = CustomEnvironment()

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = CentralizedDQNAgent(state_size=state_size, action_size=action_size)

batch_size = 32
episodes = 1000

# Training loop
for e in range(episodes):
    
    # Initialize the states
    observations = env.reset()
        
    # Reset Terminations and Truncations
    terminations = {a: False for a in env.agents}
    truncations = {a: False for a in env.agents}
    
#     while not terminations['RL0'] and not truncations['RL0']: # Steps
    while not env.terminations and not env.truncations:
        # Choose an action using the agent's policy
        actions = agent.act(observations)

        # Take the action and observe the next state, reward, and done flag
        new_observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Remember the experience
#         agent.remember(observations, actions, rewards, new_observations, terminations)
        agent.remember(observations, actions, rewards, new_observations)

        # Update the state
        observations = new_observations
        
        timestep = infos['timestep']
        
        # print('Environment Time Step = ', timestep)
        # End the episode if done is True
        if env.terminations or env.truncations:
            env.episode_reward.append(env.reward_compiler)
            print("episode: {}/{}, episode_reward: {}, Timestep: {}, e: {:.2}".format(e, episodes,
                                                                                      env.episode_reward[-1],timestep, agent.epsilon))
            break

        # Train the agent using random experiences from the memory
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    gc.collect()

agent.save("model.h5", "target_model.h5")
end_time = time.time()

total_time = end_time - start_time
print('Total Time = ', total_time)