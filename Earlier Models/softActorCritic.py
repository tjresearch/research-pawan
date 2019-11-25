from collections import deque
import numpy as np
import gym
import random
import sys

from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

EPISODES = 30000
class Agent():
    def__init(self,state_shape, action_shape, **render = False):
        self.render = render
        self.state_size = state_shape
        self.action_size = action_size

        self.gamma = 0.99 #discount factor
        self.alpha = 0.001 #learning rate
        self.batch_size = 32
        self.train_start = 1000 #start training after 1000 time steps of data has been collected

        #epsilon parameters
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = .999

        # instantiate replay memory which is used to make samples independent
        self.memory = deque(maxlen=50000)

        # instantiate models
        self.policy = self.build_model() #actor
        self.q1 = self.build_model()
        self.q2 = self.build_model()
        self.v = self.build_model()
        self.target_v = self.build_model()
        self.update_target_model()

    def update_target_model(self):
        self.target_v.set_weights(self.v.get_weights())

    def build_model(self): #the neural network
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(4, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation = 'linear', kernel_initializer='he_uniform'))
        model.compile(loss = 'mse', optimizer=Adam(self.learning_rate))
        return model

    def get_action(self, state): # e greedy exploration
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_values = self.model.predict(state)
            return np.argmax(q_values[0]) #pick action with highest action value

    def replay_memory(self, s_t1, a, r, s_t2, done): # store transition tuple
        self.memory.append((s_t1, a, r, s_t2, done))
        if self.epsilon > self.epsilon_min: #update epsilon
            self.epsilon *= self.epsilon_decay

    def train(self):
        if(len(self.memory) < self.train_start):
            return

        minibatch =  random.sample(self.memory,self.batch_size)
        state_t, action_t, reward_t, state_t1, done = zip(*minibatch) #list of states, actions, rewards...

    def load_model(self, name):
        self.model.load_weights(name)

    def save_model(self, name):
        self.model.save_weights(name)


def test(modelName):
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n #number of actions
    agent = DQN_agent(state_size, action_size,render = True)
    agent.load_model("modelName")

    done = False
    state = env.reset()
    state = np.reshape(state, [1, state_size])

    while not done:
        env.render()
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        state = next_state

def train(modelName):
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n #number of actions
    agent = DQN_agent(state_size, action_size)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n #number of actions

    scores, episodes= [], []
    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size]) #make array

        while not done:

            action = agent.get_action(state) #
            next_state, reward, done, info = env.step(action) #collect env feedback
            next_state = np.reshape(next_state, [1, state_size])
            #if action makes the episode end, give a penalty
            reward = reward if not done or score == 499 else -100
            agent.replay_memory(state, action, reward, next_state, done)
            agent.train()
            score+=reward
            state = next_state

            if done:#episode over
                env.reset()
                agent.update_target_model() #update target model after every episode
                score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)
                #pylab.plot(episodes, scores, 'b')
                #pylab.savefig("Cartpole_dqn.png")
                print("episode:", e, "  score:", score, "  memory length:", len(agent.memory),
                      "  epsilon:", agent.epsilon)
                #environment is considered solved if the mean score of 10 episodes is >490
                if np.mean(scores[-min(10,len(scores)):]) == 500:
                    agent.save_model("cartpole-dqn.h5")
                    sys.exit()
                #save model every 500 episodes
                if e % 500 == 0:
                    agent.save_model("cartpole-dqn.h5")
