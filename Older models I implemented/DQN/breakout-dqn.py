from collections import deque
import numpy as np
import gym
import random
import sys

from keras.layers import Dense
from keras.optimizers import Adam #look at adabound: https://github.com/Luolc/AdaBound
from keras.models import Sequential

EPISODES = 30000
class DQN_agent():
    def __init__(self, state_shape, action_shape):
        self.render = True #show the agent playing the game
        self.state_size = state_shape
        self.action_size = action_shape

        #hyper-parameters
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.batch_size = 32
        self.train_start = 1000
        #epsilon parameters
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = .999
        # instantiate replay memory which is used to make samples independent
        self.memory = deque(maxlen=50000)

        # instantiate models
        self.model = self.build_model()
        self.target_model = self.build_model() #used to evaluate action values
        self.update_target_model()

    def build_model(self): #the neural network
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(4, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation = 'linear', kernel_initializer='he_uniform'))
        model.compile(loss = 'mse', optimizer=Adam(self.learning_rate))
        return model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

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
        if len(self.memory) < self.train_start:
             #replay memory isnt big enough to start training yet
            return

        minibatch = random.sample(self.memory, self.batch_size)

        state_t, action_t, reward_t, state_t1, done = zip(*minibatch)
        state_t = np.concatenate(state_t) #list of states
        state_t1 = np.concatenate(state_t1) #list of next states
        targets = self.model.predict(state_t) #list of predicted next states
        q_val =self.target_model.predict(state_t1) #list of q values for next state
        #bellman equation Q(s,a) =  r + gamma*maxQ(s',a')
        targets[range(self.batch_size), action_t] = reward_t + self.discount_factor * np.max(q_val, axis = 1)
        self.model.train_on_batch(state_t, targets)

    def load_model(self, name):
        self.model.load_weights(name)

    def save_model(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    env = gym.make("Breakout-v0")
    env.render()
