from collections import deque
import numpy as np
import gym
import random
import sys

from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras.losses import mean_absolute_error

EPISODES = 30000
class Agent():
    def __init__(self,state_shape, action_shape, *render):
        self.render = render
        if render == None:
            self.render = False
        self.state_size = state_shape
        self.action_size = action_shape

        self.gamma = 0.99 #discount factor
        self.alpha = 0.001 #learning rate
        self.batch_size = 32
        self.train_start = 1000 #start training after 1000 time steps of data has been collected
        # instantiate replay memory which is used to make samples independent
        self.memory = deque(maxlen=50000)

        # instantiate models
        self.policy = self.build_model('mean_absolute_error',action_shape)
        self.q1 = self.build_model('mse',action_shape)
        self.q2 = self.build_model('mse',action_shape)
        self.v = self.build_model('mse',1)
        self.target_v = self.build_model('mse',1)
        self.update_target_model()

    def update_target_model(self):
        self.target_v.set_weights(self.v.get_weights())

    def build_model(self,loss_function,outputSize): #the neural network
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(4, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(outputSize, activation = 'linear', kernel_initializer='he_uniform'))
        model.compile(loss = loss_function, optimizer=Adam(self.alpha))
        return model

    def get_action(self, state):
        return np.argmax(self.policy.predict(state)[0]) #pick action according to policy

    def store_replay(self, s_t1, a, r, s_t2, done): # store transition tuple
        self.memory.append((s_t1, a, r, s_t2, done))

    def updateQfunctions(self,state_t,action_t,reward_t,state_t1,d):
        targets = self.q1.predict(state_t)
        # q1_predicted = np.array([q1_vals[n][a] for n,a in enumerate(action_t)])
        q2_vals = self.q2.predict(state_t)
        # q2_predicted = np.array([q2_vals[n][a] for n,a in enumerate(action_t)])
        targets[range(self.batch_size),action_t] = reward_t + self.gamma * (np.ones(self.batch_size) - d) * self.target_v.predict(state_t1)
        print(f'q_targets shape: {q_targets.shape} q1_pred shape: {q1_predicted.shape}')
        self.q1.train_on_batch(state_t, q_targets)
        self.q2.train_on_batch(state_t, q_targets)
        return targets,q2_vals

    def updateValueFunction(self,state_t,action_t,reward_t,state_t1,d, action_probabilities,q1_vals,q2_vals):
        q1_a = np.array([q1_vals[n][a] for n,a in enumerate(action_theta)])
        q2_a = np.array([q2_vals[n][a] for n,a in enumerate(action_theta)])
        q = np.array([min(a,b) for a,b in zip(q1_a,q2_a)])
        v_targets = q - self.alpha * np.log(np.max(action_probabilities,axis=1))
        v_predicted = self.v.predict(state_t)
        self.v.train_on_batch(state_t, v_targets)

    def updatePolicy(self,state_t,action_probabilities,action_theta,q1_vals):
        mean = np.mean(action_probabilities,axis=1)
        std_dev = np.std(action_probabilities,axis=1)
        noise = np.random.normal(self.action_size,self.batch_size)
        for row in range(self.action_size):
            noise[row] = noise[row]*std_dev + mean
        reparameterized = np.argmax(noise,axis=1)
        policy_predicted = np.array([q1_vals[n][a] for n,a in enumerate(reparameterized)])
        policy_targets = self.alpha * np.log(np.max(noise,axis=1))
        self.policy.train_on_batch(policy_targets, policy_predicted)

    def update(self):
        if(len(self.memory) < self.train_start):
            return

        minibatch =  random.sample(self.memory,self.batch_size)
        state_t, action_t, reward_t, state_t1, done = zip(*minibatch) #list of states in the form [[state1]], [[state2]]
        state_t = np.concatenate(state_t) #list of list of states in the form [[state1,state2]]
        state_t1 = np.concatenate(state_t1)
        action_probabilities = self.policy.predict(state_t)
        action_theta = np.argmax(action_probabilities,axis=1)
        q1_vals, q2_vals = self.updateQfunctions(state_t,action_t,reward_t,state_t1,done)
        self.updateValueFunction(state_t,action_t,reward_t,state_t1,done,action_theta,q1_vals,q2_vals)
        self.updatePolicy(state_t,action_theta,q1_vals)

    def load_model(self, name):
        self.policy.load_weights(name+'policy.h5')

    def save_model(self, name):
        self.policy.save_weights(name+'_policy.h5')
        self.q1.save_weights(name+'_q1.h5')
        self.q2.save_weights(name+'_q2.h5')
        self.v.save_weights(name+'_v.h5')
        self.policy.save_weights(name+'_policy.h5')



def test(modelName):
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n #number of actions
    agent = Agent(state_size, action_size,True)
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

def train():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n #number of actions
    print(f"Created Cartpole Environment state_shape = {state_size} action_shape = {action_size}")
    agent = Agent(state_size, action_size)

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
            #add transition to buffer
            agent.store_replay(state, action, reward, next_state, done)
            agent.update()
            score+=reward
            state = next_state

            if done:#episode over
                env.reset()
                agent.update_target_model() #update target model after every episode
                score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)
                #environment is considered solved if the mean score of 10 episodes is >490
                if np.mean(scores[-min(10,len(scores)):]) == 500:
                    agent.save_model("cartpole-sac")
                    sys.exit()
                #save model every 500 episodes
                if e % 500 == 0:
                    agent.save_model("cartpole-sac")
if __name__ == '__main__':
    if len(sys.argv) >1:
        if sys.argv[1] == 'train':
            train()
        else:
            test()
    else:
        print("enter train or test")
        exit()
