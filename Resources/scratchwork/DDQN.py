from collections import deque
import numpy as np
import gym
import random
import sys
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.optimizers import Adam
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


def graph_triathlon(results):
    x = range(1,len(results[0])+1)
    plt.plot(x,results[0],'ro')

    x = range(1,len(results[1])+1)
    plt.plot(x,results[0],'go')

    x = range(1,len(results[2])+1)
    plt.plot(x,results[0],'bo')
    plt.savefig("triathlon_training_results.png")


def triathlon(render):# solve 3 tasks simultaneously
    envs = []
    scores = [[],[],[]]
    envs.append(gym.make('Bowling-ram-v0'))
    envs.append(gym.make('Pong-ram-v0'))
    envs.append(gym.make('SpaceInvaders-ram-v0'))
    state_size = 128
    action_size = 6
    agent = DQN_agent(state_size, action_size)
    agent.render = render
    for e in range(EPISODES):
        env = envs[e%3]
        done = False
        score = 0
        state = np.reshape(env.reset(), [1, state_size])
        ep_len = 0
        while not done:
            ep_len+=1
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)  # collect env feedback
            next_state = np.reshape(next_state, [1, state_size])
            agent.replay_memory(state, action, reward, next_state, done)
            agent.train()
            score+=reward
            state = next_state
            if done:
                env.reset()
                agent.update_target_model()
                scores[e%3].append(score)
                print(f'episode: {e}\tenv: {env.env.game}\tepisode length: {ep_len}\treward: {score}\tavg of 10: {np.mean(scores[e%3][-10:])}')
                if e % 100 == 0:
                    agent.save_model("triathlon-dqn.h5")

def trainer_cartpole(render):
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n  # number of actions
    agent = DQN_agent(state_size, action_size)
    agent.render = render
    scores, episodes = [], []
    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])  # make array
        while not done:
            if agent.render:
                env.render()
            action = agent.get_action(state)  #
            next_state, reward, done, info = env.step(action)  # collect env feedback
            next_state = np.reshape(next_state, [1, state_size])
            # if action makes the episode end, give a penalty
            reward = reward if not done or score == 499 else -100
            agent.replay_memory(state, action, reward, next_state, done)
            agent.train()
            score += reward
            state = next_state
            if done:  # episode over
                env.reset()
                agent.update_target_model()  # update target model after every episode
                score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)
                # pylab.plot(episodes, scores, 'b')
                # pylab.savefig("Cartpole_dqn.png")
                print("episode:", e, "  score:", score, "  memory length:", len(agent.memory),
                     "  epsilon:", agent.epsilon)
                # environment is considered solved if the mean score of 10 episodes is >490
                if np.mean(scores[-min(10, len(scores)):]) == 500:
                    agent.save_model("cartpole-dqn.h5")
                    sys.exit()
                # save model every 500 episodes
                if e % 500 == 0:
                    agent.save_model("cartpole-dqn.h5")


def test_cartpole(file):
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    agent = DQN_agent(state_size, env.action_space.n )
    agent.load_model(file)
    for i in range(10):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        while not done:
            env.render()
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state

def test_triathlon(file):
    envs = []
    scores = [[],[],[]]
    envs.append(gym.make('Bowling-ram-v0'))
    envs.append(gym.make('Pong-ram-v0'))
    envs.append(gym.make('SpaceInvaders-ram-v0'))
    observation_size = 128
    action_size = 6
    agent = DQN_agent(state_size, action_size)
    agent.load(file)
    for i in envs:
        env = gym.make(i)
        state = np.reshape(env.reset(), [1, state_size])
        done = False
        while not done:
            env.render()
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state

if __name__ == "__main__": #allows this dqn to be imported to other files
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--env",type=str)
    parser.add_argument("--model_file", type=str)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.train:
        if args.env == "CartPole-v1":
            trainer_cartpole(args.render)
        if args.env == "triathlon":
            triathlon(args.render)
    else:
        try:
            if args.env == "CartPole-v1":
                test_cartpole(args.model_file)
            if args.env == "triathlon":
                test_triathlon(args.model_file)
        except FileNotFoundError:
            print("Must train first before testing")
