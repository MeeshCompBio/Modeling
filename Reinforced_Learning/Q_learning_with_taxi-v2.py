import numpy as np
import gym
import random

env = gym.make("Taxi-v2")
# look at what the grid entails
env.render()

# This is the following scheme
# +20 pts for successful drop off
# -1 pt for each timesetp
# -10 pts for illegal pickup or dropoff actions

# columns
action_size = env.action_space.n
print("Action size ", action_size)

# rows
state_size = env.observation_space.n
print("State size ", state_size)

# make a table based off rows and columns initialized with zeros
qtable = np.zeros((state_size, action_size))
print(qtable)

# creating the hyperparameters, this will tune the training
total_episodes = 50000     # Total episodes
# how many test we will run
total_test_episodes = 100  # Total test episodes
# prevent infinited loop
max_steps = 99             # Max steps per episode

learning_rate = 0.7        # Learning rate
gamma = 0.618              # Discounting rate

# Exploration parameters
# Set epsilon high since we want to explore a lot in beginning
epsilon = 1.0              # Exploration rate
max_epsilon = 1.0          # Exploration probability at start
min_epsilon = 0.01         # Minimum exploration probability
# changes ratio tradeoff for exploration vs exploitation
decay_rate = 0.01          # Exponential decay rate for exploration probability


# 2 For life or until learning is stopped
for episode in range(total_episodes):
    # Reset the environment
    state = env.reset()
    step = 0
    done = False

    for step in range(max_steps):
        # 3. Choose an action a in the current world state (s)
        # First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)

        # If this number > greater than epsilon -->
        # exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            # this will output index of action with highest -->
            # q-val for that state
            action = np.argmax(qtable[state, :])

        # If not, then make a random choice --> exploration
        else:
            action = env.action_space.sample()

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)

        # Use bellman equation to update Q value at that state
        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        qtable[state, action] = (qtable[state, action] +
                                 learning_rate *
                                 (reward + gamma *
                                 np.max(qtable[new_state, :]) -
                                 qtable[state, action])
                                 )

        # Our new state is state
        state = new_state

        # If done : finish episode
        if done:
            break

    # Reduce epsilon (because we need less and less exploration)
    epsilon = (min_epsilon +
               (max_epsilon - min_epsilon) *
               np.exp(-decay_rate*episode)
               )

# Time to test agent to see if it works correctly
env.reset()
rewards = []

for episode in range(total_test_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards = 0
    # print("****************************************************")
    # print("EPISODE ", episode)

    for step in range(max_steps):
        # UNCOMMENT IT IF YOU WANT TO SEE OUR AGENT PLAYING
        # env.render()
        # Take the action (index) that have the maximum expected -->
        # future reward given that state
        action = np.argmax(qtable[state, :])

        new_state, reward, done, info = env.step(action)

        total_rewards += reward

        if done:
            rewards.append(total_rewards)
            # print("Score", total_rewards)
            break
        state = new_state
env.close()
print("Score over time: " + str(sum(rewards)/total_test_episodes))
