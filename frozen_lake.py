import numpy as np
import gym
import random
import time
from IPython.display import clear_output

env = gym.make('FrozenLake-v0')

# Making q_table
state_space_size = env.observation_space.n
action_space_size = env.action_space.n
q_table = np.zeros((state_space_size, action_space_size))
print("Q Table Size:", q_table.size)

# Define Episodes
num_episodes = 25000
max_step_per_episode = 100

# Learning Rate
learning_rate = 0.1 # Alpha in the equitation
discount_rate = 0.99 # Game in the equitation (Prioratise present reward from future rewared in continues state env)

# Exploration & Exploitation
exploration_rate = 1 # This is epsilon
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001


episode_rewards = []

for episode in range(num_episodes):
  state = env.reset()
  done = False
  reward_current_episode = 0

  for step in range(max_step_per_episode):
    # Select Action by exploration or exploitation
    exploration_rate_threshold = random.uniform(0,1)
    if exploration_rate_threshold > exploration_rate:
      # select exploitaion
      action = np.argmax(q_table[state, :])
    else:
      # select exploration
      action = env.action_space.sample()

    # Using selection action to observe next step
    new_state, reward, done, info = env.step(action)

    # Update Q_table with q learning algorithm
    q_table[state, action] = (1 - learning_rate) * (q_table[state, action]) + \
    learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

    # Update current state to new state and reward
    state = new_state
    reward_current_episode += reward

    # Exit the state if tusk is done
    if done == True:
      break

  # Update Exploration rate by using exploration decay
  exploration_rate = min_exploration_rate + \
    (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)

  #print(reward_current_episode)
  episode_rewards.append(reward_current_episode)

# Avarage Reward per thousand episode
rewards_per_thosand_episodes = np.split(np.array(episode_rewards),num_episodes/1000)
count = 1000

print("********Average reward per thousand episodes********\n")
for r in rewards_per_thosand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000
# Update Qtable
print(q_table)

# Watch our agent play Frozen Lake by playing the best action
# from each state according to the Q-table

for episode in range(3):
    # initialize new episode params
    state = env.reset()
    done = False
    print("*****EPISODE ", episode+1, "*****\n\n\n\n")
    time.sleep(1)

    for step in range(max_step_per_episode):
        # Show current state of environment on screen
        # Choose action with highest Q-value for current state
        # Take new action
        clear_output(wait=True)
        env.render()
        time.sleep(0.3)

        action = np.argmax(q_table[state,:])
        new_state, reward, done, info = env.step(action)

        if done:
          clear_output(wait=True)
          env.render()
          if reward == 1:
              print("****You reached the goal!****")
              time.sleep(3)
          else:
              print("****You fell through a hole!****")
              time.sleep(3)
              clear_output(wait=True)
          break

        state = new_state


env.close()
