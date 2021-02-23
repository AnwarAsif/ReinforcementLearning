import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')

learning_rate = 0.1
discount = 0.95
episodes = 2000
show_every = 500

# Making Q-Table
Discrete_obs_size = [20] * len(env.observation_space.high)
Discrete_win_size = (env.observation_space.high - env.observation_space.low) / Discrete_obs_size
q_table = np.random.uniform(low=-2, high=0, size=(Discrete_obs_size + [env.action_space.n]))
print((Discrete_obs_size + [env.action_space.n]))
# Exploration and exploitation
epsilon = 0.5
start_epsilon_decaying = 1
end_epsilon_decaying = episodes // 2
epsilon_decay_value = epsilon/(end_epsilon_decaying - start_epsilon_decaying)

# Reward
epz_reward = []
agg_epz_reward = {'epz':[],'avg':[], 'min':[], 'max':[]}

print(env.observation_space.high)
print(env.observation_space.low)


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / Discrete_win_size
    return tuple(discrete_state.astype(int))

for episode in range(episodes):
    episode_reward = 0

    # Render in a given interval
    if episode % show_every == 0:
        print(episode)
        render = True
    else:
        render = False

    discrete_state = get_discrete_state(env.reset())

    done = False
    while not done:
        # Using the Epsilon value for exploration
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        # Get New State details
        new_state, reward, done, _ = env.step(action)

        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)

        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            # Algorithm
            new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
            # update q table
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            print("Goal achived on Epz:", episode)
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

    #Tuning the Epsilon value
    if epsilon_decay_value >= episode >= start_epsilon_decaying:
        epsilon -= epsilon_decay_value

    epz_reward.append(episode_reward)

    if not episode % show_every:
        avarage_reward = sum(epz_reward[-show_every:])/len(epz_reward[-show_every:])
        agg_epz_reward['epz'].append(episode)
        agg_epz_reward['avg'].append(avarage_reward)
        agg_epz_reward['min'].append(min(epz_reward[-show_every:]))
        agg_epz_reward['max'].append(max(epz_reward[-show_every:]))

        print(f"Episode:, {episode} Avarage:{avarage_reward}, Min:{min(epz_reward[-show_every:])} , Max:{max(epz_reward[-show_every:])}")

env.close()


plt.plot(agg_epz_reward['epz'], agg_epz_reward['avg'], label="average rewards")
plt.plot(agg_epz_reward['epz'], agg_epz_reward['max'], label="max rewards")
plt.plot(agg_epz_reward['epz'], agg_epz_reward['min'], label="min rewards")
plt.legend(loc=4)
plt.show()