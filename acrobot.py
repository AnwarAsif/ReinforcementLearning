import gym
import numpy as np

env = gym.make('Acrobot-v1')

episodes = 25000
show_now = 1000
learning_rate = 0.1
# Value of Future value
discount = 0.95


high = env.observation_space.high
low = env.observation_space.low
no_of_actions = env.action_space.n
# making q-table
obs_size = [20] * len(high)
win_size = (high - low)/obs_size
q_table = np.random.uniform(low=-2, high=0, size=(obs_size + [env.action_space.n]))
#print(q_table.shape)



def get_state(state):
    dis_state = (state - env.observation_space.low)/ obs_size
    return tuple(dis_state.astype(np.int))

for episode in range(episodes):
    if episode % show_now == 0:
        print(episode)
        render = True
    else:
        render = False
        
    dis_state = get_state(env.reset())
    #print(np.argmax(q_table[dis_state]))
    done = False
    while not done:

        action = np.argmax(q_table[dis_state])
        new_state, reward, done, _ = env.step(action)
        new_dis_state = get_state(new_state)
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_dis_state])
            current_q = q_table[dis_state +(action,)]

            new_q = (1 -learning_rate) * current_q + learning_rate * (reward * discount + max_future_q)

            q_table[dis_state +(action,)] = new_q
        elif new_state[0] >= 0:
            q_table[dis_state + (action,)] = 0

        dis_state = new_dis_state


env.close()
