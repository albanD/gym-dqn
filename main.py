import gym
import lua
import numpy as np

torch = lua.require('torch')
lua.require('trepl')
lua.require('cunn') # We run the network on GPU
dqn = lua.eval("dofile('dqn/NeuralQLearner.lua')")
tt = lua.eval("dofile('dqn/TransitionTable.lua')")

env = gym.make('CartPole-v0')

n_episode = 1000000
max_it = 250

possible_actions = lua.toTable({
        1: 0,
        2: 1
    })
input_dims = lua.toTable({
        1: env.observation_space.shape[0],
    })

dqn_args = {
    'target_q': 1000,
    'ncols': 1,
    'replay_memory': 1000,
    'min_reward': -100,
    'max_reward': 1,
    'discount': 0.99,
    'learn_start': 200,
    'hist_len': 1,
    'ep': 1,
    'network': "fcn_acrobot",
    'gpu': 0,
    'n_replay': 1,
    'clip_delta': 1,
    'valid_size': 100,
    'lr': 0.0025, # ??
    'lr_end': 0.00025, # ??
    'bufferSize': 150,
    'update_freq': 4,
    'minibatch_size': 2,
    'rescale_r': 1,
    'ep_end': 0.1,
    'state_dim': env.observation_space.shape[0],
    'actions': possible_actions,
    'verbose': False,
    'TransitionTable':tt.TransitionTable,
    'input_dims': input_dims
}

agent = dqn.NeuralQLearner(dqn_args)

running_t = 0
for i_episode in xrange(n_episode):
    observation = env.reset()
    action_index = 1
    done = False
    for t in xrange(max_it):
        #env.render()
        observation, reward, done, info = env.step(possible_actions[action_index])
        if done:
            reward = -100
        action_index = agent.perceive(agent, reward, observation, done)
        if done:
            #print "Episode finished after {} timesteps".format(t+1)
            running_t += t+1
            break
    if i_episode%1000 == 0:
        print "Episode finished after {} timesteps".format(running_t/1000)
        running_t = 0

torch.save("net.t7", agent)