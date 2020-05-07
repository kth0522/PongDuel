import numpy as np
import os
import random

class ReplayBuffer:
    def __init__(self, size=1000000, input_shape=(12,), use_per=True):
        self.size = size
        self.input_shape = input_shape
        self.count = 0 # total index of memory written to
        self.current = 0 # index to write to
        self.multistep=True
        self.n_step = 3
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.states = np.empty((self.size, self.input_shape[0]))
        self.dones = np.empty(self.size, dtype=np.bool)
        self.priorities = np.zeros(self.size, dtype=np.float32)

        self.use_per = use_per

    def add_experience(self, action, state, reward, done, clip_reward=True):
        if clip_reward:
            reward = np.sign(reward)

        self.actions[self.current] = action
        self.states[self.current, ...] = state
        self.rewards[self.current] = reward
        self.dones[self.current] = done
        self.priorities[self.current] = max(self.priorities.max(), 1) # most recent experience is the most important
        self.count = max(self.count, self.current+1)
        self.current = (self.current+1) % self.size

    def get_minibatch(self, batch_size=32, priority_scale=0.0):
        if self.use_per:
            scaled_priorities = self.priorities[:self.count-1] ** priority_scale
            sample_probabilities = scaled_priorities / sum(scaled_priorities)

        indices = []
        for i in range(batch_size):
            while True:
                if self.use_per:
                    index = np.random.choice(np.arange(0, self.count-1), p=sample_probabilities)
                elif self.multistep:
                    index = np.random.choice(np.arange(0, max(self.count-1-self.multistep, 0)))
                else:
                    index = random.randint(0, self.count-1)

                break
            indices.append(index)

        states = []
        new_states = []
        end_states = []
        multi_step_rewards = []
        multi_step_dones = []
        if self.multistep:
            for i in range(self.n_step):
                n_step_rewards = []
                for idx in indices:
                    n_step_rewards.append(self.rewards[idx+i, ...])
                multi_step_rewards.append(n_step_rewards)
            for idx in indices:
                states.append(self.states[idx, ...])
                end_states.append(self.states[idx + self.n_step, ...])
                multi_step_dones.append(self.rewards[idx+self.n_step, ...])
        else:
            for idx in indices:
                states.append(self.states[idx, ...])
                new_states.append(self.states[idx+1, ...])

        if self.use_per:
            importance = 1/self.count * 1/sample_probabilities[[index for index in indices]]
            importance = importance / importance.max()

            return (states, self.actions[indices], self.rewards[indices], new_states, self.dones[indices]), importance, indices
        elif self.multistep:
            return states, self.actions[indices], multi_step_rewards, end_states, multi_step_dones
        else:
            return states, self.actions[indices], self.rewards[indices], new_states, self.dones[indices]

    def set_priorities(self, indices, errors, offset=0.1):
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset

    def save(self, folder_name):
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        np.save(folder_name + '/actions.npy', self.actions)
        np.save(folder_name + '/states.npy', self.states)
        np.save(folder_name + '/rewards.npy', self.rewards)
        np.save(folder_name + '/dones.npy', self.dones)

    def load(self, folder_name):
        self.actions = np.load(folder_name + '/actions.npy')
        self.states = np.load(folder_name + '/states.npy')
        self.rewards = np.load(folder_name + '/rewards.npy')
        self.dones = np.load(folder_name+'/dones.npy')
