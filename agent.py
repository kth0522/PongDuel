import json
import os

import numpy as np

import tensorflow as tf

class Agent(object):
    def __init__(self,
                 dqn,
                 target_dqn,
                 replay_buffer,
                 n_actions,
                 input_shape=(12,),
                 batch_size=64,
                 eps_initial=1,
                 eps_mid=0.5,
                 eps_final=0.1,
                 eps_eval=0.0,
                 eps_annealing_states=1000000,
                 replay_buffer_start_size=50000,
                 max_states=25000000,
                 use_per=False):
        self.n_actions = n_actions
        self.input_shape = input_shape
        self.multistep = True
        self.n_step = 5

        self.gamma = 0.99

        self.replay_buffer_start_size = replay_buffer_start_size
        self.max_states = max_states
        self.batch_size = batch_size

        self.replay_buffer = replay_buffer
        self.use_per = use_per

        self.eps_initial = eps_initial
        self.eps_mid = eps_mid
        self.eps_final = eps_final
        self.eps_eval = eps_eval
        self.eps_annealing_states = eps_annealing_states

        self.slope = -(self.eps_initial - self.eps_final) / self.eps_annealing_states
        self.intercept = self.eps_initial - self.slope*self.replay_buffer_start_size
        self.slope_2 = -(self.eps_mid - self.eps_final) / (self.max_states - self.eps_annealing_states - self.replay_buffer_start_size)
        self.intercept_2 = self.eps_final - self.slope_2*self.max_states

        self.DQN = dqn
        self.target_dqn = target_dqn

    def calc_epsilon(self, state_number, evaluation=False):
        if evaluation:
            return self.eps_eval
        elif state_number < self.replay_buffer_start_size:
            return self.eps_initial
        elif state_number >= self.replay_buffer_start_size and state_number < self.replay_buffer_start_size + self.eps_annealing_states:
            return self.slope*state_number + self.intercept
        elif state_number >= self.replay_buffer_start_size + self.eps_annealing_states:
            return self.slope_2*state_number + self.intercept_2

    def get_action(self, state_number, state, evaluation=False):
        eps = self.calc_epsilon(state_number, evaluation)

        if np.random.rand(1) < eps:
            return np.random.randint(0, self.n_actions)

        q_vals = self.DQN.predict(np.atleast_2d(state.astype('float32')))[0]

        return q_vals.argmax()

    def update_target_network(self):
        self.target_dqn.set_weights(self.DQN.get_weights())

    def add_experience(self, action, state, reward, done, clip_reward=True):
        self.replay_buffer.add_experience(action, state, reward, done, clip_reward)

    def learn(self, batch_size, gamma, state_number, priority_scale=1.0):
        if self.use_per:
            if self.multistep:
                states, actions, multi_step_rewards, end_states, multi_step_dones, importance, indices = self.replay_buffer.get_minibatch(
                    batch_size=self.batch_size, priority_scale=priority_scale)
                importance = importance ** (1 - self.calc_epsilon(state_number))
            else:
                (states, actions, rewards, new_states, dones), importance, indices = self.replay_buffer.get_minibatch(batch_size=self.batch_size, priority_scale=priority_scale)
                importance = importance ** (1-self.calc_epsilon(state_number))
        elif self.multistep:
            states, actions, multi_step_rewards, end_states, multi_step_dones = self.replay_buffer.get_minibatch(batch_size=self.batch_size, priority_scale=priority_scale )
        else:
            states, actions, rewards, new_states, dones = self.replay_buffer.get_minibatch(batch_size=self.batch_size, priority_scale=priority_scale)

        if self.multistep:
            multi_step_rewards = np.asarray(multi_step_rewards)
            multi_step_dones = np.asarray(multi_step_dones)
            reward_sum = multi_step_rewards[0]
            gamma = self.gamma
            for i in range(1, self.n_step):
                step_reward = multi_step_rewards[i] * gamma
                reward_sum += step_reward
                gamma *= gamma
            end_states = np.asarray(end_states)
            states = np.asarray(states)
            actions = np.asarray(actions)
            dones = np.asarray(multi_step_dones)
            reward_sum = np.asarray(reward_sum)

            states = np.atleast_2d(states.astype('float32'))
            end_states = np.atleast_2d(end_states.astype('float32'))
            arg_q_max = self.DQN.predict(end_states).argmax(axis=1)
            future_q_vals = self.target_dqn.predict(end_states)
            double_q = future_q_vals[range(batch_size), arg_q_max]

            target_q = reward_sum + (gamma*double_q*(1-multi_step_dones))
        else:
            new_states = np.asarray(new_states)
            states = np.asarray(states)
            actions = np.asarray(actions)
            dones = np.asarray(dones)
            rewards = np.asarray(dones)

            states = np.atleast_2d(states.astype('float32'))
            new_states = np.atleast_2d(new_states.astype('float32'))
            arg_q_max = self.DQN.predict(new_states).argmax(axis=1)

            future_q_vals = self.target_dqn.predict(new_states)
            double_q = future_q_vals[range(batch_size), arg_q_max]

            target_q = rewards + (gamma*double_q*(1-dones))
        with tf.GradientTape() as tape:
            q_values = self.DQN(states)

            one_hot_actions = tf.keras.utils.to_categorical(actions, self.n_actions, dtype=np.float32)
            Q = tf.reduce_mean(tf.multiply(q_values, one_hot_actions), axis=1)

            error = Q - target_q
            loss = tf.keras.losses.MeanSquaredError()(target_q, Q)

            if self.use_per:
                loss = tf.reduce_mean(loss * importance)

        model_gradients = tape.gradient(loss, self.DQN.trainable_variables)
        self.DQN.optimizer.apply_gradients(zip(model_gradients, self.DQN.trainable_variables))

        if self.use_per:
            self.replay_buffer.set_priorities(indices, error)

        return float(loss.numpy()), error

    def save(self, folder_name, **kwargs):
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        self.DQN.save(folder_name+'/dqn.h5')
        self.target_dqn.save(folder_name+'/target_dqn.h5')

        self.replay_buffer.save(folder_name + '/replay-buffer')

        with open(folder_name+'/meta.json', 'w+') as f:
            f.write(json.dumps({**{'buff_count': self.replay_buffer.count, 'buff_curr': self.replay_buffer.current}, **kwargs}))

    def load(self, folder_name, load_replay_buffer=True):
        self.DQN = tf.keras.models.load_model(folder_name+'/dqn.h5')
        self.target = tf.keras.models.load_model(folder_name+'/target_dqn.h5')
        self.optimizer = self.DQN.optimizer

        if load_replay_buffer:
            self.replay_buffer.load(folder_name+'/replay-buffer')

        with open(folder_name+'/meta.json', 'r') as f:
            meta = json.load(f)

        if load_replay_buffer:
            self.replay_buffer.count = meta['buff_count']
            self.replay_buffer.current = meta['buff_curr']

        del meta['buff_count'], meta['buff_curr']
        return meta