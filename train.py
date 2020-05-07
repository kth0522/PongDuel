import gym
import random
import time
import numpy as np
from ma_gym.wrappers import Monitor
from model import create_model
from buffer import ReplayBuffer
from collections import deque
from agent import Agent
from datetime import datetime
import math

from config import (BATCH_SIZE, CLIP_REWARD, DISCOUNT_FACTOR,
                    INPUT_SHAPE, LOAD_FROM, LOAD_REPLAY_BUFFER,
                    MAX_EPISODE_LENGTH, MEM_SIZE,
                    MIN_REPLAY_BUFFER_SIZE, PRIORITY_SCALE,
                    UPDATE_FREQ, USE_PER)


def concat_obs(obs):
    if obs == None:
        return obs
    else:
        return np.array(obs[0][:2] + obs[1])


def random_action():
    return random.randint(0, 2)


def main():
    env = gym.make('PongDuel-v0')
    env = Monitor(env, directory='testings/PongDuel-v0', force=True)
    action_dim = env.action_space[0].n
    state_dim = env.observation_space[0].shape[0] + 2

    MAIN_DQN = create_model(state_dim, action_dim, is_dueling=True)
    TARGET_DQN = create_model(state_dim, action_dim, is_dueling=True)

    replay_buffer = ReplayBuffer(size=MEM_SIZE, input_shape=INPUT_SHAPE, use_per=USE_PER)
    agent = Agent(MAIN_DQN, TARGET_DQN, replay_buffer, action_dim, input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE,
                  use_per=USE_PER)

    if LOAD_FROM is None:
        state_number = 0
        rewards = []
        loss_list = []
    else:
        print('Loading from', LOAD_FROM)
        meta = agent.load(LOAD_FROM, LOAD_REPLAY_BUFFER)

        state_number = meta['state_number']
        rewards = meta['rewards']
        loss_list = meta['loss_list']

    try:
        last_50_l = deque(maxlen=50)
        last_50_r = deque(maxlen=50)
        recent_10_game = deque(maxlen=10)

        start_time = datetime.now()
        print(start_time)
        for ep in range(MAX_EPISODE_LENGTH):
            done_n = [False for _ in range(env.n_agents)]
            l_cnt = 0
            r_cnt = 0
            state = env.reset()
            state = concat_obs(state)
            while not all(done_n):
                trained_action = agent.get_action(state_number, state, evaluation=False)

                next_state, reward_n, done_n, _ = env.step([trained_action, random_action()])
                next_state = concat_obs(next_state)

                paddle_l = next_state[0]
                paddle_r = next_state[2]
                ball = state[4]

                delta_l = np.subtract(paddle_l, ball)
                delta_r = np.subtract(paddle_r, ball)

                # if reward_n[1] == 1:
                #     if delta_l > 0:
                #         if trained_action == 1:
                #             reward = 1
                #         else:
                #             reward = -1
                #     elif delta_l == 0:
                #         if trained_action == 0:
                #             reward = 1
                #         else:
                #             reward = -1
                #     else:
                #         if trained_action == 2:
                #             reward = 1
                #         else:
                #             reward = -1
                # elif reward_n[0] == 1:
                #     reward = abs(delta_r)*10
                # else:
                #     reward = reward_n[0]



                agent.add_experience(action=trained_action, state=state, reward=reward_n[0], clip_reward=CLIP_REWARD,
                                     done=done_n[0])

                state_number += 1
                l_reward = reward_n[0]
                r_reward = reward_n[1]
                l_cnt += l_reward
                r_cnt += r_reward

                if state_number % UPDATE_FREQ == 0 and agent.replay_buffer.count > MIN_REPLAY_BUFFER_SIZE:
                    loss, _ = agent.learn(BATCH_SIZE, gamma=DISCOUNT_FACTOR, state_number=state_number,
                                          priority_scale=PRIORITY_SCALE)
                    loss_list.append(loss)

                if state_number % UPDATE_FREQ == 0 and state_number > MIN_REPLAY_BUFFER_SIZE:
                    agent.update_target_network()

                state = next_state

            if l_cnt > r_cnt:
                is_win = 'Win'
                recent_10_game.append(1)
            elif l_cnt == r_cnt:
                is_win = 'draw'
            else:
                is_win = 'Lose'
                recent_10_game.append(0)

            last_50_l.append(l_cnt)
            last_50_r.append(r_cnt)
            avg_l, avg_r = np.mean(last_50_l), np.mean(last_50_r)
            recent_10_win_rate = np.mean(recent_10_game)
            cur_time = datetime.now()
            print("{}||Episode #{} {} left: {} right: {} / avg score {}:{} / recent 10 game win rate: {}".format(cur_time, ep, is_win, l_cnt, r_cnt, avg_l, avg_r, recent_10_win_rate))

        SAVE_PATH = 'PongDuel-saves'
        print('\nTraining end.')
        if SAVE_PATH is None:
            try:
                SAVE_PATH = input(
                    'Would you like to save the trained model? If so, type in a save path, otherwise, interrupt with '
                    'ctrl+c. ')
            except KeyboardInterrupt:
                print('\nExiting...')
        if SAVE_PATH is not None:
            print('Saving...')
            agent.save(f'{SAVE_PATH}/save-{str(state_number).zfill(8)}', state_number=state_number, rewards=rewards,
                       loss_list=loss_list)
            print('Saved.')

    except KeyboardInterrupt:
        SAVE_PATH = 'PongDuel-saves'
        print('\nTraining exited early.')
        if SAVE_PATH is None:
            try:
                SAVE_PATH = input(
                    'Would you like to save the trained model? If so, type in a save path, otherwise, interrupt with '
                    'ctrl+c. ')
            except KeyboardInterrupt:
                print('\nExiting...')
        if SAVE_PATH is not None:
            print('Saving...')
            agent.save(f'{SAVE_PATH}/save-{str(state_number).zfill(8)}', state_number=state_number, rewards=rewards,
                       loss_list=loss_list)
            print('Saved.')


if __name__ == "__main__":
    main()
