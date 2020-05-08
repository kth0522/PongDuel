import os
import gym
import argparse
from collections import deque
from ma_gym.wrappers import Monitor
from multiprocessing import Process, Queue
from buffer import ReplayBuffer
from agent import Agent
from model import create_model
from datetime import datetime
import numpy as np
from config import (BATCH_SIZE, CLIP_REWARD, DISCOUNT_FACTOR,
                    INPUT_SHAPE, LOAD_FROM, LOAD_REPLAY_BUFFER,
                    MAX_EPISODE_LENGTH, MEM_SIZE,
                    MIN_REPLAY_BUFFER_SIZE, PRIORITY_SCALE,
                    UPDATE_FREQ, USE_PER, LEFT_LOAD_FROM,RIGHT_LOAD_FROM,SAVE_PATH)

def concat_obs(obs):
    if obs == None:
        return obs
    else:
        return np.array(obs[0][:2] + obs[1])

def which_dir(dir):
    for i in range(6):
        if dir[i] == 1:
            if i < 3:
                return 0
            else:
                return 1

def is_hit(dir, next_dir, is_end_round):
    if which_dir(dir) != which_dir(next_dir):
        return True
    else:
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Agent for ma-gym')
    parser.add_argument('--env', default='PongDuel-v0',
                        help='Name of the environment (default: %(default)s)')
    parser.add_argument('--episodes', type=int, default=500000,
                        help='episodes (default: %(default)s)')
    args = parser.parse_args()

    reward_list = deque(maxlen=100)
    env = gym.make(args.env)
    env = Monitor(env, directory='testings/' + args.env, force=True)

    action_dim = 3
    state_dim = 12

    LEFT_MAIN_DQN = create_model(state_dim, action_dim, is_dueling=True)
    LEFT_TARGET_DQN = create_model(state_dim, action_dim, is_dueling=True)

    LEFT_replay_buffer = ReplayBuffer(size=MEM_SIZE, input_shape=INPUT_SHAPE, use_per=USE_PER)
    LEFT_agent = Agent(LEFT_MAIN_DQN, LEFT_TARGET_DQN, LEFT_replay_buffer, action_dim, input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE,
                  use_per=USE_PER)

    RIGHT_MAIN_DQN = create_model(state_dim, action_dim, is_dueling=True)
    RIGHT_TARGET_DQN = create_model(state_dim, action_dim, is_dueling=True)

    RIGHT_replay_buffer = ReplayBuffer(size=MEM_SIZE, input_shape=INPUT_SHAPE, use_per=USE_PER)
    RIGHT_agent = Agent(RIGHT_MAIN_DQN, RIGHT_TARGET_DQN, RIGHT_replay_buffer, action_dim, input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE,
                  use_per=USE_PER)


    if LOAD_FROM is None:
        state_number = 0
        LEFT_loss_list = []
        RIGHT_loss_list = []
    else:
        print('Loading from', LEFT_LOAD_FROM, RIGHT_LOAD_FROM)
        meta_l = LEFT_agent.load(LEFT_LOAD_FROM, LOAD_REPLAY_BUFFER)
        meta_r = RIGHT_agent.load(RIGHT_LOAD_FROM, LOAD_REPLAY_BUFFER)

        state_number = meta_l['state_number']
        LEFT_loss_list = meta_l['LEFT_loss_list']
        RIGHT_loss_list = meta_r['RIGHT_loss_list']

    try:
        for ep_i in range(args.episodes):
            done_n = [False for _ in range(env.n_agents)]
            l_cnt = 0
            r_cnt = 0

            env.seed(ep_i)
            obs_n = env.reset()

            state = concat_obs(obs_n)
            round = 0
            cur_round = 0

            is_l_hit = False
            is_r_hit = False

            while not all(done_n):
                l_action = LEFT_agent.get_action(state_number, state, evaluation=False)
                r_action = RIGHT_agent.get_action(state_number, state, evaluation=False)
                end_round = False
                next_state, reward_n, done_n, info = env.step([l_action, r_action])
                next_state = concat_obs(next_state)
                cur_round = info['rounds']

                if cur_round != round:
                    is_l_hit = False
                    is_r_hit = False
                    round = cur_round
                    end_round = True

                paddle_l = np.array([state[0], state[1]])
                paddle_r = np.array([state[2], state[3]])
                ball = np.array([state[4], state[5]])
                dir = np.array(state[6:12])
                next_dir = np.array(next_state[6:12])

                delta_l = np.square(np.subtract(paddle_l, ball)).mean()
                delta_r = np.square(np.subtract(paddle_r, ball)).mean()

                if reward_n[1] == 1:
                    if is_r_hit:
                        reward_r = abs(delta_l)*20
                        reward_l = -abs(delta_l)*15
                    else:
                        reward_r = abs(delta_l)*10
                        reward_l = -abs(delta_l)*10
                elif reward_n[0] == 1:
                    if is_l_hit:
                        reward_r = -abs(delta_r)*15
                        reward_l = abs(delta_r)*20
                    else:
                        reward_r = -abs(delta_r) * 10
                        reward_l = abs(delta_r)*10
                else:
                    if is_hit(dir, next_dir, end_round) and ball[1] < 0.5:
                        #print('left hit!')
                        is_l_hit = True
                        reward_r = 0
                        reward_l = 1
                    elif is_hit(dir, next_dir, end_round) and ball[1] > 0.5:
                        #print('right hit!')
                        is_r_hit = True
                        reward_r = 1
                        reward_l = 0
                    else:
                        reward_l = 0
                        reward_r = 0

                eps = LEFT_agent.calc_epsilon(state_number=state_number)

                LEFT_agent.add_experience(action=l_action, state=state, reward=reward_l, clip_reward=CLIP_REWARD,
                                     done=done_n[0])
                RIGHT_agent.add_experience(action=r_action, state=state, reward=reward_r, clip_reward=CLIP_REWARD,
                                          done=done_n[1])

                state_number += 1

                if state_number % UPDATE_FREQ == 0 and LEFT_agent.replay_buffer.count > MIN_REPLAY_BUFFER_SIZE:
                    LEFT_loss, _ = LEFT_agent.learn(BATCH_SIZE, gamma=DISCOUNT_FACTOR, state_number=state_number,
                                          priority_scale=PRIORITY_SCALE)
                    LEFT_loss_list.append(LEFT_loss)

                    RIGHT_loss, _ = LEFT_agent.learn(BATCH_SIZE, gamma=DISCOUNT_FACTOR, state_number=state_number,
                                               priority_scale=PRIORITY_SCALE)
                    RIGHT_loss_list.append(RIGHT_loss)

                if state_number % UPDATE_FREQ == 0 and state_number > MIN_REPLAY_BUFFER_SIZE:
                    LEFT_agent.update_target_network()
                    RIGHT_agent.update_target_network()

                l_reward = reward_n[0]
                r_reward = reward_n[1]
                l_cnt += l_reward
                r_cnt += r_reward

                state = next_state


            cur_time = datetime.now()
            print("{}||Episode #{} left: {} right: {}||state_number: {}||cur_epsilon: {}".format(cur_time, ep_i, l_cnt, r_cnt, state_number, eps))



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
            LEFT_agent.save(f'{SAVE_PATH}/save-L{str(state_number).zfill(8)}', state_number=state_number, LEFT_loss_list=LEFT_loss_list)
            RIGHT_agent.save(f'{SAVE_PATH}/save-R{str(state_number).zfill(8)}', state_number=state_number, RIGHT_loss_list=RIGHT_loss_list)
            print('Saved.')
    except KeyboardInterrupt:
        if SAVE_PATH is None:
            try:
                SAVE_PATH = input(
                    'Would you like to save the trained model? If so, type in a save path, otherwise, interrupt with '
                    'ctrl+c. ')
            except KeyboardInterrupt:
                print('\nExiting...')
        if SAVE_PATH is not None:
            print('Saving...')
            LEFT_agent.save(f'{SAVE_PATH}/save-L{str(state_number).zfill(8)}', state_number=state_number,
                            LEFT_loss_list=LEFT_loss_list)
            RIGHT_agent.save(f'{SAVE_PATH}/save-R{str(state_number).zfill(8)}', state_number=state_number,
                             RIGHT_loss_list=RIGHT_loss_list)
            print('Saved.')
