import random
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def concat_obs(obs):
    if obs == None:
        return obs
    else:
        return obs[0][:2]+obs[1]

def start(mode, to_recv, to_send):
    while True:
        obs = to_recv.get()
        if obs == None:
            break
        obs = concat_obs(obs)
        action = random.randint(0, 2)
        to_send.put(action)

def train(mode, to_recv, to_send):
    while True:
        obs = to_recv.get()
        if obs == None:
            break
        obs = concat_obs(obs)
        action = random.randint(0, 2)
        to_sent.put(action)
