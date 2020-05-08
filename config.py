# Name of the Gym environment for the agent to learn & play
ENV_NAME = 'PongDuel-v0'

# Loading and saving information.
# If LOAD_FROM is None, it will train a new agent.
# If SAVE_PATH is None, it will not save the agent
#LOAD_FROM = 'PongDuel-saves/save-L2'
RIGHT_LOAD_FROM = 'PongDuel-saves/save-R00575000'
LEFT_LOAD_FROM = 'PongDuel-saves/save-L00575000'
LOAD_FROM = None
SAVE_PATH = 'PongDuel-saves'
LOAD_REPLAY_BUFFER = True

# If True, use the prioritized experience replay algorithm, instead of regular experience replay
# This is much more computationally expensive, but will also allow for better results. Implementing
# a binary heap, as recommended in the PER paper, would make this less expensive.
# Since Breakout is a simple game, I wouldn't recommend using it here.
USE_PER = False

PRIORITY_SCALE = 0.7              # How much the replay buffer should sample based on priorities. 0 = complete random samples, 1 = completely aligned with priorities
CLIP_REWARD = False                # Any positive reward is +1, and negative reward is -1, 0 is unchanged

MAX_EPISODE_LENGTH = 20000        # Maximum length of an episode (in frames).  18000 frames / 60 fps = 5 minutes

DISCOUNT_FACTOR = 0.95            # Gamma, how much to discount future rewards
MIN_REPLAY_BUFFER_SIZE = 50000    # The minimum size the replay buffer must be before we start to update the agent
MEM_SIZE = 150000                # The maximum size of the replay buffer

UPDATE_FREQ = 100                   # Number of actions between gradient descent steps

INPUT_SHAPE = (12,)            # Size of the preprocessed input frame. With the current model architecture, anything below ~80 won't work.
BATCH_SIZE = 32                   # Number of samples the agent learns from at once
