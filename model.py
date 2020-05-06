import tensorflow as tf

import argparse

from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Add
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0005)
args = parser.parse_args()

def create_model(state_dim, action_dim, is_dueling=True):
    input_shape = (state_dim,)
    X_input = Input(shape=input_shape)
    X = X_input

    X = Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='glorot_uniform')(X)
    X = Dense(256, activation="relu", kernel_initializer='glorot_uniform')(X)
    X = Dense(64, activation="relu", kernel_initializer='glorot_uniform')(X)
    X = Dense(32, activation="relu", kernel_initializer='glorot_uniform')(X)
    X = Dense(16, activation="relu", kernel_initializer='glorot_uniform')(X)

    if is_dueling:
        state_value = Dense(1, kernel_initializer='he_uniform')(X)
        state_value = Lambda(lambda s: K.expand_dims(s[:,0], -1), output_shape=(action_dim,))(state_value)

        action_advantage = Dense(action_dim, kernel_initializer='he_uniform')(X)
        action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_dim,))(action_advantage)

        X = Add()([state_value, action_advantage])
    else:
        X = Dense(action_dim, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs=X_input, outputs=X)
    model.compile(loss='mse', optimizer=Adam(args.lr))

    model.summary()
    return model


def main():
    model = create_model(10, 3, True)
    plot_model(model, to_file='./model.png')

if __name__ == "__main__":
    main()
