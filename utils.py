import jax.numpy as jnp

def train_test_split(data, removed_steps = 1000, train_percentage = .8):

    data = data[removed_steps:]
    L_tot = len(data)
    L_train = int(len(data) * train_percentage)

    x_train = data[:L_train]
    x_test = data[L_train:]

    return x_train, x_test
