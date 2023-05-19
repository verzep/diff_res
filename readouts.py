from jax import config

config.update("jax_enable_x64", True)

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import jax
from copy import copy
import jax.numpy as jnp
from jax import random, grad, jit
from functools import partial

from utils import compute_MSE

import matplotlib.pyplot as plt


class LinearReadout:
    def __init__(self, n_dim=500, reg_param=1e-6):

        self.reg_param = reg_param
        self.n_dim = n_dim

        self.W_out = None

    def fit(self,  input, input_dot, states, states_dot, washout_steps):
        s = states[washout_steps:]
        i = input[washout_steps:]

        W_out = (jnp.linalg.pinv(s.T @ s + self.reg_param * jnp.eye(self.n_dim))) @ s.T @ i
        self.W_out = W_out
        return W_out


    def predict(self, states):
        return states @ self.W_out


    def deriv_predict(self, derivatives, states):
        return derivatives @self.W_out

class LinearReadoutWithDerivatives:
    def __init__(self, n_dim=500, reg_param=1e-6, alpha=0):
        self.reg_param = reg_param
        self.n_dim = n_dim
        self.alpha = alpha
        self.W_out = None

    def fit(self, input, input_dot, states, states_dot, washout_steps):
        s = states[washout_steps:]
        s_dot = states_dot[washout_steps:]
        i = input[washout_steps:]
        i_dot = input_dot[washout_steps:]

        ss = jnp.vstack([jnp.sqrt(1 - self.alpha) * s, self.alpha * s_dot])
        ii = jnp.vstack([jnp.sqrt(1 - self.alpha) * i, self.alpha * i_dot])

        W_out = (jnp.linalg.pinv(ss.T @ ss + self.reg_param * jnp.eye(self.n_dim))) @ ss.T @ ii
        self.W_out = W_out
        return W_out


    def predict(self, states):
        return states @ self.W_out


    def deriv_predict(self, derivatives, states):
        return derivatives @ self.W_out


class QuadraticReadout:

    def __init__(self, n_dim=500, reg_param=1e-6):
        self.reg_param = reg_param
        self.n_dim = n_dim

        self.W_l = None
        self.W_nl = None

    def fit(self,  input, input_dot, states, states_dot,washout_steps):
        s = jnp.hstack((states[washout_steps:], states[washout_steps:] ** 2))

        i = input[washout_steps:]

        W_out = (jnp.linalg.pinv(s.T @ s + self.reg_param * jnp.eye(self.n_dim*2))) @ s.T @ i
        W_l = W_out[:self.n_dim]
        W_nl = W_out[self.n_dim:]

        self.W_l = W_l
        self.W_nl = W_nl
        return W_l, W_nl


    def predict(self, states):
        return states @ self.W_l + states ** 2 @ self.W_nl


    def deriv_predict(self, derivatives, states):
        return derivatives @ self.W_l + 2 * derivatives * states @ self.W_nl
