# This is a sample Python script.
from jax import config

from utils import train_test_split, compute_forecast_horizon

import jax


import matplotlib.pyplot as plt
import numpy as np
from copy import copy
import jax.numpy as jnp
from jax import random, grad, jit
from functools import partial

from dysts.datasets import load_dataset
from utils import train_test_split, compute_forecast_horizon, calculate_lyapunov_exponent

from dysts.flows import Lorenz, Rossler
from readouts import *

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import jax
from copy import copy
import jax.numpy as jnp
from jax import random, grad, jit
from functools import partial

from utils import compute_MSE

import matplotlib.pyplot as plt
from readouts import *


def _rdot(x: jnp.ndarray, r: jnp.ndarray, gamma: float, W_in: jnp.ndarray,
          sigma: float, W: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the derivative of the reservoir state `r` with respect to time.

    Parameters
    ----------
    x :  jnp.ndarray
        The input signal.
    r :  jnp.ndarray
        The current reservoir state.
    gamma : float
        The leakage rate.
    W_in :  jnp.ndarray
        The input weights.
    sigma :  jnp.ndarray
        The scaling factor for the input signal.
    W :  jnp.ndarray
        The recurrent weights of the reservoir.
    bias :  jnp.ndarray
        The bias term.

    Returns
    -------
     jnp.ndarray
        The derivative of the reservoir state `r` with respect to time.
    """


    r_dot = gamma * (-r + jnp.tanh(x @ W_in * sigma + r @ W + bias))
    return r_dot

def _rdot_linear(x: jnp.ndarray, r: jnp.ndarray, gamma: float, W_in: jnp.ndarray,
          sigma: float, W: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the derivative of the reservoir state `r` with respect to time.

    Parameters
    ----------
    x :  jnp.ndarray
        The input signal.
    r :  jnp.ndarray
        The current reservoir state.
    gamma : float
        The leakage rate.
    W_in :  jnp.ndarray
        The input weights.
    sigma :  jnp.ndarray
        The scaling factor for the input signal.
    W :  jnp.ndarray
        The recurrent weights of the reservoir.
    bias :  jnp.ndarray
        The bias term.

    Returns
    -------
     jnp.ndarray
        The derivative of the reservoir state `r` with respect to time.
    """
    print("USING LINEAR STEP")
    r_dot = gamma * (-r + x @ W_in * sigma + r @ W + bias)
    return r_dot


def _step(x: jnp.ndarray, r: jnp.ndarray, dt: float, gamma: float,
          W_in: jnp.ndarray, sigma: float, W: jnp.ndarray, bias: jnp.ndarray):
    """
    Perform a single step of the reservoir simulation.

    Parameters
    ----------
    x : jnp.ndarray
        The input signal.
    r : jnp.ndarray
        The current reservoir state.
    dt : float
        The simulation time step.
    gamma : float
        The leakage rate of the reservoir.
    W_in : jnp.ndarray
        The input weight matrix.
    sigma : float
        The scaling factor of the input.
    W : jnp.ndarray
        The reservoir weight matrix.
    bias : jnp.ndarray
        The bias vector.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        A tuple containing the updated reservoir state and its time derivative.
    """
    r_dot = _rdot(x, r, gamma, W_in, sigma, W, bias)
    r = r + r_dot * dt
    return r, r_dot


def _step_linear(x: jnp.ndarray, r: jnp.ndarray, dt: float, gamma: float,
          W_in: jnp.ndarray, sigma: float, W: jnp.ndarray, bias: jnp.ndarray):
    """
    Perform a single step of the reservoir simulation.

    Parameters
    ----------
    x : jnp.ndarray
        The input signal.
    r : jnp.ndarray
        The current reservoir state.
    dt : float
        The simulation time step.
    gamma : float
        The leakage rate of the reservoir.
    W_in : jnp.ndarray
        The input weight matrix.
    sigma : float
        The scaling factor of the input.
    W : jnp.ndarray
        The reservoir weight matrix.
    bias : jnp.ndarray
        The bias vector.

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        A tuple containing the updated reservoir state and its time derivative.
    """
    r_dot = _rdot_linear(x, r, gamma, W_in, sigma, W, bias)
    r = r + r_dot * dt
    return r, r_dot



class RCN:

    def __init__(self,
                 key,
                 readout,
                 n_dim: int = 500,
                 n_input: int = 1,
                 spectral_radius: float = 0.9,
                 gamma: float = 10,
                 sigma: float = 0.02,
                 bias: float = 0.1,
                 dt: float = 1e-3,
                 washout_steps: int = None,
                 linear_step = False
                 ):


        """
        Initializes the RCN model.

        Parameters
        ----------
        key : jax.random.PRNGKey
            The random seed for initialization.
        n_dim : int, optional
            The dimension of the reservoir, by default 500.
        n_input : int, optional
            The number of inputs to the RCN, by default 1.
        spectral_radius : float, optional
            The desired spectral radius of the reservoir weight matrix, by default 0.9.
        gamma : float, optional
            The time constant of the neurons, by default 10.
        sigma : float, optional
            The scaling factor of the input weights, by default 0.02.
        bias : float, optional
            The bias current of the neurons, by default 0.1.
        dt : float, optional
            The time step size for numerical integration, by default 1e-3.
        washout_steps : int, optional
            The number of steps to discard from the beginning of the input sequence, by default None.
        reg_param : float, optional
            The regularization parameter for the output weight matrix, by default 1e-4.
        """


        self.key, W_key, W_in_key, bias_key = random.split(key, 4)
        self.readout = readout
        self.n_dim = n_dim
        self.n_input = n_input
        self.spectral_radius = spectral_radius
        self.gamma = gamma
        self.sigma = sigma
        self.bias = bias * jnp.array(random.normal(bias_key, [n_dim]))
        self.dt = dt
        self.washout_steps = washout_steps

        self.W = self._create_reservoir(W_key, self.spectral_radius)
        self.W_in = self._create_W_in(W_in_key)

        self.r_last = None
        self.R = None
        self.R_dot = None
        self.input = None
        self.input_dot = None


        if linear_step:
            self.rdot = jit(partial(_rdot_linear, gamma=self.gamma, W_in=self.W_in,
                                    sigma=self.sigma, W=self.W, bias=self.bias))

            self.step = jit(partial(_step_linear, dt=self.dt, gamma=self.gamma, W_in=self.W_in,
                                    sigma=self.sigma, W=self.W, bias=self.bias))
        else:
            self.rdot = jit(partial(_rdot, gamma=self.gamma, W_in=self.W_in,
                                sigma=self.sigma, W=self.W, bias=self.bias))


            self.step = jit(partial(_step, dt=self.dt, gamma=self.gamma, W_in=self.W_in,
                                sigma=self.sigma, W=self.W, bias=self.bias))

    def listen(self, input, r_0=None):
        """
                Computes the reservoir states in response to the input sequence.

                Parameters
                ----------
                input : jnp.ndarray
                    The input sequence.
                r_0 : jnp.ndarray, optional
                    The initial state of the reservoir, by default None.

                Returns
                -------
                jnp.ndarray
                    An array of the reservoir states in response to the input sequence.
                """

        if r_0 is None:
            r_0 = jnp.zeros((self.n_dim,))

        r = copy(r_0)
        R = []
        R_dot = []

        for x in input:
            R.append(r)
            r, r_dot = self.step(x, r)
            R_dot.append(r_dot)

        self.R = jnp.array(R)
        self.R_dot = jnp.array(R_dot)
        self.r_last = copy(r)

        return self.R


    def train(self, input, input_dot = None, states=None, states_dot = None):

        """
        Trains the output weights of the RCN.

        Parameters
        ----------
        input : jnp.ndarray
            The input sequence.
        states : jnp.ndarray, optional
            The reservoir states in response to the input sequence, by default None.

        Returns
        -------
        jnp.ndarray
            The trained output weight matrix.
        """
        self.input = input

        if self.washout_steps is None:
            self.washout_steps = int(len(input)/10)

        if states is None:
            states = self.listen(input)
            states_dot = self.R_dot

        self.readout.fit(input, input_dot, states, states_dot, self.washout_steps)


    def generate(self, n_steps):
        """
        Generates an output sequence from the RCN.

        Parameters
        ----------
        n_steps : int
            The number of steps to be generated

        Returns
        -------
        jnp.ndarray
            The generated output sequence.
        """

        R_test = []
        X_test = []
        r = copy(self.r_last)

        for _ in range(n_steps):
            x = self.readout.predict(r)  # change with a predict function
            R_test.append(r)
            X_test.append(x)
            r, r_dot = self.step(x, r)

        self.R_test = jnp.array(R_test)
        self.X_test = jnp.array(X_test)
        return self.X_test

    def _create_reservoir(self, res_key, spectral_radius):
        """
        Creates the reservoir of the ESN network using random normal values multiplied by the spectral radius.

        Parameters
        ----------
        res_key : numpy.ndarray
            Key used for the random generator of the reservoir.
        spectral_radius : float or None
            Spectral radius used to scale the reservoir. If None, then the instance's `spectral_radius` attribute is used.

        Returns
        -------
        W : numpy.ndarray
            Reservoir of the ESN network.
        """

        if spectral_radius == None:
            spectral_radius = self.spectral_radius
        W = random.normal(res_key, shape=(self.n_dim, self.n_dim))
        W = spectral_radius * W / jnp.sqrt(self.n_dim)
        return W

    def _create_W_in(self, input_key):
        """
        Creates the input weight matrix W_in.

        Parameters
        ----------
        input_key: PRNGKey
            The key to use for generating the random numbers.

        Returns
        -------
        W_in: ndarray of shape (n_input, n_dim)
            The input weight matrix.
        """
        W_in = random.normal(input_key, shape=(self.n_input, self.n_dim))
        return W_in


    def predict_states(self, states=None, discard=None):

        """
        Predict the output values using the trained ESN.

        Parameters
        ----------
        states : jnp.ndarray, optional
            the state matrix to be used for prediction. If not provided, the internal reservoir state `self.R` will be used.
        discard : int, optional
            the number of initial steps to discard before starting prediction. If not provided, the default `self.washout_steps` will be used.

        Returns
        -------
        prediction : jnp.ndarray
            the predicted output values
        """
        if states is None:
            states = self.R

        if discard is None:
            discard = self.washout_steps

        return self.readout.predict(states[discard:])

    def predict_state_derivative(self, states = None, discard=None):
        if states is None:
            states = self.R


        input = self.readout.predict(states)

        if discard is None:
            discard = self.washout_steps

        return  self.rdot(input[discard:], states[discard:])



    def predict_derivative(self, derivatives = None, states = None, discard=None, use_estimate=True):
        if states is None:
            states = self.R

        if use_estimate:
            print("using estimate...")
            vals = self.readout.predict(states)
            derivatives = self.rdot(vals, states)

        if derivatives is None:
            derivatives = self.R_dot


        if discard is None:
            discard = self.washout_steps

        return self.readout.deriv_predict(derivatives[discard:], states[discard:])

    def derivative_train_MSE(self, input_derivative, normalize=True, use_estimate=True, use_mae=False):
        return compute_MSE(input_derivative, self.predict_derivative(discard=0, use_estimate=use_estimate), self.washout_steps, normalize, use_mae=use_mae)


    def train_MSE(self, normalize=True, use_mae=False):
        """
           Compute the Mean Squared Error (MSE) loss between the input and the predicted output.

           Parameters
           ----------
           normalize : bool, optional
               Whether to normalize the input and predicted output, by default False.

           Returns
           -------
           float
               The MSE loss value.
           """
        return compute_MSE(self.input, self.predict_states(discard=0), self.washout_steps, normalize, use_mae=use_mae)








# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    dt = 1e-3
    train_per = 0.7
    lam_lorenz = 0.906

    ## Load and simulate an attractor

    model = Lorenz()
    model.dt = dt

    t, x_tot = model.make_trajectory(80000, return_times=True)
    x_dot_tot = jnp.array(model.rhs(x_tot, t)).T

    x_train, x_test = train_test_split(x_tot, 1000, train_percentage=train_per)
    x_dot_train, x_dot_test = train_test_split(x_dot_tot, 1000, train_percentage=train_per)

    key = random.PRNGKey(14)
    #readout = LinearReadout(500, 1e-6)
    #readout = QuadraticReadout(500, reg_param=1e-6)
    readout = LinearReadoutWithDerivatives(alpha=0)
    rcn = RCN(key=key, n_dim=500, readout=readout, n_input=3, dt=dt, washout_steps=1000, spectral_radius=0.8, sigma=0.02, gamma = 10)
    rcn.train(x_train, x_dot_train)
    y_train = rcn.predict_states()

    R_hat_dot = rcn.predict_state_derivative()
    R_dot = rcn.R_dot[1000:]

    mse = jnp.sqrt(rcn.train_MSE(normalize=False))
    print(f"MSE is {mse}")


    d_mse = jnp.sqrt(rcn.derivative_train_MSE(x_dot_train, normalize=False, use_estimate=False)) * dt
    print(f"MSE on derivative is {d_mse}")

    d_mse_x = jnp.sqrt(rcn.derivative_train_MSE(x_dot_train, normalize=False, use_estimate=True)) * dt
    print(f"MSE on estimated derivative is {d_mse_x}")


    r_mse = jnp.sqrt(compute_MSE(R_dot ,R_hat_dot, washout_steps=0 ,normalize=False))
    print(f"MSE on R_dot is {r_mse}")

    l_e = calculate_lyapunov_exponent(x_train[1000:], y_train, dt)
    print(f"M.L.E. is {l_e}")






    N = len(y_train)
    T = np.arange(N) * dt * lam_lorenz
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    axs[0].plot(T, y_train[:, 0], 'r')
    axs[0].plot(T, x_train[1000:, 0], 'k--')

    axs[0].set_title('Component x')

    axs[1].plot(T, y_train[:, 1], 'r')
    axs[1].plot(T, x_train[1000:, 1], 'k--')

    axs[1].set_title('Component y')

    axs[2].plot(T, y_train[:, 2], 'r')
    axs[2].plot(T, x_train[1000:, 2], 'k--')
    axs[2].set_title('Component z')

    plt.tight_layout()
    plt.show()



    print("generating test")
    y_test = rcn.generate(len(x_test))

    fh, f_steps = compute_forecast_horizon(x_test, y_test, dt=dt, lyap_exp=lam_lorenz, epsilon=1, normalize=True)
    print(f"forecast horizon is {fh}")
    print("figures...")
    plt.plot(y_test[:,0], y_test[:,1])
    plt.plot(x_test[:,0], x_test[:,1], 'k--')
    plt.show()


    fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    N = len(y_test)
    T = np.arange(N) * dt * lam_lorenz

    axs[0].plot(T, y_test[:, 0], 'r')
    axs[0].plot(T, x_test[:, 0], 'k--')
    axs[0].vlines(fh, ymin=x_test[:, 0].min(), ymax=x_test[:, 0].max())
    axs[0].set_title('Component x')

    axs[1].plot(T, y_test[:, 1], 'r')
    axs[1].plot(T, x_test[:, 1], 'k--')
    axs[1].vlines(fh, ymin=x_test[:, 1].min(), ymax=x_test[:, 1].max())
    axs[1].set_title('Component y')

    axs[2].plot(T, y_test[:, 2], 'r')
    axs[2].plot(T, x_test[:, 2], 'k--')
    axs[2].vlines(fh, ymin=x_test[:, 2].min(), ymax=x_test[:, 2].max())
    axs[2].set_title('Component z')

    plt.tight_layout()
    plt.show()


    l_e = calculate_lyapunov_exponent(x_test, y_test, dt)
    print(f"M.L.E. in test is {l_e}")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


