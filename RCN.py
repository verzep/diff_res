# This is a sample Python script.
from jax import config

config.update("jax_enable_x64", True)

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import jax
from copy import copy
import jax.numpy as jnp
from jax import random, grad, jit
from functools import partial

import matplotlib.pyplot as plt


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


class RCN:

    def __init__(self,
                 key,
                 n_dim: int = 500,
                 n_input: int = 1,
                 spectral_radius: float = 0.9,
                 gamma: float = 10,
                 sigma: float = 0.02,
                 bias: float = 0.1,
                 dt: float = 1e-3,
                 washout_steps: int = None,
                 reg_param: float = 1e-4
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
        self.n_dim = n_dim
        self.n_input = n_input
        self.spectral_radius = spectral_radius
        self.gamma = gamma
        self.sigma = sigma
        self.bias = bias * jnp.array(random.normal(bias_key, [n_dim]))
        self.dt = dt
        self.washout_steps = washout_steps
        self.reg_param = reg_param

        self.W = self._create_reservoir(W_key, self.spectral_radius)
        self.W_in = self._create_W_in(W_in_key)

        self.W_out = None
        self.r_last = None
        self.R = None
        self.R_dot = None
        self.input = None


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

    def train(self, input, states=None):

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
        if states is None:
            states = self.listen(input)
        W_out = self.fit_W_out(input, states)

        self.W_out = W_out

        return W_out

    def fit_W_out(self, input, states):
        """
           Computes the output weight matrix for the given input and reservoir states.

           Parameters
           ----------
           input : jnp.ndarray
               The input sequence.
           states : jnp.ndarray
               The reservoir states in response to the input sequence.

           Returns
           -------
           jnp.ndarray
               The output weight matrix.
           """
        if self.washout_steps is None:
            self.washout_steps = int(input.shape[0] / 10)
        
        s = states[self.washout_steps:]
        i = input[self.washout_steps:]

        W_out = (jnp.linalg.pinv(s.T @ s + self.reg_param * jnp.eye(self.n_dim))) @ s.T @ input[self.washout_steps:]

        return W_out

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
            x = r @ self.W_out  # change with a predict function
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


    def predict(self, states=None, discard=None):

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

        return states[discard:] @ self.W_out

    def train_MSE(self, normalize=False):
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
        return self._MSE(self.input, self.predict(discard=0), self.washout_steps, normalize)

    def _MSE(self, target, prediction, washout_steps: int, normalize: bool = False):

        """
        Compute mean-square error (MSE) between the `target` and the `prediction`.

        Parameters
        ----------
        target : jnp.ndarray
            The target output.
        prediction : jnp.ndarray
            The predicted output.
        washout_steps : int
            The number of initial steps to discard.
        normalize : bool, optional
            Whether to normalize the input and predicted output, by default False.

        Returns
        -------
        jax.interpreters.xla.DeviceArray
            The MSE loss value.
        """

        y = target[washout_steps:]
        y_hat = prediction[washout_steps:]

        if normalize:
            means = jnp.mean(y, axis=0)
            stds = jnp.std(y, axis=0)

            y = (y - means) / stds
            y_hat = (y_hat - means) / stds

        MSE = jnp.mean((y - y_hat) ** 2)
        return MSE




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    key = random.PRNGKey(42)
    rcn = RCN(key)
    import numpy as np

    # define input signal as a sine wave
    freq = 0.1  # frequency of the sine wave
    t = np.arange(0, 100, 0.1)  # time vector
    input = np.sin(2 * np.pi * freq * t).reshape(-1, 1)

    # repeat input to match original length
    input = np.tile(input, (10, 1))[:1000, :]

    # convert to jax array
    input = jnp.array(input)

    print(input.shape)
    print(rcn.train(input).shape)
    P = rcn.predict()
    # print(rcn.generate(T = 2))
    print(rcn.train_MSE())
    plt.plot(input)
    plt.plot(P)
    plt.show()
    plt.plot(rcn.generate(1))
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
