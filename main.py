from utils import train_test_split, compute_forecast_horizon

import jax

from RCN import RCN

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

dt = 1e-3
train_per = 0.7
lam_lorenz = 0.906

plot = False

## Load and simulate an attractor

model = Lorenz()
model.dt = dt

t, x_tot = model.make_trajectory(80000, return_times=True)
x_dot_tot = jnp.array(model.rhs(x_tot, t)).T

x_train, x_test = train_test_split(x_tot, 1000, train_percentage=train_per)
x_dot_train, x_dot_test = train_test_split(x_dot_tot, 1000, train_percentage=train_per)

exp_key = random.PRNGKey(42)

sr_list = np.linspace(0.5, 2, num=10)

metric_list = []

for sr in sr_list:

    exp_key, key = random.split(exp_key, 2)
    print(f"Spectral Radius is {sr}")
    #readout = LinearReadout(500, 1e-6)
    readout = QuadraticReadout(500, reg_param=1e-6)
    # readout = LinearReadoutWithDerivatives(alpha=0)
    rcn = RCN(key=key, n_dim=500, readout=readout, n_input=3, dt=dt, washout_steps=1000,
              spectral_radius=sr, sigma=0.02, gamma=10)
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

    r_mse = jnp.sqrt(compute_MSE(R_dot, R_hat_dot, washout_steps=0, normalize=False))
    print(f"MSE on R_dot is {r_mse}")

    l_e = calculate_lyapunov_exponent(x_train[1000:], y_train, dt)
    print(f"M.L.E. is {l_e}")

    N = len(y_train)
    T = np.arange(N) * dt * lam_lorenz

    print("generating test")
    y_test = rcn.generate(len(x_test))

    f_h, f_steps = compute_forecast_horizon(x_test, y_test, dt=dt, lyap_exp=lam_lorenz, epsilon=1, normalize=True)
    print(f"forecast horizon is {f_h}")

    l_e_test = calculate_lyapunov_exponent(x_test, y_test, dt)
    print(f"M.L.E. in test is {l_e_test}")

    metric_dict = {
        "mse": mse,
        "d_mse": d_mse,
        "d_mse_x": d_mse_x,
        "r_mse": r_mse,
        "l_e": l_e,
        "f_h": f_h,
        "l_e_test": l_e_test
    }

    if plot:
        print("figures...")
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

    if plot:
        plt.plot(y_test[:, 0], y_test[:, 1])
        plt.plot(x_test[:, 0], x_test[:, 1], 'k--')
        plt.show()

        fig, axs = plt.subplots(3, 1, figsize=(8, 12))
        N = len(y_test)
        T = np.arange(N) * dt * lam_lorenz

        axs[0].plot(T, y_test[:, 0], 'r')
        axs[0].plot(T, x_test[:, 0], 'k--')
        axs[0].vlines(f_h, ymin=x_test[:, 0].min(), ymax=x_test[:, 0].max())
        axs[0].set_title('Component x')

        axs[1].plot(T, y_test[:, 1], 'r')
        axs[1].plot(T, x_test[:, 1], 'k--')
        axs[1].vlines(f_h, ymin=x_test[:, 1].min(), ymax=x_test[:, 1].max())
        axs[1].set_title('Component y')

        axs[2].plot(T, y_test[:, 2], 'r')
        axs[2].plot(T, x_test[:, 2], 'k--')
        axs[2].vlines(f_h, ymin=x_test[:, 2].min(), ymax=x_test[:, 2].max())
        axs[2].set_title('Component z')

        plt.tight_layout()
        plt.show()

    metric_list.append(metric_dict)
    print()

error = [elem["r_mse"] for elem in metric_list]
horiz = [elem["f_h"] for elem in metric_list]

print(error)
print(horiz)

plt.scatter(np.log(np.array(error)), np.log(np.array(horiz).reshape(-1)))
plt.ylabel("horizon")
plt.xlabel("error")
plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
