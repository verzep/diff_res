
import matplotlib.pyplot as plt
import numpy as np
from copy import copy
import jax.numpy as jnp
from jax import random, grad, jit
from functools import partial

from dysts.datasets import load_dataset
from utils import train_test_split, compute_forecast_horizon

from dysts.flows import Lorenz
from readouts import *

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import jax
from copy import copy
import jax.numpy as jnp
from RCN import *
from copy import deepcopy


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    print("Simulation in starting")

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

    key = random.PRNGKey(42)
    readout = LinearReadout(500, 1e-6)
    #readout = QuadraticReadout(100, reg_param=1e-6)
    readout = LinearReadoutWithDerivatives(alpha=0)
    rcn = RCN(key=key, n_dim=500, readout=readout, n_input=3, dt=dt, washout_steps=3000, spectral_radius=1.2)
    rcn.train(x_train, x_dot_train)

    alpha_list = np.arange(0, 0.9, 0.1)
    MSE = []
    MSE_d = []
    MSE_dx = []
    FH = []

    for alpha in alpha_list:
        print(f"alpha is {alpha}")
        readout = LinearReadoutWithDerivatives(alpha=alpha)
        rcn = RCN(key=key, n_dim=500, readout=readout, n_input=3, dt=dt, washout_steps=3000, sigma=0.2)
        rcn.train(x_train, x_dot_train)
        y = rcn.predict_states()

        mse = rcn.train_MSE()
        print(f"MSE is {mse}")

        d_mse = rcn.derivative_train_MSE(x_dot_train, use_estimate=False)
        print(f"MSE on derivative is {d_mse}")

        d_mse_x = rcn.derivative_train_MSE(x_dot_train, use_estimate=True)
        print(f"MSE on estimated derivative is {d_mse_x}")

        print("generating test")
        y_test = rcn.generate(len(x_test))

        fh, f_steps = compute_forecast_horizon(x_test, y_test, dt=dt, lyap_exp=lam_lorenz, epsilon=1, normalize=True)
        print(f"forecast horizon is {fh}")

        MSE.append(mse)
        MSE_d.append(d_mse)
        MSE_dx.append(d_mse_x)
        FH.append(fh)



    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    # Plot the data on the first y_train-axis
    line1, = ax.plot(alpha_list, MSE, label='MSE')
    line2, = ax.plot(alpha_list, MSE_d, label='MSE_deriv')
    line3, = ax.plot(alpha_list, MSE_dx, label='MSE_deriv_estima')

    # Create a twin y_train-axis
    ax2 = ax.twinx()

    # Plot line4 on the second y_train-axis
    line4, = ax2.plot(alpha_list, FH, 'k--', label='forecasting_horizon')

    # Set the labels for the y_train-axes
    ax.set_ylabel('MSE / MSE_deriv / MSE_deriv_estima')
    ax2.set_ylabel('forecasting_horizon')

    # Combine the legends for both y_train-axes
    lines = [line1, line2, line3, line4]
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels)

    plt.show()

