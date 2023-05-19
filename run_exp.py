from RCN import RCN

import matplotlib.pyplot as plt
import numpy as np
from copy import copy
import jax.numpy as jnp
from jax import random, grad, jit
from functools import partial

from dysts.datasets import load_dataset
from utils import train_test_split, compute_forecast_horizon

from dysts.flows import Lorenz





dt = 1e-3
train_per = 0.7
lam_lorenz = 0.906

## Load and simulate an attractor

model = Lorenz()
model.dt = dt

t, x_tot = model.make_trajectory(80000, return_times=True)
x_dot_tot = jnp.array(model.rhs(x_tot, t)).T

x_train, x_test = train_test_split(x_tot,1000, train_percentage=train_per)
x_dot_train, x_dot_test = train_test_split(x_dot_tot, 1000, train_percentage=train_per)




key = random.PRNGKey(14)
rcn = RCN(key=key, n_input=3, dt =dt, washout_steps=100, reg_param=0)
rcn.train_with_derivative(x_train, x_dot_train)
y = rcn.predict()

print(f"MSE is {rcn.train_MSE()}")


y_dot_train = rcn.R_dot@ rcn.W_out


d_mce = rcn.derivative_train_MSE(x_dot_train)
print(f"MSE on derivative is {d_mce}")

print("generating test")
y_test = rcn.generate(len(x_test))




d = rcn.washout_steps
# Create a new figure
fig = plt.figure()
# Add an axis to the figure
ax = fig.add_subplot(1, 1, 1)
# Plot the data on the axis
ax.plot(y_test[:, 0], y_test[:, 1], 'r')
ax.plot(x_test[:, 0], x_test[:, 1], 'k--')

# Show the plot
plt.show()

fh, f_steps = compute_forecast_horizon(x_test,y_test, dt= dt, lyap_exp=lam_lorenz, epsilon=1, normalize=True)

print(f"forecast horizon is {fh}")


fig, axs = plt.subplots(3, 1, figsize=(8, 12))


N = len(y_test)
T = np.arange(N)*dt*lam_lorenz


axs[0].plot(T , y_test[:, 0], 'r')
axs[0].plot(T, x_test[:, 0], 'k--')
axs[0].vlines(fh, ymin=y_test[:, 0].min(), ymax=y_test[:, 0].max())
axs[0].set_title('Component x')

axs[1].plot(T,y_test[:, 1], 'r')
axs[1].plot(T,x_test[:, 1], 'k--')
axs[1].vlines(fh, ymin=y_test[:, 1].min(), ymax=y_test[:, 1].max())
axs[1].set_title('Component y_train')

axs[2].plot(T,y_test[:, 2], 'r')
axs[2].plot(T,x_test[:, 2], 'k--')
axs[2].vlines(fh, ymin=y_test[:, 2].min(), ymax=y_test[:, 2].max())
axs[2].set_title('Component z')

plt.tight_layout()
plt.show()



# vals = jnp.arange(0.1,3.1, 0.1)
# horizons = []
#
# for e in vals:
#     fh, _ = compute_forecast_horizon(x_test, y_test, dt=dt, lyap_exp=lam_lorenz, epsilon=e, normalize=True)
#     horizons.append(fh)
# fig, ax = plt.subplots(1, 1, figsize=(8, 8))
#
# ax.plot(vals, horizons)
# plt.show()
