from RCN import RCN

import matplotlib.pyplot as plt
import numpy as np
from copy import copy
import jax.numpy as jnp
from jax import random, grad, jit
from functools import partial

from dysts.datasets import load_dataset
from utils import train_test_split

from dysts.flows import Lorenz





dt = 1e-3
train_per = 0.3

## Load and simulate an attractor

model = Lorenz()
model.dt = dt

t, x_tot = model.make_trajectory(80000, return_times=True)
x_dot_tot = jnp.array(model.rhs(x_tot, t)).T.shape

x_train, x_test = train_test_split(x_tot,1000, train_percentage=train_per)
x_dot_train, x_dot_test = train_test_split(x_tot, 1000, train_percentage=train_per)




key = random.PRNGKey(42)
rcn = RCN(key=key, n_input=3, dt =dt, washout_steps=100)
rcn.train(x_train)
y = rcn.predict()
print(rcn.train_MSE())

d = rcn.washout_steps



y_test = rcn.generate(len(x_test))

# Create a new figure
fig = plt.figure()
# Add an axis to the figure
ax = fig.add_subplot(1, 1, 1)
# Plot the data on the axis
ax.plot(y_test[:, 0], y_test[:, 1], 'r')
ax.plot(x_test[:, 0], x_test[:, 1], 'k--')

# Show the plot
plt.show()

fig, axs = plt.subplots(3, 1, figsize=(8, 12))

lam_lorenz = 0.906
N = len(y_test)
T = np.arange(N)*dt/lam_lorenz


axs[0].plot(T , y_test[:, 0], 'r')
axs[0].plot(T, x_test[:, 0], 'k--')
axs[0].set_title('Component x')

axs[1].plot(T,y_test[:, 1], 'r')
axs[1].plot(T,x_test[:, 1], 'k--')
axs[1].set_title('Component y')

axs[2].plot(T,y_test[:, 2], 'r')
axs[2].plot(T,x_test[:, 2], 'k--')
axs[2].set_title('Component z')

plt.tight_layout()
plt.show()


# Show the plot
plt.show()