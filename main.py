import matplotlib.pyplot as plt
import numpy as np
from copy import copy
import jax.numpy as jnp
from jax import random, grad, jit
from functools import partial
from RCN import RCN
from dysts.datasets import load_dataset
from dysts.flows import Lorenz

model = Lorenz()
dt= 0.001
L_train = 9000
L_test = 14000
L_remove = 1000

model.dt = dt
x_tot = model.make_trajectory(L_train + L_test + L_remove, resample=False, return_times=False)


x = x_tot[L_remove:L_remove+L_train ]
x_test = x_tot[L_remove+L_train:]


key = random.PRNGKey(42)

rcn = RCN(key=key, n_input=3, dt=dt)
rcn.train(x)
y = rcn.predict()

# plt.plot(y_train[:,0], 'r')
# plt.plot(x[:,0], 'k--')
# plt.show()

print(len(x_test))

T_test = int(len(x_test)*dt)
y_new = rcn.generate(T_test)

plt.figure()
plt.plot(y_new[:,0], 'r')
plt.plot(x_test[:,0], 'k--')
plt.show()

