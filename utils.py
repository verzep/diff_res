import jax.numpy as jnp

def train_test_split(data, removed_steps = 1000, train_percentage = .8):

    data = data[removed_steps:]
    L_tot = len(data)
    L_train = int(len(data) * train_percentage)

    x_train = data[:L_train]
    x_test = data[L_train:]

    return x_train, x_test

def compute_forecast_horizon(y, y_hat, dt, lyap_exp = None, epsilon = 0.1, normalize=True):
    if normalize:
        means = jnp.mean(y, axis=0)
        stds = jnp.std(y, axis=0)

        y = (y - means) / stds
        y_hat = (y_hat - means) / stds

    diff = y-y_hat

    if diff.ndim <2:
        diff = diff.reshape((-1,1))
    norm = jnp.sqrt((diff**2).sum(axis=1))

    forecast_steps = jnp.argwhere(norm>epsilon)[0]
    forecast_time = forecast_steps * dt

    if lyap_exp is not None:
        forecast_time=forecast_time*lyap_exp

    return forecast_time, forecast_steps

def compute_MSE(target, prediction, washout_steps: int, normalize: bool = False, use_mae=False):

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
        if use_mae:
            print("using MAE instead")
            MSE = jnp.mean(jnp.abs(y - y_hat))
        else:
            MSE = jnp.mean((y - y_hat) ** 2)
        return MSE