from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "gpu")

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"

import logging

import jax
import jax.numpy as jnp
import jaxley as jx

from jaxley_retina import OPL

logger = logging.getLogger(__name__)

import pathlib
import pickle
import time as time_module

import numpy as np
import optax
import yaml
from scipy.io import loadmat
from tqdm import tqdm


def load_data(data_path: str):
    "Load stimuli and data (mat file structure)."
    data = loadmat(data_path)
    responses = [data["ConeResponses"][0][i][0][0][0][0] for i in range(6)]
    stimuli = [data["ConeResponses"][0][i][0][0][1][0] for i in range(6)]
    return responses, stimuli


def train(train_params: dict):
    """Train the parameters of the phototransduction cascade."""
    # Set hyperparams
    dt = train_params["dt"]
    cut_length = train_params["cut_length"]
    _ = np.random.seed(train_params["seed"])

    # Set up save paths
    p = pathlib.Path(__file__).parent.resolve()
    time_stamp = time_module.strftime("%m%d-%H%M%S")

    # Create the output directory
    save_path = str(p.parents[1] / "params" / "PR" / "cascade" / time_stamp)
    os.mkdir(save_path)
    logging.basicConfig(
        filename=os.path.join(save_path, "train_log.log"), level=logging.INFO
    )

    # Save config files in output directory
    with open(os.path.join(save_path, "train_config.yaml"), "w") as outfile:
        yaml.dump(train_params, outfile, default_flow_style=False)

    PR, params_mouse_cone = OPL.PR.build_PR()

    # Load the data
    data_path = str(p.parents[1] / "data" / "OPL" / "RiekeMouseConeResponses.mat")
    responses, stimuli = load_data(data_path)

    # Set up the recordings
    PR.record("PR_Phototransduction_I")

    # Set up the parameter training
    PR.delete_trainables()
    for name, param in params_mouse_cone.items():
        PR.make_trainable(name, param)
    params = PR.get_parameters()
    transform = OPL.transforms.PTC_transform

    # Write the simulation and loss functions
    def simulate(params, stim):
        data_clamps = PR.data_clamp("PR_Phototransduction_Stim", stim, None)
        soln = jx.integrate(PR, delta_t=dt, data_clamps=data_clamps, params=params)
        return soln[0, :-1]

    def loss_fn(opt_params, stim, data):
        params = transform.forward(opt_params)
        soln = simulate(params, stim)
        # Subsample (stim was interpolated, see later)
        soln = soln[::num_interpolated_points]
        # Calculate the dark current and normalize the solution by that
        I_dark = (
            params[9]["PR_Phototransduction_G_dark"]
            ** params[7]["PR_Phototransduction_n"]
            * params[5]["PR_Phototransduction_k"]
        )
        soln = -1 * soln / I_dark  # also sign flip for correct direction
        return jnp.sum((soln - data) ** 2)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=0))
    grad_fn_vmapped = jax.vmap(grad_fn, in_axes=(None, 0, 0), out_axes=(0, 0))

    opt_params = transform.inverse(params)
    init_params = opt_params.copy()  # to have for later if needed
    optimizer = optax.adam(learning_rate=train_params["lr"])
    opt_state = optimizer.init(opt_params)

    # Number of points to interpolate between each pair of points
    num_interpolated_points = int(0.1 / dt)  # 4 when dt = 0.025

    # Generate new time series with interpolated points
    def interp_stim(stim):
        interp = lambda x, y: jnp.linspace(x, y, num=num_interpolated_points)
        interpolated_stim = jax.vmap(interp)(stim[:-1], jnp.roll(stim, -1)[:-1])
        return interpolated_stim.flatten()

    # Cut the stimuli and responses into a random chunk
    def cut_batch(stims, responses, cut_length, dt):
        start = np.random.randint(0, 16 - cut_length)  # start of the cut in seconds
        stop = start + cut_length  # stop of the cut in seconds
        dt_s = dt * 0.001  # dt of stimulus (s)
        fs = 0.1 * 0.001  # sampling rate of the data (s)

        stims = stims[:, int(start / dt_s) : int(stop / dt_s)]
        responses = responses[:, int(start / fs) : int(stop / fs)]
        return stims, responses

    # Write the training loop
    k = len(stimuli)  # k-fold / loo cross-validation
    train_losses = {i: [] for i in range(k)}
    eval_losses = []

    for i in range(k):

        # Split the data
        test_stim = stimuli[i]
        train_stim = jnp.array(stimuli[:i] + stimuli[i + 1 :])
        test_response = responses[i]
        train_responses = jnp.array(responses[:i] + responses[i + 1 :])

        for j in tqdm(range(train_params["max_epochs"])):

            # Assemble the stimuli
            stims_interpolated = jax.vmap(interp_stim)(train_stim)
            responses_vec = jnp.array(train_responses)
            stims_cut, responses_cut = cut_batch(
                stims_interpolated, responses_vec, cut_length, dt
            )

            # Calculate the loss and grads for all stimulus/response pairs
            l, g = grad_fn_vmapped(opt_params, stims_cut, responses_cut)
            # Average the grads & losses
            g = jax.tree.map(jnp.mean, g)
            l = jnp.mean(l)
            train_losses[i].append(l)

            if j % 5 == 0:
                logger.info(f"l: {l}")

            updates, opt_state = optimizer.update(g, opt_state)
            opt_params = optax.apply_updates(opt_params, updates)

        # Calculate the evaluation loss
        test_stim = interp_stim(test_stim)
        eval_loss = loss_fn(opt_params, test_stim, test_response[:-1])
        eval_losses.append(eval_loss)
        logger.info(f"evaluation loss: {eval_loss}")

        # Save the parameters
        params = transform.forward(opt_params)
        with open(os.path.join(save_path, f"params_{i}.pkl"), "wb") as outfile:
            pickle.dump(params, outfile)

        # Save losses
        with open(os.path.join(save_path, f"losses_{i}.pkl"), "wb") as outfile:
            pickle.dump({"train": train_losses[i], "eval": eval_losses[i]}, outfile)

    logger.info(f"Avg. eval loss: {np.mean(eval_losses)}")


if __name__ == "__main__":
    train_params = {
        "max_epochs": 100,
        "dt": 0.025,
        "lr": 0.01,
        "cut_length": 4.0,
        "seed": 0,
    }
    train(train_params)
