from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "gpu")

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".8"

import argparse
import json
import logging
from functools import reduce

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)
import pathlib
import pickle
import time as time_module

import jaxley as jx
import numpy as np
import optax
import yaml
from jaxley.optimize.transforms import ParamTransform, SigmoidTransform
from jaxley_mech.synapses import RibbonSynapse
from tqdm import tqdm

from jaxley_retina.OPL.HC import build_HC
from jaxley_retina.OPL.PR import build_PR


def train(cell_id: int, train_params: dict):
    # Set up save paths
    p = pathlib.Path(__file__).parent.resolve()
    time_stamp = time_module.strftime("%m%d-%H%M%S")
    folder_name = time_stamp + f"-{cell_id}"

    # Create the output directory
    save_path = p.parents[1] / "params" / "PR" / "ribbon" / folder_name
    os.mkdir(str(save_path))

    # Save config files in output directory
    with open(str(save_path / "train_config.yaml"), "w") as outfile:
        yaml.dump(train_params, outfile, default_flow_style=False)

    # Create the log file
    logging.basicConfig(
        filename=str(save_path / "single_PR_train.log"), level=logging.INFO
    )
    logger.info(f"Starting cell {cell_id}!")
    # Log the train params
    formatted_dict = json.dumps(train_params, indent=4, sort_keys=True, default=str)
    log_message = f"Train params: {{\n{formatted_dict}\n}}"
    logger.info(log_message)

    # Set hyperparams
    dt = train_params["dt"]
    _ = np.random.seed(train_params["seed"])
    ramp_up = train_params["ramp_up"]
    phi_max = train_params["phi_max"]

    # Load fitted PTC params (non-transformed now)
    with open(train_params["ptc_path"], "rb") as f:
        ptc_params = pickle.load(f)
    ptc_params = reduce(lambda x, y: {**x, **y}, ptc_params)
    # Save the PTC params for future reference
    with open(str(save_path / "ptc_params.pkl"), "wb") as f:
        pickle.dump(ptc_params, f)

    # Set up the model
    PR, _ = build_PR(ptc_params)
    HC = build_HC()
    network = jx.Network([PR, HC])
    jx.connect(network.cell(0), network.cell(1), RibbonSynapse(solver="explicit"))
    network.init_states()

    # Load the data (cell_id x time)
    with open(train_params["data_path"], "rb") as f:
        data = pickle.load(f)
    data = data[cell_id]

    # Set up the recordings
    network.delete_recordings()
    network.RibbonSynapse.record("RibbonSynapse_exo")

    # Set up the parameter training
    network.delete_trainables()
    # Load the yaml file defining the trainable params (ion channel params can also be trained)
    with open(p.parents[1] / "params" / "PR" / "ribbon" / "trainables.yaml", "r") as f:
        trainables = yaml.safe_load(f)
    transform = []
    for param in trainables:
        if param.startswith("RibbonSynapse_"):
            network.RibbonSynapse.make_trainable(param, trainables[param]["init"])
        else:
            network.cell(0).make_trainable(param, trainables[param]["init"])
        transform.append(
            {
                param: SigmoidTransform(
                    jnp.array(trainables[param]["lower"]),
                    jnp.array(trainables[param]["upper"]),
                )
            }
        )
    alphas = [{"alphas": jnp.array([0.5])}]
    params = alphas + network.get_parameters()
    transform.insert(
        0, {"alphas": SigmoidTransform(jnp.array([0.0]), jnp.array([1.0]))}
    )
    transform = ParamTransform(transform)

    # Create the stimulus
    tsteps = int((2000 + 2000 + ramp_up) / dt)
    stim = jnp.zeros((2, tsteps))
    stim = stim.at[0, int(ramp_up / dt) : int((ramp_up + 1000) / dt)].set(
        phi_max
    )  # green center flash
    stim = stim.at[1, int((ramp_up + 2000) / dt) : int((ramp_up + 3000) / dt)].set(
        phi_max
    )  # uv center flash
    ramp_up_tsteps = int(ramp_up / dt)

    # Write the simulation and loss functions
    def simulate(opt_params, stim):
        params = transform.forward(opt_params)

        # Set the stimulus
        alpha = params[0]["alphas"]
        stim = (alpha * stim[0, :] + (1 - alpha) * stim[1, :].T).T
        data_clamps = network.cell(0).data_clamp(
            "PR_Phototransduction_Stim", stim, None
        )

        soln = jx.integrate(
            network,
            delta_t=dt,
            data_clamps=data_clamps,
            params=params[1:],
        )
        return soln[0, ramp_up_tsteps:-1]

    fs = (1 / 500) * 1000  # experimental sampling rate, 500Hz

    def loss_fn(opt_params, stim, data):
        soln = simulate(opt_params, stim)
        # Subsample (data not so high freq)
        soln = soln[:: int(fs / dt)]
        # Demean by the first time point of the green stimulus as in the data
        soln -= soln[0]
        return jnp.sum((soln - data) ** 2)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=0))

    # Save the initial params
    with open(str(save_path / f"init_params.pkl"), "wb") as outfile:
        pickle.dump(params, outfile)

    # Set up the optimizer
    opt_params = transform.inverse(params)
    optimizer = optax.adam(learning_rate=train_params["lr"])
    opt_state = optimizer.init(opt_params)

    # Stop criteria
    losses = []
    temp_losses = []
    avg_loss = np.inf
    stop = False
    epoch_counter = 0

    # Training loop
    with tqdm(total=train_params["max_epochs"]) as pbar:
        while not stop and len(losses) < train_params["max_epochs"]:
            pbar.update(1)
            epoch_counter += 1

            l, g = grad_fn(opt_params, stim, data)
            updates, opt_state = optimizer.update(g, opt_state)
            opt_params = optax.apply_updates(opt_params, updates)
            losses.append(l)
            temp_losses.append(l)

            if np.isnan(l):
                raise Exception("Nan loss")

            if epoch_counter % 10 == 0:
                logger.info(f"it: {epoch_counter}, loss: {l}")

                # Check for early stopping
                if np.mean(temp_losses) > 0.99 * avg_loss:
                    stop = True
                    logger.info(f"Early stopping at epoch {epoch_counter}")
                avg_loss = np.mean(temp_losses)
                temp_losses = []  # reset

                # Save the parameters
                params = transform.forward(opt_params)
                with open(str(save_path / f"trained_params.pkl"), "wb") as outfile:
                    pickle.dump(params, outfile)

                # Save losses
                with open(str(save_path / f"losses.pkl"), "wb") as outfile:
                    pickle.dump(losses, outfile)


def parse_args():
    """Get the cell number to train."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell_id", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    train_params = {
        "max_epochs": 1100,
        "dt": 0.025,
        "lr": 0.01,
        "seed": 0,
        "ramp_up": 200,  # ms
        "phi_max": 20_000,
        "ptc_path": "",
        "data_path": "",
    }
    args = parse_args()
    train(args.cell_id, train_params)
