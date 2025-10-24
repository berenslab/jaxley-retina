from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "gpu")

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_CLIENT_MEM_FRACTION"] = ".7"

import argparse
import importlib
import jax
import jax.numpy as jnp
import logging
logger = logging.getLogger(__name__)
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import warnings
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
import yaml

import jaxley as jx
from jaxley_retina.mnist import (
    stimulus_from_image,
    build_train_loader,
    log_dictionary
)

def contrast_sweep(config_path, test_seed=0, init_control=False):
    """
    Test the performance on OOD contrast levels.
    
    NOTE: config_path should contain the config of the training on mnist without distortions.
    """
    # Load the config
    with open(config_path, "r") as stream:
        configs = yaml.safe_load(stream)
    model_config = configs["Model config"]
    stim_config = configs["Stim config"]
    train_config = configs["Train config"]
    
    # Adjustments in case the config was older
    if "use_benison" not in model_config:
        model_config["use_benison"] = True
    if "stim_max" not in stim_config:
        stim_config["stim_max"] = stim_config["phi_max"]

    _ = np.random.seed(train_config["seed"])

    # Path handling (from the training directory)
    save_path = Path(config_path).parents[0]

    # Create the log file
    if init_control:
        file_name = f"contrast_sweep_{test_seed}_init_control.log"
    else:
        file_name = f"contrast_sweep_{test_seed}.log"
    logging.basicConfig(
        filename=str(save_path / file_name), 
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s',
        )
    logger.info(config_path) # some quick info about what experiment
    logger.info(f"Starting!")
    log_dictionary(train_config, logger, logging.INFO, "Train config")
    log_dictionary(stim_config, logger, logging.INFO, "Stim config")

    # Build the model
    model_builder = importlib.import_module(model_config["model_builder"])
    network = model_builder.build_mnist_model(model_config)
    log_dictionary(model_config, logger, logging.INFO, "Model config")

    # Check that using the correct number of readouts
    if stim_config["digits"] == "all":
        assert model_config["n_readouts"] == 10
    else:
        assert len(stim_config["digits"]) == model_config["n_readouts"]

    # Scale the conductance initialization by the number of synapses
    M = network.edges.groupby("post_index").apply(len).mean()
    gS_init_mean = train_config["TanhConductanceSynapse_gS_init_mean"] * 1/M
    gS_init_std = gS_init_mean / 10 # to sweep over the means
    # .item() to convert numpy generic type to python type 
    train_config["TanhConductanceSynapse_gS_init_mean"] = gS_init_mean.item()
    train_config["TanhConductanceSynapse_gS_init_std"] = gS_init_std.item()

    network, transform = model_builder.setup_param_training(network, train_config)
    
    # Record activity of readouts
    network.delete_recordings()
    readout_inds = network.readout.nodes.global_cell_index.tolist()
    network.cell(readout_inds).record("v")

    # Do the ramp-up for the network and save the states
    PR_inds = network.PR.nodes.global_cell_index.tolist()
    ramp_up_time = np.arange(0, stim_config["outer_ramp_up"], stim_config["dt"])
    ramp_up_stim = np.zeros((len(PR_inds), len(ramp_up_time)))
    params = network.get_parameters()
    data_clamps = network.cell(PR_inds).data_clamp(stim_config["var"], ramp_up_stim, None)
    init_soln, init_states = jx.integrate(
        network,
        delta_t=stim_config["dt"],
        data_clamps=data_clamps,
        params=params, 
        return_states=True
    )
    del init_soln
    starting_tsteps = int(train_config["loss_start"] / stim_config["dt"])

    if isinstance(train_config["checkpoint_lengths"], int):
        assert(train_config["checkpoint_lengths"] > 0), "Checkpoint lengths must be > 0"
        levels = train_config["checkpoint_lengths"]
        num_timesteps = int(stim_config["t_max"] / stim_config["dt"])
        checkpoints = [int(np.ceil(num_timesteps ** (1/levels))) for _ in range(levels)]
    else:
        checkpoints = train_config["checkpoint_lengths"]

    def simulate(params, stim):
        data_clamps = network.cell(PR_inds).data_clamp(stim_config["var"], stim, None)
        soln = jx.integrate(
            network, 
            delta_t=stim_config["dt"],
            data_clamps=data_clamps,
            params=params,
            all_states=init_states,
            checkpoint_lengths=checkpoints
        )
        return soln[:, starting_tsteps:-1]

    vmapped_simulate = jax.vmap(simulate, in_axes=(None, 0), out_axes=0)

    def predict(params, stim):
        soln = vmapped_simulate(params, stim) # batch x neurons x time
        response = jnp.mean(soln, axis=2) # readouts within each simulation
        v_penalty = jnp.sum(jnp.minimum(response-(-90), 0)**2 + jnp.maximum(response-150, 0)**2) # penalty for abnormal voltages
        response = response - jnp.mean(response, axis=1)[:, None] # subtract mean response of neurons
        logits = jax.nn.softmax(response, axis=1)
        return logits, v_penalty
    
    jitted_predict = jax.jit(predict)

    # Stimuli prep
    stimuli_from_imgs = jax.vmap(stimulus_from_image, in_axes=(0, None, None))
    network.PR.compute_compartment_centers()
    coords = np.vstack((network.PR.nodes.x, network.PR.nodes.y))

    # Load the trained parameters as opt_params
    if not init_control:
        with open(str(save_path / "trained_params.pkl"), "rb") as f:
            params = pickle.load(f)
    opt_params = transform.inverse(params)
        
    # Do some evaluation
    min_test_contrasts = np.arange(0, 1.01 , step=0.1)
    avg_test_accs = []
    for x in min_test_contrasts:
        ds_test, = build_train_loader(
            train_config["batch_size"], 
            digits=stim_config["digits"],
            contrast_range=(x, x),
            lum_range=(stim_config["test_lum_min"], stim_config["test_lum_max"]),
            splits=["test"],
        )
        test_accuracies = []
        for i, batch in enumerate(ds_test):
            imgs, labels = batch
            stim = stimuli_from_imgs(imgs, coords, stim_config)
            labels = jax.nn.one_hot(labels, model_config["n_readouts"])
            y = jitted_predict(opt_params, stim)
            acc = jnp.mean(jnp.argmax(y, axis=1) == jnp.argmax(labels, axis=1))
            test_accuracies.append(acc)
            if i % 10 == 0:
                logger.info(f"it: {i}, acc: {acc}")
        
        avg_test_acc = np.mean(test_accuracies)
        logger.info(f"Min test contrast: {x}, Avg test accuracy: {avg_test_acc}")
        avg_test_accs.append(avg_test_acc)
    
    if init_control:
        file_name = f"contrast_sweep_{test_seed}_init_control.pkl"
    else:
        file_name =  f"contrast_sweep_{test_seed}.pkl"
    with open(str(save_path / file_name), "wb") as f:
        pickle.dump((min_test_contrasts, avg_test_accs), f)       


def parse_args():
    """Collect the train config file"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init_control", type=bool, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    contrast_sweep(args.config, test_seed=args.seed, init_control=args.init_control)