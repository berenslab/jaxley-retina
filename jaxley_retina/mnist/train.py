from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "gpu")

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_CLIENT_MEM_FRACTION"] = ".7"  # set in batch script

import argparse
import gc
import importlib
import jax
from jax import jit, value_and_grad, vmap
import jax.numpy as jnp
import logging

logger = logging.getLogger(__name__)
import numpy as np
import optax
import pandas as pd
from pathlib import Path
import pickle
import time as time_module

import warnings

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
import yaml

import jaxley as jx
from jaxley_retina.tasks.mnist.image_to_stim import stimulus_from_image
from jaxley_retina.tasks.mnist.simple_data_prep import build_train_loader
from jaxley_retina.tasks.mnist.train_io import log_dictionary


def parse_args():
    """If not using wandb, can collect the train config file with this"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def train(cfg: dict):
    """Train outer retina models to classify MNIST"""
    start_time = time_module.time()

    # Split up the config
    model_config = cfg["model_config"]
    stim_config = cfg["stim_config"]
    train_config = cfg["train_config"]

    _ = np.random.seed(train_config["seed"])
    save_path = Path(cfg["save_path"])

    # NOTE: the creation of the logging file here was removed for wandb usage
    logging.basicConfig(filename=str(save_path / "mnist.log"), level=logging.INFO)
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

    # Set up the parameter training (according to the model)
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
    data_clamps = network.cell(PR_inds).data_clamp(
        stim_config["var"], ramp_up_stim, None
    )
    init_soln, init_states = jx.integrate(
        network,
        delta_t=stim_config["dt"],
        data_clamps=data_clamps,
        params=params,
        return_states=True,
    )
    del init_soln
    starting_tsteps = int(train_config["loss_start"] / stim_config["dt"])

    if isinstance(train_config["checkpoint_lengths"], int):
        assert train_config["checkpoint_lengths"] > 0, "Checkpoint lengths must be > 0"
        levels = train_config["checkpoint_lengths"]
        num_timesteps = int(stim_config["t_max"] / stim_config["dt"])
        checkpoints = [
            int(np.ceil(num_timesteps ** (1 / levels))) for _ in range(levels)
        ]
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
            checkpoint_lengths=checkpoints,
        )
        return soln[:, starting_tsteps:-1]

    vmapped_simulate = vmap(simulate, in_axes=(None, 0), out_axes=0)

    def predict(params, stim):
        soln = vmapped_simulate(params, stim)  # batch x neurons x time
        response = jnp.mean(soln, axis=2)  # readouts within each simulation
        v_penalty = jnp.sum(
            jnp.minimum(response - (-90), 0) ** 2 + jnp.maximum(response - 150, 0) ** 2
        )  # penalty for abnormal voltages

        logits = jnp.abs(response)  # rectify (chosen readout will hyperpolarize more)
        probs = jax.nn.softmax(logits / train_config["softmax_temperature"], axis=1)
        return probs, v_penalty

    jitted_predict = jit(predict)

    def get_accuracy(logits, labels):
        return jnp.mean(jnp.argmax(logits, axis=1) == jnp.argmax(labels, axis=1))

    def loss_fn(opt_params, stim, labels):
        """Cross entropy loss for the batch"""
        print("Compiling loss_fn")
        params = transform.forward(opt_params)
        logits, v_penalty = predict(params, stim)
        acc = get_accuracy(logits, labels)
        loss = (
            -jnp.sum(labels * jnp.log(logits) + (1 - labels) * jnp.log(1 - logits))
            / labels.shape[0]
        )
        # L1 regularization
        l1_penalty = sum(
            [jnp.sum(jnp.abs(p)) for p in jax.tree_util.tree_leaves(params)]
        )
        loss += train_config["lambda_l1"] * l1_penalty
        # Make sure the voltage is within a reasonable range
        loss += v_penalty
        return loss, acc

    grad_fn = jit(value_and_grad(loss_fn, argnums=0, has_aux=True))

    # Stimuli prep
    stimuli_from_imgs = jax.vmap(stimulus_from_image, in_axes=(0, None, None))
    network.PR.compute_compartment_centers()
    coords = np.vstack((network.PR.nodes.x, network.PR.nodes.y))

    # Set up the optimization
    scheduler = optax.exponential_decay(
        init_value=train_config["start_lr"],
        transition_steps=train_config["transition_steps"],
        decay_rate=train_config["decay_rate"],
    )
    gradient_transform = optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip by the gradient by the global norm
        optax.scale_by_adam(),  # Use the updates from adam
        optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler
        # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss
        optax.scale(-1.0),
    )
    opt_params = transform.inverse(params)
    opt_state = gradient_transform.init(opt_params)

    # Save the initial params
    with open(str(save_path / f"init_params.pkl"), "wb") as outfile:
        pickle.dump(params, outfile)

    # Stop criteria info
    epoch_counter = 0
    stop = False
    train_loss_avg = np.inf  # initialize
    all_train_losses = []
    all_train_accuracies = []

    setup_time = time_module.time() - start_time
    logger.info(
        f"Setup time: {time_module.strftime('%H:%M:%S', time_module.gmtime(setup_time))}"
    )
    # Training loop
    while not stop and epoch_counter < train_config["max_epochs"]:
        epoch_counter += 1
        print(f"Epoch {epoch_counter}")
        train_losses = []
        train_accuracies = []

        # Rebuild iterator
        (ds_train,) = build_train_loader(
            train_config["batch_size"],
            digits=stim_config["digits"],
            contrast_range=(
                stim_config["train_contrast_min"],
                stim_config["train_contrast_max"],
            ),
            lum_range=(stim_config["train_lum_min"], stim_config["train_lum_max"]),
            splits=["train"],
            data_dir=stim_config["data_dir"],
        )
        try:
            for i, batch in enumerate(ds_train):
                imgs, labels = batch

                # Process batch
                stim = stimuli_from_imgs(imgs, coords, stim_config)
                labels = jax.nn.one_hot(labels, model_config["n_readouts"])
                (l, acc), g = grad_fn(opt_params, stim, labels)

                # Explicitly convert JAX arrays to np/python primitives
                loss_scalar = float(l)
                acc_scalar = float(acc)

                # Update the parameters
                updates, opt_state = gradient_transform.update(g, opt_state)
                opt_params = optax.apply_updates(opt_params, updates)
                train_losses.append(loss_scalar)
                train_accuracies.append(acc_scalar)

                # Clean up
                del stim, g, updates, imgs, labels, batch

                if np.isnan(l):
                    raise Exception("Nan loss")

                if i % 10 == 0:
                    print(f"l: {l}")
                    logger.info(f"it: {i}, loss: {l}, acc: {acc}")

                    # Save the parameters
                    params = transform.forward(opt_params)
                    with open(str(save_path / f"trained_params.pkl"), "wb") as outfile:
                        pickle.dump(params, outfile)

        finally:
            del ds_train
            gc.collect()

        # For saving the losses and accuracies to look at later
        all_train_losses.extend(train_losses)
        all_train_accuracies.extend(train_accuracies)

        # At the end of each epoch, check if loss reduced more than X x previous loss
        stop = np.mean(train_losses) > train_config["stop_criteria"] * train_loss_avg
        train_loss_avg = np.mean(train_losses)
        logger.info(
            f"Epoch {epoch_counter}, avg train loss: {np.mean(train_losses)}, acc: {np.mean(train_accuracies)}"
        )

    # Saving the accumulated losses and accuries for completed trainings
    with open(str(save_path / f"train_losses.pkl"), "wb") as outfile:
        pickle.dump(all_train_losses, outfile)
    with open(str(save_path / f"train_accuracies.pkl"), "wb") as outfile:
        pickle.dump(all_train_accuracies, outfile)

    train_time = time_module.time() - start_time - setup_time
    logger.info(
        f"Training time: {time_module.strftime('%H:%M:%S', time_module.gmtime(train_time))}"
    )

    # Do some evaluation
    (ds_test,) = build_train_loader(
        train_config["batch_size"],
        digits=stim_config["digits"],
        contrast_range=(
            stim_config["test_contrast_min"],
            stim_config["test_contrast_max"],
        ),
        lum_range=(stim_config["test_lum_min"], stim_config["test_lum_max"]),
        splits=["test"],
        data_dir=stim_config["data_dir"],
    )
    test_accuracies = []
    # Make sure you have the latest parameters
    params = transform.forward(opt_params)
    try:
        for i, batch in enumerate(ds_test):
            imgs, labels = batch
            stim = stimuli_from_imgs(imgs, coords, stim_config)
            labels = jax.nn.one_hot(labels, model_config["n_readouts"])
            y, _ = jitted_predict(params, stim)
            acc = jnp.mean(jnp.argmax(y, axis=1) == jnp.argmax(labels, axis=1))
            test_accuracies.append(acc)
            if i % 50 == 0:
                logger.info(f"it: {i}, acc: {acc}")
    finally:
        del ds_test
        gc.collect()

    avg_test_acc = np.mean(test_accuracies)
    logger.info(f"Avg test accuracy: {avg_test_acc}")

    test_time = time_module.time() - start_time - setup_time - train_time
    logger.info(
        f"Testing time: {time_module.strftime('%H:%M:%S', time_module.gmtime(test_time))}"
    )


if __name__ == "__main__":
    # For running without wandb
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config["save_path"] = str(Path(args.config).parents[0])
    train(config)
