from jaxley.optimize.transforms import ParamTransform, SigmoidTransform, CustomTransform, ChainTransform
import jax.numpy as jnp


def mnist_transform(bounds: dict):
    """Define different param transforms for the different synapse types."""
    log_transform = CustomTransform(jnp.exp, jnp.log) # needed for the conductances

    transforms = []
    for name in bounds:
        if "gS" in name:
            t = ChainTransform([
                SigmoidTransform(jnp.array([bounds[name][0]]), jnp.array([bounds[name][1]])),
                log_transform
            ])
            transforms.append({name: t})
        elif "e_syn" in name or "k_minus" in name:
            t = SigmoidTransform(jnp.array([bounds[name][0]]), jnp.array([bounds[name][1]]))
            transforms.append({name: t})
        else:
            raise ValueError(f"Unknown parameter name: {name}")

    return ParamTransform(transforms)