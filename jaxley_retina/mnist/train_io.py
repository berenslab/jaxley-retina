from functools import reduce
from jaxley_retina.OPL.transforms import PTC_transform
import json
import logging
import pickle
from typing import Dict, Any, Optional



def load_PTC_params(ptc_path: str, transformed: bool = False) -> dict:
    """Loads phototransduction cascade params fitted with train_cascade.py"""
    with open(ptc_path, "rb") as f:        
        ptc_params = pickle.load(f)

    # Update the key names if needed (have to start with PR_ and in list for transform)
    ptc_params = reduce(lambda x, y: {**x, **y}, ptc_params)
    for name, param in ptc_params.items():
        if not name.startswith("PR_"):
            ptc_params["PR_" + name] = ptc_params.pop(name)
    
    # use PTC_transform in the parameters were transformed (to depricate)
    if transformed:
        ptc_params = [{key: val} for key, val in ptc_params.items()]
        ptc_params = PTC_transform.forward(ptc_params)
        ptc_params = reduce(lambda x, y: {**x, **y}, ptc_params)
    return ptc_params


def load_ribbon_params(ribbon_path:str) -> list:
    """These params were saved non-transformed but still need to be unpacked."""
    with open(ribbon_path, "rb") as f:
        ribbon_params = pickle.load(f)
    return [reduce(lambda x, y: {**x, **y}, sublist) for sublist in ribbon_params]


def load_conductances(g_path: str) -> dict:
    """These parameters come from mnist task training (not transformed)"""
    with open(g_path, "rb") as f:
        conductances = pickle.load(f)
    conductance_params = reduce(lambda x, y: {**x, **y}, conductances)
    return conductance_params


def log_dictionary(dictionary: Dict[str, Any], 
                   logger: Optional[logging.Logger] = None, 
                   log_level: int = logging.INFO, 
                   name: Optional[str] = None) -> None:
    """Logs a dictionary in a formatted, readable manner."""
    # If no logger is provided, use the root logger
    log = logger or logging.getLogger()
    
    # Create a formatted string representation of the dictionary
    try:
        # Use json.dumps for pretty formatting with indentation
        formatted_dict = json.dumps(dictionary, 
                                    indent=4,  # 4-space indentation 
                                    sort_keys=True,  # Sort keys for consistent output
                                    default=str)  # Convert non-serializable objects to strings
        
        # Corrected line: Use f-string correctly
        log_message = f"{name + ': ' if name else ''}{{\n{formatted_dict}\n}}"
        
        # Log the message at the specified log level
        log.log(log_level, log_message)
    
    except Exception as e:
        # Fallback logging if json.dumps fails
        log.error(f"Error logging dictionary: {e}")
        log.error(f"Dictionary contents: {str(dictionary)}")