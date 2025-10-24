from jaxley.optimize.transforms import ParamTransform, SigmoidTransform
import jax.numpy as jnp


PTC_transform = ParamTransform(
    [
        {"PR_Phototransduction_sigma": SigmoidTransform(jnp.array([5.0]), jnp.array([24.0]))},  
        {"PR_Phototransduction_phi": SigmoidTransform(jnp.array([5.0]), jnp.array([24.0]))},    
        {"PR_Phototransduction_eta": SigmoidTransform(jnp.array([750.0]), jnp.array([800.0]))}, 
        {"PR_Phototransduction_gamma": SigmoidTransform(jnp.array([1.0]), jnp.array([22.0]))},
        {"PR_Phototransduction_beta": SigmoidTransform(jnp.array([2.5]), jnp.array([10.0]))},    
        {"PR_Phototransduction_k": SigmoidTransform(jnp.array([0.008]), jnp.array([0.022]))},            
        {"PR_Phototransduction_m": SigmoidTransform(jnp.array([3.75]), jnp.array([4.25]))},
        {"PR_Phototransduction_n": SigmoidTransform(jnp.array([2.8]), jnp.array([3.2]))},        
        {"PR_Phototransduction_K_GC": SigmoidTransform(jnp.array([0.2]), jnp.array([0.6]))},     
        {"PR_Phototransduction_G_dark": SigmoidTransform(jnp.array([12.0]), jnp.array([25.0]))}, 
        {"PR_Phototransduction_C_dark": SigmoidTransform(jnp.array([0.8]), jnp.array([1.2]))}, 
    ]
)

ribbon_transform = ParamTransform(
    [
        {"alphas": SigmoidTransform(jnp.array([0.0]), jnp.array([1.0]))},
        {"RibbonSynapse_gS": SigmoidTransform(jnp.array([1e-12]), jnp.array([1.0]))},
        {"RibbonSynapse_e_max": SigmoidTransform(jnp.array([1e-5]), jnp.array([1.0]))},
        {"RibbonSynapse_r_max": SigmoidTransform(jnp.array([1e-5]), jnp.array([1.0]))},
        {"RibbonSynapse_i_max": SigmoidTransform(jnp.array([1e-5]), jnp.array([1.0]))},
        {"RibbonSynapse_d_max": SigmoidTransform(jnp.array([1e-5]), jnp.array([1.0]))},
        {"RibbonSynapse_k": SigmoidTransform(jnp.array([1e-5]), jnp.array([5.0]))},
        {"RibbonSynapse_V_half": SigmoidTransform(jnp.array([-50.0]), jnp.array([-20.0]))},
    ]
)