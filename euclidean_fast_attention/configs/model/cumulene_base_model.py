"""The default configuration for the experiments."""

import numpy as np
import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""

    config = ml_collections.ConfigDict()

    config.name = 'base_model'
    config.cutoff = 4.0  # in Angstrom
    config.num_features = 64
    config.num_layers = 3
    config.mp_max_degree = 2
    config.mp_num_basis_fn = 32
    config.radial_basis_fn = 'reciprocal_bernstein'
    config.emulate_era_block = False
    config.era_use_in_iterations = "0 1 2" # set to empty string "" to disable EFA
    config.era_max_degree = 1 # vary between 0, 1, 2 in the paper.
    config.era_include_pseudotensors = False
    config.era_activation_fn = "gelu"
    config.era_num_frequencies = None
    config.era_max_frequency = float(np.pi)
    config.era_max_length = 15.0
    config.era_lebedev_num = 50
    config.era_qk_num_features = 32
    config.era_v_num_features = 16
    config.num_post_residual_mlps = 0
    config.iterated_tensor_products = False
    config.dispersion_correction_bool = False
    
    return config
