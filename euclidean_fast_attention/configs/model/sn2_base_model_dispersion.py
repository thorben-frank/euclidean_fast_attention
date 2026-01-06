"""The default configuration for the experiments."""

import numpy as np
import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""

    config = ml_collections.ConfigDict()

    config.name = 'base_model'
    config.cutoff = 5.0  # in Angstrom
    config.num_features = 128
    config.num_layers = 2
    config.mp_max_degree = 2
    config.mp_num_basis_fn = 32
    config.radial_basis_fn = 'reciprocal_bernstein'
    config.emulate_era_block = False
    config.era_use_in_iterations = None
    config.era_max_degree = None
    config.era_include_pseudotensors = None
    config.era_activation_fn = None
    config.era_num_frequencies = None
    config.era_max_frequency = None
    config.era_max_length = None
    config.era_lebedev_num = None
    config.era_qk_num_features = None
    config.era_v_num_features = None
    config.num_post_residual_mlps = 0
    config.iterated_tensor_products = False
    config.dispersion_correction_bool = True
    
    return config
