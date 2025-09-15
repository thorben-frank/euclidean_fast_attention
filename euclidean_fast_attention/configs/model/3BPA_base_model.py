"""The default configuration for the experiments."""

import numpy as np
import ml_collections

from ml_collections import config_dict


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.name = 'base_model'
    config.cutoff = 5.0  # in Angstrom
    config.num_features = 128
    config.num_layers = 3
    config.mp_max_degree = 1
    config.mp_num_basis_fn = 32
    config.radial_basis_fn = 'reciprocal_bernstein'
    config.emulate_era_block = False
    config.era_use_in_iterations = "0 1"
    config.era_max_degree = 0
    config.era_include_pseudotensors = False
    config.era_activation_fn = 'identity'
    config.era_num_frequencies = None
    config.era_max_length = 11.0
    config.era_max_frequency = float(3*np.pi)
    config.era_lebedev_num = 146
    config.era_qk_num_features = 32
    config.era_v_num_features = 16
    config.efa_block_post_mlp_bool = False
    config.mp_block_post_mlp_bool = False
    config.num_post_residual_mlps = 1
    config.use_switch = False
    config.iterated_tensor_products = False
    config.dispersion_correction_bool = False
    config.output_is_zero_at_init = True

    return config
