"""The default configuration for the experiments."""

import numpy as np

from ml_collections import config_dict


def get_config(split: str):
    """Get the default hyperparameter configuration."""

    config = config_dict.ConfigDict()

    config.name = 'schnet'

    # Model Architecture Parameters
    config.num_layers = 3
    config.num_features = 280  # 290 for local model

    # Interaction Parameters
    config.cutoff = 5.0

    # Radial Basis Function Parameters
    config.radial_basis_fn = 'exponential_bernstein'
    config.num_basis_fn = 32

    # Atomic Number / Element Range
    config.zmax = 119

    # Euclidean Fast Attention (EFA) Block Parameters
    config.use_efa_block = True
    config.emulate_efa_block = False
    config.era_max_length = max_length_lookup[split]
    config.era_max_frequency = float(3*np.pi)
    config.era_qk_num_features = 16
    config.era_v_num_features = 32
    config.era_lebedev_num = 146
    config.era_activation_fn = 'identity'
    config.efa_block_skip_in_final_layer_bool = True
    config.efa_block_behaves_like_identity_at_init = True
    config.efa_block_layer_normalization_bool = False
    config.efa_block_mlp_hidden_features = 64
    
    return config


max_length_lookup = {
    "AT_AT": 22.0,
    "AT_AT_CG_CG": 24.0,
    "AcAla3NHMe": 12.0,
    "DHA": 16.0,
    "buckyball_catcher": 15.0,
    "nanotube": 33.0,
    "stachyose": 14.0,
}
