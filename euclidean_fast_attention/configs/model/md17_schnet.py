"""The default configuration for the experiments."""

import numpy as np

from ml_collections import config_dict


def get_config():
    """Get the default hyperparameter configuration."""

    config = config_dict.ConfigDict()

    config.name = 'schnet'

    # Model Architecture Parameters
    config.num_layers = 3
    config.num_features = 256  # 290 for local model

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
    config.era_max_length = 10.0
    config.era_max_frequency = float(np.pi)
    config.era_qk_num_features = 16
    config.era_v_num_features = 32
    config.era_lebedev_num = 50
    config.era_activation_fn = 'identity'
    config.efa_block_skip_in_final_layer_bool = False
    config.efa_block_behaves_like_identity_at_init = True
    config.efa_block_layer_normalization_bool = False
    config.efa_block_mlp_hidden_features = 64
    
    return config
