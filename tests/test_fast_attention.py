"""Test functionality for the core Euclidean fast attention operation.
"""
import e3x
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from euclidean_fast_attention import fast_attention

jax.config.update('jax_default_matmul_precision', 'highest')


@pytest.mark.parametrize("include_pseudotensors_inputs", [True, False])
@pytest.mark.parametrize("include_pseudotensors_qk", [True, False, None])
@pytest.mark.parametrize("include_pseudotensors_v", [True, False, None])
@pytest.mark.parametrize("max_degree_inputs", [0, 2])
@pytest.mark.parametrize("max_degree_qk", [None])
@pytest.mark.parametrize("max_degree_v", [None])
@pytest.mark.parametrize("num_features", [32])
def test_shapes_pseudotensors(
        include_pseudotensors_inputs,
        include_pseudotensors_qk,
        include_pseudotensors_v,
        max_degree_inputs,
        max_degree_qk,
        max_degree_v,
        num_features
):
    num_nodes = 11

    # Parameters.
    key = jax.random.PRNGKey(0)
    assert num_features % 2 == 0

    # RoPE parameters.
    max_distance = 10.0
    max_frequency = jnp.pi

    # Lebedev grid.
    num = 50  # Should be enough for theta_max = pi.

    # Draw random query/key/value features and positions.
    num_parity = 2 if include_pseudotensors_inputs else 1
    num_degrees = (max_degree_inputs + 1) ** 2
    key, inputs_key, pos_key = jax.random.split(key, num=3)

    inputs = jax.random.normal(
        inputs_key, (num_nodes, num_parity, num_degrees, num_features)
    )

    pos = jax.random.normal(pos_key, (num_nodes, 3))  # Positions.

    efa = fast_attention.EuclideanFastAttention(
        lebedev_num=num,
        parametrized=False,
        include_pseudotensors_qk=include_pseudotensors_qk,
        include_pseudotensors_v=include_pseudotensors_v,
        max_degree_qk=max_degree_qk,
        max_degree_v=max_degree_v,
        epe_max_frequency=max_frequency,
        epe_max_length=max_distance
    )

    params = efa.init(
        key,
        inputs=inputs,
        positions=pos,
        batch_segments=jnp.zeros((len(pos), )),
        graph_mask=jnp.array([True])
    )

    # test single graph
    out = efa.apply(
        params,
        inputs=inputs,
        positions=pos,
        batch_segments=jnp.ones((num_nodes, )).astype(jnp.int16),
        graph_mask=jnp.array([True])
    )

    # test batched graphs
    out_batched = efa.apply(
        params,
        inputs=inputs,
        positions=pos,
        batch_segments=jnp.concatenate([jnp.zeros((3, )), jnp.ones((6, )), 2*jnp.ones((2, ))]).astype(jnp.int16),
        graph_mask=jnp.array([True, True, False])
    )

    npt.assert_equal(out.shape, out_batched.shape)

    if include_pseudotensors_v is None:
        npt.assert_equal(out.shape, inputs.shape)
    elif not include_pseudotensors_v:
        npt.assert_equal(out.shape, (inputs.shape[0], 1, inputs.shape[2], inputs.shape[3]))
    elif include_pseudotensors_v:
        npt.assert_equal(out.shape, (inputs.shape[0], 2, inputs.shape[2], inputs.shape[3]))
    else:
        raise RuntimeError(
            'Should not end up here.'
        )


@pytest.mark.parametrize("include_pseudotensors_inputs", [True, False])
@pytest.mark.parametrize("include_pseudotensors_qk", [False, None])
@pytest.mark.parametrize("include_pseudotensors_v", [True, None])
@pytest.mark.parametrize("max_degree_inputs", [1])
@pytest.mark.parametrize("max_degree_qk", [None])
@pytest.mark.parametrize("max_degree_v", [None])
@pytest.mark.parametrize("num_features", [32])
@pytest.mark.parametrize("ti_include_pseudotensors", [True, False, None])
@pytest.mark.parametrize("ti_max_degree_sph", [None])
@pytest.mark.parametrize("ti_max_degree", [None])
def test_shapes_pseudotensors_tensor_integration(
        include_pseudotensors_inputs,
        include_pseudotensors_qk,
        include_pseudotensors_v,
        max_degree_inputs,
        max_degree_qk,
        max_degree_v,
        num_features,
        ti_include_pseudotensors,
        ti_max_degree_sph,
        ti_max_degree,
):
    num_nodes = 11

    # Parameters.
    key = jax.random.PRNGKey(0)
    assert num_features % 2 == 0

    # RoPE parameters.
    max_distance = 10.0
    max_frequency = jnp.pi

    # Lebedev grid.
    num = 50  # Should be enough for theta_max = pi.

    # Draw random query/key/value features and positions.
    num_parity = 2 if include_pseudotensors_inputs else 1
    num_degrees = (max_degree_inputs + 1) ** 2
    key, inputs_key, pos_key = jax.random.split(key, num=3)

    inputs = jax.random.normal(
        inputs_key, (num_nodes, num_parity, num_degrees, num_features)
    )

    pos = jax.random.normal(pos_key, (num_nodes, 3))  # Positions.

    efa = fast_attention.EuclideanFastAttention(
        lebedev_num=num,
        parametrized=False,
        include_pseudotensors_qk=include_pseudotensors_qk,
        include_pseudotensors_v=include_pseudotensors_v,
        max_degree_qk=max_degree_qk,
        max_degree_v=max_degree_v,
        epe_max_frequency=max_frequency,
        epe_max_length=max_distance,
        ti_include_pseudotensors=ti_include_pseudotensors,
        ti_max_degree_sph=ti_max_degree_sph,
        ti_max_degree=ti_max_degree,
        tensor_integration=True
    )

    params = efa.init(
        key,
        inputs=inputs,
        positions=pos,
        batch_segments=jnp.zeros((len(pos),)),
        graph_mask=jnp.array([True])
    )

    # test single graph
    out = efa.apply(
        params,
        inputs=inputs,
        positions=pos,
        batch_segments=jnp.ones((num_nodes,)).astype(jnp.int16),
        graph_mask=jnp.array([True])
    )

    # test batched graphs
    out_batched = efa.apply(
        params,
        inputs=inputs,
        positions=pos,
        batch_segments=jnp.concatenate([jnp.zeros((3,)), jnp.ones((6,)), 2 * jnp.ones((2,))]).astype(jnp.int16),
        graph_mask=jnp.array([True, True, False])
    )

    npt.assert_equal(out.shape, out_batched.shape)

    if ti_include_pseudotensors is None:
        npt.assert_equal(out.shape, inputs.shape)
    elif not ti_include_pseudotensors:
        npt.assert_equal(out.shape, (inputs.shape[0], 1, inputs.shape[2], inputs.shape[3]))
    elif ti_include_pseudotensors:
        npt.assert_equal(out.shape, (inputs.shape[0], 2, inputs.shape[2], inputs.shape[3]))
    else:
        raise RuntimeError(
            'Should not end up here.'
        )


@pytest.mark.parametrize("include_pseudotensors_inputs", [True])
@pytest.mark.parametrize("include_pseudotensors_qk", [None])
@pytest.mark.parametrize("include_pseudotensors_v", [None])
@pytest.mark.parametrize("max_degree_inputs", [0, 1])
@pytest.mark.parametrize("max_degree_qk", [0, 1, None])
@pytest.mark.parametrize("max_degree_v", [1, 2, None])
@pytest.mark.parametrize("num_features", [32])
def test_shapes_degrees(
        include_pseudotensors_inputs,
        include_pseudotensors_qk,
        include_pseudotensors_v,
        max_degree_inputs,
        max_degree_qk,
        max_degree_v,
        num_features
):
    num_nodes = 11

    # Parameters.
    key = jax.random.PRNGKey(0)
    assert num_features % 2 == 0

    # RoPE parameters.
    max_distance = 10.0
    max_frequency = jnp.pi

    # Lebedev grid.
    num = 50  # Should be enough for theta_max = pi.

    # Draw random query/key/value features and positions.
    num_parity = 2 if include_pseudotensors_inputs else 1
    num_degrees = (max_degree_inputs + 1) ** 2
    key, inputs_key, pos_key = jax.random.split(key, num=3)

    inputs = jax.random.normal(
        inputs_key, (num_nodes, num_parity, num_degrees, num_features)
    )

    pos = jax.random.normal(pos_key, (num_nodes, 3))  # Positions.

    efa = fast_attention.EuclideanFastAttention(
        lebedev_num=num,
        parametrized=False,
        include_pseudotensors_qk=include_pseudotensors_qk,
        include_pseudotensors_v=include_pseudotensors_v,
        max_degree_qk=max_degree_qk,
        max_degree_v=max_degree_v,
        epe_max_frequency=max_frequency,
        epe_max_length=max_distance
    )

    if max_degree_v is None and max_degree_qk is not None:
        max_degree_qkv = max_degree_qk
    elif max_degree_v is not None and max_degree_qk is None:
        max_degree_qkv = max_degree_v
    elif max_degree_v is None and max_degree_qk is None:
        max_degree_qkv = max_degree_inputs
    else:
        max_degree_qkv = max(max_degree_qk, max_degree_v)

    if max_degree_inputs < max_degree_qkv:
        with npt.assert_raises(ValueError):
            efa.init(
                key,
                inputs=inputs,
                positions=pos,
                batch_segments=jnp.zeros((len(pos), )),
                graph_mask=jnp.array([True])
            )
    else:
        params = efa.init(
            key,
            inputs=inputs,
            positions=pos,
            batch_segments=jnp.zeros((len(pos),)),
            graph_mask=jnp.array([True])
        )
        # test single graph
        out = efa.apply(
            params,
            inputs=inputs,
            positions=pos,
            batch_segments=jnp.ones((num_nodes, )).astype(jnp.int16),
            graph_mask=jnp.array([True])
        )

        # test batched graphs
        out_batched = efa.apply(
            params,
            inputs=inputs,
            positions=pos,
            batch_segments=jnp.concatenate([jnp.zeros((3, )), jnp.ones((6, )), 2*jnp.ones((2, ))]).astype(jnp.int16),
            graph_mask=jnp.array([True, True, False])
        )

        npt.assert_equal(out.shape, out_batched.shape)
        if max_degree_v is None:
            npt.assert_equal(out.shape, inputs.shape)
        else:
            npt.assert_equal(
                out.shape,
                (inputs.shape[0], inputs.shape[1], (max_degree_v+1)**2, inputs.shape[3])
            )


@pytest.mark.parametrize("include_pseudotensors_inputs", [True, False])
@pytest.mark.parametrize("include_pseudotensors_qk", [None])
@pytest.mark.parametrize("include_pseudotensors_v", [None])
@pytest.mark.parametrize("max_degree_inputs", [1])
@pytest.mark.parametrize("max_degree_qk", [None])
@pytest.mark.parametrize("max_degree_v", [None])
@pytest.mark.parametrize("num_features", [32])
@pytest.mark.parametrize("ti_include_pseudotensors", [None])
@pytest.mark.parametrize("ti_max_degree_sph", [None, 2])
@pytest.mark.parametrize("ti_max_degree", [None, 1, 2])
def test_shapes_degrees_tensor_integration(
        include_pseudotensors_inputs,
        include_pseudotensors_qk,
        include_pseudotensors_v,
        max_degree_inputs,
        max_degree_qk,
        max_degree_v,
        num_features,
        ti_include_pseudotensors,
        ti_max_degree_sph,
        ti_max_degree,
):
    num_nodes = 11

    # Parameters.
    key = jax.random.PRNGKey(0)
    assert num_features % 2 == 0

    # RoPE parameters.
    max_distance = 10.0
    max_frequency = jnp.pi

    # Lebedev grid.
    num = 50  # Should be enough for theta_max = pi.

    # Draw random query/key/value features and positions.
    num_parity = 2 if include_pseudotensors_inputs else 1
    num_degrees = (max_degree_inputs + 1) ** 2
    key, inputs_key, pos_key = jax.random.split(key, num=3)

    inputs = jax.random.normal(
        inputs_key, (num_nodes, num_parity, num_degrees, num_features)
    )

    pos = jax.random.normal(pos_key, (num_nodes, 3))  # Positions.

    efa = fast_attention.EuclideanFastAttention(
        lebedev_num=num,
        parametrized=False,
        include_pseudotensors_qk=include_pseudotensors_qk,
        include_pseudotensors_v=include_pseudotensors_v,
        max_degree_qk=max_degree_qk,
        max_degree_v=max_degree_v,
        epe_max_frequency=max_frequency,
        epe_max_length=max_distance,
        ti_include_pseudotensors=ti_include_pseudotensors,
        ti_max_degree_sph=ti_max_degree_sph,
        ti_max_degree=ti_max_degree,
        tensor_integration=True
    )

    params = efa.init(
        key,
        inputs=inputs,
        positions=pos,
        batch_segments=jnp.zeros((len(pos),)),
        graph_mask=jnp.array([True])
    )

    # test single graph
    out = efa.apply(
        params,
        inputs=inputs,
        positions=pos,
        batch_segments=jnp.ones((num_nodes,)).astype(jnp.int16),
        graph_mask=jnp.array([True])
    )

    # test batched graphs
    out_batched = efa.apply(
        params,
        inputs=inputs,
        positions=pos,
        batch_segments=jnp.concatenate([jnp.zeros((3,)), jnp.ones((6,)), 2 * jnp.ones((2,))]).astype(jnp.int16),
        graph_mask=jnp.array([True, True, False])
    )

    npt.assert_equal(out.shape, out_batched.shape)

    if ti_max_degree is None:
        npt.assert_equal(out.shape, inputs.shape)
    else:
        npt.assert_equal(
            out.shape,
            (inputs.shape[0], inputs.shape[1], (ti_max_degree+1)**2, inputs.shape[3])
        )


@pytest.mark.parametrize("include_pseudotensors_inputs", [True, False])
@pytest.mark.parametrize("include_pseudotensors_qk", [None])
@pytest.mark.parametrize("include_pseudotensors_v", [None])
@pytest.mark.parametrize("max_degree_inputs", [0, 2])
@pytest.mark.parametrize("max_degree_qk", [None])
@pytest.mark.parametrize("max_degree_v", [None])
@pytest.mark.parametrize("num_features", [32])
def test_translation_invariance(
        include_pseudotensors_inputs,
        include_pseudotensors_qk,
        include_pseudotensors_v,
        max_degree_inputs,
        max_degree_qk,
        max_degree_v,
        num_features
):
    jax.config.update('jax_enable_x64', False)

    num_nodes = 11

    # Parameters.
    key = jax.random.PRNGKey(0)
    assert num_features % 2 == 0

    # RoPE parameters.
    max_distance = 10.0
    max_frequency = jnp.pi

    # Lebedev grid.
    num = 50  # Should be enough for theta_max = pi.

    # Draw random query/key/value features and positions.
    num_parity = 2 if include_pseudotensors_inputs else 1
    num_degrees = (max_degree_inputs + 1) ** 2
    key, inputs_key, pos_key = jax.random.split(key, num=3)

    inputs = jax.random.normal(
        inputs_key, (num_nodes, num_parity, num_degrees, num_features)
    )

    pos = jax.random.normal(pos_key, (num_nodes, 3))  # Positions.
    pos_translated = pos + jnp.expand_dims(jnp.array([3.1, -5.4, 2.5]), axis=0)

    efa = fast_attention.EuclideanFastAttention(
        lebedev_num=num,
        parametrized=False,
        include_pseudotensors_qk=include_pseudotensors_qk,
        include_pseudotensors_v=include_pseudotensors_v,
        max_degree_qk=max_degree_qk,
        max_degree_v=max_degree_v,
        epe_max_frequency=max_frequency,
        epe_max_length=max_distance
    )

    params = efa.init(
        key,
        inputs=inputs,
        positions=pos,
        batch_segments=jnp.zeros((len(pos), )),
        graph_mask=jnp.array([True])
    )

    # test single graph
    out = efa.apply(
        params,
        inputs=inputs,
        positions=pos,
        batch_segments=jnp.ones((num_nodes, )).astype(jnp.int16),
        graph_mask=jnp.array([True])
    )
    out_translated = efa.apply(
        params,
        inputs=inputs,
        positions=pos_translated,
        batch_segments=jnp.ones((num_nodes,)).astype(jnp.int16),
        graph_mask=jnp.array([True])
    )

    # test batched graphs
    out_batched = efa.apply(
        params,
        inputs=inputs,
        positions=pos,
        batch_segments=jnp.concatenate([jnp.zeros((3, )), jnp.ones((6, )), 2*jnp.ones((2, ))]).astype(jnp.int16),
        graph_mask=jnp.array([True, True, False])
    )
    # test batched graphs
    out_batched_translated = efa.apply(
        params,
        inputs=inputs,
        positions=pos_translated,
        batch_segments=jnp.concatenate([jnp.zeros((3, )), jnp.ones((6, )), 2*jnp.ones((2, ))]).astype(jnp.int16),
        graph_mask=jnp.array([True, True, False])
    )

    npt.assert_equal(out.shape, out_batched.shape)
    npt.assert_equal(out_translated.shape, out_batched_translated.shape)

    npt.assert_allclose(
        out, out_translated, atol=1e-4
    )  # enabling x64 allows to decrease atol < 1e-6

    npt.assert_allclose(
        out_batched, out_batched_translated, atol=1e-4
    )  # enabling x64 allows to decrease atol < 1e-6


@pytest.mark.parametrize("include_pseudotensors_inputs", [True, False])
@pytest.mark.parametrize("include_pseudotensors_qk", [None])
@pytest.mark.parametrize("include_pseudotensors_v", [None])
@pytest.mark.parametrize("max_degree_inputs", [1])
@pytest.mark.parametrize("max_degree_qk", [None])
@pytest.mark.parametrize("max_degree_v", [None])
@pytest.mark.parametrize("num_features", [32])
@pytest.mark.parametrize("ti_include_pseudotensors", [None])
@pytest.mark.parametrize("ti_max_degree_sph", [None, 2])
@pytest.mark.parametrize("ti_max_degree", [None])
def test_translational_invariance_tensor_integration(
        include_pseudotensors_inputs,
        include_pseudotensors_qk,
        include_pseudotensors_v,
        max_degree_inputs,
        max_degree_qk,
        max_degree_v,
        num_features,
        ti_include_pseudotensors,
        ti_max_degree_sph,
        ti_max_degree,
):
    num_nodes = 11

    # Parameters.
    key = jax.random.PRNGKey(0)
    assert num_features % 2 == 0

    # RoPE parameters.
    max_distance = 10.0
    max_frequency = jnp.pi

    # Lebedev grid.
    num = 50  # Should be enough for theta_max = pi.

    # Draw random query/key/value features and positions.
    num_parity = 2 if include_pseudotensors_inputs else 1
    num_degrees = (max_degree_inputs + 1) ** 2
    key, inputs_key, pos_key = jax.random.split(key, num=3)

    inputs = jax.random.normal(
        inputs_key, (num_nodes, num_parity, num_degrees, num_features)
    )

    pos = jax.random.normal(pos_key, (num_nodes, 3))  # Positions.
    pos_translated = pos + jnp.expand_dims(jnp.array([3.1, -5.4, 2.5]), axis=0)

    efa = fast_attention.EuclideanFastAttention(
        lebedev_num=num,
        parametrized=False,
        include_pseudotensors_qk=include_pseudotensors_qk,
        include_pseudotensors_v=include_pseudotensors_v,
        max_degree_qk=max_degree_qk,
        max_degree_v=max_degree_v,
        epe_max_frequency=max_frequency,
        epe_max_length=max_distance,
        ti_include_pseudotensors=ti_include_pseudotensors,
        ti_max_degree_sph=ti_max_degree_sph,
        ti_max_degree=ti_max_degree,
        tensor_integration=True
    )

    params = efa.init(
        key,
        inputs=inputs,
        positions=pos,
        batch_segments=jnp.zeros((len(pos),)),
        graph_mask=jnp.array([True])
    )

    # test single graph
    out = efa.apply(
        params,
        inputs=inputs,
        positions=pos,
        batch_segments=jnp.ones((num_nodes,)).astype(jnp.int16),
        graph_mask=jnp.array([True])
    )
    out_translated = efa.apply(
        params,
        inputs=inputs,
        positions=pos_translated,
        batch_segments=jnp.ones((num_nodes,)).astype(jnp.int16),
        graph_mask=jnp.array([True])
    )

    # test batched graphs
    out_batched = efa.apply(
        params,
        inputs=inputs,
        positions=pos,
        batch_segments=jnp.concatenate([jnp.zeros((3,)), jnp.ones((6,)), 2 * jnp.ones((2,))]).astype(jnp.int16),
        graph_mask=jnp.array([True, True, False])
    )

    out_batched_translated = efa.apply(
        params,
        inputs=inputs,
        positions=pos_translated,
        batch_segments=jnp.concatenate([jnp.zeros((3,)), jnp.ones((6,)), 2 * jnp.ones((2,))]).astype(jnp.int16),
        graph_mask=jnp.array([True, True, False])
    )

    npt.assert_equal(out.shape, out_batched.shape)
    npt.assert_equal(out_translated.shape, out_batched_translated.shape)

    npt.assert_allclose(
        out, out_translated, atol=1e-4
    )  # enabling x64 allows to decrease atol < 1e-6

    npt.assert_allclose(
        out_batched, out_batched_translated, atol=1e-4
    )  # enabling x64 allows to decrease atol < 1e-6


@pytest.mark.parametrize("include_pseudotensors_inputs", [True, False])
@pytest.mark.parametrize("include_pseudotensors_qk", [None])
@pytest.mark.parametrize("include_pseudotensors_v", [None])
@pytest.mark.parametrize("max_degree_inputs", [0, 2])
@pytest.mark.parametrize("max_degree_qk", [None])
@pytest.mark.parametrize("max_degree_v", [None])
@pytest.mark.parametrize("num_features", [32])
@pytest.mark.parametrize("lebedev_num", [6, 50])
def test_rotation_equivariance(
        include_pseudotensors_inputs,
        include_pseudotensors_qk,
        include_pseudotensors_v,
        max_degree_inputs,
        max_degree_qk,
        max_degree_v,
        num_features,
        lebedev_num
):
    jax.config.update('jax_enable_x64', True)

    num_nodes = 11

    # Parameters.
    key = jax.random.PRNGKey(0)
    assert num_features % 2 == 0

    # RoPE parameters.
    max_distance = 10.0
    max_frequency = jnp.pi

    # Draw random query/key/value features and positions.
    num_parity = 2 if include_pseudotensors_inputs else 1
    num_degrees = (max_degree_inputs + 1) ** 2
    key, inputs_key, pos_key, rot_key = jax.random.split(key, num=4)

    inputs = jax.random.normal(
        inputs_key, (num_nodes, num_parity, num_degrees, num_features)
    )

    pos = jax.random.normal(pos_key, (num_nodes, 3))  # Positions.

    rot = e3x.so3.random_rotation(rot_key)
    wigner_d = e3x.so3.wigner_d(rot, max_degree=max_degree_inputs)
    inputs_rot = jnp.einsum('...af,ab->...bf', inputs, wigner_d)
    pos_rot = jnp.einsum('...a,ab->...b', pos, rot)

    efa = fast_attention.EuclideanFastAttention(
        lebedev_num=lebedev_num,
        parametrized=False,
        include_pseudotensors_qk=include_pseudotensors_qk,
        include_pseudotensors_v=include_pseudotensors_v,
        max_degree_qk=max_degree_qk,
        max_degree_v=max_degree_v,
        epe_max_frequency=max_frequency,
        epe_max_length=max_distance
    )

    params = efa.init(
        key,
        inputs=inputs,
        positions=pos,
        batch_segments=jnp.zeros((len(pos), )),
        graph_mask=jnp.array([True])
    )

    # test single graph
    out = efa.apply(
        params,
        inputs=inputs,
        positions=pos,
        batch_segments=jnp.ones((num_nodes, )).astype(jnp.int16),
        graph_mask=jnp.array([True])
    )
    out_rot = efa.apply(
        params,
        inputs=inputs_rot,
        positions=pos_rot,
        batch_segments=jnp.ones((num_nodes,)).astype(jnp.int16),
        graph_mask=jnp.array([True])
    )

    # test batched graphs
    out_batched = efa.apply(
        params,
        inputs=inputs,
        positions=pos,
        batch_segments=jnp.concatenate([jnp.zeros((3, )), jnp.ones((6, )), 2*jnp.ones((2, ))]).astype(jnp.int16),
        graph_mask=jnp.array([True, True, False])
    )
    # test batched graphs
    out_batched_rot = efa.apply(
        params,
        inputs=inputs_rot,
        positions=pos_rot,
        batch_segments=jnp.concatenate([jnp.zeros((3, )), jnp.ones((6, )), 2*jnp.ones((2, ))]).astype(jnp.int16),
        graph_mask=jnp.array([True, True, False])
    )

    npt.assert_equal(out.shape, out_batched.shape)
    npt.assert_equal(out_rot.shape, out_batched_rot.shape)

    num_degrees = out.shape[-2]
    max_degree_out = int(np.rint(np.sqrt(num_degrees) - 1))

    # output features can be of different degree than the input features.
    wigner_d_out = e3x.so3.wigner_d(rot, max_degree=max_degree_out)

    # Lebedev number is sufficiently large.
    if lebedev_num == 50:
        # for max degree out = 0, it is invariant so rotated and non-rotated output should be the same.
        # Otherwise it should fail.
        if max_degree_out > 0:
            with npt.assert_raises(AssertionError):
                npt.assert_allclose(out, out_rot, atol=1e-4)

            with npt.assert_raises(AssertionError):
                npt.assert_allclose(out_batched, out_batched_rot, atol=1e-4)

        npt.assert_allclose(
            out, jnp.einsum('...af,ab->...bf', out_rot, wigner_d_out.T), atol=1e-4
        )  # enabling x64 allows to decrease atol < 1e-6

        npt.assert_allclose(
            out_batched, jnp.einsum('...af,ab->...bf', out_batched_rot, wigner_d_out.T), atol=1e-4
        )  # enabling x64 allows to decrease atol < 1e-6

    # Lebedev grid to small. Now its features are neither invariant nor equivariant anymore.
    elif lebedev_num == 6:
        with npt.assert_raises(AssertionError):
            npt.assert_allclose(
                out, jnp.einsum('...af,ab->...bf', out_rot, wigner_d_out.T), atol=1e-4
            )  # enabling x64 allows to decrease atol < 1e-6

            npt.assert_allclose(
                out_batched, jnp.einsum('...af,ab->...bf', out_batched_rot, wigner_d_out.T), atol=1e-4
            )  # enabling x64 allows to decrease atol < 1e-6


@pytest.mark.parametrize("include_pseudotensors_inputs", [True, False])
@pytest.mark.parametrize("include_pseudotensors_qk", [None])
@pytest.mark.parametrize("include_pseudotensors_v", [None])
@pytest.mark.parametrize("max_degree_inputs", [0, 2])
@pytest.mark.parametrize("max_degree_qk", [None])
@pytest.mark.parametrize("max_degree_v", [None])
@pytest.mark.parametrize("num_features", [32])
@pytest.mark.parametrize("ti_include_pseudotensors", [None])
@pytest.mark.parametrize("ti_max_degree_sph", [None, 1])
@pytest.mark.parametrize("ti_max_degree", [None])
@pytest.mark.parametrize("lebedev_num", [6, 50])
def test_rotation_equivariance_tensor_integration(
        include_pseudotensors_inputs,
        include_pseudotensors_qk,
        include_pseudotensors_v,
        max_degree_inputs,
        max_degree_qk,
        max_degree_v,
        num_features,
        ti_include_pseudotensors,
        ti_max_degree_sph,
        ti_max_degree,
        lebedev_num,
):
    jax.config.update('jax_enable_x64', True)

    num_nodes = 11

    # Parameters.
    key = jax.random.PRNGKey(0)
    assert num_features % 2 == 0

    # RoPE parameters.
    max_distance = 10.0
    max_frequency = jnp.pi

    # Draw random query/key/value features and positions.
    num_parity = 2 if include_pseudotensors_inputs else 1
    num_degrees = (max_degree_inputs + 1) ** 2
    key, inputs_key, pos_key, rot_key = jax.random.split(key, num=4)

    inputs = jax.random.normal(
        inputs_key, (num_nodes, num_parity, num_degrees, num_features)
    )

    pos = jax.random.normal(pos_key, (num_nodes, 3))  # Positions.

    rot = e3x.so3.random_rotation(rot_key)
    wigner_d = e3x.so3.wigner_d(rot, max_degree=max_degree_inputs)
    inputs_rot = jnp.einsum('...af,ab->...bf', inputs, wigner_d)
    pos_rot = jnp.einsum('...a,ab->...b', pos, rot)

    efa = fast_attention.EuclideanFastAttention(
        lebedev_num=lebedev_num,
        parametrized=False,
        include_pseudotensors_qk=include_pseudotensors_qk,
        include_pseudotensors_v=include_pseudotensors_v,
        max_degree_qk=max_degree_qk,
        max_degree_v=max_degree_v,
        epe_max_frequency=max_frequency,
        epe_max_length=max_distance,
        ti_include_pseudotensors=ti_include_pseudotensors,
        ti_max_degree_sph=ti_max_degree_sph,
        ti_max_degree=ti_max_degree,
        tensor_integration=True
    )

    params = efa.init(
        key,
        inputs=inputs,
        positions=pos,
        batch_segments=jnp.zeros((len(pos),)),
        graph_mask=jnp.array([True])
    )

    # test single graph
    out = efa.apply(
        params,
        inputs=inputs,
        positions=pos,
        batch_segments=jnp.ones((num_nodes,)).astype(jnp.int16),
        graph_mask=jnp.array([True])
    )
    out_rot = efa.apply(
        params,
        inputs=inputs_rot,
        positions=pos_rot,
        batch_segments=jnp.ones((num_nodes,)).astype(jnp.int16),
        graph_mask=jnp.array([True])
    )

    # test batched graphs
    out_batched = efa.apply(
        params,
        inputs=inputs,
        positions=pos,
        batch_segments=jnp.concatenate([jnp.zeros((3,)), jnp.ones((6,)), 2 * jnp.ones((2,))]).astype(jnp.int16),
        graph_mask=jnp.array([True, True, False])
    )
    # test batched graphs
    out_batched_rot = efa.apply(
        params,
        inputs=inputs_rot,
        positions=pos_rot,
        batch_segments=jnp.concatenate([jnp.zeros((3,)), jnp.ones((6,)), 2 * jnp.ones((2,))]).astype(jnp.int16),
        graph_mask=jnp.array([True, True, False])
    )

    npt.assert_equal(out.shape, out_batched.shape)
    npt.assert_equal(out_rot.shape, out_batched_rot.shape)

    num_degrees = out.shape[-2]
    max_degree_out = int(np.rint(np.sqrt(num_degrees) - 1))

    # output features can be of different degree than the input features.
    wigner_d_out = e3x.so3.wigner_d(rot, max_degree=max_degree_out)

    # Lebedev number is sufficiently large.
    if lebedev_num == 50:
        # for max degree out = 0, it is invariant so rotated and non-rotated output should be the same.
        # Otherwise it should fail.
        if max_degree_out > 0:
            with npt.assert_raises(AssertionError):
                npt.assert_allclose(out, out_rot, atol=1e-4)

            with npt.assert_raises(AssertionError):
                npt.assert_allclose(out_batched, out_batched_rot, atol=1e-4)

        npt.assert_allclose(
            out, jnp.einsum('...af,ab->...bf', out_rot, wigner_d_out.T), atol=1e-4
        )  # enabling x64 allows to decrease atol < 1e-6

        npt.assert_allclose(
            out_batched, jnp.einsum('...af,ab->...bf', out_batched_rot, wigner_d_out.T), atol=1e-4
        )  # enabling x64 allows to decrease atol < 1e-6

    # Lebedev grid to small. Now its features are neither invariant nor equivariant anymore.
    elif lebedev_num == 6:
        with npt.assert_raises(AssertionError):
            npt.assert_allclose(
                out, jnp.einsum('...af,ab->...bf', out_rot, wigner_d_out.T), atol=1e-4
            )  # enabling x64 allows to decrease atol < 1e-6

            npt.assert_allclose(
                out_batched, jnp.einsum('...af,ab->...bf', out_batched_rot, wigner_d_out.T), atol=1e-4
            )  # enabling x64 allows to decrease atol < 1e-6
