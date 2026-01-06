"""Test functionality for the Euclidean RoPe attention mechanism.
"""
import e3x
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from euclidean_fast_attention import rope

jax.config.update('jax_default_matmul_precision', 'highest')


@pytest.mark.parametrize("num_nodes", [10])
@pytest.mark.parametrize("include_pseudotensors", [True])
@pytest.mark.parametrize("max_degree", [2])
@pytest.mark.parametrize("num_features", [32])
def test_translation_invariance(
        num_nodes,
        include_pseudotensors,
        max_degree,
        num_features
):
    # Parameters.
    key = jax.random.PRNGKey(0)
    assert num_features % 2 == 0

    # RoPE parameters.
    max_distance = 10.0
    theta = jnp.linspace(0, jnp.pi / max_distance, num_features // 2)

    # Lebedev grid.
    num = 26  # Should be enough for theta_max = pi.
    grid_u, grid_w = e3x.so3.lebedev_quadrature(precision=None, num=num)

    # Draw random query/key/value features and positions.
    num_parity = 2 if include_pseudotensors else 1
    num_degrees = (max_degree + 1) ** 2
    key, q_key, k_key, v_key, pos_key = jax.random.split(key, num=5)
    q = jax.random.normal(
        q_key, (num_nodes, num_parity, num_degrees, num_features)
    )
    k = jax.random.normal(
        k_key, (num_nodes, num_parity, num_degrees, num_features)
    )
    v = jax.random.normal(
        v_key, (num_nodes, num_parity, num_degrees, num_features)
    )
    pos = jax.random.normal(pos_key, (num_nodes, 3))  # Positions.

    _, key_translation = jax.random.split(key, num=2)
    pos_translated = pos + jax.random.normal(key_translation, (1, 3))

    # Create batch segments and graph mask, which is trivial here
    # since we only consider a single graph
    batch_segments = jnp.zeros((num_nodes,), dtype=jnp.int16)
    graph_mask = jnp.array([True])

    out = rope.apply(
        q, k, v, pos, theta, grid_u, grid_w, batch_segments, graph_mask
    )  # original output

    out_translated = rope.apply(
        q,
        k,
        v,
        pos_translated,
        theta,
        grid_u,
        grid_w,
        batch_segments,
        graph_mask,
    )  # rotated output

    # Check for invariance.
    npt.assert_allclose(out, out_translated, atol=1e-5)


@pytest.mark.parametrize("num_nodes", [10])
@pytest.mark.parametrize("include_pseudotensors", [True])
@pytest.mark.parametrize("max_degree", [2])
@pytest.mark.parametrize("num_features", [32])
def test_rotation_equivariance(
        num_nodes,
        include_pseudotensors,
        max_degree,
        num_features
):
    # Parameters.
    key = jax.random.PRNGKey(0)
    assert num_features % 2 == 0

    # RoPE parameters.
    max_distance = 10.0
    theta = jnp.linspace(0, jnp.pi / max_distance, num_features // 2)

    # Lebedev grid.
    num = 26  # Should be enough for theta_max = pi.
    grid_u, grid_w = e3x.so3.lebedev_quadrature(precision=None, num=num)

    # Draw random query/key/value features and positions.
    num_parity = 2 if include_pseudotensors else 1
    num_degrees = (max_degree + 1) ** 2
    key, q_key, k_key, v_key, pos_key = jax.random.split(key, num=5)
    q = jax.random.normal(
        q_key, (num_nodes, num_parity, num_degrees, num_features)
    )
    k = jax.random.normal(
        k_key, (num_nodes, num_parity, num_degrees, num_features)
    )
    v = jax.random.normal(
        v_key, (num_nodes, num_parity, num_degrees, num_features)
    )
    pos = jax.random.normal(pos_key, (num_nodes, 3))  # Positions.

    # Draw random rotation matrix and calculate rotated quantities.
    _, rot_key = jax.random.split(key, num=2)
    rot = e3x.so3.random_rotation(rot_key)
    wigner_d = e3x.so3.wigner_d(rot, max_degree=max_degree)
    q_rot = jnp.einsum('...af,ab->...bf', q, wigner_d)
    k_rot = jnp.einsum('...af,ab->...bf', k, wigner_d)
    v_rot = jnp.einsum('...af,ab->...bf', v, wigner_d)
    pos_rot = jnp.einsum('...a,ab->...b', pos, rot)

    # Create batch segments and graph mask, which is trivial here
    # since we only consider a single graph
    batch_segments = jnp.zeros((num_nodes,), dtype=jnp.int16)
    graph_mask = jnp.array([True])

    out = rope.apply(
        q, k, v, pos, theta, grid_u, grid_w, batch_segments, graph_mask
    )  # original output

    out_rot = rope.apply(
        q_rot,
        k_rot,
        v_rot,
        pos_rot,
        theta,
        grid_u,
        grid_w,
        batch_segments,
        graph_mask,
    )  # rotated output

    # Check for equivariance.
    rot_out = jnp.einsum('...af,ab->...bf', out, wigner_d)
    npt.assert_allclose(out_rot, rot_out, atol=1e-5)

    npt.assert_equal(v.shape, out.shape)
    npt.assert_equal(v.shape, out_rot.shape)


@pytest.mark.parametrize("num", [50])
def test_sinc_equivalence(num):

    # Define the sinc function.
    def sinc(x):
        return jnp.sinc(x / jnp.pi)

    # Choose theta.
    theta = jnp.array([20 * jnp.pi])

    # We check two different numbers of Lebedev grid points, of which one is too
    # small to fulfill equivalence to the sinc on the expected interval.
    for num_is in ['to_small', 'correct']:
        # x range for plotting
        x = jnp.linspace(0, 1, 512)

        # Get the grid values and integration coefficients.
        if num_is == 'correct':
            grid_u, grid_w = e3x.so3.lebedev_quadrature(precision=None, num=num)
        elif num_is == 'to_small':
            grid_u, grid_w = e3x.so3.lebedev_quadrature(precision=None, num=num // 2)
        else:
            raise RuntimeError('Unknown test case.')

        # Draw random vectors.
        vec_key, rot_key = jax.random.split(jax.random.PRNGKey(0))
        v1 = jax.random.normal(vec_key, (x.size, 3))
        rot = e3x.so3.random_rotation(rot_key, num=x.shape[0])
        u = jnp.einsum(
            '...a,...ab->...b', jnp.zeros((x.size, 3)).at[:, 0].set(x), rot
        )
        v2 = v1 + u

        pos = jnp.stack([v1, v2], axis=1).reshape(-1, 3)

        # Due to internal attention normalization, one of the coefficients has to
        # be sqrt(2)
        q, k = jnp.array([jnp.sqrt(2), 0]), jnp.array([1, 0])
        v = jnp.array([1.0])

        q = jnp.repeat(q[None], int(2 * x.shape[0]), axis=0)
        k = jnp.repeat(k[None], int(2 * x.shape[0]), axis=0)
        v = jnp.repeat(v[None], int(2 * x.shape[0]), axis=0)

        batch_segments = jnp.stack(
            [jnp.arange(x.shape[0]), jnp.arange(x.shape[0])], axis=1
        ).reshape(-1)

        # Evaluate Eculidean RoPE.
        y_rope = rope.apply(
            q=q[:, None, None, :],
            k=k[:, None, None, :],
            v=v[:, None, None, :],
            pos=pos,
            grid_u=grid_u,
            grid_w=grid_w,
            theta=theta,
            batch_segments=batch_segments,
            graph_mask=jnp.array([True] * x.shape[0]),
        )

        # Evaluate sinc reference.
        y_sinc = sinc(theta * x)

        diff = jnp.abs(y_sinc.reshape(-1) - (y_rope.squeeze()[::2] - 1))
        cross = jnp.argwhere(diff > 1e-5).min()
        boundary = theta * x[cross]
        check = boundary >= rope.LEBEDEV_FREQUENCY_LOOKUP[num]
        if num_is == 'correct':
            npt.assert_equal(check, np.array([True]))
        if num_is == 'to_small':
            npt.assert_equal(check, np.array([False]))
