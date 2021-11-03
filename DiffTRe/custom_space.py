from jax_md import space
from jax import ops
import jax.numpy as jnp


def rectangular_boxtensor(box, spacial_dim):
    return ops.index_update(jnp.eye(spacial_dim), jnp.diag_indices(spacial_dim), box)


def scale_to_fractional_coordinates(R_init, box):
    spacial_dim = R_init.shape[1]
    box_tensor = rectangular_boxtensor(box, spacial_dim)
    inv_box_tensor = space.inverse(box_tensor)
    R_init = jnp.dot(R_init, inv_box_tensor)  # scale down to hypercube
    return R_init, box_tensor
