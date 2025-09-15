import numpy as np
import jax
import jax.numpy as jnp

from flax import struct
from jaxtyping import Array, Float, Int, Bool
from itertools import product
from typing import Optional

from euclidean_fast_attention.utils import space_utils

import e3x


def sparse_pairwise_indices_np(num: int, mask_self: bool = True):
  if num < 1:
    raise ValueError(f'num must be larger than 0, received {num}')

  dst_idx = np.repeat(jnp.arange(num), num)
  src_idx = np.tile(jnp.arange(num), num)
  if mask_self:
    dst_idx = np.delete(dst_idx, slice(0, num * num, num + 1))
    src_idx = np.delete(src_idx, slice(0, num * num, num + 1))
  return dst_idx, src_idx


def stable_erf_coulomb(x, a=1.0):
  mask = x < jnp.sqrt(jnp.finfo(x.dtype).eps)
  safe_x = jnp.where(mask, 1.0, x)
  a2x2 = a**2 * x**2
  taylor = (  # Definitely overkill, but better safe than sorry.
      2
      * a
      / jnp.sqrt(jnp.pi)
      * (
          1
          - a2x2 / 3
          + a2x2**2 / 10
          - a2x2**3 / 42
          + a2x2**4 / 216
          - a2x2**5 / 1320
          + a2x2**6 / 9360
      )
  )
  normal = jax.lax.erf(a * safe_x) / safe_x
  return jnp.where(mask, taylor, normal)


def get_kmax(Nxmax: int, Nymax: int, Nzmax: int) -> Array:
    kx = np.arange(0, Nxmax + 1)
    kx = np.concatenate([kx, -kx[1:]])
    ky = np.arange(0, Nymax + 1)
    ky = np.concatenate([ky, -ky[1:]])
    kz = np.arange(0, Nzmax + 1)
    kz = np.concatenate([kz, -kz[1:]])
    kmul = np.array(list(product(kx, ky, kz)))[1:]  # 0th entry is 0 0 0
    kmax = max(max(Nxmax, Nymax), Nzmax)
    return kmul[jnp.sum(kmul ** 2, axis=-1) <= kmax ** 2]


@struct.dataclass
class NaClPotential:
    C: float = struct.field(pytree_node=False)
    alpha: float = struct.field(pytree_node=False)
    beta: float = struct.field(pytree_node=False)
    vdW_radii: Array = struct.field(pytree_node=False)
    Nxmax: int = struct.field(pytree_node=False)
    Nymax: int = struct.field(pytree_node=False)
    Nzmax: int = struct.field(pytree_node=False)
    pbc_bool: bool = struct.field(pytree_node=False)
    kmul: Array = struct.field(pytree_node=False)
    cell: Array = struct.field(pytree_node=False)
    repulsion_bool: bool = struct.field(pytree_node=False)
    
    @classmethod
    def create(
        cls,
        C: float = 30.0,
        alpha: float = 0.5,
        beta: float = 0.25,
        pbc_bool: bool = False,
        repulsion_bool: bool = True,
        Nxmax: Optional[int] = None,
        Nymax: Optional[int] = None,
        Nzmax: Optional[int] = None,
        cell: Optional[Array] = None,
    ):
        vdW_radii = np.zeros((119, ))
        vdW_radii[11] = 0.95
        vdW_radii[17] = 1.81

        if pbc_bool == True:
            if Nxmax is None:
                raise ValueError(f"Nxmax must be set if pbc_bool is True. received {Nxmax=} and {pbc_bool=}.")
            if Nymax is None:
                raise ValueError(f"Nymax must be set if pbc_bool is True. received {Nymax=} and {pbc_bool=}.")
            if Nzmax is None:
                raise ValueError(f"Nzmax must be set if pbc_bool is True. received {Nzmax=} and {pbc_bool=}.")
            if cell is None:
                raise ValueError(f"cell must be set if pbc_bool is True. received {cell=} and {pbc_bool=}.")
            
            kmul = get_kmax(Nxmax, Nymax, Nzmax)  # (num_k, 3)
        else:
            kmul = None

        return NaClPotential(
            C=C,
            alpha=alpha,
            beta=beta,
            vdW_radii=vdW_radii,
            Nxmax=Nxmax,
            Nymax=Nymax,
            Nzmax=Nzmax,
            pbc_bool=pbc_bool,
            kmul=kmul,
            cell=cell,
            repulsion_bool=repulsion_bool
        )


    def repulsive_energy(
        self,
        distances: Float[Array, "num_pairs"],
        atomic_numbers: Int[Array, "num_atoms"],
        src_idx: Int[Array, "num_pairs"], 
        dst_idx: Int[Array, "num_pairs"]
    ) -> Float[Array, "num_pairs"]:

        vdW_radii = jnp.take(
            self.vdW_radii,
            atomic_numbers
        )  # (num_atoms, )

        r0 = vdW_radii[src_idx] + vdW_radii[dst_idx]  # (num_pairs, )

        return self.C * jnp.exp(-distances / self.beta / r0)


    def real_space(
        self,
        positions: Float[Array, "num_atoms 3"], 
        atomic_numbers: Int[Array, "num_atoms"],
        charges: Float[Array, "num_atoms"],
        src_idx: Int[Array, "num_pairs"], 
        dst_idx: Int[Array, "num_pairs"],
        batch_segments: Int[Array, "num_atoms"],
        graph_mask: Bool[Array, "num_graphs"]
    ) -> Float[Array, "num_graphs"]:

        # that might seem overkill, but it in principle allows for easy batching in the future
        
        ke = 14.399645351950548
        if src_idx is None:
            assert dst_idx is None
            pair_charges = charges[None] * charges[:, None]
            displacements = positions[None] - positions[:, None]
            distances = e3x.ops.norm(displacements, axis=-1)
            pairwise_energy = ke * stable_erf_coulomb(distances, self.alpha) * pair_charges
            pairwise_energy = jnp.fill_diagonal(pairwise_energy, 0.0, inplace=False)
            
            energy_per_graph = jnp.sum(pairwise_energy.reshape(-1)) / 2.0
        else:
            positions_dst = positions[dst_idx]
            positions_src = positions[src_idx]
            
            displacements = positions_src - positions_dst
            distances = e3x.ops.norm(displacements, axis=-1)
            
            electrostatics = ke * charges[src_idx] * charges[dst_idx] * stable_erf_coulomb(distances, self.alpha)
            pairwise_energy = electrostatics

            if self.repulsion_bool == True:
                repulsion = self.repulsive_energy(
                    distances=distances,
                    atomic_numbers=atomic_numbers,
                    src_idx=src_idx,
                    dst_idx=dst_idx
                )
                pairwise_energy += repulsion

            pairwise_energy = pairwise_energy / 2.0   # (num_pairs, )
            
            energy_per_atom = jax.ops.segment_sum(pairwise_energy, segment_ids=dst_idx, num_segments=len(batch_segments))
            energy_per_graph = jax.ops.segment_sum(energy_per_atom, segment_ids=batch_segments, num_segments=len(graph_mask))
            energy_per_graph = jnp.where(graph_mask, energy_per_graph, 0.0)
        
        return -jnp.sum(energy_per_graph), energy_per_graph

    def reciprocal_space(
        self,
        positions: Float[Array, "num_atoms 3"], 
        atomic_numbers: Int[Array, "num_atoms"],
        charges: Float[Array, "num_atoms"],
        src_idx: Int[Array, "num_pairs"], 
        dst_idx: Int[Array, "num_pairs"],
        cell_offsets: Int[Array, "num_pairs 3"],
        batch_segments: Int[Array, "num_atoms"] = None,
        graph_mask: Bool[Array, "num_graphs"] = None
    ) -> Float[Array, "num_pairs"]:

        ke = 14.399645351950548
        if batch_segments is not None or graph_mask is not None:
            raise NotImplemented(
                'Reciprocal space sum currently only supported for non-batched inputs.'
            )

        box_length = e3x.ops.norm(self.cell, axis=-1)  # (3)
        
        k = 2 * jnp.pi * self.kmul / jnp.expand_dims(box_length, axis=0)  # (num_k, 3)
        k2 = jnp.sum(jnp.square(k), axis=-1)  # (num_k)  
        
        # k grid does not include 0 0 0 so division is safe
        qg = jnp.exp(-0.25 * k2 / jnp.square(self.alpha)) / k2  # (num_k)

        # calculate the projections on the k-points
        r_dot_k = jnp.einsum("nd, kd -> nk", positions, k)  # (num_atoms, num_k)
        
        q_real = jnp.sum(jnp.expand_dims(charges, axis=1) * jnp.cos(r_dot_k), axis=0)  # (num_k)
        q_imag = jnp.sum(jnp.expand_dims(charges, axis=1) * jnp.sin(r_dot_k), axis=0)  # (num_k)

        qf = jnp.square(q_real) + jnp.square(q_imag)  # (num_k)
    
        # reciprocal energy
        box_volume = jnp.prod(box_length, axis=-1)  # ()
        reciprocal_energy = 2 * jnp.pi / box_volume * jnp.sum(qf * qg)  # ()
        
        # self energy
        self_energy = self.alpha / jnp.sqrt(jnp.pi) * jnp.sum(jnp.square(charges))  # ()
    
        # total electrostatic energy is reciprocal minus self energy
        total_electrostatic_energy = ke * (reciprocal_energy - self_energy)  # ()

        if self.repulsion_bool == True:
            # Need this for repulsive part.
            displacements = space_utils.calculate_displacement_vectors(
                positions=positions,
                dst_idx=dst_idx,
                src_idx=src_idx,
                batch_segments=jnp.zeros(len(positions)).astype(int),
                lattice_vectors=jnp.expand_dims(self.cell, axis=0),
                cell_offsets=cell_offsets,
                pbc_bool=True,
            )  # (num_pairs, 3)
            
            distances = e3x.ops.norm(displacements, axis=-1)  # (num_pairs)
        
            pairwise_repulsive_energy = self.repulsive_energy(
                distances=distances, 
                atomic_numbers=atomic_numbers, 
                src_idx=src_idx, 
                dst_idx=dst_idx
            )  # (num_pairs)

            # Account for double counting in repulsive energy.
            total_repulsive_energy = jnp.sum(pairwise_repulsive_energy) / 2.0  # ()
        else:
            total_repulsive_energy = 0.0

        pot_energy = total_electrostatic_energy + total_repulsive_energy
        
        return  -pot_energy, pot_energy


def place_atoms_in_sphere(
    diameter, 
    n_total,
):
    """
    Randomly places Na and Cl atoms in a sphere, returning separate arrays for
    atomic numbers and positions.

    Args:
        diameter (float): The diameter of the sphere.
        n_total (int): The total number of atoms (must be an even number).
        total_charge (int): The total charge.

    Returns:
        tuple: A tuple containing two NumPy arrays:
               - atomic_numbers (np.ndarray): An array of atomic numbers (11 for Na, 17 for Cl).
               - atomic_positions (np.ndarray): An Nx3 array of atom coordinates.
    """

    radius = diameter / 2.0
    
    n_na = n_total // 2
    n_cl = n_total - n_na
    
    # Correct for the fact that we will always have one Cl more for uneven number of n_total.
    # At the same time, it allows to sample 2 Na or 2 Cl for only two atoms or generally total charge +2 and -2 states.
    if n_total % 2 == 0:
        offset = np.random.randint(low=-1, high=2, size=(1, )).item()
    else:
        offset = np.random.randint(low=0, high=2, size=(1, )).item()
    
    n_na += offset
    n_cl -= offset

    atom_specs = np.zeros((17 + 1, ))
    atom_specs[11] = 0.95*0.3
    atom_specs[17] = 1.81*0.3

    atoms_to_place = [11] * n_na + [17] * n_cl
    np.random.shuffle(atoms_to_place)

    # Pre-allocate arrays for placed types and positions
    placed_types = np.full(n_total, np.inf, dtype=int)  # "inf" is a placeholder for empty slots
    placed_positions = np.empty((n_total, 3), dtype=float)
    n_placed = 0

    # Set a maximum number of attempts to avoid infinite loops in dense cases
    max_attempts_per_atom = 1000 * n_total

    for atomic_number in atoms_to_place:
        is_placed = False
        for _ in range(max_attempts_per_atom):
            # Generate a random point within a cube and check if it's in the sphere
            pos = np.random.uniform(-radius, radius, 3)
            if np.sum(pos**2) > radius**2:
                continue

            # Check for collisions with already placed atoms using numpy arrays
            if n_placed == 0:
                has_collision = False
            else:
                compare_types = placed_types[:n_placed]
                compare_positions = placed_positions[:n_placed]
                
                min_dists = np.square(atom_specs[compare_types] + atom_specs[atomic_number])
                dists_sq = np.sum((compare_positions - pos) ** 2, axis=1)
                
                has_collision = np.any(dists_sq < min_dists)

            if not has_collision:
                placed_types[n_placed] = atomic_number
                placed_positions[n_placed] = pos
                n_placed += 1
                is_placed = True
                break
        
        if not is_placed:
            print(f"⚠️ Warning: Could not place an atom of type {atomic_number} after {max_attempts_per_atom} attempts.")
            print(f"Stopping placement. Successfully placed {len(placed_types)} out of {n_total} atoms.")
            break

    return placed_types, placed_positions



def place_atoms_in_cube(
    L, 
    n_total,
):
    """
    Randomly places Na and Cl atoms in a cube with periodic boundary conditions,
    returning separate arrays for atomic numbers and positions.

    Args:
        L (float): The side length of the cubic box.
        n_total (int): The total number of atoms.

    Returns:
        tuple: A tuple containing two NumPy arrays:
               - atomic_numbers (np.ndarray): An array of atomic numbers (11 for Na, 17 for Cl).
               - atomic_positions (np.ndarray): An Nx3 array of atom coordinates.
    """
    
    # --- Atom Type and Count Setup (similar to original code) ---
    n_na = n_total // 2
    n_cl = n_total - n_na
    
    # # Randomly introduce a small charge imbalance if desired
    # if n_total % 2 == 0:
    #     offset = np.random.randint(low=-1, high=2)
    # else:
    #     offset = np.random.randint(low=0, high=2)
    
    # n_na += offset
    # n_cl -= offset

    # Define atomic radii for collision checks (Na: 11, Cl: 17)
    # Using an array for direct indexing by atomic number
    atom_radii = np.zeros((18, )) # Max atomic number is 17
    atom_radii[11] = 0.95*0.5  # Na radius
    atom_radii[17] = 1.81*0.5  # Cl radius

    atoms_to_place = [11] * n_na + [17] * n_cl
    np.random.shuffle(atoms_to_place)

    # --- Atom Placement Loop ---
    placed_types = np.zeros(n_total, dtype=int)
    placed_positions = np.zeros((n_total, 3), dtype=float)
    n_placed = 0

    max_attempts_per_atom = 1000 * n_total

    for atomic_number in atoms_to_place:
        is_placed = False
        for _ in range(max_attempts_per_atom):
            # Generate a random position inside the cube (centered at origin)
            pos = np.random.uniform(-L / 2.0, L / 2.0, 3)

            # Check for collisions with already placed atoms
            if n_placed == 0:
                has_collision = False
            else:
                # Get types and positions of atoms already in the box
                compare_types = placed_types[:n_placed]
                compare_positions = placed_positions[:n_placed]
                
                # Calculate the vector distance
                delta = compare_positions - pos
                
                # Apply periodic boundary conditions (minimum image convention)
                # This "wraps around" the box to find the shortest distance
                delta = delta - L * np.round(delta / L)
                
                # Calculate squared distances from the new atom to all others
                dists_sq = np.sum(delta ** 2, axis=1)
                
                # Calculate the minimum allowed squared distance (r1 + r2)^2
                min_dists_sq = (atom_radii[compare_types] + atom_radii[atomic_number])**2
                
                # A collision occurs if any squared distance is less than the minimum allowed
                has_collision = np.any(dists_sq < min_dists_sq)

            if not has_collision:
                placed_types[n_placed] = atomic_number
                placed_positions[n_placed] = pos
                n_placed += 1
                is_placed = True
                break
        
        if not is_placed:
            print(f"⚠️ Warning: Could not place an atom of type {atomic_number} after {max_attempts_per_atom} attempts.")
            print(f"Stopping placement. Successfully placed {n_placed} out of {n_total} atoms.")
            # Return only the atoms that were successfully placed
            return placed_types[:n_placed], placed_positions[:n_placed]

    return placed_types, placed_positions
