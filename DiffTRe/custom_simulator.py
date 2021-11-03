from typing import Callable, Tuple, TypeVar
import jax.numpy as jnp
from jax import jit, lax, random
from jax_md import dataclasses, quantity, util, space

Array = util.Array
f32 = util.f32

T = TypeVar('T')
ShiftFn = space.ShiftFn
InitFn = Callable[..., T]
ApplyFn = Callable[[T], T]
Simulator = Tuple[InitFn, ApplyFn]


def run_to_next_printout_fn_neighbors(apply_fn, neighbor_fn,
                                      n_steps_per_printout):
    """
    Initializes a function that runs simulation to next printout state and returns that state.

    Run simulation forward to each printout point and return state.
    Used to sample a specified number of states

    Args:
      apply_fn: Apply function from initialization of simulator
      neighbor_fn: Neighbor function
      n_steps_per_printout: Number of time steps to run for each printout state

    Returns:
      A function that takes the current simulation state and runs the simulation forward
      to the next printout state and returns that state

    """

    def do_step(cur_state, t):
        state, nbrs = cur_state
        new_state = apply_fn(state, neighbor=nbrs)
        nbrs = neighbor_fn(new_state.position, nbrs)
        new_sim_state = (new_state, nbrs)
        return new_sim_state, t

    @jit
    def run_small_simulation(start_state, t):
        printout_state, _ = lax.scan(do_step, start_state,
                                     xs=jnp.arange(n_steps_per_printout))
        cur_state, _ = printout_state
        return printout_state, cur_state

    return run_small_simulation


@dataclasses.dataclass
class TimingClass:
    """
    A dataclass containing runtimes of simulation

    Attributes:
    num_printouts_production: Number of states to save during production run
    num_dumped: Number of states to drop for equilibration
    timesteps_per_printout: Number of simulation timesteps to run for each for each printout

    """
    num_printouts_production: int
    num_dumped: int
    timesteps_per_printout: int


def process_printouts(time_step, total_time, t_equilib, print_every):
    """
    Initializes a dataclass containing information for the simulator on which states to save.

    Args:
        time_step: Time step size
        total_time: Total simulation time
        t_equilib: Equilibration time
        print_every: Time after which a state is saved: increase for improved decorrelation

    Returns:
        A class containing information for the simulator on which states to save.

    """
    timesteps_per_printout = int(print_every / time_step)
    num_printouts_production = int((total_time - t_equilib) / print_every)
    num_dumped = int(t_equilib / print_every)
    timings_struct = TimingClass(num_printouts_production, num_dumped,
                                 timesteps_per_printout)
    return timings_struct


@dataclasses.dataclass
class NVTLangevinState:
    """A struct containing state information for the Langevin thermostat.

  Attributes:
    position: The current position of the particles. An ndarray of floats with
      shape [n, spatial_dimension].
    velocity: The velocity of particles. An ndarray of floats with shape
      [n, spatial_dimension].
    force: The (non-stochistic) force on particles. An ndarray of floats with
      shape [n, spatial_dimension].
    mass: The mass of particles. Will either be a float or an ndarray of floats
      with shape [n].
    rng: The current state of the random number generator.
  """
    position: Array
    velocity: Array
    force: Array
    mass: Array
    rng: Array


def nvt_langevin(energy_or_force: Callable[..., Array],
                 shift: ShiftFn,
                 dt: float,
                 kT: float,
                 gamma: float = 0.1) -> Simulator:
    """Fixes the bug in the Langevin thermostat present in jax-md 0.1.13,
     where the force is not devided by the mass. Original docstring below.

    Simulation in the NVT ensemble using the Langevin thermostat.

    Samples from the canonical ensemble in which the number of particles (N),
    the system volume (V), and the temperature (T) are held constant. Langevin
    dynamics are stochastic and it is supposed that the system is interacting with
    fictitious microscopic degrees of freedom. An example of this would be large
    particles in a solvent such as water. Thus, Langevin dynamics are a stochastic
    ODE described by a friction coefficient and noise of a given covariance.

    Our implementation follows the excellent set of lecture notes by Carlon,
    Laleman, and Nomidis [1].

    Args:
        energy_or_force: A function that produces either an energy or a force from
            a set of particle positions specified as an ndarray of shape
            [n, spatial_dimension].
        shift_fn: A function that displaces positions, R, by an amount dR. Both R
            and dR should be ndarrays of shape [n, spatial_dimension].
        dt: Floating point number specifying the timescale (step size) of the
            simulation.
        kT: Floating point number specifying the temperature inunits of Boltzmann
            constant. To update the temperature dynamically during a simulation one
            should pass `kT` as a keyword argument to the step function.
        gamma: A float specifying the friction coefficient between the particles
            and the solvent.
    Returns:
        See above.

    [1] E. Carlon, M. Laleman, S. Nomidis. "Molecular Dynamics Simulation."
        http://itf.fys.kuleuven.be/~enrico/Teaching/molecular_dynamics_2015.pdf
        Accessed on 06/05/2019.
  """

    force_fn = quantity.canonicalize_force(energy_or_force)

    dt_2 = f32(dt / 2)
    dt2 = f32(dt ** 2 / 2)
    dt32 = f32(dt ** (3.0 / 2.0) / 2)

    kT = f32(kT)

    gamma = f32(gamma)

    def init_fn(key, R, mass=f32(1), **kwargs):
        _kT = kT if 'kT' not in kwargs else kwargs['kT']
        mass = quantity.canonicalize_mass(mass)

        key, split = random.split(key)

        V = jnp.sqrt(_kT / mass) * random.normal(split, R.shape, dtype=R.dtype)
        V = V - jnp.mean(V, axis=0, keepdims=True)

        return NVTLangevinState(R, V, force_fn(R, **kwargs) / mass, mass,
                                key)  # pytype: disable=wrong-arg-count

    def apply_fn(state, **kwargs):
        R, V, F, mass, key = dataclasses.astuple(state)

        _kT = kT if 'kT' not in kwargs else kwargs['kT']
        N, dim = R.shape

        key, xi_key, theta_key = random.split(key, 3)
        xi = random.normal(xi_key, (N, dim), dtype=R.dtype)
        theta = random.normal(theta_key, (N, dim), dtype=R.dtype) / jnp.sqrt(
            f32(3))

        # NOTE(schsam): We really only need to recompute sigma if the temperature
        # is nonconstant. @Optimization
        # TODO(schsam): Check that this is really valid in the case that the masses
        # are non identical for all particles.
        sigma = jnp.sqrt(f32(2) * _kT * gamma / mass)
        C = dt2 * (F - gamma * V) + sigma * dt32 * (xi + theta)

        R = shift(R, dt * V + C, **kwargs)
        F_new = force_fn(R, **kwargs) / mass
        V = (f32(1) - dt * gamma) * V + dt_2 * (F_new + F)
        V = V + sigma * jnp.sqrt(dt) * xi - gamma * C

        return NVTLangevinState(R, V, F_new, mass,
                                key)  # pytype: disable=wrong-arg-count

    return init_fn, apply_fn
