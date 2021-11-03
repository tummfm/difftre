import jax.numpy as jnp
from jax import value_and_grad, checkpoint, jit, lax
import optax
import time
import warnings

from jax_md import util, dataclasses
from DiffTRe import custom_simulator


def compute_energy_trajectory(trajectory, fixed_reference_nbrs, neighbor_fn, energy_fn):
    """
    Computes potential energy values for all states in a trajectory using lax.scan.

    Args:
        trajectory: Trajectory of states from simulation
        fixed_reference_nbrs: A reference neighbor list to recompute neighbors for each snapshot allowing jit
        neighbor_fn: Neighbor function
        energy_fn: Energy function

    Returns:
        An array of potential energy values containing the energy of each state in trajectory
    """
    def energy_trajectory(dummy_carry, state):
        R = state.position
        nbrs = neighbor_fn(R, fixed_reference_nbrs)
        energy = energy_fn(R, neighbor=nbrs)
        return dummy_carry, energy

    _, U_traj = lax.scan(energy_trajectory, jnp.array(0., dtype=jnp.float32), trajectory)
    return U_traj


def compute_quantity_traj(traj_state, quantities, neighbor_fn, energy_params=None):
    """
    Computes quantities of interest for all states in a trajectory using lax.scan.

    Arbitrary quantity functions can be provided via the quantities dict.
    The quantities dict should provide each quantity function via its own key
    that contains another dict containing the function under the 'compute_fn' key.
    The resulting quantity trajectory will be saved in a dict under the same key
    as the input quantity function.

    Args:
        traj_state: Trajectory state as output from trajectoty generator
        quantities: The quantity dict containing for each target quantity a dict containing
                    the quantity function under 'compute_fn'
        neighbor_fn: Neighbor function
        energy_params: Energy params for energy_fn_template to initialize the current energy_fn

    Returns:
        A dict of quantity trajectories saved under the same key as the input quantity function.
    """
    sim_state, trajectory, _ = traj_state
    _, fixed_reference_nbrs = sim_state

    @jit
    def quantity_trajectory(dummy_carry, state):
        R = state.position
        nbrs = neighbor_fn(R, fixed_reference_nbrs)
        computed_quantities = {quantity_fn_key: quantities[quantity_fn_key]['compute_fn'](
            state, neighbor=nbrs, energy_params=energy_params) for quantity_fn_key in quantities}
        return dummy_carry, computed_quantities

    _, quantity_trajs = lax.scan(quantity_trajectory, 0., trajectory)
    return quantity_trajs


def trajectory_generator_init(sim_funs, timings_struct):
    """
    Initializes a trajectory_generator function that computes a new trajectory stating at the last state.
    Additionally computes energy values for each state used during the reweighting step.

    Args:
        sim_funs: A tuple containing simulation functions (simulator_template, energy_fn_template, neighbor_fn)
        timings_struct: Instance of TimingClass containing information about which states to retain

    Returns:
        A function taking energy params and the current state (including neighbor list) that runs
        the simulation forward generating the next trajectory state: (new_sim_state, traj, U_traj)
    """
    num_printouts_production, num_dumped, timesteps_per_printout = dataclasses.astuple(timings_struct)
    simulator_template, energy_fn_template, neighbor_fn = sim_funs

    def generate_reference_trajectory(params, sim_state):
        energy_fn = energy_fn_template(params)
        _, apply_fn = simulator_template(energy_fn)
        run_small_simulation = custom_simulator.run_to_next_printout_fn_neighbors(apply_fn, neighbor_fn,
                                                                                  timesteps_per_printout)

        sim_state, _ = lax.scan(run_small_simulation, sim_state, xs=jnp.arange(num_dumped))  # equilibrate
        new_sim_state, traj = lax.scan(run_small_simulation, sim_state, xs=jnp.arange(num_printouts_production))
        final_state, nbrs = new_sim_state

        # always recompute neighbor list from last, fixed neighbor list.
        # Note: one could save all the neighbor lists at the printout times, if memory permits
        #       In principle this energy computation could be omitted altogether if ones saves the energy
        #       from the simulator for each printout state.
        #       We did not opt for this optimization to not destroy compatibility with Jax MD simulators
        U_traj = compute_energy_trajectory(traj, nbrs, neighbor_fn, energy_fn)
        return new_sim_state, traj, U_traj

    return generate_reference_trajectory


def weight_computation_init(energy_fn_template, neighbor_fn, kbT):
    """
    Initializes a function that computes weights for the reweighting approach in the NVT ensemble.

    Args:
        energy_fn_template: Energy function template
        neighbor_fn: Neighbor function
        kbT: Temperature in kbT

    Returns:
        A function computing weights and the effective sample size given current energy params and
        trajectory state (from most recent trajectory generation)
    """

    def estimate_effective_samples(weights):
        weights = jnp.where(weights > 1.e-10, weights, 1.e-10)  # mask to avoid NaN from log(0) if a few weights are 0.
        exponent = - jnp.sum(weights * jnp.log(weights))
        return jnp.exp(exponent)

    def compute_weights(params, traj_state):
        sim_state, trajectory, U_traj = traj_state
        _, nbrs = sim_state

        energy_fn = checkpoint(energy_fn_template(params))  # whole backward pass too memory consuming
        U_traj_new = compute_energy_trajectory(trajectory, nbrs, neighbor_fn, energy_fn)

        # Difference in pot. Energy is difference in total energy as kinetic energy is the same and cancels
        exponent = -(1. / kbT) * (U_traj_new - U_traj)
        # Note: The reweighting scheme is a softmax, where the exponent above represents the logits.
        #       To improve numerical stability and guard against overflow it is good practice to subtract
        #       the max of the exponent using the identity softmax(x + c) = softmax(x). With all values
        #       in the exponent <=0, this rules out overflow and the 0 value guarantees a denominator >=1.
        exponent -= jnp.max(exponent)
        prob_ratios = jnp.exp(exponent)
        weights = prob_ratios / util.high_precision_sum(prob_ratios)
        n_eff = estimate_effective_samples(weights)
        return weights, n_eff

    return compute_weights


def mse_loss(predictions, targets):
    """Computes mean squared error loss for given predictions and targets."""
    squared_difference = jnp.square(targets - predictions)
    mean_of_squares = util.high_precision_sum(squared_difference) / predictions.size
    return mean_of_squares


def init_independent_mse_loss_fn(quantities):
    """
    Initializes the default loss function, where MSE errors of destinct quantities are added.

    First, observables are computed via the reweighting scheme. These observables can be ndarray
    valued, e.g. vectors for RDF / ADF or matrices for stress. For each observable, the element-wise
    MSE error is computed wrt. the target provided in "quantities[quantity_key]['target']".
    This per-quantity loss is multiplied by gamma in "quantities[quantity_key]['gamma']". The final loss is
    then the sum over all of these weighted per-quantity MSE losses.
    A pre-requisite for using this function is that observables are simply ensemble averages of
    instantaneously fluctuating quantities. If this is not the case, a custom loss_fn needs to be defined.
    The custom loss_fn needs to have the same input-output signuture as the loss_fn implemented here.


    Args:
        quantities: The quantity dict with 'compute_fn', 'gamma' and 'target' for each observable

    Returns:
        The loss_fn taking trajectories of fluctuating properties, computing ensemble averages via the
        reweighting scheme and outputs the loss and predicted observables.

    """
    def loss_fn(quantity_trajs, weights):
        loss = 0.
        predictions = {}
        for quantity_key in quantities:
            quantity_snapshots = quantity_trajs[quantity_key]
            weighted_snapshots = (quantity_snapshots.T * weights).T
            ensemble_average = util.high_precision_sum(weighted_snapshots, axis=0)  # weights account for "averaging"
            predictions[quantity_key] = ensemble_average
            loss += quantities[quantity_key]['gamma'] * mse_loss(ensemble_average, quantities[quantity_key]['target'])
        return loss, predictions
    return loss_fn


def gradient_and_propagation_init(compute_weights, trajectory_generation_fn, loss_fn, quantities,
                                  neighbor_fn, reweight_ratio=0.9):
    """
    Initializes the main DiffTRe function that recomputes trajectories when needed 
    and computes gradients of the loss wrt. energy function parameters.
    
    Args:
        compute_weights: Initialized weight compute function from weight_computation_init
        trajectory_generation_fn: Initialized trajectory generator function from trajectory_generator_init
        loss_fn: Loss function, e.g. from init_independent_mse_loss_fn or custom
        quantities: The quantity dict containing for each target quantity a dict containing
                    at least the quantity compute function under 'compute_fn'.
        neighbor_fn: Neighbor function
        reweight_ratio: Ratio of reference samples required for n_eff to surpass to allow
                        re-use of previous reference trajectory state

    Returns:
        A function that takes current energy parameters and trajectory state and returns the
        new trajectory state, the gradient of the loss wrt. energy parameters, the loss value,
        an error code indicating caught exceptions and predicted quantities of the current model state
    """
    def reweighting_loss(params, traj_state):
        """
        Computes the loss using the DiffTRe formalism and additionally return predictions of the current model.
        """
        weights, _ = compute_weights(params, traj_state)
        quantity_trajs = compute_quantity_traj(traj_state, quantities, neighbor_fn, params)
        loss, predictions = loss_fn(quantity_trajs, weights)
        return loss, predictions

    def trajectory_identity_mapping(input):
        """Helper function to re-use trajectory if no recomputation is necessary."""
        _, traj_state = input
        error_code = 0
        return traj_state, error_code

    def recompute_trajectory(input):
        """
        Recomputes the reference trajectory, starting from the last state of the previous trajectory
        to save equilibration time.
        """
        params, traj_state = input
        sim_state = traj_state[0]
        updated_traj_state = trajectory_generation_fn(params, sim_state)
        _, nbrs = updated_traj_state[0]
        error_code = lax.cond(nbrs.did_buffer_overflow, lambda _: 1, lambda _: 0, None)
        return updated_traj_state, error_code

    @jit
    def propagation_fn(params, traj_state):
        # check if trajectory re-use is possible; otherwise recompute trajectory
        weights, n_eff = compute_weights(params, traj_state)
        n_snapshots = traj_state[2].size
        recompute = n_eff < reweight_ratio * n_snapshots
        traj_state, error_code = lax.cond(recompute, recompute_trajectory, trajectory_identity_mapping,
                                          (params, traj_state))

        outputs, curr_grad = value_and_grad(reweighting_loss, has_aux=True)(params, traj_state)
        loss_val, predictions = outputs
        return traj_state, curr_grad, loss_val, error_code, predictions

    return propagation_fn


def update_init(propagation_fn, optimizer, neighbor_fn):
    """
    Initializes the update function that is called iteratively to updates the energy parameters.

    The returned function computes the gradient used by the optimizer to update the energy parameters
    and the opt_state and returns loss and predictions at the current step. Additionally it handles the
    case of neighborlist buffer overflow by recomputing the neighborlist from the last state of the
    new reference trajectory and resets the trajectoty state such that the update is only performed
    with a proper neighbor list. The update function is not jittable for this reason.

    Args:
        propagation_fn: DiffTRe gradient and propagation function as initialized from gradient_and_propagation_init
        optimizer: Optimizer from optax
        neighbor_fn: Neighbor function

    Returns:
        A function that will be called iteratively by the user to update energy params via
        the optimizer and output model predictions at the current state
    """

    def update(step, params, opt_state, traj_state):
        new_traj_state, curr_grad, loss_val, error_code, predictions = propagation_fn(params, traj_state)

        if error_code == 1:  # neighbor list buffer overflowed: Enlarge neighbor list and rerun with old traj_state
            warnings.warn('Neighborlist buffer overflowed at step ' + str(step) + '. Initializing larger neighborlist.')
            new_state = new_traj_state[0][0]
            enlarged_nbrs = neighbor_fn(new_state.position)
            reset_traj_state = ((traj_state[0][0], enlarged_nbrs), traj_state[1], traj_state[2])
            new_traj_state = reset_traj_state
            new_params = params
        else:  # only use gradient if neighbor list did not overflowed
            scaled_grad, opt_state = optimizer.update(curr_grad, opt_state, params)
            new_params = optax.apply_updates(params, scaled_grad)
        return new_params, opt_state, new_traj_state, loss_val, predictions
    return update


def DiffTRe_init(simulation_fns, timings_struct, quantities, kbT, init_params, reference_state, optimizer,
                 loss_fn=None, reweight_ratio=0.9):
    """
    Initializes all functions for DiffTRe and returns an update function and the first reference trajectory.

    This function needs definition of all parameters of the simulation and DiffTRe.
    Simulation_fns is a tuple defining all constant parameters of the simulation:
    Simulator_template is a function that takes an energy function and returns a simulator function.
    Energy_fn_template is a function that takes energy parameters and initializes an new energy function.
    The quantity dict defines the way target observations contribute to the loss function.
    Each target observable needs to be saved in the quantity dict via a unique key. Model predictions
    will be output under the same key. Each unique observable needs to provide another dict containing
    a function computing the observable under 'compute_fn', a multiplier controlling the weight of the
    observable in the loss function under 'gamma' as well as the prediction target under 'target'.

    In many applications, the default loss function will be sufficient. If a target observable cannot
    be described directly as an average over instantaneous quantities (e.g. stiffness in the diamond example),
    a custom loss_fn needs to be defined. The signature of the loss_fn needs to be the following:
    It takes the trajectory of computed instantaneous quantities saved in a dict under its respective key of the
    quantities_dict. Additionally, it receives corresponding weights w_i to perform ensemble averages under the
    reweighting scheme. With these components, ensemble averages of more complex observables can be computed.
    The output of the function is (loss value, predicted ensemble averages). The latter is only necessary
    for post-processing the optimization process. See 'init_independent_mse_loss_fn' for an example implementation.

    Args:
        simulation_fns: A tuple containing simulation functions (simulator_template, energy_fn_template, neighbor_fn)
        timings_struct: Instance of TimingClass containing information about which states to retain
        quantities: The quantity dict with 'compute_fn', 'gamma' and 'target' for each observable
        kbT: Temperature in kbT
        init_params: Initial energy parameters
        reference_state: Tuple of initial siumlation state and neighbor list
        optimizer: Optimizer from optax
        loss_fn: Custom loss function taking the trajectory of quantities and weights and returning
                 the loss and predictions; Default None initializes independent MSE loss
        reweight_ratio: Ratio of reference samples required for n_eff to surpass to allow
                        re-use of previous reference trajectory state

    Returns:
        An update function and the first reference trajectory
    """
    if loss_fn is None:
        loss_fn = init_independent_mse_loss_fn(quantities)
    else:
        print('Using custom loss function. Ignoring \'gamma\' and \'target\' in  \"quantities\".')

    _, energy_fn_template, neighbor_fn = simulation_fns
    trajectory_generation_fn = trajectory_generator_init(simulation_fns, timings_struct)
    compute_weights = weight_computation_init(energy_fn_template, neighbor_fn, kbT)
    propagation_fn = gradient_and_propagation_init(compute_weights, trajectory_generation_fn, loss_fn, quantities,
                                                   neighbor_fn, reweight_ratio=reweight_ratio)
    update_fn = update_init(propagation_fn, optimizer, neighbor_fn)

    t_start = time.time()
    traj_initstate = trajectory_generation_fn(init_params, reference_state)
    runtime = (time.time() - t_start) / 60.
    print('Time for a single trajectory generation:', runtime, 'mins')

    return update_fn, traj_initstate
