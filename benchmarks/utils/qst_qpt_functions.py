import os
import numpy as np
import qililab as ql
import qutip as qt

from qibo import gates
from qibo.models import Circuit
from itertools import product

from scipy.optimize import minimize


def basic_rotations_36states(qubit):
    return [
        (gates.I(qubit), "id"),
        (gates.RX(qubit, np.pi), "x_pi"),
        (gates.RX(qubit, np.pi / 2), "x_pi_half"),
        (gates.RX(qubit, -np.pi / 2), "x_minus_pi_half"),
        (gates.RY(qubit, np.pi / 2), "y_pi_half"),
        (gates.RY(qubit, -np.pi / 2), "y_minus_pi_half"),
    ]


def get_probabilities_data_combined_bitstrings(
    data_combined_bitstrings, nsubsets: int | None = None
):
    ncircuits, nbins, _ = np.shape(data_combined_bitstrings)
    if nsubsets is not None:
        divided_subsets = data_combined_bitstrings.reshape(
            ncircuits, nsubsets, nbins // nsubsets, 4
        )
        means_subsets = np.mean(divided_subsets, axis=2)
        means = np.mean(means_subsets, axis=1)
        stdevs = np.std(means_subsets, axis=1)
    else:
        means = np.mean(data_combined_bitstrings, axis=1)
        stdevs = None
    return means, stdevs


def obtain_expectation_values_2qubits(array_probs_binary):  ## 00 01 10 11
    """
    The ordering that we set here is in order to automatically be aligned with the resulting
    order of the product iterator, but it exchanges the order of first and second qubit (i.e.,
    the qubit we set for the inner loop will appear as the outer one).
    """
    signs_op_diagonal = np.array(
        [[1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]  # IZ  # ZI  # ZZ
    )
    observables = ["IZ", "ZI", "ZZ"]  # ['ZI', 'IZ', 'ZZ'] #['IZ', 'ZI', 'ZZ']
    observables_probabilities = np.zeros(3)
    for i, _ in enumerate(observables):
        observables_probabilities[i] = array_probs_binary @ signs_op_diagonal[i]

    return observables_probabilities


def get_probabilities_data_combined_measurementops(
    data_combined_bitstrings, nsubsets: int | None = None
):
    ncircuits, nbins, _ = np.shape(data_combined_bitstrings)

    data_combined_measurementops = np.zeros((ncircuits, nbins, 3))
    for i in range(ncircuits):
        for j in range(nbins):
            data_combined_measurementops[i, j] += obtain_expectation_values_2qubits(
                data_combined_bitstrings[i, j]
            )
    if nsubsets is not None:
        divided_subsets = data_combined_measurementops.reshape(
            ncircuits, nsubsets, nbins // nsubsets, 3
        )
        means_subsets = np.mean(divided_subsets, axis=2)
        means = np.mean(means_subsets, axis=1)
        stdevs = np.std(means_subsets, axis=1).flatten()
    else:
        means = np.mean(data_combined_bitstrings, axis=1)
        stdevs = None
    return means, stdevs


def prepare_linear_system_measurement_calibration(processed_data):
    """
    processed_data (array): numpy array of dimension (ncircuits=4, nbasiselements=4) containing the means of the bitstring experiment outcomes.
    """
    ideal_measurements = np.zeros(12)
    coeffs_mat = np.zeros((12, 12))

    coeffs_mat = np.zeros((12, 12))

    ## stacked IZ, ZI and ZZ ideal measurements of each experiment, array of 3x3x3x3
    ideal_measurements = np.array(
        [obtain_expectation_values_2qubits(x) for x in np.eye(4)]
    ).flatten()

    for alpha, (experiment_index, operator_index) in enumerate(
        product(range(4), range(3))
    ):
        data_currentcirc = processed_data[experiment_index]
        means_pauliobs = obtain_expectation_values_2qubits(data_currentcirc)
        coeffs_mat[alpha, operator_index * 4 : 4 * (operator_index + 1)] += np.append(
            1, means_pauliobs
        )

    return coeffs_mat, ideal_measurements


def solve_linear_system_measurement_calibration(
    coeff_matrix, independent_term, wlsq_stds=None, initial_guess=None
):
    """
    Returns a 3x4 matrix that takes in the measured probabilities for II (always 1), IZ, ZI and ZZ and turns them into
    the corrected expectation values of IZ, ZI and ZZ.
    """

    if wlsq_stds is None:
        calibration_matrix = np.linalg.solve(coeff_matrix, independent_term)
    else:
        if initial_guess is None:
            initial_guess = independent_term

        def residuals_weighted_least_squares(x):
            """We define the chisq function here in order to optimise the least squares with weights."""
            return np.sum(((independent_term - coeff_matrix @ x) / wlsq_stds) ** 2)

        res = minimize(fun=residuals_weighted_least_squares, x0=initial_guess)
        calibration_matrix = res.x

    return calibration_matrix.reshape(3, 4)


def get_measurement_calibration_bitstrings(
    processed_data, wlsq_stds=None, initial_guess=None
):
    """
    Returns a 4x4 matrix that takes in the measured bitstring probabilities in the order 00, 01, 10, 11 (as (1, 4) vectors)
    and turns them into the corrected probabilities of measuring 00, 01, 10, 11.
    Args:
        processed data (array): array of shape (ncircuits=4, nbitstrings=4) containing the measured probabilities of
                                    finding the 4 basis states for each basis preparation circuit.
        wlsq_stds (array|None): only not None if we want to do statistics over several subsets of data within the number
                                of bins. We are not taking that many measurements anyway at the moment, so currently we
                                don't gain much from it.
        initial_guess (array|None): only considered if wlsq_stds is not None. Initial guess required by the optimizer, if None
                                    it defaults to the independent term.
    """

    coeff_matrix = np.zeros((16, 16))
    ideal_measurements = np.eye(4).flatten()
    for alpha, (experiment_index, bitstring_index) in enumerate(
        product(range(4), range(4))
    ):
        data_currentcirc = processed_data[experiment_index]
        coeff_matrix[
            alpha, bitstring_index * 4 : 4 * (bitstring_index + 1)
        ] += data_currentcirc

    ## Solve linear system and solve it
    independent_term = ideal_measurements
    if wlsq_stds is None:
        calibration_matrix = np.linalg.solve(coeff_matrix, independent_term)
    else:
        if initial_guess is None:
            initial_guess = independent_term

        def residuals_weighted_least_squares(x):
            """We define the chisq function here in order to optimise the least squares with weights."""
            return np.sum(((independent_term - coeff_matrix @ x) / wlsq_stds) ** 2)

        res = minimize(fun=residuals_weighted_least_squares, x0=initial_guess)
        calibration_matrix = res.x

    return calibration_matrix.reshape(4, 4)


Pn_ops = [gates.I(0), gates.X(0), gates.Y(0), gates.Z(0)]
Mi_ops = [gates.I(0), gates.Z(0)]


def get_weights_state_tomography():  ### METHOD FOR QST
    """Method for QST.
    Calculates Tr[U_k^dag M_i U_k P_n] where P_n = {I, X, Y, Z}^{\otimes 2} and M_i = {II, IZ, ZI, ZZ}
    with the U_k set = basic_rotations.
    """
    combs = list(product(basic_rotations_36states(0), basic_rotations_36states(0)))
    weights_state_tomography = np.zeros((3, 16, len(combs)))

    Mi_circuit_gates = list(product(Mi_ops, repeat=2))[
        1:
    ]  ## we remove the identity, so that we are left with IZ, ZI, ZZ
    for k, comb in enumerate(combs):
        R1, _ = comb[0]
        R2, _ = comb[1]
        c = Circuit(2)
        c.add(R1.on_qubits({R1.target_qubits[0]: 0}))
        c.add(R2.on_qubits({R2.target_qubits[0]: 1}))
        u_k = c.unitary()

        for i, mi_gates in enumerate(Mi_circuit_gates):
            d = Circuit(2)
            d.add(mi_gates[0])
            d.add(mi_gates[1].on_qubits({mi_gates[1].target_qubits[0]: 1}))
            m_i = d.unitary()

            for n, pn_gates in enumerate(product(Pn_ops, repeat=2)):
                f = Circuit(2)
                f.add(pn_gates[0])
                f.add(pn_gates[1].on_qubits({pn_gates[1].target_qubits[0]: 1}))
                P_n = f.unitary()

                weights_state_tomography[i, n, k] = np.trace(
                    np.real(np.conj(u_k).T @ m_i @ u_k @ P_n)
                )

    return weights_state_tomography


def prepare_linear_system_QST(
    measurement_calibration_weights, means_ops_Rcircuit, stds_ops_Rcircuit
):
    """
    Returns all the components of the linear system to be solved to perform process tomography.
    """
    weights_state_tomography = (
        get_weights_state_tomography()
    )  # has indices i, n, k, i.e., shape (3, 16, len(combs))
    beta_Mi_calibration_matrix = measurement_calibration_weights  # this should have dim (4, 3), i.e., each column
    # is a measurement operator and each row
    # the coefficient accompanying II, IZ, ZI, ZZ
    # in that order
    len_combs = 36
    coeff_matrix = np.zeros(
        (3 * len_combs, 15)
    )  # we only need to solve 15 variables because of normalisation (p0 = 1-\sum_{n=1} pn)
    independent_term = np.zeros(
        3 * len_combs
    )  # will be filled by self.post_processed_results
    wlsq_stds = np.zeros(3 * len_combs)

    #### means_ops_Rcircuit has shape (36, 3)
    if stds_ops_Rcircuit is not None:
        stds_ops_Rcircuit = stds_ops_Rcircuit.reshape(
            36, 3
        )  ## we flatten it automatically when we return it in
        ## get_probabilities_data_combined_measurementops() because of convenience for the measurement
        ### calibration experiment, but for qst it's convenient to keep it separated to ensure we are not mirring the order.
        #### TODO: check whether the resulting order is indeed different from having it flattened in the first place
        # (it seems so)
    else:
        stds_ops_Rcircuit = np.zeros(
            (36, 0)
        )  ## so that the loop below doesn't complain

    for alpha, (i, k) in enumerate(
        product(range(3), range(len_combs))
    ):  # alpha from 0 to 107; i from 0 to 2; k from 0 to 35
        for n in range(15):
            coeff_matrix[alpha, n] += np.sum(
                beta_Mi_calibration_matrix[1:, i]
                * weights_state_tomography[:, n + 1, k]
            )
        # the n+1 is because the weight matrix had the identity there
        dat_means_ops = means_ops_Rcircuit[k]
        dat_stds_ops = stds_ops_Rcircuit[k]

        independent_term[alpha] = (
            dat_means_ops[i] - beta_Mi_calibration_matrix[0, i] / 4
        )
        wlsq_stds[alpha] = dat_stds_ops[i]

    if stds_ops_Rcircuit is None:
        wlsq_stds = None

    return coeff_matrix, independent_term, wlsq_stds


def solve_linear_system_QST(
    coeff_matrix, independent_term, wlsq_stds=None, initial_guess=np.zeros(15)
):
    """
    Returns reconstructed state in the pauli string basis without the first entry, corresponding to II and which is always equal to 1/4.
    """

    if wlsq_stds is None or (np.round(wlsq_stds, 6) == 0).all():
        reconstructed_state_paulibasis_noII = (
            np.linalg.pinv(coeff_matrix) @ independent_term
        )  # pseudo inverse @ indep terms

    else:

        def residuals_weighted_least_squares(x):
            """We define the chisq function here in order to optimise the least squares with weights."""
            return np.sum(((independent_term - coeff_matrix @ x) / wlsq_stds) ** 2)

        res = minimize(fun=residuals_weighted_least_squares, x0=initial_guess)
        reconstructed_state_paulibasis_noII = res.x
    return reconstructed_state_paulibasis_noII


def get_binbitstrings_from_loadresults(mmt_data):
    """
    Joins data from qubits 1 and 2 into a single bitstring from the format qililab returns it in.
    """
    ncircuits, _, nbins = np.shape(mmt_data)
    data_combined_bitstrings = np.zeros((ncircuits, nbins, 4))
    for k_c in range(ncircuits):
        for k_b in range(nbins):
            data_q0, data_q1 = mmt_data[k_c, :, k_b]
            binary_str_column = str(int(data_q0)) + str(int(data_q1))

            data_combined_bitstrings[k_c, k_b, 0] = binary_str_column == "00"
            data_combined_bitstrings[k_c, k_b, 1] = binary_str_column == "01"
            data_combined_bitstrings[k_c, k_b, 2] = binary_str_column == "10"
            data_combined_bitstrings[k_c, k_b, 3] = binary_str_column == "11"
    return data_combined_bitstrings


def take_pauli_reconstructed_state_to_density_matrix(rho_paulivec):
    """
    Takes in the Pauli vector of length 16 and returns the operator
    it describes in state space.
    """
    dim_hilbert_space = int(np.sqrt(len(rho_paulivec)))
    rho = 1j * 0.0 * np.eye(dim_hilbert_space)
    for n, pn_gates in enumerate(product(Pn_ops, repeat=2)):
        f = Circuit(2)
        f.add(pn_gates[0])
        f.add(pn_gates[1].on_qubits({pn_gates[1].target_qubits[0]: 1}))
        P_n = f.unitary()
        rho += round(rho_paulivec[n], 3) * P_n

    return rho


def take_density_matrix_to_pauli_basis(rho):
    dim_hilbert_space = np.shape(rho)[0]
    pn_vector = np.zeros(4**2)
    for n, pn_gates in enumerate(product(Pn_ops, repeat=2)):
        f = Circuit(2)
        f.add(pn_gates[0])
        f.add(pn_gates[1].on_qubits({pn_gates[1].target_qubits[0]: 1}))
        P_n = f.unitary()
        pn_vector[n] = np.trace(np.real(rho @ P_n)) / dim_hilbert_space
    return pn_vector


## additional QPT functions
def get_additional_weights_process_tomography():
    """
    Calculates Tr[U_k^dag M_i U_k P_n] where P_n = {I, X, Y, Z}^{\otimes 2} and M_i = {II, IZ, ZI, ZZ}
    with the U_k set = basic_rotations.
    """
    combs = list(product(basic_rotations_36states(0), basic_rotations_36states(0)))
    weights_process_tomography = np.zeros((16, len(combs)))

    for l, comb in enumerate(combs):
        R1, _ = comb[0]
        R2, _ = comb[1]
        c = Circuit(2)
        c.add(R1.on_qubits({R1.target_qubits[0]: 0}))
        c.add(R2.on_qubits({R2.target_qubits[0]: 1}))
        u_l = c.unitary()

        for m, pn_gates in enumerate(product(Pn_ops, repeat=2)):
            f = Circuit(2)
            f.add(pn_gates[0])
            f.add(pn_gates[1].on_qubits({pn_gates[1].target_qubits[0]: 1}))
            P_m = f.unitary()

            weights_process_tomography[m, l] = np.real(
                (np.conj(u_l).T @ P_m @ u_l)[0, 0]
            )

    return weights_process_tomography


def prepare_linear_system_QPT(state_reconstruction_paulibasis):
    weights_process_tomography = get_additional_weights_process_tomography()
    independent_term = np.zeros(16 * 36)
    coeffs_mat = np.zeros((16 * 36, 16 * 16))
    pn_out_data = state_reconstruction_paulibasis

    for n in range(16):
        for l in range(36):
            alpha = 36 * n + l
            independent_term[alpha] = 4 * pn_out_data[l, n]

            for m in range(16):
                beta = 16 * n + m
                coeffs_mat[alpha, beta] += weights_process_tomography[m, l]
    return coeffs_mat, independent_term


def get_ideal_R_matrix(ideal_operator):
    R_mat = np.zeros((16, 16))
    for n, _ in enumerate(product(Pn_ops, repeat=2)):
        state_in = np.zeros(16)
        state_in[n] = 1
        rho_in = take_pauli_reconstructed_state_to_density_matrix(
            state_in
        )  # ideal transformation

        rho_out = ideal_operator @ rho_in @ np.conj(ideal_operator).T
        state_out = take_density_matrix_to_pauli_basis(rho_out)
        R_mat[:, n] = state_out
    return R_mat


def compute_gate_fidelity(R_ideal, R_exp):
    return (np.trace(np.conj(R_ideal).T @ R_exp) + 2 * 2) / ((2 * 2) ** 2 + 2 * 2)


def compute_process_fidelity(R_ideal, R_exp):
    return np.trace(np.conj(R_ideal).T @ R_exp) / (2 * 2) ** 2


from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib as mpl
from qutip.matplotlib_utilities import complex_phase_cmap


def plot_matrix(M: np.ndarray, ax: mpl.axes.Axes | None = None):
    """ Adapted from qtup `matrix_histogram_complex`. Draw a histogram for the amplitudes of matrix M, using the argument
    of each element for coloring the bars.

    Args:
        M (np.ndarray): matrix to plot
        ax (matplotlib.axes.Axes): matplotlib axis
    """
    if ax is None:
        fig=plt.figure()
        ax = fig.add_subplot(projection='3d')

    n = np.size(M)
    xpos, ypos = np.meshgrid(range(M.shape[0]), range(M.shape[1]))
    xpos = xpos.T.flatten() - 0.5
    ypos = ypos.T.flatten() - 0.5
    zpos = np.zeros(n)
    dx = dy = 0.8 * np.ones(n)
    Mvec = M.flatten()
    dz = abs(Mvec)

    # make small numbers real, to avoid random colors
    idx, = np.where(abs(Mvec) < 0.001)
    Mvec[idx] = abs(Mvec[idx])

    phase_min = -np.pi
    phase_max = np.pi

    norm = mpl.colors.Normalize(phase_min, phase_max)
    cmap = complex_phase_cmap()

    colors = cmap(norm(np.angle(Mvec)))

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors)

    xtics = -0.5 + np.arange(M.shape[0])
    ax.axes.xaxis.set_major_locator(plt.FixedLocator(xtics))
    ax.tick_params(axis='x', labelsize=12)
    ytics = -0.5 + np.arange(M.shape[1])
    ax.axes.yaxis.set_major_locator(plt.FixedLocator(ytics))
    ax.tick_params(axis='y', labelsize=12)
    ax.set_zlim3d([0, 1])  # use min/max

    return ax
