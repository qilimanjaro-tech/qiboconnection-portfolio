import numpy as np
from itertools import product

from qibo.models import Circuit
from qibo.gates import I, X, Y, Z, RX, RY

def six_operators(qubit):
    return [
        I(qubit),
        X(qubit),
        RX(qubit, np.pi / 2),
        RX(qubit, -np.pi / 2),
        RY(qubit, np.pi / 2),
        RY(qubit, -np.pi / 2),
    ]

Pn_ops = [I(0), X(0), Y(0), Z(0)]
Mi_ops = [I(0), Z(0)]


### Data format processing
def get_basis_elements_dict(nqubits):
    basis_elements = {}
    for i, x in enumerate(product(['0', '1'], repeat = nqubits)):
        basis_elements[''.join(x)] = i
    return basis_elements

def process_returned_dataformat(results, nqubits=2):
    """Organises the results returned by qiboconnection into a matrix.

    Args:
        results (list): list of result objects returned by qiboconnection
        nqubits (int, optional): number of qubits. Defaults to 2.

    Returns:
        res (array): matrix of dimensions (len(results), 2**nqubits) containing the
                    probabilities with which each bitstring was found for each circuit.
    """
    res = np.zeros((len(results), 2**nqubits))
    basis_elements_dict = get_basis_elements_dict(nqubits)
    for i, result in enumerate(results):
        for key, val in result["probabilities"].items():
            res[i, basis_elements_dict[key]] = val
    return res


def obtain_expectation_values_2qubits(array_probs_binary):  ## 00 01 10 11
    """
    The ordering that we set here is in order to automatically be aligned with the resulting
    order of the product iterator, but it exchanges the order of first and second qubit (i.e.,
    the qubit we set for the inner loop will appear as the outer one).
    """
    signs_op_diagonal = np.array([
                                [1, -1, 1, -1], #IZ
                                [1, 1, -1, -1], #ZI
                                [1, -1, -1, 1]  #ZZ
                                ])
    observables = ['IZ', 'ZI', 'ZZ']
    observables_probabilities = np.zeros(3)
    for i, _ in enumerate(observables):
        observables_probabilities[i] = array_probs_binary @ signs_op_diagonal[i]
    
    return observables_probabilities

  
def convert_probabilities2measurementops(data_matrix_probs):
    """Turns bitstring probabilities into the expectation values of the IZ, ZI and ZZ
    operators.

    Args:
        data_matrix_probs (array): matrix containing the probabilities for each bitstring
                                    for a set of circuits.
                                    
    Returns:
        array: matrix containing the expectation values of operators IZ, ZI and ZZ for each circuit
    """
    ncircuits = len(data_matrix_probs)
    data_measurementops = np.zeros((ncircuits, 3))
    for i in range(ncircuits):
        data_measurementops[i] += obtain_expectation_values_2qubits(data_matrix_probs[i])

    return data_measurementops

  
def prepare_linear_system_measurement_calibration(processed_data):
    """
    processed_data (array): numpy array of dimension (ncircuits=4, nbasiselements=4) containing the means of the bitstring experiment outcomes.
    """
    ideal_measurements = np.zeros(12)
    coeffs_mat = np.zeros((12, 12))

    ## stacked IZ, ZI and ZZ ideal measurements of each experiment, array of 3x3x3x3
    ideal_measurements = np.array([obtain_expectation_values_2qubits(x) for x in np.eye(4)]).flatten()

    for alpha, (experiment_index, operator_index) in enumerate(product(range(4), range(3))):
        data_currentcirc = processed_data[experiment_index]
        means_pauliobs = obtain_expectation_values_2qubits(data_currentcirc)
        coeffs_mat[alpha, operator_index * 4 : 4 * (operator_index + 1)] += np.append(1, means_pauliobs)

    return coeffs_mat, ideal_measurements


## QST system functions
def get_weights_state_tomography():  ### METHOD FOR QST
    """Method for QST.
    Calculates Tr[U_k^dag M_i U_k P_n] where P_n = {I, X, Y, Z}^{\otimes 2} and M_i = {II, IZ, ZI, ZZ}
    with the U_k set = basic_rotations.
    """
    basic_rotations_36states = six_operators(0)
    combs = list(product(basic_rotations_36states, basic_rotations_36states))
    weights_state_tomography = np.zeros((3, 16, len(combs)))
    Mi_circuit_gates = list(product(Mi_ops, repeat=2))[1:] ## we remove the identity, so that we are left with IZ, ZI, ZZ
    for k, comb in enumerate(combs):
        R1 = comb[0]
        R2 = comb[1]
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
                
                weights_state_tomography[i, n, k] = np.trace(np.real(np.conj(u_k).T @ m_i @ u_k @ P_n))

    return weights_state_tomography


def prepare_linear_system_QST(measurement_calibration_weights, means_ops_Rcircuit):
    """
    Returns all the components of the linear system to be solved to perform process tomography.
    """
    weights_state_tomography = get_weights_state_tomography()  # has indices i, n, k, i.e., shape (3, 16, len(combs))
    beta_Mi_calibration_matrix = measurement_calibration_weights  # this should have dim (4, 3), i.e., each column
    # is a measurement operator and each row
    # the coefficient accompanying II, IZ, ZI, ZZ
    # in that order
    len_combs = 36
    coeff_matrix = np.zeros(
        (3 * len_combs, 15)
    )  # we only need to solve 15 variables because of normalisation (p0 = 1-\sum_{n=1} pn)
    independent_term = np.zeros(3 * len_combs)

    for alpha, (i, k) in enumerate(
        product(range(3), range(len_combs))
    ):  # alpha from 0 to 107; i from 0 to 2; k from 0 to 35
        for n in range(15):
            coeff_matrix[alpha, n] += np.sum(beta_Mi_calibration_matrix[1:, i] * weights_state_tomography[:, n + 1, k])
        # the n+1 is because that part goes into the independent term
        dat_means_ops = means_ops_Rcircuit[k]

        independent_term[alpha] = dat_means_ops[i] - beta_Mi_calibration_matrix[0, i] / 4

    return coeff_matrix, independent_term


##  Additional QPT functions


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
    basic_rotations_36states = six_operators(0)
    combs = list(product(basic_rotations_36states, basic_rotations_36states))
    weights_process_tomography = np.zeros((16, len(combs)))

    for l, comb in enumerate(combs):
        R1 = comb[0]
        R2 = comb[1]
        c = Circuit(2)
        c.add(R1.on_qubits({R1.target_qubits[0]: 0}))
        c.add(R2.on_qubits({R2.target_qubits[0]: 1}))
        u_l = c.unitary()

        for m, pn_gates in enumerate(product(Pn_ops, repeat=2)):
            f = Circuit(2)
            f.add(pn_gates[0])
            f.add(pn_gates[1].on_qubits({pn_gates[1].target_qubits[0]: 1}))
            P_m = f.unitary()

            weights_process_tomography[m, l] = np.real((np.conj(u_l).T @ P_m @ u_l)[0, 0])

    return weights_process_tomography


def prepare_linear_system_QPT(state_reconstruction_paulibasis):
    """Generates linear system to be inverted in order to perform QPT

    Args:
        state_reconstruction_paulibasis (array): matrix containing the reconstruction
                                of the 6**nqubits basis states in the pauli basis

    Returns:
        coeffs_mat (array): coefficient matrix
        independent_term (array): independent term
    """
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
    """Computes the R matrix of an ideal process

    Args:
        ideal_operator (array): theoretical matrix describing the process

    Returns:
        R_mat (array): R matrix corresponding to ideal_operator.
    """
    R_mat = np.zeros((16, 16))
    for n, _ in enumerate(product(Pn_ops, repeat=2)):
        vec_in = np.zeros(16)
        vec_in[n] = 1
        rho_in = take_pauli_reconstructed_state_to_density_matrix(vec_in)  # ideal transformation
        rho_out = ideal_operator @ rho_in @ np.conj(ideal_operator).T
        vec_out = take_density_matrix_to_pauli_basis(rho_out)
        R_mat[:, n] = vec_out
    return R_mat
  

def compute_gate_fidelity(R_ideal, R_exp):
    return (np.trace(np.conj(R_ideal).T @ R_exp) + 2 * 2) / ((2 * 2) ** 2 + 2 * 2)


def compute_process_fidelity(R_ideal, R_exp):
    """Returns measure of the fidelity of 2QB gates (float).

    Args:
        R_ideal (array): R matrix of the ideal process
        R_exp (array): R matrix of the reconstructed process
    """
    return np.trace(np.conj(R_ideal).T @ R_exp) / (2 * 2) ** 2


def trace_out_extra_qubits(data_all_qubits, qubits2keep, nqubits):
    len_circuits = np.shape(data_all_qubits)[0]
    qb_assignment_idx = [(np.arange(2**nqubits) // 2 ** (nqubits - 1 - q)) % 2 for q in qubits2keep]
    traced_out_data = np.zeros((len_circuits, 2 ** len(qubits2keep)))
    for l_c in range(len_circuits):
        for idx in range(2**nqubits):
            idx_contributes_to_string = ("").join([str(x[idx]) for x in qb_assignment_idx])
            index_reduced_string = int(idx_contributes_to_string, 2)
            traced_out_data[l_c, index_reduced_string] += data_all_qubits[l_c, idx]
    return traced_out_data
