# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 17:10:41 2020

@author: Muqing Zheng
"""

# Qiskit
from qiskit import QuantumCircuit, transpile

# Numerical/Stats pack
import csv
import pandas as pd
import numpy as np
import json
import scipy.stats as ss
import numpy.linalg as nl
# For optimization
from cvxopt import matrix, solvers

import measfilter
import importlib
importlib.reload(measfilter)
from measfilter import *
# For plot
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
width = 6.72 # plot width
height = 4.15 # plot height


######################## For Parameter Characterzation ########################
def gate_circ(nGates, gate_type, interested_qubit, itr, backend):
    """
      Generate circuits for gate error experiment

    Parameters
    ----------
    nGates : int
        number of gates.
    gate_type : String
        Chosen between "X", "Y", and "Z".
    interested_qubit : int
        on which qubit that those gates apply on.
    itr : int
        number of iteration to submit on the same qubit 
        with defacult 8192 shots. (So in total runs itr*8192 times)

    Returns
    -------
    None.

    """
    circ = QuantumCircuit(1, 1)
    for _ in range(nGates):
        if gate_type == 'X':
            circ.x(0)
        elif gate_type == 'Y':
            circ.y(0)
        elif gate_type == 'Z':
            circ.z(0)
        else:
            raise Exception('Choose gate_type from X, Y, Z')
        # Barriers prevent compiler optimization (crucial for noise characterization)
        circ.barrier(0)
    circ.measure([0], [0])

    circ_trans = transpile(circ, backend, initial_layout=[interested_qubit])
    print('Circ depth is ', circ_trans.depth())
    
    circs = []
    for i in range(itr):
        new_circ = circ_trans.copy()
        new_circ.name = 'itr' + str(i)
        circs.append(new_circ)
    return circs


def Gateexp(nGates,
            gate_type,
            interested_qubit,
            itr,
            backend,
            file_address='',
            job_id=''):
    """
      Function for collect data for gate error experiment

    Parameters
    ----------
    nGates : int
        number of gates.
    gate_type : String
        Chosen between "X", "Y", and "Z".
    interested_qubit : int
        on which qubit that those gates apply on.
    itr : int
        number of iteration to submit on the same qubit 
        with defacult 8192 shots. (So in total runs itr*8192 times)
    backend: Robust backend object
        backend primarily supports Amazon Braket, before trying to call
        `retrive_job` from qiskit-aer. Functional as of Qiskit 2.2.3.
    file_address: string, optional
        The relative file address to save data file. 
        Ends with '/' if not empty
        The default is ''.
    job_id: string
        If the job has been run before, put job id here;
        otherwise a new job will be created and submiited to selected backend

    Returns
    -------
    None.

    """

    circs = gate_circ(nGates, gate_type, interested_qubit, itr, backend)
    
    if job_id:
            try:
                # Attempt 1: Retrieve via provider (Common for Amazon Braket / IBM)
                if hasattr(backend, 'provider') and backend.provider:
                    job_exp = backend.provider.retrieve_job(job_id)
                
                # Attempt 2: Retrieve directly from backend (Legacy support)
                elif hasattr(backend, 'retrieve_job'):
                    job_exp = backend.retrieve_job(job_id)
                
                # Attempt 3: If using Qiskit Runtime Service 
                elif hasattr(backend, 'service'):
                    job_exp = backend.service.job(job_id)
                    
                else:
                    raise NotImplementedError("This backend does not support direct job retrieval.")
                
                print("Successfully retrieved Job ID:", job_exp.job_id())

            except Exception as e:
                print(f"\n[Error] Could not retrieve Job ID {job_id}")
                print(f"Reason: {str(e)}")
                print("Note: Qiskit Aer (Simulators) cannot retrieve jobs from previous sessions.")
                return # Exit to avoid crashing later when trying to access results
    else:
        # Run new job if no ID provided
        # Optimization_level=0 is crucial to prevent removing the calibration gates
        try:
            # Handle shots depending on backend signature
            job_exp = backend.run(circs, shots=8192, memory=True)
        except TypeError:
            # Fallback if backend doesn't accept 'memory' kwarg (some simulators)
            job_exp = backend.run(circs, shots=8192)
            
        print("New Job submitted. Job ID:", job_exp.job_id())

    # Record bit string
    try:
        exp_results = job_exp.result()
        readout = np.array([])
        for i in range(itr):
            # Use integer index instead of name to ensure compatibility across providers
            readout = np.append(readout, exp_results.get_memory(i))

        # Save to file
        filename = f'Readout_{nGates}{gate_type}Q{interested_qubit}.csv'
        full_path = file_address + filename
        
        with open(full_path, mode='w') as sgr:
            read_writer = csv.writer(sgr, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            read_writer.writerow(readout)
        print(f"Data saved to {full_path}")
        
    except Exception as e:
        print(f"Failed to process/save results: {e}")

# used to call QoI
def QoI_gate(prior_lambdas, ideal_p0, gate_num):
    """
    Function equivalent to Q(lambda) in https://doi.org/10.1137/16M1087229

    Parameters
    ----------
    prior_lambdas : numpy array
        each subarray is an individual prior lambda.
    ideal_p0: float
        probability of measuring 0 without any noise.
    gate_num: int
        number of gates in the circuit. 
        Should be the same as nGates in Gateexp().

    Returns
    -------
    qs : numpy array
        QoI's. Here they are the probability of measuring 0 with each given
        prior lambdas in prior_lambdas.

    """
    shape = prior_lambdas.shape
    nQubit = 1

    # Initialize the output array
    qs = np.array([], dtype=np.float64)

    # Smiluate measurement error, assume independence
    p0 = ideal_p0
    p1 = 1 - p0
    # Compute Fourier coefficient
    phat0 = 1 / 2 * (p0 * (-1)**(0 * 0) + p1 * (-1)**(0 * 1))
    phat1 = 1 / 2 * (p0 * (-1)**(1 * 0) + p1 * (-1)**(1 * 1))

    for i in range(shape[0]):
        eps = prior_lambdas[i][2]
        noisy_p0 = ((1 - eps)**(0))**gate_num * phat0 * (-1)**(0 * 0) + (
            (1 - eps)**(1))**gate_num * phat1 * (-1)**(1 * 0)
        noisy_p1 = ((1 - eps)**(0))**gate_num * phat0 * (-1)**(0 * 1) + (
            (1 - eps)**(1))**gate_num * phat1 * (-1)**(1 * 1)
        M_ideal = np.array([noisy_p0, noisy_p1])

        A = errMitMat(prior_lambdas[i])
        M_meaerr = np.dot(A, M_ideal)

        # Only record interested qubits
        qs = np.append(qs, M_meaerr[0])
    return qs

def data_readout(datafile = None, data = None):
    """
    Function to readout json files or suitable numpy arrays
    """

    if datafile.endswith('.json'):
        with open(str(datafile), 'r') as file:
            data = json.load(file)
        qubit_props = data['oneQubitProperties']
        num_qubits = len(qubit_props)

        epsilon01 = np.zeros(num_qubits)
        epsilon10 = np.zeros(num_qubits)
        gate_epsilon = np.zeros(num_qubits)

        for qub_ind,qub in enumerate(qubit_props):
            e01 = qubit_props[qub]['oneQubitFidelity'][1]['fidelity'] # Notice that even though it says "fidelity", we get error rate...
            e10 = qubit_props[qub]['oneQubitFidelity'][2]['fidelity'] # Notice that even though it says "fidelity", we get error rate...
            gate_err = qubit_props[qub]['oneQubitFidelity'][0]['fidelity']
            epsilon01[qub_ind] = e01
            epsilon10[qub_ind] = e10
            gate_epsilon[qub_ind] = gate_err
        
    elif type(data) == np.ndarray:
        if data.shape[1] != 3:
            raise Exception("Warning: Data array shape does not match simulation setting!\nData shape: "
                    + str(data.shape) + "\nSimulation qubits: " + str(num_qubits))
        epsilon01 = data[:,0].tolist()
        epsilon10 = data[:,1].tolist()
        gate_epsilon = data[:,2].tolist()

    elif datafile is None and data is None:
        raise Exception("Error: No data or datafile provided for the `data_readout()`")
    else:
        raise Exception("Must either:\nsupply datafile as path string to a json data file, i.e. '../data/datafile.json'\nor supply a nx3 numpy array with 0 -> 1 errors, 1 -> 0 errors, and gate errors in each row (ONLY AVAILABLE FOR SINGLE QUBIT GATES)")

    lambdas = np.array([epsilon01,epsilon10,gate_epsilon])
    return lambdas

# Used to call output, delete ideal_p0 parameter
# WARNING: After 2021 Summer, 0.5*eps_in_code = eps_in_paper
def output_gate(d,
                interested_qubit,
                M,
                params,
                gate_sd,
                meas_sd,
                gate_type,
                gate_num,
                datafile = None,
                data = None,
                seed=127,
                file_address='',
                meas_prior = None,
                gate_prior = None,
                prep_state = '0'):
    """
      The main function that do all Bayesian inferrence part

    Parameters
    ----------
    d : array
        array of data (Observed QoI). Here, it is array of prob. of meas. 0.
    interested_qubit : int
        The index of qubit that we are looking at. 
        For naming the figure file only.
    M : int
        Number of priors required.
    params : dict
        A dictionary records backend properties. Must have
        {'pm1p0': float # Pr(Meas. 1| Prep. 0)
         'pm0p1': float # Pr(Meas. 0| Prep. 1)
         }
    gate_sd : float
        standard deviation for truncated normal distribution when generating 
        prior gate error parameters.
    meas_sd : float
        same as gate_sd but for meausrement error parameters.
    gate_type : String
        Chosen between "X", "Y", and "Z". 
        Should be the same as gate_type in Gateexp().
    gate_num : int
        number of gates in the experiment circuit.
        Should be the same as nGates in Gateexp().
    seed : int, optional
        Seed for random numbers. The default is 127.
    file_address: string, optional
        The relative file address to save data file. 
        Ends with '/' if not empty
        The default is ''.
    write_data_for_SB : boolean, optional
        If write data to execute standard Bayesian.
        This parameter is only for the purpose of writing paper.
        Just IGNORE IT.
        The default is False.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    prior_lambdas : numpy array
        prior lambdas in the form of a-by-b matrix where 
        a is the number of priors and m is the number of model parameters
    post_lambdas : numpy array
        prior lambdas in the form of a-by-b matrix where 
        a is the number of posterior and m is the number of model parameters

    """
    np.random.seed(seed)
    # Algorithm 1 of https://doi.org/10.1137/16M1087229

    average_lambdas = np.array([
        1 - params[interested_qubit].get('pm1p0', 0.05),
        1 - params[interested_qubit].get('pm0p1', 0.05)
    ])

    if datafile is not None:
        data_readout(datafile=datafile)
    elif data is not None:
        data_readout(data=data)
    else:
        raise Exception("Must supply priors through either datafile as a directory string referring to a AWS Braket calibration .json or a nx3 numpy array")

    # if gate_type == 'X':
    #     average_lambdas = np.append(average_lambdas,
    #                                 2 * params[interested_qubit]['u3_error'])
    #     if gate_num % 2:
    #         ideal_p0 = 0
    #     else:
    #         ideal_p0 = 1
    # elif gate_type == 'Y':
    #     average_lambdas = np.append(average_lambdas,
    #                                 2 * params[interested_qubit]['u3_error'])
    #     if gate_num % 2:
    #         ideal_p0 = 0
    #     else:
    #         ideal_p0 = 1
    # elif gate_type == 'Z':
    #     average_lambdas = np.append(average_lambdas,
    #                                 2 * params[interested_qubit]['u1_error'])
    #     ideal_p0 = 0
    # elif gate_type == 'CZ':

    # else:
    #     raise Exception('Only accept X gate now')

    # write data for standard bayesian inference
    if write_data_for_SB:
        with open(file_address + 'Qubit{}.csv'.format(interested_qubit),
                  mode='w',
                  newline='') as sgr:
            read_writer = csv.writer(sgr,
                                     delimiter=',',
                                     quotechar='"',
                                     quoting=csv.QUOTE_MINIMAL)
            read_writer.writerow(['x', 'y'])
            for i in range(len(d)):
                read_writer.writerow([ideal_p0, d[i]])

    np.random.seed(seed)
    num_lambdas = 3
    # Get distribution of data (Gaussian KDE)
    d_ker = ss.gaussian_kde(d)  # i.e., pi_D^{obs}(q), q = Q(lambda)

    if average_lambdas[0] == 1 or average_lambdas[0] < 0.7:
        average_lambdas[0] = 0.9
    if average_lambdas[1] == 1 or average_lambdas[1] < 0.7:
        average_lambdas[1] = 0.9

    # Sample prior lambdas, assume prior distribution is Normal distribution with mean as the given probality from IBM
    # Absolute value is used here to avoid negative values, so it is little twisted, may consider Gamma Distribution
    prior_lambdas = np.zeros(M * num_lambdas).reshape((M, num_lambdas))

    for i in range(M):
        one_sample = np.zeros(num_lambdas)
        for j in range(num_lambdas):
            if j < 2:
                one_sample[j] = tnorm01(average_lambdas[j], meas_sd)
                # while one_sample[j]<= 0 or one_sample[j] > 1:
                #     one_sample[j] = np.random.normal(average_lambdas[j],meas_sd,1)
            else:
                one_sample[j] = tnorm01(average_lambdas[j], gate_sd)
                # while one_sample[j]<= 0 or one_sample[j] > 1:
                #     one_sample[j] = np.random.normal(average_lambdas[j],gate_sd,1)
        prior_lambdas[i] = one_sample

    # Produce prior QoI
    qs = QoI_gate(prior_lambdas, ideal_p0, gate_num)
    #print(qs)
    qs_ker = ss.gaussian_kde(qs)  # i.e., pi_D^{Q(prior)}(q), q = Q(lambda)

    # Plot and Print
    print('Given Lambdas', average_lambdas)

    # Algorithm 2 of https://doi.org/10.1137/16M1087229

    # Find the max ratio r(Q(lambda)) over all lambdas
#    max_r, max_ind = findM(qs_ker, d_ker, qs)
#    # Print and Check
#    print('Final Accepted Posterior Lambdas')
#    print('M: %.6g Index: %.d pi_obs = %.6g pi_Q(prior) = %.6g' %
#          (max_r, max_ind, d_ker(qs[max_ind]), qs_ker(qs[max_ind])))
    
    max_r, max_q = findM(qs_ker, d_ker)
    # Print and Check
    print('Final Accepted Posterior Lambdas')
    print('M: %.6g Maximizer: %.6g pi_obs = %.6g pi_Q(prior) = %.6g' %
          (max_r, max_q, d_ker(max_q), qs_ker(max_q)))

    post_lambdas = np.array([])
    # Go to Rejection Iteration
    for p in range(M):
        r = d_ker(qs[p]) / qs_ker(qs[p])
        eta = r / max_r
        if eta > np.random.uniform(0, 1, 1):
            post_lambdas = np.append(post_lambdas, prior_lambdas[p])

    post_lambdas = post_lambdas.reshape(
        int(post_lambdas.size / num_lambdas),
        num_lambdas)  # Reshape since append destory subarrays
    post_qs = QoI_gate(post_lambdas, ideal_p0, gate_num)
    post_ker = ss.gaussian_kde(post_qs)

    xs = np.linspace(0, 1, 1000)
    xsd = np.linspace(0.1, 0.9, 1000)

    I = 0
    for i in range(xs.size - 1):
        q = xs[i]
        if qs_ker(q) > 0:
            r = d_ker(q) / qs_ker(q)
            I += r * qs_ker.pdf(q) * (xs[i + 1] - xs[i])  # Just Riemann Sum

    print('Accepted Number N: %.d, fraction %.3f' %
          (post_lambdas.shape[0], post_lambdas.shape[0] / M))
    print('I(pi^post_Lambda) = %.5g' % (I))  # Need to close to 1
    print('Posterior Lambda Mean', closest_average(post_lambdas))
    print('Posterior Lambda Mode', closest_mode(post_lambdas))

    print('0 to 1: KL-Div(pi_D^Q(post),pi_D^obs) = %6g' %
          (ss.entropy(post_ker(xs), d_ker(xs))))
    print('0 to 1: KL-Div(pi_D^obs,pi_D^Q(post)) = %6g' %
          (ss.entropy(d_ker(xs), post_ker(xs))))
    
    # File name change from Post_Qubit{} to Gate_Post_Qubit{}
    with open(file_address + 'Gate_Post_Qubit{}.csv'.format(interested_qubit),
              mode='w',
              newline='') as sgm:
        writer = csv.writer(sgm,
                            delimiter=',',
                            quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        for i in range(post_lambdas.shape[0]):
            writer.writerow(post_lambdas[i])

    # Plots
    plt.figure(figsize=(width,height), dpi=120, facecolor='white')
    plt.plot(xsd,
             d_ker(xsd),
             color='Red',
             linestyle='dashed',
             linewidth=3,
             label=r'$\pi^{\mathrm{obs}}_{\mathcal{D}}$')
    plt.plot(xsd, post_ker(xsd), color='Blue', label=r'$\pi_{\mathcal{D}}^{Q(\mathrm{post})}$')
    plt.xlabel('Pr(Meas. 0)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_address + 'QoI-Qubit%g.pdf' % interested_qubit)
    plt.show()
    
    plt.figure(figsize=(width,height), dpi=100, facecolor='white')
    # figure(num=None, figsize=fig_size, dpi=fig_dpi, facecolor='w', edgecolor='k')
    eps_ker = ss.gaussian_kde(post_lambdas[:, 2])
    eps_line = np.linspace(
        np.min(post_lambdas, axis=0)[2],  
        np.max(post_lambdas, axis=0)[2], 1000) 
    plt.plot(eps_line*0.5, eps_ker(eps_line), color='Blue') # WARNING: After 2021 Summer, 0.5*eps_in_code = eps_in_paper
    plt.ticklabel_format(axis="x", style="sci", scilimits=(-5, 1))
    plt.xlabel(r'$\epsilon_g$')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig(file_address + 'Eps-Qubit%g.pdf' % interested_qubit)
    plt.show()
    return prior_lambdas, post_lambdas


def read_data(interested_qubit, gate_type, gate_num, file_address=''):
    """Read out bit string data from csv file generated by collect_filter_data().

    Args:
      file_address:
        file address, string ends with '/' if not empty

    Returns:
      An array of bit strings.
    """
    with open(file_address + 'Readout_{}{}Q{}.csv'.format(
            gate_num, gate_type, interested_qubit),
              mode='r') as measfile:
        reader = csv.reader(measfile)
        cali01 = np.asarray([row for row in reader][0])

    return cali01

    # File name change from Post_Qubit{} to Gate_Post_Qubit{}
def read_post(ideal_p0, itr,
              shots,
              interested_qubits,
              gate_type,
              gate_num,
              file_address='', interested_qubit=0):
    """
        Read posteror from files
        See output_gate() for explaintion for arguments and returns.
    """
    post = {}
    for q in interested_qubits:
        data = read_data(q, gate_type, gate_num, file_address=file_address)
        d = getData0(data, int(itr * shots / 1024), q)
        post_lambdas = pd.read_csv(file_address +
                                   'Gtae_Post_Qubit{}.csv'.format(q),
                                   header=None).to_numpy()
        post['Qubit' + str(q)] = post_lambdas

        # information part
        xs = np.linspace(0, 1, 1000)
        xsd = np.linspace(0.2, 0.8, 1000)
        d_ker = ss.gaussian_kde(d)
        print('Posterior Lambda Mean', closest_average(post_lambdas))
        print('Posterior Lambda Mode', closest_mode(post_lambdas))
        print('0 to 1: KL-Div(pi_D^Q(post),pi_D^obs) = %6g' %
              (ss.entropy(post_ker(xs), d_ker(xs))))
        print('0 to 1: KL-Div(pi_D^obs,pi_D^Q(post)) = %6g' %
              (ss.entropy(d_ker(xs), post_ker(xs))))
        
        # figure(num=None, figsize=fig_size, dpi=fig_dpi, facecolor='w', edgecolor='k')
        plt.figure(figsize=(width,height), dpi=100, facecolor='white')
        post_qs = QoI_gate(post_lambdas, ideal_p0, gate_num)
        post_ker = ss.gaussian_kde(post_qs)
        plt.plot(xsd,
                 d_ker(xsd),
                 color='Red',
                 linestyle='dashed',
                 linewidth=3,
                 label='Observed QoI')
        plt.plot(xsd, post_ker(xsd), color='Blue', label='QoI by Posterior')
        plt.xlabel('Pr(Meas. 0)')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(file_address + 'QoI-Qubit%g.pdf' % interested_qubit)
        plt.show()
        
        # figure(num=None, figsize=fig_size, dpi=fig_dpi, facecolor='w', edgecolor='k')
        plt.figure(figsize=(width,height), dpi=120, facecolor='white')
        eps_ker = ss.gaussian_kde(post_lambdas[:, 2])
        eps_line = np.linspace(
            np.min(post_lambdas, axis=0)[2],
            np.max(post_lambdas, axis=0)[2], 1000)
        plt.plot(eps_line, eps_ker(eps_line), color='Blue')
        plt.ticklabel_format(axis="x", style="sci", scilimits=(-5, 1))
        plt.xlabel(r'$\epsilon_g$')
        plt.ylabel('Density')
        plt.tight_layout()
        plt.savefig(file_address + 'Eps-Qubit%g.pdf' % interested_qubit)
        plt.show()

    return post


def plotComparsion(data, post_lambdas, q, file_address=''):
    """
        Plot comparision between BJW bayesian and standard bayesian.
        For writing paper only.

    """
    postSB = pd.read_csv(file_address +
                         'StandPostQubit{}.csv'.format(q)).to_numpy()
    SB = QoI_gate(postSB, 1, 200)
    OB = QoI_gate(post_lambdas, 1, 200)
    minSB = min(SB) 
    minOB = min(OB)
    maxSB = max(SB)
    maxOB = max(OB)
    line = np.linspace(min(minSB, minOB), max(maxSB, maxOB), 1000)
    SBker = ss.gaussian_kde(SB)
    OBker = ss.gaussian_kde(OB)
    dker = ss.gaussian_kde(data)
    
    # figure(num=None, figsize=fig_size, dpi=fig_dpi, facecolor='w', edgecolor='k')
    plt.figure(figsize=(width,height), dpi=120, facecolor='white')
    plt.plot(line, SBker(line), color='Green', linewidth=2, label='Stand.')
    plt.plot(line, OBker(line), color='Blue', linewidth=2, label='Cons.')
    plt.plot(line,
             dker(line),
             color='Red',
             linestyle='dashed',
             linewidth=4,
             label='Data')
    plt.xlabel('Pr(Meas. 0)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_address + 'SBOB-Qubit%g.pdf' % q)
    plt.show()


#################### For Error Filtering #########################
def sxpower(s, x):
    total = 0
    length = len(s)
    for i in range(length):
        si = int(s[i])
        xi = int(x[i])
        total += si * xi
    return total


def count1(s):
    count = 0
    for i in range(len(s)):
        if s[i] == '1':
            count += 1
    return count


def gate_matrix(length, eps, m):
    """
        Generate matrix for denosing gate error

    """
    size = 2**length
    mat = np.empty([size, size], dtype=np.float64)
    for row in range(size):
        for col in range(size):
            x = ("{0:0" + str(length) + "b}").format(row)
            s = ("{0:0" + str(length) + "b}").format(col)
            power = sxpower(s, x)
            mat[row, col] = ((-1)**power) * ((1 - eps)**(count1(s) * m))
    return mat


def find_least_norm_gate(ptilde):
    """
        Only for one-qubit. Similar to find_least_norm() in measfilter.py.

    """
    # Formulation
    Q = 2 * matrix(np.identity(2))
    p = -2 * matrix(ptilde)

    G = matrix(np.array([[0, 1], [0, -1]]), (2, 2), 'd')
    h = 0.5 * matrix(np.ones(2))

    A = matrix(np.array([1, 0]), (1, 2), 'd')
    b = matrix(0.5)

    solvers.options['show_progress'] = False
    sol = solvers.qp(Q, p, G, h, A, b)
    return sol['status'], sol['x']


def gate_denoise(m, p0s, lambdas):
    """
        Complete function for filter gate and measurement errors.

    """
    denoised = []
    meas_err_mat = errMitMat([lambdas[0], lambdas[1]])
    M = gate_matrix(1, lambdas[2], m)
    for p0 in p0s:
        ptilde = np.array([p0, 1 - p0])
        gate_ptilde = np.linalg.solve(meas_err_mat, ptilde)
        phat = np.linalg.solve(M, gate_ptilde)
        status, opt_phat = find_least_norm_gate(phat)
        opt_recovered_p0 = opt_phat[0] + opt_phat[1] * (-1)**(
            1 * 0)  # phat(0) + phat(1)
        opt_recovered_p1 = opt_phat[0] + opt_phat[1] * (-1)**(
            1 * 1)  # phat(0) - phat(1)
        denoised.append(opt_recovered_p0)

    return denoised


########################## Class for Error Filtering ########################
class GMFilter:
    """ Gate and Measurement error filter.
        WARNING: After 2021 Summer, 0.5*eps_in_code = eps_in_paper
        TODO: Change Code!!!!
    Attributes:
        interested_qubits: array,
          qubit indices that experiments are applied on.
        gate_type : String
            Chosen between "X", "Y", and "Z". 
            Should be the same as gate_type in Gateexp().
        gate_num : int
            number of gates in the experiment circuit.
            Should be the same as nGates in Gateexp().
        device_param_address: string
          the address for saving Params.csv. 
          End with '/' if not empty.
        data_file_address: string
          the address for saving 
          Readout_{gate_num}{gate_type}Q{qubit_index}.csv. 
          End with '/' if not empty.
        post: dict
          posterior of each qubit. {'Qubit0':[...], 'Qubit1': [...], ...}
        params: dict
          backend properties. Not None after execute inference()
        modes: numpy array
          posterior MAP. {'Qubit0':[...], 'Qubit1': [...], ...}
          Not None after execute inference()
        means: numpy array
          posterior mean. {'Qubit0':[...], 'Qubit1': [...], ...}
          Not None after execute inference()
          
    """
    def __init__(self,
                 interested_qubits,
                 gate_num,
                 gate_type,
                 device_param_address='',
                 data_file_address=''):
        self.interested_qubits = interested_qubits
        self.device_param_address = device_param_address
        self.data_file_address = data_file_address
        self.gate_type = gate_type
        self.gate_num = gate_num
        self.params = None
        self.post = {}
        self.modes = None
        self.means = None

    def mean(self):
        res = {}
        for q in self.interested_qubits:
            res['Qubit' + str(q)] = closest_average(self.post['Qubit' +
                                                              str(q)])
        return res

    def mode(self):
        res = {}
        for q in self.interested_qubits:
            res['Qubit' + str(q)] = closest_mode(self.post['Qubit' + str(q)])
        return res

    def inference(self,
                  nPrior=40000,
                  meas_sd=0.1,
                  gate_sd=0.01,
                  seed=127,
                  shots_per_point=1024,
                  write_data_for_SB=False):
        """
          Same as output_gate().

        Parameters
        ----------
        nPrior : int, optional
            Number of priors required. The default is 40000.
        meas_sd : float, optional
            standard deviation for truncated normal distribution 
            when generating prior measurment error parameters. 
            The default is 0.1.
        gate_sd : TYPE, optional
            standard deviation for truncated normal distribution 
            when generating prior gate error parameters. 
            The default is 0.01.
        seed : int, optional
            Seed for random numbers. The default is 127.
        shots_per_point : int, optional
            how many shots you want to estimate one QoI (prob. of meas. 0).
            Used to control number of data points and accuracy.
            The default is 1024.
        write_data_for_SB : boolean, optional
            If write data to execute standard Bayesian.
            This parameter is only for the purpose of writing paper.
            Just IGNORE IT.
            The default is False.

        Returns
        -------
        None.

        """
        self.params = read_params(self.device_param_address)

        itr = self.params[0]['itr']
        shots = self.params[0]['shots']
        info = {}
        for i in self.interested_qubits:
            print('Qubit %d' % (i))
            data = read_data(i,
                             self.gate_type,
                             self.gate_num,
                             file_address=self.data_file_address)
            d = getData0(data, int(itr * shots / shots_per_point), i)
            _, post_lambdas = output_gate(d,
                                          i,
                                          nPrior,
                                          self.params,
                                          gate_sd,
                                          meas_sd,
                                          self.gate_type,
                                          self.gate_num,
                                          file_address=self.data_file_address)
            self.post['Qubit' + str(i)] = post_lambdas
        self.modes = self.mode()
        self.means = self.mean()
        
        # File name change from Post_Qubit{} to Gate_Post_Qubit{}
    def post_from_file(self):
        """
          Read posterior from file directly if inference() is already run once.

        Returns
        -------
        None.

        """
        for i in self.interested_qubits:
            post_lambdas = pd.read_csv(self.data_file_address +
                                       'Gate_Post_Qubit{}.csv'.format(i),
                                       header=None).to_numpy()
            self.post['Qubit' + str(i)] = post_lambdas
        self.modes = self.mode()
        self.means = self.mean()

    def filter_mean(self, p0s, qubit_index):
        """
          Use posteror mean to filter measurement and gate error out.

        Parameters
        ----------
        p0s : array
            An array of probabilities of measuring 0.
        qubit_index : int
            which qubit that p0s is corresponds to.

        Returns
        -------
        denoised_p0s: array
            p0s w/o gate and measurment error.

        """
        return gate_denoise(self.gate_num, p0s,
                            self.means['Qubit' + str(qubit_index)])

    def filter_mode(self, p0s, qubit_index):
        """
          Use posteror MAP to filter measurement and gate error out.

        Parameters
        ----------
        p0s : array
            An array of probabilities of measuring 0.
        qubit_index : int
            which qubit that p0s is corresponds to.

        Returns
        -------
        denoised_p0s: array
            p0s w/o gate and measurment error.

        """
        return gate_denoise(self.gate_num, p0s,
                            self.modes['Qubit' + str(qubit_index)])
