# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 18:19:38 2020

@author: Muqing Zheng
"""
# print("Formal Installation")
# Qiskit
from qiskit import QuantumCircuit, transpile

# Numerical/Stats pack
import csv
import pandas as pd
import numpy as np
import scipy.stats as ss
import numpy.linalg as nl
from math import pi
# For optimization
from cvxopt import matrix, solvers
from scipy.optimize import minimize_scalar,minimize
# For plot
from qiskit import QuantumCircuit, transpile
from braket.aws import AwsDevice

import matplotlib.pyplot as plt
width = 6.72 # plot width
height = 4.15 # plot height



def get_braket_calibration_dict(device_arn, n_qubits=None):
    """
    Returns the params list in the exact format required by MeasFilter.
    """
    device = AwsDevice(device_arn)
    properties = device.properties
    
    # Try to find qubit count if not provided
    if n_qubits is None:
        try:
            n_qubits = properties.paradigm.qubitCount
        except:
            n_qubits = 5 # Manual override if needed

    formatted_params = []

    for q in range(n_qubits):
        # 1. Default conservative error (5% error)
        p_meas0_prep1 = 0.05 
        p_meas1_prep0 = 0.05
        
        # 2. Try to extract real Fidelity from Braket Properties
        # This structure depends on the provider (Rigetti, OQC, etc.)
        try:
            # Example for Rigetti (Standardized in Braket properties)
            # We look for "fRO" (Readout Fidelity)
            provider_specs = properties.provider.specs
            qubit_specs = provider_specs.get(f"{q}Q", {}).get(f"{q}", {})
            fRO = qubit_specs.get('fRO', None)
            
            if fRO:
                error = 1.0 - fRO
                # Assume symmetric error if specific p(0|1) isn't detailed
                p_meas0_prep1 = error
                p_meas1_prep0 = error
        except:
            # If extraction fails, we stick to the 0.05 default
            pass

        # 3. Create the dictionary for this qubit
        qubit_cal = {
            'qubit': q,
            'pm0p1': p_meas0_prep1, # PROB MEASURING 0 GIVEN PREP 1
            'pm1p0': p_meas1_prep0, # PROB MEASURING 1 GIVEN PREP 0
            # Extra fields required by the code logic, even if unused
            'itr': 32,
            'shots': 8192
        }
        
        formatted_params.append(qubit_cal)

    return np.array(formatted_params)


def param_record(backend, itr=32, shots=8192, if_write=True, file_address=''):
    """Write backend property into an array of dict 
       and save as csv if permissible.

    Args:
      backend: Backend
        A Qiskit backend instance.
      itr: int
        number of iterations of job submission.
      shots: int
        number of shots per each job submission.
      if_write: boolean
        True if save the properties as a csv file.
      file_address: string
        The relative file address to save backend properties. 
        Ends with '/' if not empty
        The default is ''.

    Returns: numpy array
      An array of dicts. Each dict records all characterization of one qubit.
    """
    allParam = np.array([])
    
    try:
        # Check if backend supports properties interface
        if not hasattr(backend, 'properties') or backend.properties() is None:
            raise Exception("Backend properties not available.")

        prop_dict = backend.properties().to_dict()
        nQubits = len(prop_dict['qubits'])
        
        # Helper to safely extract values from property lists
        def get_val(props, name):
            for item in props:
                if item.get('name') == name:
                    return item.get('value')
            return 0.0 # Default if not found

        target_qubits = range(nQubits)
        
        for target_qubit in target_qubits:
            qubit_props = prop_dict['qubits'][target_qubit]
            
            # Basic params
            params = {
                'qubit': target_qubit,
                'update_date': prop_dict.get('last_update_date', 'N/A'),
                'T1': get_val(qubit_props, 'T1'),
                'T2': get_val(qubit_props, 'T2'),
                'freq': get_val(qubit_props, 'frequency'),
                'readout_err': get_val(qubit_props, 'readout_error'),
                'pm0p1': get_val(qubit_props, 'prob_meas0_prep1'),
                'pm1p0': get_val(qubit_props, 'prob_meas1_prep0'),
                'itr': itr,
                'shots': shots,
            }

            # Try to fetch gate errors if available in legacy format
            # Many non-IBM backends won't have this specific structure, so we wrap in try
            try:
                gates = prop_dict.get('gates', [])
                # Simplified extraction logic or defaults
                # This part is highly specific to IBM's old map; defaulting if fails
                params['id_error'] = 0.001
                params['u3_error'] = 0.001 
                # (Real extraction logic omitted for brevity/compatibility)
            except:
                pass

            allParam = np.append(allParam, params)

        if if_write:
            with open(file_address + 'Params.csv', mode='w', newline='') as sgm:
                param_writer = csv.writer(sgm,
                                          delimiter=',',
                                          quotechar='"',
                                          quoting=csv.QUOTE_MINIMAL)
                for pa in allParam:
                    for key, val in pa.items():
                        param_writer.writerow([key, val])
                    param_writer.writerow(['End'])

    except Exception as e:
        print(f"Note: Backend parameters could not be recorded ({str(e)}). Using defaults for inference.")
        # Return empty; inference engine handles empty params by setting defaults
        return np.array([])

    return allParam


def meas_circ(nQubits, backend=None, itr=32):
    """
    Generates 2 * itr circuits compatible with Qiskit 2.x and Braket.
    - First half prepares |0>
    - Second half prepares |1>

    Args:
        nQubits: int
            number of qubits.
        backend: Backend
            A Qiskit backend instance. Used for transpilation.
        itr: int
            number of iterations of each state preparation.

    Returns: list of form [{QuantumCircuit preparing |0>}, {QuantumCircuit preparing |1>}]
    """
    circs = []
    
    # Create |0> circuits (Identity)
    # Note: Explicit 'id' gates often optimized away, so we use barriers 
    # or simple measurements to define the state.
    c0 = QuantumCircuit(nQubits, nQubits)

    # No gates needed for |0>, just measure. 
    # We add a barrier to prevent transpiler from merging if we added ops later.
    c0.barrier() 
    c0.measure(range(nQubits), range(nQubits))
    
    # Create |1> circuits (X gate)
    c1 = QuantumCircuit(nQubits, nQubits)
    c1.x(range(nQubits)) # Broadcast X to all qubits
    c1.barrier()
    c1.measure(range(nQubits), range(nQubits))

    # Transpile if backend provided (Optional but recommended for ISA)
    if backend:
        # In Qiskit 2.x, it's best to transpile once before copying
        c0 = transpile(c0, backend)
        c1 = transpile(c1, backend)

    # Create the batch
    # We use metadata or naming to track them
    ## Perhaps change to one request with many shots?
    for i in range(itr):
        # Create copies
        c0_copy = c0.copy()
        c0_copy.name = f"cal_0_itr{i}"
        circs.append(c0_copy)
        
    for i in range(itr):
        c1_copy = c1.copy()
        c1_copy.name = f"cal_1_itr{i}"
        circs.append(c1_copy)
        
    return circs

def collect_filter_data(backend,
                        itr=32,
                        shots=8192,
                        if_write=True,
                        file_address='',
                        job_id=''):
    """
    Collects measurement data compatible with Amazon Braket Qiskit Provider.
    """
    # 1. Determine qubit count safely
    try:
        # Braket backends usually have 'num_qubits' attribute
        nQubits = backend.num_qubits
    except:
        nQubits = 5 # Fallback

    readout_m0 = np.array([])
    
    # 2. Generate the circuits (One circuit repeated 'itr' times with unique names)
    circs = meas_circ(nQubits, backend, itr=itr)

    # 3. Job Execution or Retrieval
    if job_id:
        print(f"Retrieving existing Braket Job ID: {job_id}")
        try:
            # Braket Provider usually allows retrieving via the backend or service
            job_m0 = backend.retrieve_job(job_id)
        except Exception as e:
            print(f"Failed to retrieve job: {e}")
            return np.array([])
    else:
        print(f"Submitting new batch job with {itr} circuits to {backend.name}...")
        try:
            # Execute all circuits in one batch
            # 'memory=True' is CRITICAL to get the bitstrings required for this inference
            job_m0 = backend.run(circs, shots=shots, memory=True)
            print(f"Job submitted. ID: {job_m0.job_id()}")
            
            # Optional: Explicit wait loop if the provider doesn't block automatically
            # while not job_m0.in_final_state():
            #     print("Status:", job_m0.status())
            #     time.sleep(5)
                
        except Exception as e:
            print(f"Job submission failed: {e}")
            return np.array([])

    # 4. Extract Results
    try:
        # This blocks until the job is done
        m0_res = job_m0.result()
        
        print("Job complete. extracting memory...")

        # Loop through the experiments (one for each circuit in the batch)
        for i in range(itr):
            # get_memory(i) retrieves the list of bitstrings for the i-th circuit
            # e.g., ['000', '001', '000', ...]
            memory_data = m0_res.get_memory(i)
            readout_m0 = np.append(readout_m0, memory_data)

        # 5. Save to CSV (Original script logic)
        if if_write:
            filename = file_address + 'Filter_data.csv'
            print(f"Saving data to {filename}...")
            # Using 'w' mode with standard csv writer
            with open(filename, mode='w', newline='') as sgr:
                read_writer = csv.writer(sgr, quoting=csv.QUOTE_MINIMAL)
                # The original script writes the entire flattened array as one row
                read_writer.writerow(readout_m0)
                
    except Exception as e:
        print(f"Error processing results: {e}")
        # If accessing memory fails, print available keys to help debug
        try:
            print("Result keys available:", m0_res.get_counts())
        except:
            pass

    return readout_m0
def read_params(file_address=''):
    """Read out backend properties from csv file generated by param_record().

    Args:
      file_address: string
        The relative file address to read backend properties. 
        Ends with '/' if not empty
        The default is ''.

    Returns: numpy array
      An array of dicts. Each dict records all characterization of one qubit.
    """
    textKeys = ['name', 'update_date', 'qubit']
    intKeys = ['itr', 'shots']
    # Read Parameters
    with open(file_address + 'Params.csv', mode='r') as sgm:
        reader = csv.reader(sgm)
        params = np.array([])
        singleQubit = {}
        first = True
        for row in reader:
            if row[0] == 'End':
                params = np.append(params, singleQubit)
                singleQubit = {}
            else:
                singleQubit[row[0]] = row[1]

    # Convert corresponding terms into floats or ints
    for qubit in params:
        for key in qubit.keys():
            if key not in textKeys:
                qubit[key] = float(qubit[key])
            if key in intKeys:
                qubit[key] = int(qubit[key])

    return params


def read_filter_data(file_address=''):
    """Read out bit string data from csv file generated by collect_filter_data().

    Args:
      file_address: string
        The relative file address to read data for filter generation. 
        Ends with '/' if not empty
        The default is ''.

    Returns: numpy array
      An array of bit strings.
    """
    # Should be able to use np.genfromtxt
    cali01 = np.genfromtxt(file_address + 'Filter_data.csv', delimiter=',',dtype=str)
    if len(cali01) < 2:
        with open(file_address + 'Filter_data.csv', mode='r') as measfile:
            reader = csv.reader(measfile)
            cali01 = np.asarray([row for row in reader][0])
    return cali01


def tnorm01(center, sd, size=1):
    """ Generate random numbers for truncated normal with range [0,1]

    Args:
      center: float
        mean of normal distribution
      sd: float
        standard deviation of normal distribution
      size: int
        number of random numbers

    Returns: array
       an array of random numbers
    """
    upper = 1
    lower = 0
    a, b = (lower - center) / sd, (upper - center) / sd
    return ss.truncnorm.rvs(a, b, size=size) * sd + center


def find_mode(data):
    """Find the mode through Gaussian KDE.

    Args:
      data: array
        an array of floats

    Returns: float
      the mode.
    """
    kde = ss.gaussian_kde(data)
    line = np.linspace(min(data), max(data), 10000)
    return line[np.argmax(kde(line))]


def closest_mode(post_lambdas):
    """Find the tuple of model parameters that closed to 
       the Maximum A Posteriori (MAP) of 
       posterior distribution of each parameter

    Args:
      post_lambdas: numpy array
        an n-by-m array where n is the number of posteriors and m is number 
        of parameters in the model

    Returns: numpy array
      an array that contains the required model parameters.
    """

    mode_lam = []
    for j in range(post_lambdas.shape[1]):
        mode_lam.append(find_mode(post_lambdas[:, j]))

    sol = np.array([])
    smallest_norm = nl.norm(post_lambdas[0])
    mode_lam = np.array(mode_lam)
    for lam in post_lambdas:
        norm_diff = nl.norm(lam - mode_lam)
        if norm_diff < smallest_norm:
            smallest_norm = norm_diff
            sol = lam
    return sol


def closest_average(post_lambdas):
    """Find the tuple of model parameters that closed to 
       the mean of posterior distribution of each parameter

    Args:
      post_lambdas: numpy array
        an n-by-m array where n is the number of posteriors and m is number 
        of parameters in the model

    Returns: numpy array
      an array that contains the required model parameters.
    """
    sol = np.array([])
    smallest_norm = nl.norm(post_lambdas[0])

    ave_lam = np.mean(post_lambdas, axis=0)

    for lam in post_lambdas:
        norm_diff = nl.norm(lam - ave_lam)

        if norm_diff < smallest_norm:
            smallest_norm = norm_diff
            sol = lam
    return sol


def dictToVec(nQubits, counts):
    """ Transfer counts to vec

    Args:
      nQUbits: int
        number of qubits
      counts: dict
        an dictionary in the form {basis string: frequency}. E.g.
        {"01": 100
         "11": 100}
        dict key follow little-endian convention

    Returns: numpy array
      an probability vector (array). E.g.
      [0, 100, 0, 100] is the result from example above.
    """
    vec = np.zeros(2**nQubits)
    form = "{0:0" + str(nQubits) + "b}"
    for i in range(2**nQubits):
        key = form.format(i) # consider key = format(i,'0{}b'.format(nQubits))
                             # and delete variable form
        if key in counts.keys():
            vec[i] = int(counts[key])
        else:
            vec[i] = 0
    return vec


def dictToVec_inv(nQubits, counts):
    """ 
      Same as dictToVec() but key uses big-endian convention
    """
    vec = np.zeros(2**nQubits)
    form = "{0:0" + str(nQubits) + "b}"
    for i in range(2**nQubits):
        key = form.format(i)[::-1]
        if key in counts.keys():
            vec[i] = int(counts[key])
        else:
            vec[i] = 0
    return vec


def vecToDict(nQubits, shots, vec):
    """ Transfer probability vector to dict in the form 
        {basis string: frequency}. E.g. [0, 0.5, 0, 0.5] in 200 shots becomes
            {"01": 100
             "11": 100}
        dict key follow little-endian convention

    Parameters
    ----------
    nQubits : int
        number of qubits.
    shots : int
        number of shots.
    vec : array
        probability vector that sums to 1.

    Returns
    -------
    counts : dict
        Counts for each basis.

    """
    counts = {}
    form = "{0:0" + str(nQubits) + "b}"
    for i in range(2**nQubits):
        key = form.format(i)
        counts[key] = int(vec[i] * shots)
    return counts


def vecToDict_inv(nQubits, shots, vec):
    """ 
      Same as dictToVec() but key uses big-endian convention
    """
    counts = {}
    form = "{0:0" + str(nQubits) + "b}"
    for i in range(2**nQubits):
        key = form.format(i)[::-1]
        counts[key] = int(vec[i] * shots)
    return counts


# Functions
def getData0(data, num_group, interested_qubit):
    """ Get the probabilities of measuring 0 from binay readouts
        **Binary number follows little-endian convention**

    Parameters
    ----------
    data : array
        an array of binary readouts.
    num_group : int
        number of probabilities. E.g. If you have 1000 binary string readouts,
        you can set num_group = 10 so that you will have 10 probabilities,
        each probability is calculated from 100 binary string readouts
    interested_qubit : int
        which qubit you interested in.

    Returns
    -------
    prob0 : numpy array
        array of proabilitilies of measuring 0.

    """
    prob0 = np.zeros(num_group)
    groups = np.split(data, num_group)
    for i in range(num_group):
        count = 0
        for d in groups[i]:
            d_rev = d[::-1]
            try:
                if d_rev[interested_qubit] == '0':
                    count += 1
            except: # usually because only the measurement of the interest qubit is returned
                if d_rev[0] == '0':
                    count += 1

        prob0[i] = count / groups[i].size
    return prob0


def errMitMat(lambdas_sample):
    """
    Compute the matrix A from
    Ax = b, where A is the error mitigation matrix (transition matrix),
    x is the number appearence of a basis in theory
    b is the number appearence of a basis in practice with noise

    Parameters
    ----------
    lambdas_sample : numpy array
        first two entry must be 
        Pr(Measuring 0|Preparing 0) and Pr(Measuring 1|Preparing 1)

    Returns
    -------
    A : numpy array
        Transition matrix that applies classical measurement error.

    """
    #
    # Input; lambdas_sample - np.array, array of (1 - error rate) whose length is number of qubits
    # Output; A - np.ndarray, as described above
    pm0p0 = lambdas_sample[0]
    pm1p1 = lambdas_sample[1]
    # Initialize the matrix
    A = np.array([[pm0p0, 1 - pm1p1], [1 - pm0p0, pm1p1]])
    return (A)


def err2MitMat(lambdas_sample):
    """
    Compute the matrix A from
    Ax = b, where A is the error mitigation matrix (transition matrix),
    x is the number appearence of a basis in theory
    b is the number appearence of a basis in practice with noise

    Parameters
    ----------
    lambdas_sample : numpy array
        entries should be 
        Pr(Measuring first 0|Preparing first 0), Pr(Measuring first 1|Preparing first 1), Pr(Measuring second 0|Preparing second 0), Pr(Measuring second 1|Preparing second 1)

    Returns
    -------
    A : numpy array
        Transition matrix that applies classical measurement error.

    """
    #
    # Input; lambdas_sample - np.array, array of (1 - error rate) whose length is number of qubits
    # Output; A - np.ndarray, as described above
    pm0xp0x = lambdas_sample[0]
    pm1xp1x = lambdas_sample[1]
    pmx0px0 = lambdas_sample[2]
    pmx1px1 = lambdas_sample[3]
    
    # Initialize the matrix

    A = np.array( [ [pm0xp0x * pmx0px0, pm0xp0x * (1-pmx1px1),  (1-pm1xp1x) * pmx0px0,  (1-pm1xp1x) * (1-pmx1px1)],
                    [pm0xp0x * (1-pmx0px0), pm0xp0x * pmx1px1,  (1-pm1xp1x) * (1-pmx0px0),  (1-pm1xp1x) * pmx1px1],
                    [(1-pm0xp0x) * pmx0px0, (1-pm0xp0x) * (1-pmx1px1),  pm1xp1x * pmx0px0,  pm1xp1x * (1-pmx1px1)],
                    [(1-pm0xp0x) * (1-pmx0px0), (1-pm0xp0x) * pmx1px1,  pm1xp1x * (1-pmx0px0),  pm1xp1x * pmx1px1]])
                  
    return (A)

def QoI(prior_lambdas, prep_state='0'):
    """
    Function equivalent to Q(lambda) in https://doi.org/10.1137/16M1087229

    Parameters
    ----------
    prior_lambdas : numpy array
        each subarray is an individual prior lambda.

    prep_state : string, optional
        The state prepared to, 

    Returns
    -------
    qs : numpy array
        QoI's. Here they are the probability of measuring 0 with each given
        prior lambdas in prior_lambdas.

    """
    shape = prior_lambdas.shape
    nQubit = 1

    # Initialize the output array
    qs = np.array([])

    # Define Ideal Vector based on what we prepared
    if prep_state == '0':
        # [Prob(0), Prob(1)] -> [100%, 0%]
        M_ideal = np.array([1.0, 0.0]) 
    elif prep_state == '1':
        # [Prob(0), Prob(1)] -> [0%, 100%]
        M_ideal = np.array([0.0, 1.0])
    else: # Default to '+'
        print('Default to |+>, set prep_state = "0" or "1" for these preparations.')
        M_ideal = np.ones(2**nQubit) / 2**nQubit

    # Simulate measurement error
    for i in range(shape[0]):
        A = errMitMat(prior_lambdas[i])
        M_noisy = np.dot(A, M_ideal)

        # Only record interested qubits (Prob of measuring 0)
        qs = np.append(qs, M_noisy[0])
    return qs


def dq(x, qs_ker, d_ker):
    # if np.abs(qs_ker(x)[0])  > 0:
    #     if d_ker(x) == 0: # A lot of 0s in both sides may cause opt algorithm terminates
    #         # return np.abs(0.5-x)
    #         return np.infty
    #     else :
    #         return - d_ker(x)[0] / qs_ker(x)[0] 
    # else:
    #     return np.infty
    if np.abs(qs_ker(x)[0])  > 1e-6 and np.abs(d_ker(x)[0])  > 1e-6:
        return - d_ker(x)[0] / qs_ker(x)[0] 
    else:
        return np.inf




def findM(qs_ker, d_ker):
    """
    Updated findM to restrict optimization bounds to the effective support
    of the prior distribution, preventing edge explosions.
    """
    # 1. Scan the space to find where the prior actually exists
    x_scan = np.linspace(0, 1, 1000)
    prior_density = qs_ker(x_scan)
    
    # 2. Define Effective Support: 
    # Only consider regions where prior density is at least 1% of its peak.
    # This cuts off the unstable "tails" at 0 and 1.
    threshold = 0.01 * np.max(prior_density)
    valid_indices = np.where(prior_density > threshold)[0]
    
    if len(valid_indices) > 1:
        # Set bounds to the range where prior is significant
        lower_bound = x_scan[valid_indices[0]]
        upper_bound = x_scan[valid_indices[-1]]
        bds = (lower_bound, upper_bound)
    else:
        # Fallback if density is flat or error
        bds = (0.2, 0.8) 

    # --- Debugging Plot (Optional: Keep your plot to verify) ---
    ys = np.array([dq(x, qs_ker, d_ker) for x in x_scan])
    # plt.figure(figsize=(width,height), dpi=100, facecolor='white')
    # plt.plot(x_scan, ys)
    # plt.title("Objective Function with Edge Explosions")
    # plt.ylim(-50, 0) # Cap the view to see the middle trough
    # plt.show()
    # -----------------------------------------------------------

    # 3. Find a safe starting point (x0) inside the valid bounds
    # We look for the minimum of y only within our valid indices
    valid_ys = ys[valid_indices]
    valid_xs = x_scan[valid_indices]
    x0 = valid_xs[np.argmin(valid_ys)]

    # 4. Run Optimizer with safe bounds
    res = minimize(dq, (x0,), args=(qs_ker, d_ker), bounds=(bds,), method='L-BFGS-B')
    
    try:
        return -res.fun[0], res.x[0]
    except Exception:
        return -res.fun, res.x

def output(d,
           interested_qubit,
           M,
           params,
           prior_sd,
           seed=127,
           show_denoised=False,
           file_address='',
           prep_state='0'):
    """
      The main function that do all Bayesian inferrence part

    Parameters
    ----------
    d : array
        array of data (Observed QoI). Here, it is array of prob. of meas. 0.
    interested_qubit : int
        The index of qubit that we are looking at. 
        For the use of naming the figure file only.
    M : int
        Number of priors required.
    params : dict
        A dictionary records backend properties. Must have
        {'pm1p0': float # Pr(Meas. 1| Prep. 0)
         'pm0p1': float # Pr(Meas. 0| Prep. 1)
         }
    prior_sd : float
        standard deviation for truncated normal distribution when generating 
        prior parameters (for measurement error).
    seed : int, optional
        Seed for random numbers. The default is 127.
    show_denoised : boolean, optional
        If plot the comparision between post. parameter and 
        given parameters in params. The default is False since 
        it is very time consuming and unnecessary in most of case
    file_address : String, optional
        The relative file address to save posteriors and figures. 
        Ends with '/' if not empty
        The default is ''.

    Returns
    -------
    prior_lambdas : numpy array
        prior lambdas in the form of a-by-b matrix where 
        a is the number of priors and m is the number of model parameters
    post_lambdas : numpy array
        prior lambdas in the form of a-by-b matrix where 
        a is the number of posterior and m is the number of model parameters

    """
    # Algorithm 1 of https://doi.org/10.1137/16M1087229
    np.random.seed(seed)
    num_lambdas = 2
    # Get distribution of data (Gaussian KDE)
    d_ker = ss.gaussian_kde(d)  # i.e., pi_D^{obs}(q), q = Q(lambda)
    average_lambdas = np.array([
        1 - params[interested_qubit]['pm1p0'],
        1 - params[interested_qubit]['pm0p1']
    ])

    # Compute distribution of Pr(meas. 0) from Qiskit results
    given_errmat = errMitMat(average_lambdas)
    # qiskit_p0 = np.empty(len(d))
    # for i in range(len(d)):
    #     single_res = nl.solve(given_errmat, [d[i], 1 - d[i]])
    #     qiskit_p0[i] = single_res[0]
    # qiskit_ker = ss.gaussian_kde(qiskit_p0)

    if average_lambdas[0] == 1 or average_lambdas[0] < 0.7:
        average_lambdas[0] = 0.95
    if average_lambdas[1] == 1 or average_lambdas[1] < 0.7:
        average_lambdas[1] = 0.95

    # Sample prior lambdas, assume prior distribution is Normal distribution with mean as the given probality from IBM
    # Absolute value is used here to avoid negative values, so it is little twisted, may consider Gamma Distribution
    prior_lambdas = np.zeros(M * num_lambdas).reshape((M, num_lambdas))

    for i in range(M):
        one_sample = np.zeros(num_lambdas)
        for j in range(num_lambdas):
            one_sample[j] = tnorm01(average_lambdas[j], prior_sd)
            # while one_sample[j]<= 0 or one_sample[j] > 1:
            #     one_sample[j] = np.random.normal(average_lambdas[j],prior_sd,1)
        prior_lambdas[i] = one_sample

    # Produce prior QoI
    qs = QoI(prior_lambdas, prep_state=prep_state) 
    qs_ker = ss.gaussian_kde(qs) # i.e., pi_D^{Q(prior)}(q), q = Q(lambda)

    # Plot and Print
    print('Given Lambdas', average_lambdas)

    # Algorithm 2 of https://doi.org/10.1137/16M1087229

    # Find the max ratio r(Q(lambda)) over all lambdas

    max_r, max_q = findM(qs_ker, d_ker)
    # Print and Check
    print('Final Accepted Posterior Lambdas')
    print('M: %.6g Maximizer: %.6g pi_obs = %.6g pi_Q(prior) = %.6g' %
          (max_r, max_q, d_ker(max_q), qs_ker(max_q)))

    post_lambdas = np.array([])
    # Go to Rejection Iteration
    for p in range(M):
        # Monitor Progress
        print('Progress: {:.3%}'.format(p/M), end='\r')
        
        r = d_ker(qs[p]) / qs_ker(qs[p])
        eta = r / max_r
        if eta > np.random.uniform(0, 1, 1):
            post_lambdas = np.append(post_lambdas, prior_lambdas[p])
    print()
    
    
    post_lambdas = post_lambdas.reshape(
        int(post_lambdas.size / num_lambdas),
        num_lambdas)  # Reshape since append destory subarrays
    post_qs = QoI(post_lambdas, prep_state=prep_state)
    post_ker = ss.gaussian_kde(post_qs)

    xs = np.linspace(0, 1, 1000)
    xsd = np.linspace(0.3, 0.7, 1000)

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
    # print('0 to 1: KL-Div(qiskit,pi_D^obs) = %6g' %
    #       (ss.entropy(qiskit_ker(xs), d_ker(xs))))
    # print('0 to 1: KL-Div(pi_D^obs,qiskit) = %6g' %
    #       (ss.entropy(d_ker(xs), qiskit_ker(xs))))

    # print('Post and Data: Sum of Differences ',
    #       np.sum(np.abs(post_ker(xs) - d_ker(xs)) / 1000))
    # print('Qisk and Data: Sum of Differences ',
    #       np.sum(np.abs(qiskit_ker(xs) - d_ker(xs)) / 1000))

    with open(file_address + 'Post_Qubit{}.csv'.format(interested_qubit),
              mode='w',
              newline='') as sgm:
        writer = csv.writer(sgm,
                            delimiter=',',
                            quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        for i in range(post_lambdas.shape[0]):
            writer.writerow(post_lambdas[i])

    # Plots
    # fig = plot_setup()
    # figure(num=None, figsize=fig_size, dpi=fig_dpi, facecolor='w', edgecolor='k')
    plt.figure(figsize=(width,height), dpi=100, facecolor='white')
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

    if show_denoised:
        res_proc = np.array([])
        for lam in post_lambdas:
            M_sub = errMitMat(lam)
            res_proc = np.append(
                res_proc,
                np.array([
                    nl.solve(M_sub, [d[ind], 1 - d[ind]])[0]
                    for ind in range(len(d))
                ]))
        proc_ker = ss.gaussian_kde(res_proc)

        # Denoised by Qiskit Parameters
        M_qsub = errMitMat(
            np.array([
                1 - params[interested_qubit]['pm1p0'],
                1 - params[interested_qubit]['pm0p1']
            ]))
        res_qisk = np.array([
            nl.solve(M_qsub, [d[ind], 1 - d[ind]])[0] for ind in range(len(d))
        ])
        qisk_ker = ss.gaussian_kde(res_qisk)
        
        # fig = plot_setup()
        # figure(num=None, figsize=fig_size, dpi=fig_dpi, facecolor='w', edgecolor='k')
        #plt.plot(xsd,d_ker(xsd),color='Red',linestyle='dashed',label = '(Noisy) Data')
        plt.figure(figsize=(width,height), dpi=100, facecolor='white')
        plt.plot(xsd, proc_ker(xsd), color='Blue', label='By Post')
        plt.plot(xsd, qisk_ker(xsd), color='green', label='By Prior')
        plt.axvline(x=0.5, color='black', label='Ideal')
        plt.xlabel('Pr(Meas. 0)')
        plt.ylabel('Density')
        #plt.title('Denoised Pr(Meas. 0), Qubit %g'%interested_qubit)
        plt.legend()
        plt.tight_layout()
        plt.savefig(file_address + 'DQoI-Qubit%g.pdf' % interested_qubit)
        plt.show()
    return prior_lambdas, post_lambdas


class SplitMeasFilter:
    """Measurement error filter.

    Attributes:
        qubit_order: array,
          using order[LastQubit, ..., FirstQubit].
        data: bit string array,
          Has to be bit string array from circuit of all hadamard gates, leave as [] if data is saved already
        file_address: string
          the address for saving Params.csv and Filter_data.csv. 
          End with '/' if not empty.
        prior: dict
          priors of each qubit. {'Qubit0':[...], 'Qubit1': [...], ...}
        post: dict
          posterior of each qubit. {'Qubit0':[...], 'Qubit1': [...], ...}
        params: dict
          backend properties. Not None after execute inference()
        data: array
          return of read_filter_data(). Not None after execute inference()
        mat_mean: numpy array
          transition matrix created from posterior mean.
          Not None after execute inference()
        mat_mode: numpy array
          transition matrix created from posterior mode.
          Not None after execute inference()
          
    """
    def __init__(self, qubit_order, data=None, file_address=''):
        self.file_address = file_address
        self.qubit_order = qubit_order
        
        # Initialize storage structures
        self.prior = {}
        self.post = {}
        self.post_marginals = {f'Qubit{q}': {'0': None, '1': None} for q in qubit_order}
        self.params = None
        self.mat_mean = None
        self.mat_mode = None

        # LOGIC FIX: Handle data loading gracefully
        if data is not None and len(data) > 0:
            self.data = data
        else:
            # Try to read from file, but default to empty if file doesn't exist yet
            try:
                self.data = read_filter_data(self.file_address)
            except (FileNotFoundError, OSError):
                # This is normal if we are just starting a new calibration
                self.data = np.array([])
            
    def create_filter_mat(self):
        """
        Calculates and stores the 2x2 inverse matrices for each qubit individually.
        This replaces the creation of the massive 2^N x 2^N matrix.
        """
        self.inv_matrices_mean = []
        self.inv_matrices_mode = []

        for q in self.qubit_order:
            q_key = f'Qubit{q}'
            
            # --- MEAN STRATEGY ---
            # Retrieve the marginal measurement errors from posterior
            res_0 = self.post_marginals[q_key]['0']
            res_1 = self.post_marginals[q_key]['1']
            
            lam0_mean = np.mean(res_0) # 1 - error_on_0
            lam1_mean = np.mean(res_1) # 1 - error_on_1
            
            # Build scalable 2x2 A matrix and invert it
            A_mean = np.array([[lam0_mean, 1 - lam1_mean],
                               [1 - lam0_mean, lam1_mean]])
            try:
                self.inv_matrices_mean.append(np.linalg.inv(A_mean))
            except np.linalg.LinAlgError:
                self.inv_matrices_mean.append(np.eye(2)) # Fallback if singular

            # --- MODE STRATEGY ---
            lam0_mode = find_mode(res_0)
            lam1_mode = find_mode(res_1)
            
            A_mode = np.array([[lam0_mode, 1 - lam1_mode],
                               [1 - lam0_mode, lam1_mode]])
            try:
                self.inv_matrices_mode.append(np.linalg.inv(A_mode))
            except np.linalg.LinAlgError:
                self.inv_matrices_mode.append(np.eye(2))

    def inference(self,
                  nPrior=40000,
                  Priod_sd=0.1,
                  seed=227,
                  shots_per_point=1024,
                  show_denoised=False,
                  prep_state='0'):
        """
          Do Bayesian interence

        Parameters
        ----------
        nPrior : int, optional
            Number of priors required. The default is 40000.
            Same as M in output().
        Priod_sd : float, optional
            standard deviation for truncated normal distribution 
            when generating prior parameters. The default is 0.1.
            Same as prior_sd in output().
        seed : int, optional
            Seed for random numbers. The default is 227.
            Same as seed in output().
        shots_per_point : int, optional
            how many shots you want to estimate one QoI (prob. of meas. 0).
            Used to control number of data points and accuracy.
            The default is 1024.
        show_denoised : boolean, optional
            If plot the comparision between post. parameter and 
            given parameters in params. The default is False since 
            it is very time consuming and unnecessary in most of case.
            Same as show_denoised in output().

        Returns
        -------
        None.

        """

        # Ensure data is valid and has a length (Fixes 'unsized object' error)
        if self.data is None:
            raise ValueError("No data provided to inference engine.")
        
        # Force data to be at least 1D array so len() works
        self.data = np.atleast_1d(self.data)
        
        if len(self.data) == 0:
             raise ValueError("Data array is empty.")

        # we check if we actually have params. If not, we use defaults immediately.        
        try:
            if self.params is None:
                 raise Exception("No params, skip to default.")
            
            itr = self.params[0]['itr']
            shots = self.params[0]['shots']
            num_points = int(itr * shots / shots_per_point)
            
        except Exception:
            # Fallback: Calculate points based on data length
            num_points = int(len(self.data) / shots_per_point)
            
            # Ensure at least 1 point
            if num_points < 1: 
                num_points = 1
                
            # Set default error parameters (0.5% error) if not present
            self.params = {}
            for q in self.qubit_order:
                self.params[q] = {}
                self.params[q]['pm1p0'] = 0.005 
                self.params[q]['pm0p1'] = 0.005

        # RUN INFERENCE LOOP
        info = {}
        for i in self.qubit_order:
            print(f'Inferring Qubit {i} for State |{prep_state}>')
            
            d = getData0(self.data, num_points, i)
            
            # Construct filename safely
            state_prefix = self.file_address + f"State{prep_state}_"
            
            prior_lambdas, post_lambdas = output(
                d,
                i,
                nPrior,
                self.params,
                Priod_sd,
                seed=seed,
                show_denoised=show_denoised,
                file_address=state_prefix,
                prep_state=prep_state)
            
            # Store results
            self.prior['Qubit' + str(i)] = prior_lambdas
            self.post['Qubit' + str(i)] = post_lambdas
            
            if prep_state == '0':
                # 0 error rate
                self.post_marginals[f'Qubit{i}']['0'] = post_lambdas[:, 0]
            elif prep_state == '1':
                # 1 error rate
                self.post_marginals[f'Qubit{i}']['1'] = post_lambdas[:, 1]
                
        # Check if we can build the full matrix (only if we have both 0 and 1 data)
        first_q = f'Qubit{self.qubit_order[0]}'
        if (self.post_marginals[first_q]['0'] is not None and 
            self.post_marginals[first_q]['1'] is not None):
            self.create_filter_mat()

    def post_from_file(self):
        """
        Loads the separated posterior files into post_marginals.
        Expects files named: 'State0_Post_QubitX.csv' and 'State1_Post_QubitX.csv'
        """
        for i in self.qubit_order:
            # 1. Load State 0 Data
            try:
                file_0 = self.file_address + f"State0_Post_Qubit{i}.csv"
                data_0 = pd.read_csv(file_0, header=None).to_numpy()
                self.post_marginals[f'Qubit{i}']['0'] = data_0[:, 0] # Col 0 is valid for State 0
            except FileNotFoundError:
                print(f"Warning: Could not find State 0 calibration for Qubit {i}")

            # 2. Load State 1 Data
            try:
                file_1 = self.file_address + f"State1_Post_Qubit{i}.csv"
                data_1 = pd.read_csv(file_1, header=None).to_numpy()
                self.post_marginals[f'Qubit{i}']['1'] = data_1[:, 1] # Col 1 is valid for State 1
            except FileNotFoundError:
                print(f"Warning: Could not find State 1 calibration for Qubit {i}")

        # Re-build matrix
        self.create_filter_mat()

    def mean(self):
        """
           return posterior mean. Now uses the post_marginals for calculation, as fitting new paradigm

        Returns
        -------
        res : dict
            posterior mean of qubits. E.g. 
            {'Qubti0': [...], 'Qubti1': [...], ...}

        """
        res = {}
        for q in self.qubit_order:
            q_key = f'Qubit{q}'
            # Calculate mean of the valid columns
            m0 = np.mean(self.post_marginals[q_key]['0'])
            m1 = np.mean(self.post_marginals[q_key]['1'])
            res[q_key] = np.array([m0, m1])
        return res

    def mode(self):
        """
           return posterior MAP.

        Returns
        -------
        res : dict
            posterior mean of qubits. E.g. 
            {'Qubti0': [...], 'Qubti1': [...], ...}

        """
        res = {}
        for q in self.qubit_order:
            q_key = f'Qubit{q}'
            # Calculate mode of the valid columns
            m0 = find_mode(self.post_marginals[q_key]['0'])
            m1 = find_mode(self.post_marginals[q_key]['1'])
            res[q_key] = np.array([m0, m1])
        return res

    def _apply_tensor_inversion(self, counts, inv_matrices):
        """
        Helper function to apply the chain of inverses to a probability vector.
        """
        shots = sum(counts.values())
        n = len(self.qubit_order)
        
        # Convert Dictionary counts to Vector
        vec = dictToVec(n, counts) / shots
        
        # Reshape to Tensor [2, 2, ..., 2]
        current_tensor = vec.reshape([2] * n)
        
        # Apply inverses sequentially
        for i, inv_mat in enumerate(inv_matrices):
            # Swap target axis 'i' to front (axis 0)
            current_tensor = np.swapaxes(current_tensor, 0, i)
            
            # Flatten to (2, -1) and multiply: M @ vec
            shape_now = current_tensor.shape
            flat = current_tensor.reshape(2, -1)
            new_flat = np.dot(inv_mat, flat)
            
            # Reshape back and swap axis back
            current_tensor = new_flat.reshape(shape_now)
            current_tensor = np.swapaxes(current_tensor, 0, i)
            
        # Flatten and Normalize
        corrected_vec = current_tensor.flatten()
        
        # OPTIONAL: Enforce physical constraints (probabilities >= 0)
        # This replaces the 'find_least_norm' QP solver which is slow.
        # Simple projection: Clip negatives and re-normalize.
        corrected_vec[corrected_vec < 0] = 0
        total_p = np.sum(corrected_vec)
        if total_p > 0:
            corrected_vec = corrected_vec / total_p
            
        return vecToDict(n, shots, corrected_vec)

    def filter_mean(self, counts):
        return self._apply_tensor_inversion(counts, self.inv_matrices_mean)

    def filter_mode(self, counts):
        return self._apply_tensor_inversion(counts, self.inv_matrices_mode)

def error_distributions(self, plotting=True, save_plots=False):
        """
        Calculates statistics and (optionally) plots the posterior distribution 
        of measurement errors for each qubit.
        
        Returns:
            stats (dict): Dictionary containing mean, std, and 95% Confidence Intervals
                          for both error_0 (p(1|0)) and error_1 (p(0|1)).
        """
        stats = {}
        
        for q in self.qubit_order:
            q_key = f'Qubit{q}'
            
            # 1. Retrieve Success Rates (Lambdas)
            # These are samples of "Probability of measuring Correctly"
            lam0_samples = self.post_marginals[q_key]['0']
            lam1_samples = self.post_marginals[q_key]['1']
            
            if lam0_samples is None or lam1_samples is None:
                print(f"Skipping Qubit {q}: Inference not complete.")
                continue

            # 2. Convert to Error Rates (Epsilons)
            # Error = 1 - Success
            err0_samples = 1.0 - lam0_samples # Prob(Meas 1 | Prep 0)
            err1_samples = 1.0 - lam1_samples # Prob(Meas 0 | Prep 1)
            
            # 3. Calculate Statistics
            stats[q_key] = {
                'err0_mean': np.mean(err0_samples),
                'err0_std': np.std(err0_samples),
                'err0_95_CI': np.percentile(err0_samples, [2.5, 97.5]),
                
                'err1_mean': np.mean(err1_samples),
                'err1_std': np.std(err1_samples),
                'err1_95_CI': np.percentile(err1_samples, [2.5, 97.5])
            }
            
            # 4. Plotting (KDE + Histogram)
            if plotting:
                plt.figure(figsize=(8, 5), dpi=100)
                
                # Plot Error 0 (Readout error on |0>)
                kde0 = ss.gaussian_kde(err0_samples)
                x0 = np.linspace(min(err0_samples), max(err0_samples), 200)
                plt.plot(x0, kde0(x0), color='blue', label=r'$P(1|0)$ (Error on 0)')
                plt.fill_between(x0, kde0(x0), alpha=0.2, color='blue')
                
                # Plot Error 1 (Readout error on |1>)
                kde1 = ss.gaussian_kde(err1_samples)
                x1 = np.linspace(min(err1_samples), max(err1_samples), 200)
                plt.plot(x1, kde1(x1), color='red', label=r'$P(0|1)$ (Error on 1)')
                plt.fill_between(x1, kde1(x1), alpha=0.2, color='red')
                
                plt.title(f'Posterior Error Distributions - Qubit {q}')
                plt.xlabel('Error Rate')
                plt.ylabel('Density')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                if save_plots:
                    plt.savefig(self.file_address + f'ErrorDist_Qubit{q}.pdf')
                
                plt.show()
                
                # Print Summary
                print(f"--- Qubit {q} Summary ---")
                print(f"Error on |0>: {stats[q_key]['err0_mean']:.4f} "
                      f"(95% CI: {stats[q_key]['err0_95_CI'][0]:.4f} - {stats[q_key]['err0_95_CI'][1]:.4f})")
                print(f"Error on |1>: {stats[q_key]['err1_mean']:.4f} "
                      f"(95% CI: {stats[q_key]['err1_95_CI'][0]:.4f} - {stats[q_key]['err1_95_CI'][1]:.4f})")
                print("-" * 30)

        return stats
