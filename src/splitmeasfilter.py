# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 13:18:38 2025

@author: Jude L. Metcalf
"""
# Qiskit
from qiskit import QuantumCircuit, transpile
from braket.aws import AwsDevice

# Numerical/Stats pack
import csv
import pandas as pd
import numpy as np
import scipy.stats as ss
import numpy.linalg as nl
from math import pi
# For optimization
from scipy.optimize import minimize
# For plotting
import matplotlib.pyplot as plt
from aquarel import load_theme

theme = load_theme("arctic_light")
theme.apply()
theme.apply_transforms()
width = 6.72 # plot width
height = 4.15 # plot height



# def get_braket_calibration_dict(device_arn, n_qubits=None):
#     """
#     Returns the params list in the exact format required by MeasFilter.
#     """
#     device = AwsDevice(device_arn)
#     properties = device.properties
    
#     # Try to find qubit count if not provided
#     if n_qubits is None:
#         try:
#             n_qubits = properties.paradigm.qubitCount
#         except:
#             n_qubits = 5 # Manual override if needed

#     formatted_params = []

#     for q in range(n_qubits):
#         # 1. Default conservative error (5% error)
#         p_meas0_prep1 = 0.05 
#         p_meas1_prep0 = 0.05
        
#         # 2. Try to extract real Fidelity from Braket Properties
#         # This structure depends on the provider (Rigetti, OQC, etc.)
#         try:
#             # Example for Rigetti (Standardized in Braket properties)
#             # We look for "fRO" (Readout Fidelity)
#             provider_specs = properties.provider.specs
#             qubit_specs = provider_specs.get(f"{q}Q", {}).get(f"{q}", {})
#             fRO = qubit_specs.get('fRO', None)
            
#             if fRO:
#                 error = 1.0 - fRO
#                 # Assume symmetric error if specific p(0|1) isn't detailed
#                 p_meas0_prep1 = error
#                 p_meas1_prep0 = error
#         except:
#             # If extraction fails, we stick to the 0.05 default
#             pass

#         # 3. Create the dictionary for this qubit
#         qubit_cal = {
#             'qubit': q,
#             'pm0p1': p_meas0_prep1, # PROB MEASURING 0 GIVEN PREP 1
#             'pm1p0': p_meas1_prep0, # PROB MEASURING 1 GIVEN PREP 0
#             # Extra fields required by the code logic, even if unused
#             'itr': 32,
#             'shots': 8192
#         }
        
#         formatted_params.append(qubit_cal)

#     return np.array(formatted_params)


# def param_record(backend, itr=32, shots=8192, if_write=True, file_address=''):
#     """Write backend property into an array of dict 
#        and save as csv if permissible.

#     Args:
#       backend: Backend
#         A Qiskit backend instance.
#       itr: int
#         number of iterations of job submission.
#       shots: int
#         number of shots per each job submission.
#       if_write: boolean
#         True if save the properties as a csv file.
#       file_address: string
#         The relative file address to save backend properties. 
#         Ends with '/' if not empty
#         The default is ''.

#     Returns: numpy array
#       An array of dicts. Each dict records all characterization of one qubit.
#     """
#     allParam = np.array([])
    
#     try:
#         # Check if backend supports properties interface
#         if not hasattr(backend, 'properties') or backend.properties() is None:
#             raise Exception("Backend properties not available.")

#         prop_dict = backend.properties().to_dict()
#         nQubits = len(prop_dict['qubits'])
        
#         # Helper to safely extract values from property lists
#         def get_val(props, name):
#             for item in props:
#                 if item.get('name') == name:
#                     return item.get('value')
#             return 0.0 # Default if not found

#         target_qubits = range(nQubits)
        
#         for target_qubit in target_qubits:
#             qubit_props = prop_dict['qubits'][target_qubit]
            
#             # Basic params
#             params = {
#                 'qubit': target_qubit,
#                 'update_date': prop_dict.get('last_update_date', 'N/A'),
#                 'T1': get_val(qubit_props, 'T1'),
#                 'T2': get_val(qubit_props, 'T2'),
#                 'freq': get_val(qubit_props, 'frequency'),
#                 'readout_err': get_val(qubit_props, 'readout_error'),
#                 'pm0p1': get_val(qubit_props, 'prob_meas0_prep1'),
#                 'pm1p0': get_val(qubit_props, 'prob_meas1_prep0'),
#                 'itr': itr,
#                 'shots': shots,
#             }

#             # Try to fetch gate errors if available in legacy format
#             # Many non-IBM backends won't have this specific structure, so we wrap in try
#             try:
#                 gates = prop_dict.get('gates', [])
#                 # Simplified extraction logic or defaults
#                 # This part is highly specific to IBM's old map; defaulting if fails
#                 params['id_error'] = 0.001
#                 params['u3_error'] = 0.001 
#                 # (Real extraction logic omitted for brevity/compatibility)
#             except:
#                 pass

#             allParam = np.append(allParam, params)

#         if if_write:
#             with open(file_address + 'Params.csv', mode='w', newline='') as sgm:
#                 param_writer = csv.writer(sgm,
#                                           delimiter=',',
#                                           quotechar='"',
#                                           quoting=csv.QUOTE_MINIMAL)
#                 for pa in allParam:
#                     for key, val in pa.items():
#                         param_writer.writerow([key, val])
#                     param_writer.writerow(['End'])

#     except Exception as e:
#         print(f"Note: Backend parameters could not be recorded ({str(e)}). Using defaults for inference.")
#         # Return empty; inference engine handles empty params by setting defaults
#         return np.array([])

#     return allParam


# def meas_circ(nQubits, backend=None, itr=32):
#     """
#     Generates 2 * itr circuits compatible with Qiskit 2.x and Braket.
#     - First half prepares |0>
#     - Second half prepares |1>

#     Args:
#         nQubits: int
#             number of qubits.
#         backend: Backend
#             A Qiskit backend instance. Used for transpilation.
#         itr: int
#             number of iterations of each state preparation.

#     Returns: list of form [{QuantumCircuit preparing |0>}, {QuantumCircuit preparing |1>}]
#     """
#     circs = []
    
#     # Create |0> circuits (Identity)
#     # Note: Explicit 'id' gates often optimized away, so we use barriers 
#     # or simple measurements to define the state.
#     c0 = QuantumCircuit(nQubits, nQubits)

#     # No gates needed for |0>, just measure. 
#     # We add a barrier to prevent transpiler from merging if we added ops later.
#     c0.barrier() 
#     c0.measure(range(nQubits), range(nQubits))
    
#     # Create |1> circuits (X gate)
#     c1 = QuantumCircuit(nQubits, nQubits)
#     c1.x(range(nQubits)) # Broadcast X to all qubits
#     c1.barrier()
#     c1.measure(range(nQubits), range(nQubits))

#     # Transpile if backend provided (Optional but recommended for ISA)
#     if backend:
#         # In Qiskit 2.x, it's best to transpile once before copying
#         c0 = transpile(c0, backend)
#         c1 = transpile(c1, backend)

#     # Create the batch
#     # We use metadata or naming to track them
#     ## Perhaps change to one request with many shots?
#     for i in range(itr):
#         # Create copies
#         c0_copy = c0.copy()
#         c0_copy.name = f"cal_0_itr{i}"
#         circs.append(c0_copy)
        
#     for i in range(itr):
#         c1_copy = c1.copy()
#         c1_copy.name = f"cal_1_itr{i}"
#         circs.append(c1_copy)
        
#     return circs

# def collect_filter_data(backend,
#                         itr=32,
#                         shots=8192,
#                         if_write=True,
#                         file_address='',
#                         job_id=''):
#     """
#     Collects measurement data compatible with Amazon Braket Qiskit Provider.
#     """
#     # Determine qubit count 
#     try:
#         nQubits = backend.num_qubits
#     except:
#         nQubits = 5 # Fallback
#         raise Exception("Number of Qubits not found!!!!")

#     readout_m0 = np.array([])
    
#     # Generate the circuits (One circuit repeated 'itr' times with unique names)
#     circs = meas_circ(nQubits, backend, itr=itr)

#     # Job Execution or Retrieval (Sim or Real)
#     if job_id:
#         print(f"Retrieving existing Braket Job ID: {job_id}")
#         try:
#             # Braket Provider usually allows retrieving via the backend or service
#             job_m0 = backend.retrieve_job(job_id)
#         except Exception as e:
#             print(f"Failed to retrieve job: {e}")
#             return np.array([])
#     else:
#         print(f"Submitting new batch job with {itr} circuits to {backend.name}...")
#         try:
#             # Executing all circuits in one batch
#             # 'memory=True' to get the bitstrings required for this inference
#             job_m0 = backend.run(circs, shots=shots, memory=True)
#             print(f"Job submitted. ID: {job_m0.job_id()}")
            
#             # For later, as of 03/01: Explicit wait loop if the provider doesn't block automatically
#             # while not job_m0.in_final_state():
#             #     print("Status:", job_m0.status())
#             #     time.sleep(5)
                
#         except Exception as e:
#             print(f"Job submission failed: {e}")
#             return np.array([])

#     # Extract Results
# # try:
#     # This blocks until the job is done
#     m0_res = job_m0.result()
    
#     print("Job complete. extracting memory...")

#     # Loop through the experiments (one for each circuit in the batch)
#     for i in range(itr):
#         # get_memory(i) retrieves the list of bitstrings for the i-th circuit
#         # e.g., ['000', '001', '000', ...]
#         memory_data = m0_res.get_memory(i)
#         readout_m0 = np.append(readout_m0, memory_data)

#     # Save to CSV
#     if if_write:
#         filename = file_address + 'Filter_data.csv'
#         print(f"Saving data to {filename}...")
#         # Using 'w' mode with standard csv writer
#         with open(filename, mode='w', newline='') as sgr:
#             read_writer = csv.writer(sgr, quoting=csv.QUOTE_MINIMAL)
#             read_writer.writerow(readout_m0)
                
# # except Exception as e:
# #     print(f"Error processing results: {e}")
# #     # If accessing memory fails, print available keys to help debug
# #     try:
# #         print("Result keys available:", m0_res.get_counts())
# #     except:
# #         pass

#     return readout_m0


# def read_params(file_address=''):
#     """Read out backend properties from csv file generated by param_record().

#     Args:
#       file_address: string
#         The relative file address to read backend properties. 
#         Ends with '/' if not empty
#         The default is ''.

#     Returns: numpy array
#       An array of dicts. Each dict records all characterization of one qubit.
#     """
#     textKeys = ['name', 'update_date', 'qubit']
#     intKeys = ['itr', 'shots']
#     # Read Parameters
#     with open(file_address + 'Params.csv', mode='r') as sgm:
#         reader = csv.reader(sgm)
#         params = []
#         singleQubit = {}
#         first = True
#         for row in reader:
#             if row[0] == 'End':
#                 params.append(singleQubit)
#                 singleQubit = {}
#             else:
#                 singleQubit[row[0]] = row[1]

#     # Convert to numpy array
#     params = np.array(params)

#     # Convert corresponding terms into floats or ints
#     for qubit in params:
#         for key in qubit.keys():
#             if key not in textKeys:
#                 qubit[key] = float(qubit[key])
#             if key in intKeys:
#                 qubit[key] = int(qubit[key])

#     return params


# def read_filter_data(file_address=''):
#     """Read out bit string data from csv file generated by collect_filter_data().

#     Args:
#       file_address: string
#         The relative file address to read data for filter generation. 
#         Ends with '/' if not empty
#         The default is ''.

#     Returns: numpy array
#       An array of bit strings.
#     """
#     # Should be able to use np.genfromtxt
#     cali01 = np.genfromtxt(file_address + 'Filter_data.csv', delimiter=',',dtype=str)
#     if len(cali01) < 2:
#         with open(file_address + 'Filter_data.csv', mode='r') as measfile:
#             reader = csv.reader(measfile)
#             cali01 = np.asarray([row for row in reader][0])
#     return cali01


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


# def closest_mode(post_lambdas):
#     """Find the tuple of model parameters that closed to 
#        the Maximum A Posteriori (MAP) of 
#        posterior distribution of each parameter

#     Args:
#       post_lambdas: numpy array
#         an n-by-m array where n is the number of posteriors and m is number 
#         of parameters in the model

#     Returns: numpy array
#       an array that contains the required model parameters.
#     """

#     mode_lam = []
#     for j in range(post_lambdas.shape[1]):
#         mode_lam.append(find_mode(post_lambdas[:, j]))

#     sol = np.array([])
#     smallest_norm = nl.norm(post_lambdas[0])
#     mode_lam = np.array(mode_lam)
#     for lam in post_lambdas:
#         norm_diff = nl.norm(lam - mode_lam)
#         if norm_diff < smallest_norm:
#             smallest_norm = norm_diff
#             sol = lam
#     return sol


# def closest_average(post_lambdas):
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

def safe_mean(data) -> float:
    """
    Safely compute the mean of a parameter, raising a type error if needed
    """
    if type(data) == list or type(data) == np.ndarray:
        mean = float(np.mean(data))
    else:
        raise Exception(f'Input {data} is not of type "list" or "numpy.ndarray", instead it is of type {type(data)}')

    return mean


### Needed? What for?
# def vecToDict_inv(nQubits, shots, vec):
#     """ 
#       Same as dictToVec() but key uses big-endian convention
#     """
#     counts = {}
#     form = "{0:0" + str(nQubits) + "b}"
#     for i in range(2**nQubits):
#         key = form.format(i)[::-1]
#         counts[key] = int(vec[i] * shots)
#     return counts

def dict_filter(data_dict: dict[str, int], percent: float | int = 99.0) -> dict[str, int]:
        """
        Filters a dictionary to retain entries that make up the top x% of total counts, the default set to 99%.

        Args:
            data_dict (dict): The input dictionary with counts.
            percent (float): The percentage (0-100) threshold to retain (default is 99).

        Returns:
            dict: A new dictionary containing only the top 99% of entries.
        """
        total_sum = sum(data_dict.values())
        if total_sum == 0:
            return {}

        # Sort descending
        sorted_items = sorted(data_dict.items(), key=lambda item: item[1], reverse=True)

        filtered_dict = {}
        cumulative_sum = 0
        # Smart way to ensure the correct percentage scale (0.99 vs 99)
        # Shouldn't have to think about it, but still!! Cool!!!
        if percent > 1.0: 
            percent = percent / 100.0
        
        threshold = percent * total_sum

        for key, value in sorted_items:
            if cumulative_sum < threshold:
                filtered_dict[key] = value
                cumulative_sum += value
            else:
                break
                
        return filtered_dict


# Functions
def getData0(data, num_group, interested_qubit):
    """ Get the probabilities of measuring 0 from binary readouts
        **Binary number follows little-endian convention**

    Parameters
    ----------
    data : array_like
        An array of binary readout strings (e.g., ['001', '010']).
    num_group : int
        Number of groups to split the data into for probability calculation.
        Data length must be divisible by num_group.
    interested_qubit : int
        The index of the qubit to analyze (0-indexed).

    Returns
    -------
    prob0 : numpy array
        Array of probabilities of measuring 0 for each group.

    """
    # Ensure input is a numpy array of strings (bytes for performance)
    data_arr = np.array(data, dtype='S')
    
    # Check if data can be evenly divided
    if data_arr.size % num_group != 0:
        raise ValueError(f"Data size {data_arr.size} is not divisible by num_group {num_group}")

    # Since the input is little-endian (qubit 0 is at the end of the string),
    # the index is: Length - 1 - interested_qubit.
    
    # length of the strings 
    str_len = data_arr.itemsize 
    
    # Calculate the target index in the string (Big-Endian representation in memory)
    col_idx = str_len - 1 - interested_qubit

    # replicate the 'try/except' fallback logic from the original code:
    # If the qubit index is out of bounds (string too short), look at the 
    # 0-th qubit (which corresponds to the last character in the string).
    if col_idx < 0:
        col_idx = str_len - 1

    # Convert array of strings into a 2D matrix of single characters
    # e.g., ['01', '10'] becomes [['0', '1'], ['1', '0']]
    # reshape into (total_samples, string_length)
    chars = data_arr.view('S1').reshape(data_arr.size, -1)
    
    # extract only the column corresponding to the interested qubit
    qubit_measurements = chars[:, col_idx]
    
    # Reshape to (num_group, samples_per_group)
    grouped_measurements = qubit_measurements.reshape(num_group, -1)
    
    # Check where measurements are '0' (results in a boolean matrix)
    # Taking the mean of booleans converts True to 1 and False to 0, giving the probability.
    prob0 = (grouped_measurements == b'0').mean(axis=1)
    
    return prob0

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
    num_samples = prior_lambdas.shape[0]

    # Define Ideal Vector based on what we prepared
    if prep_state == '0':
        # [Prob(0), Prob(1)] -> [100%, 0%]
        M_ideal = np.array([[1.0, 0.0]]*num_samples) 
    elif prep_state == '1':
        # [Prob(0), Prob(1)] -> [0%, 100%]
        M_ideal = np.array([[0.0, 1.0]]*num_samples) 
    else: # Default to '+'
        print('Default to |+>, set prep_state = "0" or "1" for these preparations.')
        M_ideal = np.array([[0.5, 0.5]]*num_samples) 

    # Vectorised Forward noising place of previous errMitMat()

    pm0p0 = prior_lambdas[:, 0]
    pm1p1 = prior_lambdas[:, 1]

    A_batch = np.zeros((num_samples, 2, 2))

    # Row 0
    A_batch[:, 0, 0] = pm0p0
    A_batch[:, 0, 1] = 1 - pm1p1
    
    # Row 1
    A_batch[:, 1, 0] = 1 - pm0p0
    A_batch[:, 1, 1] = pm1p1

    M_observed = A_batch @ M_ideal
    
    # The result is the Probability of Measuring 0 (First component)
    qs = M_observed[:, 0, 0]

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




def findM(qs_ker, d_ker, prep_state):
    """
    The function used to find the M that maximises the d/q ratio
    """
    # Scan the space to find where the prior actually exists
    if prep_state == '0':
        x_scan = np.linspace(0.95, 1, 1000)
    elif prep_state == '1':
        x_scan = np.linspace(0, 0.05, 1000)
    else:
        x_scan = np.linspace(0, 1, 1000)
    prior_density = qs_ker(x_scan)
    
    # Define Effective Support: 
    # Only consider regions where prior density is at least 1% of its peak.
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

    # Find a safe starting point (x0) inside the valid bounds
    # We look for the minimum of y only within our valid indices
    ys = np.array([dq(x, qs_ker, d_ker) for x in x_scan])
    valid_ys = ys[valid_indices]
    valid_xs = x_scan[valid_indices]
    x0 = valid_xs[np.argmin(valid_ys)]

    # Run Optimizer with safe bounds
    res = minimize(dq, (x0,), args=(qs_ker, d_ker), bounds=(bds,), method='L-BFGS-B')

    # --- Plot --------------------------------------------------
    plt.figure(figsize=(width,height), dpi=100, facecolor='white')
    plt.plot(1 - x_scan, ys, c = 'crimson', lw = 2, label = '-Data/Prior')
    plt.xlabel("Qubit Value")
    plt.ylabel("Relative Rate (-Data/Prior)")
    plt.title("-Data/Prior PDFs Minimised for Efficient Inference")
    # -----------------------------------------------------------
    
    try:
        plt.axhline(res.fun[0], label = r'$\mu$ minimiser', c = 'purple')
        plt.legend()
        plt.show()
        return -res.fun[0], res.x[0]
    except Exception:
        plt.axhline(res.fun, label = r'$\mu$ minimiser', c = 'purple')
        plt.legend()
        plt.show()
        return -res.fun, res.x
    
def QoI_single(lambdas, prep_state='0'):
    """
    Optimized QoI that maps Success Rate (lambda) to Prob(Measuring 0).
    """
    if prep_state == '0':
        # If Prep 0, P(Meas 0) = Success Rate
        return lambdas 
    elif prep_state == '1':
        # If Prep 1, P(Meas 0) = 1.0 - Success Rate
        return 1.0 - lambdas
    return lambdas

def output(d,
           interested_qubit,
           M,
           params,
           prior_sd,
           seed=127,
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
    if prep_state == '0':
        # We are looking for Success Rate on 0 (1 - p(1|0))
        prior_mean = 1.0 - params[interested_qubit].get('pm1p0', 0.05)
    else:
        # We are looking for Success Rate on 1 (1 - p(0|1))
        prior_mean = 1.0 - params[interested_qubit].get('pm0p1', 0.05)

    # Sanity check constraints
    if prior_mean > 1.0 or prior_mean < 0.7:
        prior_mean = 0.95

    prior_lambdas = tnorm01(prior_mean, prior_sd, size=M) # 1D single lambda array of size M

    qs = QoI_single(prior_lambdas, prep_state=prep_state) # Single lambda QoI, optimising to find e0 and e1 separately

    d_ker = ss.gaussian_kde(d)
    qs_ker = ss.gaussian_kde(qs)

    print(f'Given Lambda preparing {prep_state}): success rate = {prior_mean:.4f}')

    # Find the max ratio r(Q(lambda)) over a single lambda

    max_r, max_q = findM(qs_ker, d_ker, prep_state)

    # Print and Check
    print('Final Accepted Posterior Lambdas')
    print(r'$\mu$' + f': %.6g Maximizer: %.6g pi_obs = %.6g pi_Q(prior) = %.6g' %
          (max_r, max_q, d_ker(max_q), qs_ker(max_q)))

    post_lambdas = np.array([])
    # Rejection Iteration (vectorized!)
    r_vals = d_ker(qs) / qs_ker(qs)
    eta = r_vals / max_r 

    # Accept based on uniform random draw
    accept_mask = eta > np.random.uniform(0, 1, M)
    post_lambdas = prior_lambdas[accept_mask]
    
    post_qs = QoI_single(post_lambdas, prep_state=prep_state)
    post_ker = ss.gaussian_kde(post_qs)

    # Logging
    print('Accepted N: %d (%.1f%%)' % (len(post_lambdas), 100*len(post_lambdas)/M))
    print(f'Posterior Mean for preparing {prep_state}: success rate ~ {np.mean(post_lambdas):.6f}')

    # Save results as a 1D array
    filename = file_address + f'Post_Qubit{interested_qubit}.csv'
    np.savetxt(filename, post_lambdas, delimiter=',')

    # --------------------- Plotting ---------------------
    xs = np.linspace(0, 1, 1000)
    # Plotting range depends on prep_state (0 is high, 1 is low)
    if prep_state == '0':
        xsd = np.linspace(0.96, 1.0, 500)
    elif prep_state == '1':
        xsd = np.linspace(0.0, 0.04, 500)
    else:
        print('Plotting full range for unknown prep_state, please supply prep_state "0" or "1".')
        xsd = xs

    plt.figure(figsize=(6, 4), dpi=100, facecolor='white')
    plt.plot(np.ones_like(xsd)-xsd, d_ker(xsd), 'r--', lw=2, label='Observed Data')
    plt.plot(np.ones_like(xsd)-xsd, post_ker(xsd), 'b-', label='Posterior Model')
    plt.xlabel('Average Qubit Value')
    plt.ylabel('Density')
    plt.title(f'Calibration Qubit {interested_qubit} |{prep_state}>')
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_address + f'QoI-Qubit{interested_qubit}.pdf')
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
        self.post_marginals = {f'Qubit{q}': {'0': np.array([]), '1': np.array([])} for q in qubit_order}
        self.params = None
        self.mat_mean = None
        self.mat_mode = None

        # Load Data
        if data is not None and len(data) > 0:
            self.data = np.atleast_1d(data)
        else:
            self.data = self._load_data_from_file()
            
    def _load_data_from_file(self):
        """Internal helper to load raw bitstrings safely."""
        try:
            # Use pandas for robust CSV reading (handles newlines/headers better)
            path = self.file_address + 'Filter_data.csv'
            df = pd.read_csv(path, header=None, dtype=str)
            return df.values.flatten()
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return np.array([])
        
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
            
            lam0_mean = safe_mean(res_0) # 1 - error_on_0
            lam1_mean = safe_mean(res_1) # 1 - error_on_1
            
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
                file_address=state_prefix,
                prep_state=prep_state)
            
            # Store results
            self.prior['Qubit' + str(i)] = prior_lambdas
            self.post['Qubit' + str(i)] = post_lambdas
            
            if prep_state == '0':
                # 0 error rate
                self.post_marginals[f'Qubit{i}']['0'] = post_lambdas
            elif prep_state == '1':
                # 1 error rate
                self.post_marginals[f'Qubit{i}']['1'] = post_lambdas
                
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

            m0 = safe_mean(self.post_marginals[q_key]['0'])
            m1 = safe_mean(self.post_marginals[q_key]['1'])

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
    
    def get_inverse_element(self, target_bitstring, source_bitstring):
        """
        Calculates the probability transition factor from Source (Noisy) -> Target (Clean).
        Mathematically: returns the element (Row=Target, Col=Source) of the Inverse Matrix.
        
        Args:
            target_bitstring (str): The 'clean' state we are calculating probability for.
            source_bitstring (str): The 'noisy' state we actually measured.
        """
        # Ensure we have the matrices
        if self.inv_matrices_mean is None:
            self.create_filter_mat()
            
        probability_factor = 1.0
        
        # Iterate through qubits to multiply local probabilities
        # We assume Little Endian (Qubit 0 is right-most character)
        for i, q in enumerate(self.qubit_order):
            # Parse the specific bit for this qubit from the strings
            # String index must be reversed: string[-1] is Qubit 0
            t_val = int(target_bitstring[-(i+1)]) # Row Index (Clean)
            s_val = int(source_bitstring[-(i+1)]) # Col Index (Noisy)
            
            # Retrieve the specific value from the 2x2 inverse matrix
            # inv_matrices_mean is a list of 2x2 matrices for [q0, q1, ...]
            # We must access the matrix corresponding to 'i' (the current qubit loop index)
            elem = self.inv_matrices_mean[i][t_val, s_val]
            
            probability_factor *= elem
            
        return probability_factor
    
    def get_forward_element(self, observed_bitstring, hidden_bitstring):
        """
        Calculates P(Observed | Hidden).
        Used for the forward pass of Gradient Descent.
        Very similar to get_inverse_element, but uses the forward error model.
        """
        # Ensure marginals are available
        if not hasattr(self, 'post_marginals'):
            raise RuntimeError("Filter not calibrated. Run inference() first.")

        prob = 1.0
        # Iterate qubits (Little Endian: String[-1] is Qubit 0)
        for i, q in enumerate(self.qubit_order):
            obs_bit = int(observed_bitstring[-(i+1)]) # Row Index
            hid_bit = int(hidden_bitstring[-(i+1)])   # Col Index
            
            q_key = f'Qubit{q}'
            
            # Retrieve single-qubit error rates from calibration
            # P(Meas 0 | Prep 0) = Mean of Post Lambda 0
            p0_g_0 = safe_mean(self.post_marginals[q_key]['0'])
            # P(Meas 1 | Prep 1) = Mean of Post Lambda 1
            p1_g_1 = safe_mean(self.post_marginals[q_key]['1'])
            
            # Build the 2x2 Forward Error Matrix for this qubit
            # M = [[P(0|0), P(0|1)], 
            #      [P(1|0), P(1|1)]]
            m_single = np.array([
                [p0_g_0,     1 - p1_g_1],
                [1 - p0_g_0, p1_g_1]
            ])
            
            prob *= m_single[obs_bit, hid_bit]
            
        return prob
    

    def eff_DeNoise(self, datadict, percentage=100, verbose=True, GD = False, lr=0.1, max_iter=50):
        """
        Efficient DeNoiser function that applies the SplitMeasFilter to the provided data dictionary.
        
        Parameters:
        - datadict: Dictionary containing measurement data.
        - SplitMeasFilter: An instance of the SplitMeasFilter class with calibrated errors.
        - percentage: percentage threshold for filtering (default is 99).
        - verbose: If True, prints progress information.
        - GD: If True, performs optional Gradient Descent refinement.
        - lr: Learning rate for Gradient Descent (if GD is True).
        - max_iter: Maximum number of iterations for Gradient Descent (if GD is True).
        
        Returns:
        - denoised_data: Dictionary containing denoised measurement data.
        """
        
        # We only sum over source bitstrings that are most computationally significant
        # Defined from the `percentage` parameter.
        important_dict = dict_filter(datadict, percent=percentage)
        
        input_len = len(important_dict)
        # Defining our targets in a sparse approach, only those significant sources 
        # of the bitstrings we actually saw will remain (noise won't make our output vanish completely!)
        targets = list(important_dict.keys())
        
        denoised_data = {k: 0.0 for k in targets}
        
        # Sparse matrix mult (a (pretty unavoidable) double loop over sources and targets)
        # Equation: P_denoised(target) = Sum_over_sources( M_inv[target, source] * P_noisy(source) )
        
        # Pre-calculate total shots for normalization later
        filtered_shots = sum(important_dict.values())
        
        for t_str in targets:
            new_prob = 0.0
            for s_str, s_count in important_dict.items():
                # Calculate probability of Source(s) flipping to Target(t)
                # This is the "Likelihood of changing/remaining"
                weight = self.get_inverse_element(t_str, s_str) # M_inv(target_bitstring=t_str, source_bitstring=s_str)
                
                # Add contribution: (Inverse Element) * (Observed Probability)
                s_prob = s_count / filtered_shots
                new_prob += weight * s_prob
                
            denoised_data[t_str] = new_prob

        # Constrained optimisation using clipping, a good heuristic.
        # We minimize ||Ap - p_tilde|| clipping to to p >= 0 and re-normalizing.
        
        final_data = {}
        sum_p = 0.0
        
        for k, v in denoised_data.items():
            if v > 1e-9: # Clip negatives and near-zeros
                final_data[k] = v
                sum_p += v
                
        # Re-normalize to ensure sum is 1.0 (or original shot count)
        if sum_p > 0:
            factor = filtered_shots / sum_p
            final_data = {k: v * factor for k, v in final_data.items()}
        
        # -------------------------------------------------------------------------
        # OPTIONAL: GRADIENT DESCENT REFINEMENT
        # -------------------------------------------------------------------------
        if GD:
            if verbose: print(f"Analytic pass done. Starting Gradient Descent refinement...")
            
            n_dim = len(targets)
            
            # A. Vectorize Data
            # 'y' = The actual noisy observations we want to match
            p_noisy_obs = np.array([important_dict[t] for t in targets]) / filtered_shots
            
            # 'x' = Our initial guess (The result from the Analytic pass)
            # Using the analytic result as a "warm start" makes GD extremely fast.
            p_est = np.zeros(n_dim)
            for i, t in enumerate(targets):
                p_est[i] = final_data.get(t, 0.0) / filtered_shots
            
            # B. Build Forward Matrix M_sub for this subspace
            # We need M (Forward), not M_inv, to calculate the Loss: || M*x - y ||^2
            M_sub = np.zeros((n_dim, n_dim))
            
            for r, obs_bit in enumerate(targets):     # Row: Observed
                for c, hid_bit in enumerate(targets): # Col: Hidden (Clean)
                    M_sub[r, c] = self.get_forward_element(obs_bit, hid_bit)
            
            # C. Projected Gradient Descent Loop
            for step in range(max_iter-1):
                # Forward: p_pred = M * p_est
                p_pred = M_sub @ p_est
                
                # Gradient of MSE: grad = M.T * (p_pred - p_obs)
                diff = p_pred - p_noisy_obs
                grad = M_sub.T @ diff
                
                # Update
                p_est = p_est - lr * grad
                
                # Projection (Constraint: p >= 0 and sum(p) = 1)
                p_est[p_est < 0] = 0 # Clip negatives
                
                curr_sum = np.sum(p_est)
                if curr_sum > 0:
                    p_est /= curr_sum # Normalize
                
                # Convergence Check (Small gradient magnitude)
                if np.linalg.norm(lr * grad) < 1e-18:
                    print(f"----  GD Converged after {step+1} steps!  ----")
                    break

                print(f"----  GD Step {step+1}/{max_iter} complete...  ----")
            
            # D. Update final_data with refined values
            final_data = {}
            for i, t in enumerate(targets):
                if p_est[i] > 1e-9:
                    final_data[t] = p_est[i] * filtered_shots # Scale back to counts
        
            if verbose:
                print(f"Sparse DeNoising Complete. Gradient Descent Used")
                print(f"Input Keys: {len(datadict)} -> Filtered Sources: {input_len} -> Output Targets: {len(final_data)}")
                print(f"Total Shots (Input): {sum(datadict.values())} -> (Filtered): {filtered_shots} -> (Output): {sum(final_data.values())}")
        else:
            if verbose:
                print(f"Sparse DeNoising Complete. No Gradient Descent included")
                print(f"Input Keys: {len(datadict)} -> Filtered Sources: {input_len} -> Output Targets: {len(final_data)}")
                print(f"Total Shots (Input): {sum(datadict.values())} -> (Filtered): {filtered_shots} -> (Output): {sum(final_data.values())}")


        # -------------------------------------------------------------------------

        final_data = {k: float(v) for k, v in final_data.items()} # np.float64 -> float

        return final_data
        

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
