# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 13:18:38 2025

@author: Jude L. Metcalf
"""
import os
import csv
from pathlib import Path
import pandas as pd
import numpy as np
import scipy.stats as ss
import numpy.linalg as nl
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from aquarel import load_theme
from braket.aws import AwsDevice


# Optional: Apply theme if available
try:
    theme = load_theme("arctic_light")
    theme.apply()
    theme.apply_transforms()
except:
    pass

width = 6.72 
height = 4.15 

def get_braket_calibration_dict(device_arn, n_qubits=None):
    """
    Returns the params list in the exact format required by MeasFilter.
    """
    device = AwsDevice(device_arn)
    properties = device.properties
    standardized = getattr(properties, "standardized", None)
    if standardized and hasattr(standardized, "dict"):
        qubit_props = standardized.dict()
    else:
        try:
            qubit_props = properties.dict()
        except:
            qubit_props = {}
    
    one_props = qubit_props["oneQubitProperties"]
    two_props = qubit_props["twoQubitProperties"]
    one_props_len = len(one_props)

    if n_qubits is None:
        n_qubits = one_props_len
    else:
        if one_props_len != n_qubits:
            print(f"Warning: Provided n_qubits={n_qubits} does not match device oneQubitProperties length={one_props_len}. Using device length, as qubits are likely down.")
        else:
            n_qubits = one_props_len
            
    properties = {}
    for q in range(n_qubits):
        q_one_prop = one_props[str(q)]["oneQubitFidelity"]
        q_two_props = {k: v["twoQubitGateFidelity"][0]["fidelity"]
                        for k, v in two_props.items() if str(q) in k.split("-")}

        one_gate_fidelity = q_one_prop[0]["fidelity"]
        pm1p0 = q_one_prop[1]["fidelity"]
        pm0p1 = q_one_prop[2]["fidelity"]
        properties[str(q)] = {"pm1p0": pm1p0, "pm0p1": pm0p1, "oneQubitFidelity": one_gate_fidelity, "twoQubitProperties": q_two_props}

    return properties




# --- Helper Functions ---

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
    if sd == 0: return np.full(size, center)
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
    if len(data) < 2: return np.mean(data)
    kde = ss.gaussian_kde(data)
    line = np.linspace(min(data), max(data), 1000)
    return line[np.argmax(kde(line))]

def safe_mean(data) -> float:
    """ Safely compute the mean, returning 0.0 if empty to avoid crashes. """
    if len(data) == 0: return 0.0
    return float(np.mean(data))

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
        key = form.format(i) 
        if key in counts:
            vec[i] = int(counts[key])
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

def dict_filter(data_dict: dict, percent: float = 99.0) -> dict:
    """
    Filters a dictionary to retain entries that make up the top x% of total counts, the default set to 99%.

    Args:
        data_dict (dict): The input dictionary with counts.
        percent (float): The percentage (0-100) threshold to retain (default is 99).

    Returns:
        dict: A new dictionary containing only the top 99% of entries.
    """
    total_sum = sum(data_dict.values())
    if total_sum == 0: return {}
    sorted_items = sorted(data_dict.items(), key=lambda item: item[1], reverse=True)
    filtered_dict = {}
    cumulative_sum = 0
    if percent > 1.0: percent /= 100.0
    threshold = percent * total_sum
    for key, value in sorted_items:
        if cumulative_sum < threshold:
            filtered_dict[key] = value
            cumulative_sum += value
        else:
            break
    return filtered_dict

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
    # Force conversion to string list to handle numpy ints or bytes
    data_list = [str(x).strip() for x in data]
    
    if len(data_list) == 0:
        return np.array([])

    # Determine index for interested qubit (Little Endian)
    # If string is '00', Qubit 0 is at index 1 (end), Qubit 1 at index 0.
    str_len = len(data_list[0])
    target_idx = str_len - 1 - interested_qubit
    
    # Fallback if qubit index out of bounds
    if target_idx < 0: target_idx = 0 

    is_zero_counts = []
    
    # Parse Bits
    for s in data_list:
        try:
            bit = s[target_idx]
        except IndexError:
            bit = '0' # Fallback
            
        # Check for '0' char. 
        if bit == '0':
            is_zero_counts.append(1.0)
        else:
            is_zero_counts.append(0.0)

    # Reshape and Average
    total_len = len(is_zero_counts)
    if num_group > total_len or num_group <= 0:
        # Prevent zero division or impossible reshape
        return np.array(is_zero_counts)

    trunc_len = total_len - (total_len % num_group)
    
    if trunc_len == 0:
        return np.array([])
        
    arr = np.array(is_zero_counts[:trunc_len])
    grouped = arr.reshape(num_group, -1)
    
    # Calculate mean (probability of being 0)
    prob0 = grouped.mean(axis=1)
    
    return prob0

def dq(x, qs_ker, d_ker):
    if np.abs(qs_ker(x)[0]) > 1e-6 and np.abs(d_ker(x)[0]) > 1e-6:
        return - d_ker(x)[0] / qs_ker(x)[0] 
    else:
        return np.inf

def findM(qs_ker, d_ker, prep_state):
    """ Find M that maximises the d/q ratio """
    if prep_state == '0':
        x_scan = np.linspace(0.95, 1, 1000)
    elif prep_state == '1':
        x_scan = np.linspace(0, 0.05, 1000)
    else:
        x_scan = np.linspace(0, 1, 1000)
    
    prior_density = qs_ker(x_scan)
    threshold = 0.01 * np.max(prior_density)
    valid_indices = np.where(prior_density > threshold)[0]
    
    if len(valid_indices) > 1:
        lower_bound = x_scan[valid_indices[0]]
        upper_bound = x_scan[valid_indices[-1]]
        bds = (lower_bound, upper_bound)
    else:
        bds = (0.0, 1.0) 

    ys = np.array([dq(x, qs_ker, d_ker) for x in x_scan])
    
    # Safe argmin
    if len(valid_indices) > 0:
        valid_ys = ys[valid_indices]
        valid_xs = x_scan[valid_indices]
        x0 = valid_xs[np.argmin(valid_ys)]
    else:
        x0 = 0.5

    # Run Optimizer
    try:
        res = minimize(dq, (x0,), args=(qs_ker, d_ker), bounds=(bds,), method='L-BFGS-B')
        
        # Plotting (Optional - can comment out for speed)
        plt.figure(figsize=(width,height), dpi=100)
        plt.plot(1 - x_scan, ys, c='crimson', lw=2)
        plt.title("-Data/Prior PDFs")
        plt.show()
        
        return -res.fun[0] if isinstance(res.fun, np.ndarray) else -res.fun, res.x[0] if isinstance(res.x, np.ndarray) else res.x
    except:
        return 1.0, x0 # Fallback

def QoI_single(lambdas, prep_state='0'):
    if prep_state == '0':
        return lambdas 
    elif prep_state == '1':
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
    np.random.seed(seed)
    
    # Determine Prior Mean from Params
    if prep_state == '0':
        prior_mean = 1.0 - params[interested_qubit].get('pm1p0', 0.05)
    else:
        prior_mean = 1.0 - params[interested_qubit].get('pm0p1', 0.05)

    if prior_mean > 1.0 or prior_mean < 0.7: prior_mean = 0.95

    # Generate Priors (Success Rates)
    success_rates = tnorm01(prior_mean, prior_sd, size=M)

    # Map to Observable
    if prep_state == '0':
        prob_meas_0 = success_rates
    elif prep_state == '1':
        prob_meas_0 = 1.0 - success_rates
    else:
        raise ValueError('prep_state must be "0" or "1".')
    
    qs = QoI_single(success_rates, prep_state=prep_state)

    # KDE (Protected)
    try:
        d_ker = ss.gaussian_kde(d)
        qs_ker = ss.gaussian_kde(qs)
    except Exception as e:
        print(f"   KDE Failed (Data likely singular): {e}")
        # Fallback: Return original priors if we can't update them
        return success_rates, success_rates

    print(f'Given Lambda |{prep_state}>: success rate = {prior_mean:.4f}')

    # Optimization
    max_r, max_q = findM(qs_ker, d_ker, prep_state)

    # Rejection Sampling
    r_vals = d_ker(qs) / qs_ker(qs)
    eta = r_vals / max_r 
    accept_mask = eta > np.random.uniform(0, 1, M)
    post_success_rates = success_rates[accept_mask]

    # Handle case where rejection sampling rejects everything
    if len(post_success_rates) == 0:
        print("   Warning: Rejection sampling rejected all points. Returning priors.")
        return success_rates, success_rates

    print('   Accepted N: %d (%.1f%%)' % (len(post_success_rates), 100*len(post_success_rates)/M))
    
    # Save
    filename = file_address + f'Post_Qubit{interested_qubit}.csv'
    np.savetxt(filename, post_success_rates, delimiter=',')

    return success_rates, post_success_rates


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
    def __init__(self, qubit_order, home_dir=None, data=None, file_address='data/', device=None):
        if home_dir:
            self.home_dir = home_dir
            try:
                os.chdir(home_dir)
            except FileNotFoundError:
                print(f"Warning: home_dir {home_dir} not found.")

        self.file_address = file_address
        self.qubit_order = qubit_order
        
        self.prior = {}
        self.post_marginals = {f'Qubit{q}': {'0': np.array([]), '1': np.array([])} for q in qubit_order}
        posterior0path = Path(os.path.join(self.file_address, 'State0_Post_Qubit' + str(self.qubit_order[0]) + '.csv'))
        posterior1path = Path(os.path.join(self.file_address, 'State1_Post_Qubit' + str(self.qubit_order[0]) + '.csv'))
        if (posterior0path.is_file() and posterior1path.is_file()):
            self._load_posterior_marginals()
        self.params = None if device is None else get_braket_calibration_dict(device, n_qubits=len(qubit_order))
        self.inv_matrices_mean = []
        self.inv_matrices_mode = []
        if self.post_marginals[f'Qubit{self.qubit_order[0]}']['0'].size > 0 and self.post_marginals[f'Qubit{self.qubit_order[0]}']['1'].size > 0:
            self.create_filter_mat()
            self.post = {f'{q}': {'e0_mean': float(np.mean(self.post_marginals[f'Qubit{q}']['0'])), 'e1_mean': float(np.mean(self.post_marginals[f'Qubit{q}']['1'])),
                                       'e0_mode': float(ss.mode(self.post_marginals[f'Qubit{q}']['0'])[0]), 'e1_mode': float(ss.mode(self.post_marginals[f'Qubit{q}']['1'])[0]),
                                       } for q in qubit_order}
        

        self.data = {'0': np.array([]), '1': np.array([])}
        if data is not None:
            if isinstance(data, dict):
                self.data = data
            else:
                self.data['0'] = np.atleast_1d(data)
        else:
            self._load_data_from_files()

    def _load_posterior_marginals(self):
        for prep_state in ['0', '1']:
            for q in self.qubit_order:
                path = os.path.join(self.file_address, f'State{prep_state}_Post_Qubit{q}.csv')
                try:
                    self.post_marginals[f'Qubit{q}'][prep_state] = pd.read_csv(path, header=None, dtype=float).values.flatten()
                    print(f"Loaded {len(self.post_marginals[f'Qubit{q}'][prep_state])} Posterior Values for Qubit {q}, State {prep_state}.")
                except Exception as e:
                    print(f"Error reading {path}: {e}")

    def _load_data_from_files(self):
        # State 0
        path0 = os.path.join(self.file_address, 'State0.csv')
        if os.path.exists(path0):
            try:
                self.data['0'] = pd.read_csv(path0, header=None, dtype=str).values.flatten()
                print(f"Loaded {len(self.data['0'])} shots for State 0.")
            except Exception as e:
                print(f"Error reading {path0}: {e}")
        else:
            print(f"Warning: {path0} not found.")

        # State 1
        p1 = os.path.join(self.file_address, 'State1.csv')
        if os.path.exists(p1):
            try:
                self.data['1'] = pd.read_csv(p1, header=None, dtype=str).values.flatten()
                print(f"Loaded {len(self.data['1'])} shots for State 1.")
            except Exception as e:
                print(f"Error reading {p1}: {e}")
        else:
            print(f"Warning: {p1} not found.")

    def inference(self,
                  nPrior=40000,
                  prior_sd=0.1,
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
        prior_sd : float, optional
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
        # Select Data
        current_data = self.data.get(prep_state, np.array([]))
        total_shots = len(current_data)

        if total_shots == 0:
             print(f"CRITICAL: No data for prep_state='{prep_state}'. Skipping.")
             return

        # Robust Grouping Calculation
        num_points = int(total_shots / shots_per_point)
        # Ensure at least 20 points for KDE
        if num_points < 20:
            print(f"   Notice: Adjusting grouping. (Original points: {num_points})")
            num_points = 20
            # Fallback if total data is tiny
            if total_shots < 20: 
                num_points = total_shots

        # Trim data for perfect division
        remainder = total_shots % num_points
        if remainder != 0:
            valid_data = current_data[:-remainder]
        else:
            valid_data = current_data
            
        print(f"   Debug: Using {len(valid_data)} shots split into {num_points} points for KDE.")

        # Setup Params
        if self.params is None:
            self.params = {}
            for q in self.qubit_order:
                self.params[q] = {'pm1p0': 0.005, 'pm0p1': 0.005}

        # Inference Loop
        for i in self.qubit_order:
            print(f'Inferring Qubit {i} for State |{prep_state}>')
            
            try:
                d = getData0(valid_data, num_points, i)
            except Exception as e:
                print(f"   Error in getData0: {e}")
                continue

            if len(d) < 5:
                print(f"   Error: Resulting dataset too small (len={len(d)}). Skipping Qubit {i}.")
                continue

            state_prefix = self.file_address + f"State{prep_state}_"
            
            try:
                prior_lambdas, post_lambdas = output(
                    d, i, nPrior, self.params, prior_sd, 
                    seed=seed, file_address=state_prefix, prep_state=prep_state
                )
                
                self.prior[f'Qubit{i}'] = prior_lambdas
                self.post_marginals[f'Qubit{i}'][prep_state] = post_lambdas
            except Exception as e:
                print(f"   Inference failed for Qubit {i}: {e}")
                
        # Matrix Creation (Safe Check)
        first_q = f'Qubit{self.qubit_order[0]}'
        has_0 = len(self.post_marginals[first_q]['0']) > 0
        has_1 = len(self.post_marginals[first_q]['1']) > 0
        
        if has_0 and has_1:
            self.create_filter_mat()

    def create_filter_mat(self):
        self.inv_matrices_mean = []
        self.inv_matrices_mode = []

        for q in self.qubit_order:
            q_key = f'Qubit{q}'
            res_0 = self.post_marginals[q_key]['0']
            res_1 = self.post_marginals[q_key]['1']
            
            # Check for empty data before calculating mean
            if len(res_0) == 0 or len(res_1) == 0:
                print(f"Warning: Skipping Matrix for Qubit {q} (Incomplete Inference)")
                self.inv_matrices_mean.append(np.eye(2))
                self.inv_matrices_mode.append(np.eye(2))
                continue

            # MEAN
            lam0_mean = safe_mean(res_0)
            lam1_mean = safe_mean(res_1)
            A_mean = np.array([[lam0_mean, 1 - lam1_mean], [1 - lam0_mean, lam1_mean]])
            try:
                self.inv_matrices_mean.append(np.linalg.inv(A_mean))
            except:
                self.inv_matrices_mean.append(np.eye(2))

            # MODE
            lam0_mode = find_mode(res_0)
            lam1_mode = find_mode(res_1)
            A_mode = np.array([[lam0_mode, 1 - lam1_mode], [1 - lam0_mode, lam1_mode]])
            try:
                self.inv_matrices_mode.append(np.linalg.inv(A_mode))
            except:
                self.inv_matrices_mode.append(np.eye(2))

    def mean(self):
        res = {}
        for q in self.qubit_order:
            q_key = f'Qubit{q}'
            m0 = safe_mean(self.post_marginals[q_key]['0'])
            m1 = safe_mean(self.post_marginals[q_key]['1'])
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
    
    def get_inverse_element(self, target_bitstring, source_bitstring, method = 'mean'):
        """
        Calculates the probability transition factor from Source (Noisy) -> Target (Clean).
        Mathematically: returns the element (Row=Target, Col=Source) of the Inverse Matrix.
        
        Args:
            target_bitstring (str): The 'clean' state we are calculating probability for.
            source_bitstring (str): The 'noisy' state we actually measured.
        """
        # Ensure we have the matrices
        if not self.inv_matrices_mean:
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
            if method == 'mean':
                elem = self.inv_matrices_mean[i][t_val, s_val]
            elif method == 'mode':
                elem = self.inv_matrices_mode[i][t_val, s_val]
            else:
                raise ValueError("method must be 'mean' or 'mode'")
            
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
    

    def eff_DeNoise(self, datadict, method = 'mean', percentage=100, verbose=True, GD = False, lr=0.1, max_iter=50):
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
                weight = self.get_inverse_element(t_str, s_str, method = method) # M_inv(target_bitstring=t_str, source_bitstring=s_str)
                
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
