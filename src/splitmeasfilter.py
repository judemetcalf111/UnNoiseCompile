# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 13:18:38 2025

@author: Jude L. Metcalf
"""
from datetime import datetime
import glob
import json
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


# Optional: Apply theme if available
try:
    theme = load_theme("arctic_light")
    theme.apply()
    theme.apply_transforms()
except:
    pass

width = 6.72 
height = 4.15 



# --- Helper Functions ---

class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return super(NumpyEncoder, self).default(o)

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

def getData0(data, num_group, interested_qubit_index):
    """ Get the probabilities of measuring 0 from binary readouts
        **Binary number follows little-endian convention**

    Parameters
    ----------
    data : array_like
        An array of binary readout strings (e.g., ['001', '010']).
    num_group : int
        Number of groups to split the data into for probability calculation.
        Data length must be divisible by num_group.
    interested_qubit_index : int
        The index of the qubit to analyze (0-indexed).

    Returns
    -------
    prob0 : numpy array
        Array of probabilities of measuring 0 for each group.

    """
    target_idx = interested_qubit_index
    
    # Fallback if qubit index out of bounds
    if target_idx < 0: target_idx = 0 

    data_list = data[:,interested_qubit_index]

    # Reshape and Average
    total_len = len(data_list)
    if num_group > total_len or num_group <= 0:
        # Prevent zero division or impossible reshape
        return np.array(data_list)

    trunc_len = total_len - (total_len % num_group)
    
    if trunc_len == 0:
        return np.array([])
        
    arr = np.array(data_list[:trunc_len])
    grouped = arr.reshape(num_group, -1)
    
    # Calculate mean (probability of being 0)
    mean_of_entries = grouped.mean(axis=1)

    prob0 = 1 - mean_of_entries
    
    return prob0

def dq(x, qs_ker, d_ker):
    if np.abs(qs_ker(x)[0]) > 1e-6 and np.abs(d_ker(x)[0]) > 1e-6:
        return - d_ker(x)[0] / qs_ker(x)[0] 
    else:
        return np.inf

def findM(qs_ker, d_ker):
    """ Find M that maximises the d/q ratio for rejection sampling """
    
    # RECOMMENDED: Increase scan range. 0.1 is too narrow for high-error qubits.
    x_scan = np.linspace(0, 0.5, 2000) 
    
    prior_density = qs_ker(x_scan)
    threshold = 0.01 * np.max(prior_density)
    
    # Get indices where prior is significant
    valid_indices = np.where(prior_density > threshold)[0]
    
    if len(valid_indices) > 1:
        # Calculate -Ratio
        ys = np.array([dq(x_scan[i], qs_ker, d_ker) for i in valid_indices])
        res = np.min(ys) # This is the most negative value (e.g., -50)
        # Plot the rejection sampling diagnostic
        plt.figure(figsize=(8, 5))
        plt.plot(x_scan[valid_indices], ys, label='−d/q ratio', color='blue', alpha=0.7)
        plt.axhline(res, color='red', linestyle='--', label='Threshold (1% of max)')
        plt.xlabel('Error rate')
        plt.title('Rejection Sampling Diagnostic, -data/prior')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        plt.figure(figsize=(8, 5))
        plt.plot(x_scan, prior_density, label='Prior density', color='red', alpha=0.7)
        plt.xlabel('Error rate')
        plt.title('Prior Distribution')
        plt.grid(True, alpha=0.3)
        plt.show()

    else:
        raise Exception


    
    # FIX IS HERE: Return the positive magnitude (e.g., 50)
    return -res

def meas_data_readout(qubit, prep_state, datafile: str = ''):
    """
    Function to readout json files or suitable numpy arrays
    """

    if type(qubit) != int:
        raise Exception("Must supply an integer label for the interested qubit in `meas_data_readout()`") 

    if datafile.endswith('.json'):
        with open(str(datafile), 'r') as file:
            data = json.load(file)
        qubit_props = data['oneQubitProperties']
        
        epsilon01 = qubit_props[str(qubit)]['oneQubitFidelity'][1]['fidelity'] # Notice that even though it says "fidelity", we get error rate...
        epsilon10 = qubit_props[str(qubit)]['oneQubitFidelity'][2]['fidelity'] # Notice that even though it says "fidelity", we get error rate...

    elif not datafile:
        raise Exception("Error: No data or datafile provided for the `meas_data_readout()`")
    else:
        raise Exception("Must either:\nsupply datafile as path string to a json data file, i.e. '../data/datafile.json'\nor supply a nx3 numpy array with 0 -> 1 errors, 1 -> 0 errors, and gate errors in each row (ONLY AVAILABLE FOR SINGLE QUBIT GATES)")

    if prep_state == '0':
        return epsilon01
    else:
        return epsilon10

class SplitMeasFilter:
    """Measurement error filter.

    Attributes:
        qubit_order: array,
          using order given by the Braket [q0, q1, ...].
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
    def __init__(self, qubit_order, load_data=True, home_dir=None, data=None, file_address='data/', device=None):
        if home_dir:
            self.home_dir = home_dir
            try:
                os.chdir(home_dir)
            except FileNotFoundError:
                print(f"Warning: home_dir {home_dir} not found.")

        self.file_address = file_address
        self.qubit_order = [int(q) for q in qubit_order]
        
        self.prior = {f'Qubit{q}': np.array([]) for q in qubit_order}
        self.post_full = {f'Qubit{q}': {'0': np.array([]), '1': np.array([])} for q in self.qubit_order}
        self.post = {f'Qubit{q}': {} for q in self.qubit_order}
        full_post_path = Path(os.path.join(self.file_address, 'Post_Full_Current.json'))
        
        if full_post_path.is_file() and load_data == True:
            print(f"Loading historical data from {full_post_path}...")
            try:
                with open(full_post_path, 'r') as f:
                    loaded_data = json.load(f)
                
                # RECONSTRUCT NUMPY ARRAYS
                # JSON loads lists, but code expects numpy arrays (e.g. for .size or .mean)
                for q_key, states in loaded_data.items():
                    if q_key in self.post_full:
                        self.post_full[q_key]['0'] = np.array(states['0'])
                        self.post_full[q_key]['1'] = np.array(states['1'])
            except Exception as exc:
                print(f"Failed to load JSON data: {exc}")

        # Device calibration loading
        self.params = None #if device is None else get_braket_calibration_dict(device, n_qubits=len(qubit_order))
        self.inv_matrices_mean = []
        self.inv_matrices_mode = []

        # Check if data was loaded successfully (checks the first qubit's size)
        last_q = f'Qubit{self.qubit_order[-1]}'
        if self.post_full[last_q]['0'].size > 0 and self.post_full[last_q]['1'].size > 0:
            self.create_filter_mat() 
            
            # Recalculate Mean/Mode from the loaded full data
            self.post = {f'Qubit{q}': {
                'e0_mean': float(np.mean(self.post_full[f'Qubit{q}']['0'])), 
                'e1_mean': float(np.mean(self.post_full[f'Qubit{q}']['1'])),
                'e0_mode': float(ss.mode(self.post_full[f'Qubit{q}']['0'])[0]), 
                'e1_mode': float(ss.mode(self.post_full[f'Qubit{q}']['1'])[0]),
            } for q in self.qubit_order}

        self.data = {'0': np.array([]), '1': np.array([])}
        if data is not None:
            if isinstance(data, dict):
                self.data = data
            else:
                self.data['0'] = np.atleast_1d(data)
        else:
            self._load_data_from_files() 
            pass

    def output(self,
            d,
            qubit,
            M,
            params,
            prior_sd,
            seed=127,
            State_Data_address='',
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
            {'err0': float # Pr(Meas. 1| Prep. 0)
            'err1': float # Pr(Meas. 0| Prep. 1)
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
        qubit = int(qubit)
        # Determine Prior Mean from Params
        try:
            if self.params:
                if prep_state == '0':
                    prior_mean = params[f'Qubit{qubit}'].get('err0', 0.05)
                else:
                    prior_mean = params[f'Qubit{qubit}'].get('err1', 0.05)
            else:
                prior_mean = meas_data_readout(qubit=qubit, prep_state=prep_state, datafile=os.path.join(self.file_address, 'Braket_Qubit_Calibration.json'))
                print(f"From the Braket Calibration, we have a prior error rate of: {prior_mean}")
        except Exception as e:
            print("Either didn't provide 'params = {...}' (not recommended)\n" 
                  f"and no file named {os.path.join(self.file_address, 'Braket_Qubit_Calibration.json')} located in the {self.file_address} directory: {e}")
            prior_mean = 0.05
            

        # Generate Priors (Success Rates)
        prior_error_rates = tnorm01(prior_mean, prior_sd, size=M)
        qs = prior_error_rates

        if prep_state == '0':
            data_errors = 1 - d
        else:
            data_errors = d
        try:
            d_ker = ss.gaussian_kde(data_errors)
            qs_ker = ss.gaussian_kde(qs)
        except Exception as e:
            print(f"   KDE Failed (Data likely singular): {e}")
            # Fallback: Return original priors if we can't update them
            return prior_error_rates, prior_error_rates

        print(f'Given Lambda |{prep_state}>: prior error rate = {prior_mean:.4f}')

        # Optimization
        max_r = findM(qs_ker, d_ker)

        # Rejection Sampling
        r_vals = d_ker(qs) / qs_ker(qs)
        eta = r_vals / max_r 
        accept_mask = eta > np.random.uniform(0, 1, M)
        post_error_rates = prior_error_rates[accept_mask]

        # Handle case where rejection sampling rejects everything
        if len(post_error_rates) == 0:
            print("   Warning: Rejection sampling rejected all points. Returning priors.")
            return prior_error_rates, prior_error_rates

        print('   Accepted N: %d (%.1f%%)' % (len(post_error_rates), 100*len(post_error_rates)/M))
        


        # Save
        filename = State_Data_address + f'Post_Qubit{qubit}.csv'
        np.savetxt(filename, post_error_rates, delimiter=',')

        return prior_error_rates, post_error_rates

    def _load_posterior_marginals(self):
        for prep_state in ['0', '1']:
            for q in self.qubit_order:
                path = os.path.join(self.file_address, f'State{prep_state}_Post_Qubit{q}.csv')
                try:
                    self.post_full[f'Qubit{q}'][prep_state] = pd.read_csv(path, header=None, dtype=float).values.flatten()
                    print(f"Loaded {len(self.post_full[f'Qubit{q}'][prep_state])} Posterior Values for Qubit {q}, State {prep_state}.")
                except Exception as e:
                    print(f"Error reading {path}: {e}")

    def _load_data_from_files(self):
        # State 0
        path0 = os.path.join(self.file_address, 'State0.csv')
        if os.path.exists(path0):
            try:
                self.data['0'] = np.loadtxt(path0, dtype = float,delimiter=',')
                print(f"Loaded {np.size(self.data['0'],axis=0)} shots for State 0.")
            except Exception as e:
                print(f"Error reading {path0}: {e}")
        else:
            print(f"Warning: {path0} not found.")

        # State 1
        path1 = os.path.join(self.file_address, 'State1.csv')
        if os.path.exists(path1):
            try:
                self.data['1'] = np.loadtxt(path1, dtype = float,delimiter=',')
                print(f"Loaded {np.size(self.data['1'],axis=0)} shots for State 1.")
            except Exception as e:
                print(f"Error reading {path1}: {e}")
        else:
            print(f"Warning: {path1} not found.")

    def inference(self,
                  num_points,
                  nPrior=40000,
                  prior_sd=0.1,
                  seed=227,
                  prep_state='0'):
        """
          Do Bayesian interence

        Parameters
        ----------
        num_points : int
            Number of samples to use in the kernel density estimation.
            Default not given, as this is an important parameter,
            to be considered based on error rate magnitude and shot number.
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
        total_shots = np.size(current_data,axis=0)

        if total_shots == 0:
             print(f"CRITICAL: No data for prep_state='{prep_state}'. Skipping.")
             return

        # Robust Grouping Calculation
        # Ensure at least 20 points for KDE
        if num_points < 16:
            print(f"-----------------------------------\n"
                    f"Notice: Consider Adjusting grouping to ensure good sample size. num_points > 16 is recommended, whereas {num_points} is given\n"
                    f"-----------------------------------")
            # Fallback if total data is tiny
            if total_shots < 20: 
                num_points = total_shots

        # Trim data for perfect division
        remainder = total_shots % num_points
        if remainder != 0:
            valid_data = current_data[:-remainder]
        else:
            valid_data = current_data
            
        print(f"   Using {np.size(valid_data, axis=0)} shots split into {num_points} points for KDE.")

        # Inference Loop
        for idx,q in enumerate(self.qubit_order):                       # The qubit array order giving the index is given by the aws bracket as `task.result().measurement_qubits`
            print(f'Inferring Qubit {q} for State |{prep_state}>')
            
            try:
                d = getData0(valid_data, num_points, idx)
            except Exception as e:
                print(f"   Error in getData0: {e}")
                continue

            if len(d) < 5:
                print(f"   Error: Resulting dataset too small (len={len(d)}). Skipping Qubit {q}.")
                continue
            

            Posterior_dir = os.path.join(self.file_address, "Posterior Data")
            os.makedirs(Posterior_dir, exist_ok=True)
            state_prefix = os.path.join(Posterior_dir, f"State{prep_state}_")
            
            try:
                prior_lambdas, post_lambdas = self.output(
                    d, q, nPrior, self.params, prior_sd, 
                    seed=seed, State_Data_address=state_prefix, prep_state=prep_state
                )                
                self.prior[f'Qubit{q}'] = prior_lambdas
                self.post_full[f'Qubit{q}'][prep_state] = post_lambdas
            except Exception as e:
                print(f"   Inference failed for Qubit {q}: {e}")
                
        # Matrix Creation (Safe Check)
        last_q = f'Qubit{self.qubit_order[-1]}'
        has_0 = len(self.post_full[last_q]['0']) > 0
        has_1 = len(self.post_full[last_q]['1']) > 0
        
        if has_0 and has_1:
            self.create_filter_mat()

        if has_0:
            for q in self.qubit_order: self.post[f'Qubit{q}']['e0_mean'] = float(np.mean(self.post_full[f'Qubit{q}']['0']))
            for q in self.qubit_order: self.post[f'Qubit{q}']['e0_mode'] = float(ss.mode(self.post_full[f'Qubit{q}']['0'])[0]) 
        elif has_1:
            for q in self.qubit_order: self.post[f'Qubit{q}']['e1_mean'] = float(np.mean(self.post_full[f'Qubit{q}']['1']))
            for q in self.qubit_order: self.post[f'Qubit{q}']['e1_mode'] = float(ss.mode(self.post_full[f'Qubit{q}']['1'])[0]) 

        print("Inference Complete, Saving Results...")

        # Update extensions to .json
        full_path_datetime = os.path.join(self.file_address, f'Post_Full_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        full_path_current = os.path.join(self.file_address, 'Post_Full_Current.json')
        meanmode_path_datetime = os.path.join(self.file_address, f'Post_MeanMode_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        meanmode_path_current = os.path.join(self.file_address, 'Post_MeanMode_Current.json')

        # Save using the NumpyEncoder
        with open(full_path_datetime, 'w') as f:
            json.dump(self.post_full, f, cls=NumpyEncoder)

        with open(meanmode_path_datetime, 'w') as f:
            json.dump(self.post, f, cls=NumpyEncoder)

        with open(full_path_current, 'w') as f:
            json.dump(self.post_full, f, cls=NumpyEncoder)

        with open(meanmode_path_current, 'w') as f:
            json.dump(self.post, f, cls=NumpyEncoder)

        print(f"Saved Full Posterior to:\n{full_path_datetime}\nand\n{full_path_current}")
        print(f"Saved Mean/Mode Summary to:\n{meanmode_path_datetime}\nand\n{meanmode_path_current}")
        
        self.error_distributions(plotting=True, save_plots=True, num_points=num_points)

    def create_filter_mat(self):
        self.inv_matrices_mean = []
        self.inv_matrices_mode = []

        for q in self.qubit_order:
            q_key = f'Qubit{q}'
            res_0 = self.post_full[q_key]['0']
            res_1 = self.post_full[q_key]['1']
            
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
            m0 = safe_mean(self.post_full[q_key]['0'])
            m1 = safe_mean(self.post_full[q_key]['1'])
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
        if not hasattr(self, 'post_full'):
            raise RuntimeError("Filter not calibrated. Run inference() first.")

        prob = 1.0
        # Iterate qubits (Little Endian: String[-1] is Qubit 0)
        for i, q in enumerate(self.qubit_order):
            obs_bit = int(observed_bitstring[-(i+1)]) # Row Index
            hid_bit = int(hidden_bitstring[-(i+1)])   # Col Index
            
            q_key = f'Qubit{q}'
            
            # Retrieve single-qubit error rates from calibration
            # P(Meas 0 | Prep 0) = Mean of Post Lambda 0
            p0_g_0 = safe_mean(self.post_full[q_key]['0'])
            # P(Meas 1 | Prep 1) = Mean of Post Lambda 1
            p1_g_1 = safe_mean(self.post_full[q_key]['1'])
            
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
        

    def error_distributions(self, plotting=True, save_plots=False, num_points=32):
            """
            Calculates statistics and (optionally) plots the posterior distribution 
            of measurement errors for each qubit.
            
            Returns:
                stats (dict): Dictionary containing mean, std, and 95% Confidence Intervals
                            for both error_0 (p(1|0)) and error_1 (p(0|1)).
            """
            stats = {}

            for idx,q in enumerate(self.qubit_order):
                q_key = f'Qubit{q}'
                
                # Success Rates (Lambdas)
                # These are samples of "Probability of measuring Correctly"
                lam0_samples = self.post_full[q_key]['0'] if '0' in self.post_full[q_key] else np.array([])
                lam1_samples = self.post_full[q_key]['1'] if '1' in self.post_full[q_key] else np.array([])
                
                if (not lam0_samples.any()) and (not lam1_samples):
                    print(f"Skipping Qubit {q}: Inference not complete.")
                    continue
                elif (not lam0_samples.any()) or (not lam1_samples.any()):
                    print("Partial Inference complete, continuing")

                # Convert to Error Rates (Epsilons)
                # Error = 1 - Success
                if lam0_samples.any():
                    err0_samples = lam0_samples # Prob(Meas 1 | Prep 0)
                else:
                    err0_samples = np.array([])

                if lam1_samples.any():
                    err1_samples = lam1_samples # Prob(Meas 0 | Prep 1)
                else:
                    err1_samples = np.array([])

                # Calculate Statistics
                stats[q_key] = {}
                if err0_samples.any():
                    stats[q_key]['err0_mean'] = np.mean(err0_samples)
                    stats[q_key]['err0_std'] = np.std(err0_samples)
                    stats[q_key]['err0_95_CI'] = np.percentile(err0_samples, [2.5, 97.5])

                if err1_samples.any():
                    stats[q_key]['err1_mean'] = np.mean(err1_samples)
                    stats[q_key]['err1_std'] = np.std(err1_samples)
                    stats[q_key]['err1_95_CI'] = np.percentile(err1_samples, [2.5, 97.5])
                
                # Plotting (KDE + Histogram)
                if plotting:
                    plt.figure(figsize=(8, 5), dpi=100)
                    xs = np.linspace(0, 0.1, 1000)
                    
                    if err0_samples.any():
                        # Plot Error 0 (Readout error on |0>)
                        kde0 = ss.gaussian_kde(err0_samples)
                        plt.plot(xs, kde0(xs), color='blue', label=r'$P(1|0)$ (Error on 0)')
                        plt.fill_between(xs, kde0(xs), alpha=0.2, color='blue')

                        # Plot Data KDE for comparison
                        d0 = getData0(self.data.get('0', np.array([])), num_points, idx)
                        err0 = 1 - d0
                        if len(d0) > 0:
                            d0_ker = ss.gaussian_kde(err0)
                            plt.plot(xs, d0_ker(xs), color='darkslateblue', linestyle='--', label='Data KDE (State |0>)')
                            plt.fill_between(xs, d0_ker(xs), alpha=0.1, color='darkslateblue')

                    if err1_samples.any():
                        # Plot Error 1 (Readout error on |1>)
                        kde1 = ss.gaussian_kde(err1_samples)
                        plt.plot(xs, kde1(xs), color='red', label=r'$P(0|1)$ (Error on 1)')
                        plt.fill_between(xs, kde1(xs), alpha=0.2, color='red')

                        # Plot Data KDE for comparison
                        d1 = getData0(self.data.get('1', np.array([])), num_points, idx)
                        err1 = d1
                        if len(d1) > 0:
                            d1_ker = ss.gaussian_kde(err1)
                            plt.plot(xs, d1_ker(xs), color='firebrick', linestyle='--', label='Data KDE (State |1>)')
                            plt.fill_between(xs, d1_ker(xs), alpha=0.1, color='firebrick')

                    plt.title(f'Posterior Error Distributions - Qubit {q}')
                    plt.xlabel('Error Rate')
                    plt.ylabel('Density')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.xlim([0,0.1])
                    
                    if save_plots:
                        plt.savefig(self.file_address + f'ErrorDist_Qubit{q}.pdf')
                    
                    plt.show()
                    
                    # Print Summary
                    print(f"--- Qubit {q} Summary ---")
                    if err0_samples.any():
                        print(f"Error on |0>: {stats[q_key]['err0_mean']:.4f} "
                            f"(95% CI: {stats[q_key]['err0_95_CI'][0]:.4f} - {stats[q_key]['err0_95_CI'][1]:.4f})")
                    if err1_samples.any():
                        print(f"Error on |1>: {stats[q_key]['err1_mean']:.4f} "
                            f"(95% CI: {stats[q_key]['err1_95_CI'][0]:.4f} - {stats[q_key]['err1_95_CI'][1]:.4f})")
                    print("-" * 30)

            return stats
