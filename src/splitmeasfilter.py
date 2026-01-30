"""
Created on Wed Dec 31 13:18:38 2025

@author: Jude L. Metcalf
"""

# Imports
from datetime import datetime
import json
import os
from pathlib import Path

import pandas as pd
import numpy as np
import scipy.stats as ss

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from aquarel import load_theme


# Optional: Apply theme if available
try:
    theme = load_theme("arctic_light")
    theme.apply()
    theme.apply_transforms()
except:
    pass


# --- Helper Functions ---
# Helper Class to encode numpy objects as standard python objects (i.e. lists, floats, ints)
class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return super(NumpyEncoder, self).default(o)

def tnorm01(center:float, sd:float, size:int=1) -> np.ndarray:
    """Generate random numbers for truncated normal with range [0,1]

    Args:
      center:float
        mean of normal distribution
      sd:float
        standard deviation of normal distribution
      size:int
        number of random numbers

    Returns: np.ndarray
       an array of random numbers
    """ 
    upper = 1.
    lower = 0.
    if sd == 0: return np.full(size, center)
    a, b = (lower - center) / sd, (upper - center) / sd
    return np.array(ss.truncnorm.rvs(a, b, size=size) * sd + center)

def find_mode(data:np.ndarray) -> float:
    """Find the mode through Gaussian KDE.

    Args:
      data: array
        an array of floats

    Returns: float
      the mode
    """
    if len(data) < 2: return float(np.mean(data))
    kde = ss.gaussian_kde(data)
    line = np.linspace(min(data), max(data), 1000)
    return line[np.argmax(kde(line))]

def safe_mean(data:np.ndarray) -> float:
    """ Safely compute the mean, returning 0.0 if empty to avoid crashes. """
    if len(data) == 0: return 0.0
    return float(np.mean(data))

def dict_filter(data_dict:dict, percent:float = 99.0) -> dict:
    """Filters a dictionary to retain entries that make up the top x% of total counts, the default set to 99%.
    Built to ensure scalability, as this will only track at most N

    Args:
        data_dict: dict
            The input dictionary with counts.
        percent: float
            The percentage (0-100) threshold to retain (default is 99).

    Returns:
        dict: A new dictionary containing only the top 99% of entries.
    """
    # Create sum of entries to get total shot count
    total_sum = sum(data_dict.values())
    if total_sum == 0: return {}

    # Sort items into array of tuple pairs ordered by values [('00',102),('11',90),...]
    sorted_items = sorted(data_dict.items(), key=lambda item: item[1], reverse=True)
    filtered_dict = {}
    cumulative_sum = 0

    # Allow percentage range and [0-1] range
    if percent > 1.0: percent /= 100.0; print(f"'percent' interpreted as a [0-1] value, taken as {percent:.2g}%")
    threshold = percent * total_sum
    for key, value in sorted_items:
        if cumulative_sum < threshold:
            filtered_dict[key] = value
            cumulative_sum += value
        else:
            # If we reach more than the threshold, break, in order to remain scalable and only track the top x percentage of 
            break
    return filtered_dict

def getData0(data:np.ndarray, interested_qubit_index:int) -> float:
    """Get the probabilities of measuring 0 from bitstring measurements

    Parameters
    ----------
    data : np.ndarray
        A 2D array of the bitstrings, columns defining qubit number, rows defining shot
          (e.g., [0 0
                  1 2]).
    interested_qubit_index : int
        The index of the qubit to analyze (different from Qubit number, defined by measurement order).

    Returns
    -------
    prob0 : float
        float of the probability of measuring 0 for each group, in range [0,1].

    """
    target_idx = interested_qubit_index
    
    # Fallback if qubit index out of bounds
    if target_idx < 0: target_idx = 0 

    try:
        data_list = data[:,interested_qubit_index]
    except Exception as ex:
        raise(ex)

    mean_of_entries = np.mean(data_list)

    prob0 = float(1 - mean_of_entries)
    
    return prob0

def dq(x, qs_ker, d_pdf):
    """Calculates the ratio of the prior pdf to the data kernel safely at some point 'x'"""
    if np.abs(qs_ker(x)[0]) > 1e-6 and np.abs(d_pdf(x)) > 1e-6:
        return - d_pdf(x) / qs_ker(x)[0]
    else:
        return np.inf

def findM(qs_ker, d_pdf):
    """Find M that maximises the d/q ratio for rejection sampling """
    
    # Find the min value in log-space (thus negative)
    x_scan = np.linspace(0, 0.1, 4000) 
    
    prior_density = qs_ker(x_scan)
    threshold = 0.01 * np.max(prior_density)
    
    # Get indices where prior is significant
    valid_indices = np.where(prior_density > threshold)[0]

    if len(valid_indices) > 1:
        # Calculate -Ratio
        ys = np.array([dq(x_scan[i], qs_ker, d_pdf) for i in valid_indices])

        res = np.min(ys) # This is the most negative value (e.g., -50)

        # Plot the rejection sampling diagnostic
        plt.figure(figsize=(8, 5))
        plt.plot(x_scan[valid_indices], ys, label='−d/q ratio', color='blue', alpha=0.7)
        plt.axhline(res, color='red', linestyle='--', label='Threshold (1% of max)')
        plt.xlabel('Error rate')
        plt.title('Rejection Sampling Diagnostic, -data/prior')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Plot the prior and data pdfs
        x_plot = np.linspace(0,0.2,2000)
        plot_prior_density = qs_ker(x_plot)
        plot_data_density = d_pdf(x_plot)
        plt.figure(figsize=(8, 5))
        plt.plot(x_plot, plot_prior_density, label='Prior Density', color='red', alpha=0.7)
        plt.fill_between(x_plot, plot_prior_density,color='red', alpha=0.1)
        plt.plot(x_plot, plot_data_density, label='Data Density', color='blue', alpha=0.7)
        plt.fill_between(x_plot, plot_data_density,color='blue', alpha=0.1)
        plt.xlabel('Error rate')
        plt.title('Prior and Data Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    else:
        raise Exception
    
    return -res # Return the maximiser of the ratio (minimiser of the negative ratio)

def meas_data_readout(qubit, prep_state:str = '', datafile:str = ''):
    """
    Function to readout json files or suitable numpy arrays
    """

    if type(qubit) != int:
        raise Exception("Must supply an integer label for the interested qubit in `meas_data_readout()`") 
    
    if not prep_state:
        raise Exception("Must supply a `prep_state` parameter to meas_data_readout()")

    # Read in the json datafile which is called /Braket_Qubit_Calibration.json located in the self.file_address directory
    if datafile.endswith('.json'):
        with open(str(datafile), 'r') as file:
            data = json.load(file)
        qubit_props = data['oneQubitProperties']

        try:
            zero_error_entry = qubit_props[str(qubit)]['oneQubitFidelity'][1]
            one_error_entry  = qubit_props[str(qubit)]['oneQubitFidelity'][2]
            if  ((zero_error_entry['fidelityType']['name'] == 'READOUT_ERROR_0_TO_1') and 
                 (one_error_entry['fidelityType']['name'] == 'READOUT_ERROR_1_TO_0')):
                
                epsilon01 = zero_error_entry['fidelity'] # Notice that even though it says "fidelity", we get error rate... Except for ankaa-3 where we do
                epsilon10 = one_error_entry['fidelity'] # Notice that even though it says "fidelity", we get error rate... Except for ankaa-3 where we do
            else:
                pass
            
        except:
            pass # Just means we didn't get the correct format!

        try: 
            readout_error_entry = qubit_props[str(qubit)]['oneQubitFidelity'][2]

            if readout_error_entry['fidelityType']['name'] == 'READOUT':
                epsilon01 = readout_error_entry['fidelity']
                epsilon10 = readout_error_entry['fidelity']
            else:
                raise Exception("Couldn't properly parse the Braket calibration file, which should be functional for IQM and Rigetti systems as of 29/01/2026")
        except:
            raise Exception("Couldn't properly parse the Braket calibration file, which should be functional for IQM and Rigetti systems as of 29/01/2026")


    elif not datafile:
        raise Exception("Error: No data or datafile provided for the `meas_data_readout()`")
    else:
        raise Exception("Must either:\nsupply datafile as path string to a json data file, i.e. '../data/datafile.json'\nor supply a nx3 numpy array with 0 -> 1 errors, 1 -> 0 errors, and gate errors in each row (ONLY AVAILABLE FOR SINGLE QUBIT GATES)")

    if prep_state == '0':
        epsilon = epsilon01
    else:
        epsilon = epsilon10
    
    if epsilon < 0.5:
        return epsilon
    elif epsilon >= 0.5:
        return 1 - epsilon
    else:
        raise Exception("Error in calculating the value of the error rate, either as a fidelity or an error rate.")

class SplitMeasFilter:
    """A scalable and accurate measurement error object, 
        designed to measure, infer, and mitigate measurement bitflip errors.

    Attributes:
        qubit_order: list,
          Using measurement order given by the Braket [q0, q1, ...].
        load_data: bool,
          Boolean value controlling if infered data are to be loaded back in.
          Set to False if intending to complete many inference runs 
        home_dir: str,
          String containing the 'home' of the inference project,
          location of where files, plots, etc. are to be created
        data: np.ndarray
          data containing calibrations. Not fully supported.
          Strongly recommended to use a .json saved as `Braket_Qubit_Calibration.json` in self.file_address
        file_address: string
          the address for reading and saving data, both raw braket readings, and posteriors.
          End with '/' if not empty.
        prior: dict
          dict containing all the qubits in qubit_order, with prior arrays as values
        post_full: dict
          dict containing a dicts of all the qubits in qubit_order, 
          with posterior arrays as values for '0' and '1' as keys
        post: dict
          dict containing a dicts of all the qubits in qubit_order, 
          with posterior mean and mode as values and keys:
          'e0_mean'
          'e0_mode'
          'e1_mean'
          'e1_mode'
        params: dict
          Parameter dict with 'err0' and 'err1' as keys, and errors as values.
          Non-standard, preferred to use a .json saved as `Braket_Qubit_Calibration.json` in self.file_address
        inv_matrices_mean: np.ndarray
          inverse matrix to mitigate measurement noise from mean of posterior distributions
        inv_matrices_mode: np.ndarray
          inverse matrix to mitigate measurement noise from mode of posterior distributions

    """
    def __init__(self, qubit_order:list, load_data:bool=True, home_dir:str='.', data:np.ndarray=np.array([]), file_address:str='data/'):
        if home_dir:
            self.home_dir = home_dir
            try:
                os.chdir(home_dir)
            except FileNotFoundError:
                print(f"Warning: home_dir {home_dir} not found.")

        self.file_address = file_address
        self.qubit_order = [int(q) for q in qubit_order]

        self.load_data = load_data
        
        self.prior = {f'Qubit{q}': np.array([]) for q in qubit_order}
        self.post_full = {f'Qubit{q}': {'0': np.array([]), '1': np.array([])} for q in self.qubit_order}
        self.post = {f'Qubit{q}': {} for q in self.qubit_order}
        full_post_path = Path(os.path.join(self.file_address, 'Post_Full_Current.json'))
        
        if full_post_path.is_file() and self.load_data == True: self._load_historical_data(full_post_path)
            
        # Device calibration loading
        self.params = None # Use must define, if they intend to not use the braket .json
        self.inv_matrices_mean = []
        self.inv_matrices_mode = []

        # Check if data was loaded successfully (checks the first qubit's size)
        last_q = f'Qubit{self.qubit_order[-1]}'
        if self.post_full[last_q]['0'].size > 0 and self.post_full[last_q]['1'].size > 0:
            self.create_filter_mat() 
            
            # Calculate mean/mode from the loaded full data
            self.post = {f'Qubit{q}': {
                'e0_mean': float(np.mean(self.post_full[f'Qubit{q}']['0'])), 
                'e1_mean': float(np.mean(self.post_full[f'Qubit{q}']['1'])),
                'e0_mode': float(ss.mode(self.post_full[f'Qubit{q}']['0'])[0]), 
                'e1_mode': float(ss.mode(self.post_full[f'Qubit{q}']['1'])[0]),
            } for q in self.qubit_order}

        self.data = {'0': np.array([]), '1': np.array([])}
        if data.size != 0:
            if isinstance(data, dict):
                self.data = data
            else:
                self.data['0'] = np.atleast_1d(data)
        else:
            self._load_data_from_files() 

    def _load_historical_data(self,full_post_path):
        """Loading in historical posterior inference data"""
        print(f"Loading historical data from {full_post_path}...")
        try:
            with open(full_post_path, 'r') as f:
                loaded_data = json.load(f)
        except Exception as exc:
            raise Exception(f"Failed to load JSON data: {exc}")
        
        # Create np.ndarrays from .json outputs
        for q_key, states in loaded_data.items():
            if q_key in self.post_full:
                self.post_full[q_key]['0'] = np.array(states['0'])
                self.post_full[q_key]['1'] = np.array(states['1'])


    def output(self,
            d,
            qubit,
            total_shots,
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
        d : list
            array of data (Observed QoI). Here, it is array of prob. of meas. 0.
        qubit : int
            The number of qubit that we are looking at. 
            For the use of naming the figure file only.
        total_shots : int
            Total number of shots used in the construction of the data pdf
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
        State_Data_address : str
            Where plots and posterior data are store.
        prep_state : str
            State prepared and errors calculated.

        Returns
        -------
        prior_error_rates : np.ndarray
            prior lambdas in the form of a-by-b matrix where 
            a is the number of priors and m is the number of model parameters
        posterior_error_rates : np.ndarray
            prior lambdas in the form of a-by-b matrix where 
            a is the number of posterior and m is the number of model parameters

        """
        np.random.seed(seed)
        qubit = int(qubit)
        # Determine prior mean from params
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

        # Generate Priors (Error Rates)
        prior_error_rates = tnorm01(prior_mean, prior_sd, size=M)
        qs = prior_error_rates

        if prep_state == '0':
            data_error = 1 - d
        else:
            data_error = d
        try:
            # To define P(p|error_count), we use the Beta distribution, as this is the inverse of the Binomial distribution. 
            # Proportional to Beta(alpha = successes+1, beta = total_trials - successes + 1)
            def d_pdf(x): return ss.beta.pdf(x, (data_error*total_shots) + 1, total_shots*(1 - data_error) + 1)
            qs_ker = ss.gaussian_kde(qs)
        except Exception as e:
            print(f"   KDE Failed (Data likely singular): {e}")
            # Fallback: Return original priors if we can't update them
            return prior_error_rates, prior_error_rates

        print(f'Given Lambda |{prep_state}>: prior error rate = {prior_mean:.4f}')

        # Optimization
        max_r = findM(qs_ker, d_pdf)

        # Rejection Sampling
        r_vals = d_pdf(qs) / qs_ker(qs)
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
        try:
            np.savetxt(filename, post_error_rates, delimiter=',')
        except Exception as ex:
            print(f"Could not save posterior to {filename}, perhaps already saved posteriors?: {ex}")

        return prior_error_rates, post_error_rates

    def _load_posterior_marginals(self):
        for q in self.qubit_order:
            path = os.path.join(self.file_address, f'State0_Post_Qubit{q}.csv')
            path = os.path.join(self.file_address, f'State1_Post_Qubit{q}.csv')
            try:
                self.post_full[f'Qubit{q}']['0'] = pd.read_csv(path, header=None, dtype=float).values.flatten()
                print(f"Loaded {len(self.post_full[f'Qubit{q}']['0'])} Posterior Values for Qubit {q}, State 0.")
                print(f"Loaded {len(self.post_full[f'Qubit{q}']['1'])} Posterior Values for Qubit {q}, State 1.")
            except Exception as e:
                print(f"Error reading {path}: {e}")

    def _load_data_from_files(self):
        # State 0
        path0 = os.path.join(self.file_address, 'State0.csv')
        if os.path.exists(path0):
            try:
                self.data['0'] = np.loadtxt(path0, dtype = int, delimiter=',')
                print(f"Loaded {np.size(self.data['0'],axis=0)} shots for State 0.")
            except Exception as e:
                print(f"Error reading {path0}: {e}")
        else:
            print(f"Warning: {path0} not found.")

        # State 1
        path1 = os.path.join(self.file_address, 'State1.csv')
        if os.path.exists(path1):
            try:
                self.data['1'] = np.loadtxt(path1, dtype = int,delimiter=',')
                print(f"Loaded {np.size(self.data['1'],axis=0)} shots for State 1.")
            except Exception as e:
                print(f"Error reading {path1}: {e}")
        else:
            print(f"Warning: {path1} not found.")

    def inference(self,
                  nPrior=40000,
                  prior_sd=0.1,
                  seed=227,
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
        prep_state : str
            '0' or '1', the prepared state to calculate noise from

        Returns
        -------
        None.

        """
        full_post_path = Path(os.path.join(self.file_address, 'Post_Full_Current.json'))
        if full_post_path.is_file() and self.load_data == True: self._load_historical_data(full_post_path)
        
        # Select Data
        current_data = self.data.get(prep_state, np.array([]))
        total_shots = np.size(current_data,axis=0)

        if total_shots == 0:
             print(f"CRITICAL: No data for prep_state='{prep_state}'. Skipping.")
             return

        for idx,q in enumerate(self.qubit_order):                       
        
            # The qubit array order giving the index is given by the aws bracket as `task.result().measurement_qubits`
        
            print(f'Inferring Qubit {q} for State |{prep_state}>')
            
            if f"Qubit{q}" not in self.prior:
                self.post[f'Qubit{q}'] = {}
            if f"Qubit{q}" not in self.post_full:
                self.post_full[f'Qubit{q}'] = {'0': np.array([]), '1': np.array([])}
            if f"Qubit{q}" not in self.prior:
                self.prior[f'Qubit{q}'] = np.array([])
            try:
                d = getData0(current_data, idx)
            except Exception as e:
                print(f"   Error in getData0(): {e}")
                continue

            Posterior_dir = os.path.join(self.file_address, "Posterior Data")
            os.makedirs(Posterior_dir, exist_ok=True)
            state_prefix = os.path.join(Posterior_dir, f"State{prep_state}_")
            
            try:
                prior_lambdas, post_lambdas = self.output(
                    d, q,total_shots, nPrior, self.params, prior_sd, 
                    seed=seed, State_Data_address=state_prefix, prep_state=prep_state
                )                
                self.prior[f'Qubit{q}'] = prior_lambdas
                self.post_full[f'Qubit{q}'][prep_state] = post_lambdas
                print(f'   Qubit{q} has a mean error rate from the prepared state of {prep_state} of {np.mean(post_lambdas):.5}')
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
        
        self.error_distributions(plotting=True, save_plots=True)

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
        
    def get_inverse_element(self, target_bitstring, source_bitstring, method = 'mean'):
        """
        Calculates the probability transition factor from Source (Noisy) -> Target (Clean).
        Mathematically: returns the element (Row=Target, Col=Source) of the Inverse Matrix.
        
        Args:
            target_bitstring : str
                The 'clean' state we are calculating probability for.
            source_bitstring : str
                The 'noisy' state we actually measured.
            method : str
                Either 'mean' or 'mode', which average of the posterior to use
        """
        # Ensure we have the matrices
        if not self.inv_matrices_mean:
            self.create_filter_mat()
            
        probability_factor = 1.0
        
        # Iterate through qubits to multiply local probabilities
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
    

    def eff_DeNoise(self, datadict, method = 'mean', percentage: int|float =100, verbose=True, GD = False, lr=0.1, max_iter=50):
        """
        Efficient DeNoiser function that applies the SplitMeasFilter to the provided data dictionary.
        
        Parameters:
        datadict: dict
            Dictionary containing measurement data.
        method : str
            'mean' or 'mode'
        percentage: int|float
            percentage threshold for filtering (default is 100).
        verbose : bool
            If True, prints progress information.
        GD : bool
            If True, performs optional Gradient Descent refinement.
        lr : float
            Learning rate for Gradient Descent (if GD is True).
        max_iter : int
            Maximum number of iterations for Gradient Descent (if GD is True).
        
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
        
        # Overkill Gradient Descent procedure 
        if GD:
            if verbose: print(f"Analytic pass done. Starting Gradient Descent refinement...")
            
            n_dim = len(targets)
            
            # Vectorize Data
            # 'y' = The actual noisy observations we want to match
            p_noisy_obs = np.array([important_dict[t] for t in targets]) / filtered_shots
            
            # 'x' = Our initial guess (The result from the Analytic pass)
            # Using the analytic result as a "warm start" makes GD extremely fast.
            p_est = np.zeros(n_dim)
            for i, t in enumerate(targets):
                p_est[i] = final_data.get(t, 0.0) / filtered_shots
            
            # Build Forward Matrix M_sub for this subspace
            # We need M (Forward), not M_inv, to calculate the Loss: || M*x - y ||^2
            M_sub = np.zeros((n_dim, n_dim))
            
            for r, obs_bit in enumerate(targets):     # Row: Observed
                for c, hid_bit in enumerate(targets): # Col: Hidden (Clean)
                    M_sub[r, c] = self.get_forward_element(obs_bit, hid_bit)
            
            # Projected Gradient Descent Loop
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
            
            # Update final_data with refined values
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

        for idx,q in enumerate(self.qubit_order):
            q_key = f'Qubit{q}'
            
            # Success Rates (Lambdas)
            # These are samples of "Probability of measuring Correctly"
            lam0_samples = self.post_full[q_key]['0'] if '0' in self.post_full[q_key] else np.array([])
            lam1_samples = self.post_full[q_key]['1'] if '1' in self.post_full[q_key] else np.array([])
            
            if (not lam0_samples.any()) and (not lam1_samples.any()):
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
                    plt.plot(xs, kde0(xs), color='blue', label=r'Posterior Error on 0', alpha = 0.2)
                    plt.fill_between(xs, kde0(xs), alpha=0.2, color='blue')

                    # Plot Data KDE for comparison
                    raw_data = self.data.get('0', np.array([]))
                    total_shots = np.size(raw_data,axis=0)
                    d0 = getData0(raw_data, idx)
                    err0 = 1 - d0

                    def d0_pdf(x): return ss.beta.pdf(x, (err0*total_shots) + 1, total_shots*(1 - err0) + 1)
                    plt.plot(xs, d0_pdf(xs), color='darkslateblue', linestyle='--', label='Data KDE (State |0>)')


                if err1_samples.any():
                    # Plot Error 1 (Readout error on |1>)
                    kde1 = ss.gaussian_kde(err1_samples)
                    plt.plot(xs, kde1(xs), color='red', label=r'Posterior Error on 1', alpha = 0.2)
                    plt.fill_between(xs, kde1(xs), alpha=0.2, color='red')

                    # Plot Data KDE for comparison
                    raw_data = self.data.get('1', np.array([]))
                    total_shots = np.size(raw_data,axis=0)
                    d1 = getData0(raw_data, idx)
                    err1 = d1

                    def d1_pdf(x): return ss.beta.pdf(x, (err1*total_shots) + 1, total_shots*(1 - err1) + 1)
                    plt.plot(xs, d1_pdf(xs), color='firebrick', linestyle='--', label='Data KDE (State |1>)')


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
