# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 15:13:15 2025

@author: Jude L. Metcalf
"""

# Numerical/Stats pack
import pandas as pd
import numpy as np
import json
import scipy.stats as ss
import numpy.linalg as nl
from collections import defaultdict
# For optimization

import src.splitmeasfilter
import importlib
importlib.reload(src.splitmeasfilter) # what?
from pathlib import Path
from src.splitmeasfilter import *
from src.splitmeasfilter import NumpyEncoder

# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
width = 6.72 # plot width
height = 4.15 # plot height

######################### Numerical Helper Functions #########################

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

def dq_Gate(qs_val, d_val):
    """Calculates the ratio of the prior pdf to the data kernel safely at some point 'x'"""
    if np.abs(qs_val) > 1e-6 and np.abs(d_val) > 1e-6:
        return - d_val / qs_val
    else:
        return np.inf

def findGateM(qs_ker, d_pdf):
    """Find M that maximises the d/q ratio for rejection sampling """
    
    # Find the min value in log-space (thus negative)
    x_scan = np.linspace(0, 0.1, 40000) 
    
    prior_density = qs_ker(x_scan)
    threshold = 0.01 * np.max(prior_density)
    
    # Get indices where prior is significant
    valid_indices = np.where(prior_density > threshold)[0]

    if len(valid_indices) > 1:
        # We use .item() to safely turn array([0.5]) into 0.5, calculating negative ratio
        ys = np.array([
            dq_Gate(qs_ker(x_scan[i]).item(), d_pdf(x_scan[i]).item()) 
            for i in valid_indices
        ])

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


def gate_error_posterior_with_readout_noise(
    gate_num: int, 
    prep_state: str,
    total_shots: int, 
    observed_flips: int,  # Raw count of '1's (or 'errors') observed
    prior_errors: np.ndarray, # Shape (N, 2): [[P(0|1), P(1|0)], ...]
    num_posterior_samples: int = 10000
):
    """
    Calculates the posterior of gate error p, marginalizing over readout error uncertainty.
    
    Parameters
    ----------
    readout_error_samples : np.ndarray
        Samples from your measurement error characterization.
        Column 0: p(0|1) - False Negative (read 0 when state is 1)
        Column 1: p(1|0) - False Positive (read 1 when state is 0)
    """
    
    # 1. Define the Grid for Gate Error p
    # Use geomspace for precision near zero
    p_grid = np.geomspace(1e-9, 0.4999, 5000)
    
    # 2. Pre-calculate the Ideal Theory (State Probability)
    # Probability of the state being flipped after N gates (without readout noise)
    # shape: (1, 5000)
    p_state_flip = ((1 - (1 - 2 * p_grid)**gate_num) / 2).reshape(1, -1)
    p_state_correct = 1 - p_state_flip
    
    # 3. Incorporate Readout Noise (The "Forward Noise" Step)
    # We broadcast Readout Error Samples (N, 1) against p-grid (1, 5000)
    
    # Extract SPAM parameters
    # prob_read_1_given_0 
    err_0 = prior_errors[:, 0].reshape(-1, 1) 
    # prob_read_0_given_1 
    err_1 = prior_errors[:, 1].reshape(-1, 1) 
    
    # Calculate Probability of OBSERVING a flip (Outcome 1)
    # P(obs 1) = P(state 1)*P(read 1|1) + P(state 0)*P(read 1|0)
    # P(read 1|1) = 1 - err_0
    # P(read 1|0) = err_1
    
    if prep_state == '0':
        p_obs_flip = (p_state_flip * (1 - err_1)) + (p_state_correct * err_0)
        p_obs_no_flip = 1 - p_obs_flip
    elif prep_state == '0':
        p_obs_flip = (p_state_flip * (1 - err_0)) + (p_state_correct * err_1)
        p_obs_no_flip = 1 - p_obs_flip
    else:
        raise Exception("`prep_state` neither '0' nor '1'")
    
    # 4. Compute Likelihood of Raw Data
    # L = P(obs_flip)^k * P(obs_no_flip)^(n-k)
    # We do this in log space to avoid underflow
    
    # Clamp for numerical safety
    p_obs_flip = np.maximum(p_obs_flip, 1e-300)
    p_obs_no_flip = np.maximum(p_obs_no_flip, 1e-300)
    
    log_likelihoods = (
        observed_flips * np.log(p_obs_flip) + 
        (total_shots - observed_flips) * np.log(p_obs_no_flip)
    )
    
    # 5. Marginalize over Readout Errors
    # We have a matrix of log_likelihoods: (Num_SPAM_Samples, Num_P_Grid_Points)
    # We need to average over the SPAM samples (rows) to get the marginal likelihood for p
    
    # Convert back to probability space safely
    # We subtract the global max to keep exp() in range
    global_max = np.max(log_likelihoods)
    likelihoods = np.exp(log_likelihoods - global_max)
    
    # Average over the nuisance parameters (readout noise samples)
    # Axis 0 is the SPAM samples
    marginal_likelihood = np.mean(likelihoods, axis=0)
    
    # 6. Normalize to get PDF
    posterior_pdf = marginal_likelihood / np.sum(marginal_likelihood)
    
    # 7. Sample
    samples = np.random.choice(p_grid, size=num_posterior_samples, p=posterior_pdf)
    
    return samples

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


######################## For Parameter Characterzation ########################

# used to call QoI
def QoI_to_noised_errors(data_obs: float, prior_errors: np.ndarray, total_shots, gate_type, gate_num:int) -> np.ndarray:
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
    num_samples = prior_errors.shape[0]
    
    gates = ['RY', 'RX', 'RZ', 'CZ', 'X', 'Y']
    
    if (gate_type == 'X') or (gate_type == 'RX') or (gate_type == 'CZ') or (gate_type == 'RY') or (gate_type == 'RY'):
        if (gate_num % 2 == 0):
            prep_state = '0'
        else:
            prep_state = '1'
    elif gate_type == 'RZ':
        prep_state = '0'
    else:
        raise Exception(f"Gate Type {gate_type} not recognised, recognised gates are: {gates}")
    

    if prep_state == '0':
        data_error = 1 -data_obs
    elif prep_state == '1':
        data_error = data_obs
    else:
        raise Exception(f"Error in calculating data_error: prepared state is {prep_state}\n"
              "Returning p0s")
    
    flip_count = int(data_error * total_shots)
    
    print("------ Calculating the data distribution of error rates... ------")
    sampled_gate_errors = gate_error_posterior_with_readout_noise(gate_num,prep_state,total_shots,flip_count,prior_errors,num_samples)
    print("------  Discovered the data distribution of error rates!   ------")

    return sampled_gate_errors


def data_readout(qubit, datafile: str = '', data: np.ndarray = np.array([])):
    """
    Function to readout json files or suitable numpy arrays
    """

    if type(qubit) != int and type(qubit) != str:
        raise Exception("Must supply an integer label for the interested qubit in `data_readout()`") 

    if datafile.endswith('.json'):
        with open(str(datafile), 'r') as file:
            data = json.load(file)
        qubit_props = data['oneQubitProperties']
        
        epsilon01 = qubit_props[str(qubit)]['oneQubitFidelity'][1]['fidelity'] # Notice that even though it says "fidelity", we get error rate...
        epsilon10 = qubit_props[str(qubit)]['oneQubitFidelity'][2]['fidelity'] # Notice that even though it says "fidelity", we get error rate...
        gate_epsilon = 1 - qubit_props[str(qubit)]['oneQubitFidelity'][0]['fidelity'] # Except here where we do! error = 1 - fidelity

    elif not (datafile or data):
        raise Exception("Error: No data or datafile provided for the `data_readout()`")
    else:
        raise Exception("Must either:\nsupply datafile as path string to a json data file, i.e. '../data/datafile.json'\nor supply a nx3 numpy array with 0 -> 1 errors, 1 -> 0 errors, and gate errors in each row (ONLY AVAILABLE FOR SINGLE QUBIT GATES)")

    lambdas = np.array([epsilon01,epsilon10,gate_epsilon])
    return lambdas

def read_data(interested_circuit, gate_type, gate_num, file_address=''):
    """Read out bitstring data from csv file generated by collect_filter_data().

    Args:
      file_address:
        file address, string ends with '/' if not empty

    Returns:
      An array of bit strings.
    """

    circuit_location = None
    try:
        if type(interested_circuit) == int:
            circuit_location = os.path.join(file_address,f'Readout_{gate_num}_{gate_type}_Circuit{interested_circuit}.csv')
        elif type(interested_circuit) == str:
            circuit_location = os.path.join(file_address,interested_circuit)
        else:
            print("Unsupported input as interested circuit.\n"
                   f"Should be either an integer to readReadout_{gate_num}_{gate_type}" + '_Circuit{interested_circuit}.csv, or the string of the name of the file itself\n'
                   f"Value given: interested_circuit={interested_circuit}")
    except Exception as e:
        print(f"2 Qubit data unable to be read. Error:{e}")
    
    if circuit_location is None:
        raise ValueError("circuit_location could not be determined. Please check your inputs.")
    
    data_array = np.loadtxt(circuit_location, dtype=float, delimiter=',')

    return data_array


########################## Class for Error Filtering ########################
class SplitGateFilter:
    """ Gate and Measurement error filter.
    Attributes:
        qubit_order: array,
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
                 home_dir='./',
                 meas_cal_dir=None,
                 data_file_address='./data/'):
        self.home_dir = home_dir
        try:
            os.chdir(home_dir)
        except FileNotFoundError:
            pass
        self.meas_cal_dir = meas_cal_dir if meas_cal_dir else home_dir
        self.data_file_address = data_file_address
        
        # --- Initialize Data Structures ---
        self.meas_prior = {}    # Not yet loaded measurement error prior
        self.post_full = {}     # Full posterior
        self.post = {}          # Collapsed mean/mode of full posterior above
        self.means = {}
        self.modes = {}

        self.params = None

        self.current_file = os.path.join(self.data_file_address, 'Post_Gate_Full_Current.json')
        self._load_history()

    def _load_history(self):
        """Internal method to load JSON history and convert lists back to arrays."""
        if os.path.exists(self.current_file):
            print(f"Loading history from {self.current_file}...")
            try:
                with open(self.current_file, 'r') as f:
                    self.post_full = json.load(f)
                
                # Convert lists back to numpy arrays for math
                for q_key, gates in self.post_full.items():
                    for g_key, couples in gates.items():
                        for c_key, data_obj in couples.items():
                            if 'samples' in data_obj:
                                data_obj['samples'] = np.array(data_obj['samples'])
            except Exception as e:
                print(f"Warning: Failed to load history JSON: {e}")
                self.post_full = {}
        else:
            print("No history file found. Starting fresh.")
            self.post_full = {}

    def _save_history(self):
        """Save the current state of post_full to JSON using NumpyEncoder."""
        print(f"Saving updated calibration to {self.current_file}...")
        with open(self.current_file, 'w') as f:
            json.dump(self.post_full, f, cls=NumpyEncoder, indent=4)

    def load_informed_json(self):
        # Assuming meas_cal_dir is a string, convert it to a Path object
        file_path = Path(self.meas_cal_dir) / 'Post_Full_Current.json'
        
        if not self.meas_prior:
            if file_path.is_file():
                try:
                    with open(file_path, 'r') as f:
                        calibration_data = json.load(f)
                        print(f"{file_path} Calibration file loaded successfully!")
                        # 'data' now contains your JSON object (dict or list)
                    
                    self.meas_prior = calibration_data
                        
                except FileNotFoundError:
                    print(f"Error: The file {file_path} was not found.")
                except json.JSONDecodeError:
                    print(f"Error: The file {file_path} contains invalid JSON.")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
            else:
                print(f"Tried to load informed measurement priors from {self.meas_cal_dir} called 'Post_Full_Current.json', none found")
        else:
            print("Measurement error priors already loaded")
            

    def load_informed_priors(self, qubit_identity, n_samples, seed):
        """
        Loads measurement posteriors from splitmeasfilter files and resamples them.
        Returns columns for P(meas 0|prep 0) and P(meas 1|prep 1).
        """

        try:
            post_0 = self.meas_prior[f"Qubit{qubit_identity}"]["0"]
            post_1 = self.meas_prior[f"Qubit{qubit_identity}"]["1"]
            
            # Create KDEs
            kde_0 = ss.gaussian_kde(post_0)
            kde_1 = ss.gaussian_kde(post_1)
            
            # Resample to match the required N priors
            # .resample() returns shape (1, N), so we flatten
            col_0 = kde_0.resample(n_samples, seed=seed).flatten()
            col_1 = kde_1.resample(n_samples, seed=seed).flatten()
            
            # Enforce physical constraints [0, 1]
            col_0 = np.clip(col_0, 0.0, 1.0)
            col_1 = np.clip(col_1, 0.0, 1.0)
            
            return col_0, col_1
        
        except (FileNotFoundError, IOError):
            print(f"Warning: Measurement calibration not found for Qubit{qubit_identity} in {self.meas_cal_dir}")
            return None, None

    def inference(self,
                  qubit_orders,
                  gate_type,
                  gate_num,
                  interested_circuits=[],
                  qubit_couplings=[],
                  nPrior=40000,
                  meas_sd=0.05,
                  gate_sd=0.1,
                  seed=127,
                  use_informed_priors=True,
                  plotting=True):
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
        use_informed_priors : bool, optional
            Whether to use informed priors from measurement calibration.
            The default is True.
        plotting : bool, optional
            Whether to plot the posterior results. The default is True.

        Returns
        -------
        None.

        """

        num_circuits = len(qubit_couplings)

        # if the interested_circuits parameter representing the location isn't given
        if not interested_circuits:
            interested_circuits = list(range(1,num_circuits+1))
            print(f"Since `interested_circuits` isn't provided, using an integer array of the length of qubit_couplings: interested_circuits={interested_circuits}\n"
                  "NOTE: best practise is to provide the names of the files in the correct order alongwith `qubit_couplings` here to ensure that the correct data is located\n" \
                  "----- if testing single qubit gates, provide the name of the data file as `interested_circuits` as a singleton array.")
        
        self.load_informed_json()

        for circuit_number,interested_circuit in enumerate(interested_circuits):
            
            if (qubit_couplings) and (gate_type in ['CZ', 'iSWAP', 'CNOT']):
                qubit_coupling_set = qubit_couplings[circuit_number]
                qubit_order = qubit_orders[circuit_number]
            else:
                qubit_coupling_set = None
                qubit_order = qubit_orders


            print(f"Beginning Inference Run. Circuit index used: {circuit_number}.")
            # Loop over Qubits

            for idx,q in enumerate(qubit_order):
                q = int(q)
                print(f'--- Inferring Gate Errors for Qubit {q} ---')

                # Determine a 'coupling key' to correspond to whether we have 2-qubit gate or not
                qubit_couple_key = "Single"
                if gate_type in ['CZ', 'iSWAP', 'CNOT'] and qubit_coupling_set:
                    # Find which pair in the corresponding set in qubit_couplings contains q
                    found = False
                    for pair in qubit_coupling_set:
                        if str(q) in pair:
                            qubit_couple_key = f"{pair[0]}-{pair[1]}"
                            found = True
                            break
                    if not found:
                        print(f"Warning: No coupling found for Q{q} in provided list. processing as Single.")

                # Load Observed Data
                raw_data = read_data(interested_circuit, gate_type, gate_num, 
                                    file_address=self.data_file_address)
                
                total_shots = np.size(raw_data,axis=0)
                
                # Calculate the probability of measuring 0 from the total shots
                d_obs = getData0(raw_data, idx)

                # Construct Prior Matrix [M x 3]
                prior_errors = np.zeros((nPrior, 3))
                
                # --- Measurement Error Columns (p(0|0) & p(1|1)) ---
                loaded_priors = False
                if use_informed_priors:
                    p0_col, p1_col = self.load_informed_priors(q, nPrior, seed)
                    if (p0_col is not None) and (p1_col is not None):
                        prior_errors[:, 0] = p0_col
                        prior_errors[:, 1] = p1_col
                        loaded_priors = True
                        print(f"-> Successfully injected informed measurement priors.")


                # --- Gate Error {Column (2)} ---
                # We always generate this using the calibration data as an uncertain prior
                # Retrieve calibration center (i.e. mean) using existing data_readout()
                try:
                    datafile = os.path.join(self.meas_cal_dir,'Braket_Qubit_Calibration.json')
                    cal_data = data_readout(q, datafile=datafile)
                    gate_center = cal_data[2] # 3rd element is gate error
                    print(f"Loaded Qubit calibration from the Amazon Braket. The estimated gate error for {gate_type} is {gate_center:.3}")
                    if not loaded_priors:
                        try:
                            prior_errors[:, 0] = tnorm01(cal_data[0], meas_sd, size=nPrior) # Using calibrated priors based on AWS calibration
                            prior_errors[:, 1] = tnorm01(cal_data[1], meas_sd, size=nPrior)
                            print(f"Failed to load precise informative priors, Calibrated prior measurement error rates from AWS:\n"
                                f"err0={cal_data[0]:.2}, err1={cal_data[1]:.2}\n"
                                f"Standard Deviation of each is std={meas_sd}")                            
                        except Exception as e:
                                # Fallback to Uninformed (Gaussian around calibration/default)
                                print(f'-> Using uninformed Gaussian priors for measurement. 5% error assumed for Qubit{q}: {e}')
                                prior_errors[:, 0] = tnorm01(0.05, meas_sd, size=nPrior) # Default 5% error
                                prior_errors[:, 1] = tnorm01(0.05, meas_sd, size=nPrior)
                    else:
                        pass

                except Exception as e:
                    gate_center = 0.1 # Broad and poor default
                    print(f"Warning: Could not read calibration data for gate error on Qubit{q}.\n"
                          "Ensure that the Braket Calibration .json file is location in the SplitGateFilter.meas_cal_dir folder and named 'Braket_Qubit_Calibration.json'\n"
                          f"Using default calibration value: {gate_center}: {e}")
                
                prior_errors[:, 2] = tnorm01(gate_center, gate_sd, size=nPrior)
                qs = prior_errors[:, 2]
                ### Run Rejection Sampling ###

                # Calculate Densities
                data_errors = QoI_to_noised_errors(d_obs, prior_errors, total_shots, gate_type, gate_num)
                d_pdf = ss.gaussian_kde(data_errors)
                qs_ker = ss.gaussian_kde(qs)

                #########################
                xs = np.linspace(0,0.05,10000)
                plt.plot(xs,d_pdf(xs))
                plt.plot(xs,qs_ker(xs))
                plt.show()
                #########################


                # Find Maximum Ratio (Optimisation)
                # Uses existing helper findM
                max_r = findGateM(qs_ker, d_pdf) # Gate exp usually targets 0 or 1, check logic
                
                # Rejection Sampling
                r_vals = d_pdf(qs) / qs_ker(qs)
                eta = r_vals / max_r
                accept_mask = eta > np.random.uniform(0, 1, nPrior)
                post_errors = prior_errors[accept_mask]
                post_err0 = post_errors[:,0]
                post_err1 = post_errors[:,1]
                post_gaterr = post_errors[:,2]


                if len(post_gaterr) == 0:
                    print("   Warning: Rejection sampling rejected all points. Returning priors.")
                    return post_errors, post_errors

                print('   Accepted N: %d (%.1f%%)' % (len(post_gaterr), 100*len(post_gaterr)/len(prior_errors[0])))

                # Initialise the post dictionaries, including the key to store the qubit data
                q_key = f"Qubit{q}"
                if q_key not in self.post_full: self.post_full[q_key] = {}
                if gate_type not in self.post_full[q_key]: self.post_full[q_key][gate_type] = {}
                if q_key not in self.post: self.post[q_key] = {}
                if gate_type not in self.post[q_key]: self.post[q_key][gate_type] = {}
                

                # We store the samples in the post jsons
                # Store with the coupling_key, denotating which connection is tested, otherwise just store as a dict:

                if qubit_couple_key != 'Single':
                    self.post_full[q_key][gate_type] = {'FULL_GATE_ERROR': post_gaterr,
                        'FULL_MEAS0_ERROR': post_err0,
                        'FULL_MEAS1_ERROR': post_err1}
                    
                    self.post[q_key][gate_type] = {'GATE_ERROR_MEAN': float(np.mean(post_gaterr)),
                        'GATE_ERROR_MODE': float(ss.mode(post_gaterr)[0]),
                        'MEAS0_ERR_MEAN': float(np.mean(post_err0)),
                        'MEAS0_ERR_MODE': float(ss.mode(post_err0)[0]),
                        'MEAS1_ERR_MEAN': float(np.mean(post_err1)),
                        'MEAS1_ERR_MODE': float(ss.mode(post_err1)[0])}
                else:
                    self.post_full[q_key][gate_type][qubit_couple_key] = {'FULL_GATE_ERROR': post_gaterr,
                        'FULL_MEAS0_ERROR': post_err0,
                        'FULL_MEAS1_ERROR': post_err1}
                    self.post[q_key][gate_type][qubit_couple_key] = {'GATE_ERROR_MEAN': float(np.mean(post_gaterr)),
                        'GATE_ERROR_MODE': float(ss.mode(post_gaterr)[0]),
                        'MEAS0_ERR_MEAN': float(np.mean(post_err0)),
                        'MEAS0_ERR_MODE': float(ss.mode(post_err0)[0]),
                        'MEAS1_ERR_MEAN': float(np.mean(post_err1)),
                        'MEAS1_ERR_MODE': float(ss.mode(post_err1)[0])}
                    
                
                print(f"-> Inferred {len(post_errors)} samples for {q_key} ({qubit_couple_key})")

                if plotting:
                    post_ker = ss.gaussian_kde(post_gaterr)

                    # Integration for validation (Riemann Sum)
                    xs = np.linspace(0, 1, 1000)

                    # Vectorised integration calculation 
                    q_eval = xs[:-1]
                    pdf_vals = qs_ker(q_eval)
                    # Avoid division by zero if pdf is 0
                    valid_indices = pdf_vals > 1e-12
                    
                    r_int = np.zeros_like(q_eval)
                    r_int[valid_indices] = d_pdf(q_eval[valid_indices]) / pdf_vals[valid_indices]
                    
                    delta_x = xs[1:] - xs[:-1]
                    I = np.sum(r_int * pdf_vals * delta_x)

                    print('Accepted Number N: %d, fraction %.3f' %
                        (post_errors.shape[0], post_errors.shape[0] / nPrior))
                    print('I(pi^post_Lambda) = %.5g' % I)
                    
                    # Assuming closest_average and closest_mode are defined externally
                    print('Posterior Lambda Mean', closest_average(post_errors))
                    print('Posterior Lambda Mode', closest_mode(post_errors))

                    print('0 to 1: KL-Div(pi_D^Q(post),pi_D^obs) = %6g' %
                        (ss.entropy(post_ker(xs), d_pdf(xs))))
                    print('0 to 1: KL-Div(pi_D^obs,pi_D^Q(post)) = %6g' %
                        (ss.entropy(d_pdf(xs), post_ker(xs))))

                    # Defining matplotlib width and height here as in the preamble for clarity
                    width, height = 10, 6 
                    
                    xsd = np.linspace(0,0.2,2000)
                    plt.figure(figsize=(width, height), dpi=120, facecolor='white')
                    plt.plot(xsd, d_pdf(xsd), color='Red', linestyle='dashed', linewidth=3, label=r'$\pi$')
                    plt.plot(xsd, post_ker(xsd), color='Blue', label=r'$\pi_{\mathcal{D}}^{Q(\mathrm{post})}$')
                    plt.xlabel('Pr(Meas. 0)')
                    plt.ylabel('Density')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(self.data_file_address + 'QoI-Qubit%g.pdf' % q)
                    plt.show()
                    
                    plt.figure(figsize=(width, height), dpi=100, facecolor='white')
                    eps_vals = post_errors[:, 2]
                    eps_ker = ss.gaussian_kde(eps_vals)
                    eps_line = np.linspace(np.min(eps_vals), np.max(eps_vals), 1000)
                    
                    plt.plot(eps_line * 0.5, eps_ker(eps_line), color='Blue') 
                    plt.ticklabel_format(axis="x", style="sci", scilimits=(-5, 1))
                    plt.xlabel(r'$\epsilon_g$')
                    plt.ylabel('Density')
                    plt.tight_layout()
                    plt.savefig(self.data_file_address + 'Eps-Qubit%g.pdf' % q)
                    plt.show()


            full_path_datetime = os.path.join(self.data_file_address, f'Post_Gate_Full_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            full_path_current = os.path.join(self.data_file_address, f'Post_Gate_Full_Current.json')
            meanmode_path_datetime =os.path.join(self.data_file_address, f'Post_Gate_MeanMode_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            meanmode_path_current = os.path.join(self.data_file_address, f'Post_Gate_MeanMode_Current.json')
            
            with open(full_path_datetime, 'w') as f:
                json.dump(self.post_full, f, cls=NumpyEncoder, indent=4)

            with open(meanmode_path_datetime, 'w') as f:
                json.dump(self.post, f, cls=NumpyEncoder, indent=4)

            with open(full_path_current, 'w') as f:
                json.dump(self.post_full, f, cls=NumpyEncoder, indent=4)

            with open(meanmode_path_current, 'w') as f:
                json.dump(self.post, f, cls=NumpyEncoder, indent=4)

            print(f"Saved Full Gate Posterior to:\n{full_path_datetime}\nand\n{full_path_current}")
            print(f"Saved Mean/Mode Gate Summary to:\n{meanmode_path_datetime}\nand\n{meanmode_path_current}")
