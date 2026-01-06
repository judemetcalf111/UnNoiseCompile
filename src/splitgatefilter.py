# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 15:13:15 2025

@author: Jude L. Metcalf
"""

# Numerical/Stats pack
import csv
import pandas as pd
import numpy as np
import json
import scipy.stats as ss
import numpy.linalg as nl
# For optimization
from cvxopt import matrix, solvers

import src.splitmeasfilter
import importlib
importlib.reload(src.splitmeasfilter)
from src.splitmeasfilter import *

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
def QoI_gate(prior_lambdas: np.ndarray, gate_type, gate_num):
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
    num_samples = prior_lambdas.shape[0]
    
    gates = ['RY', 'RX', 'RZ', 'CZ']
    ideal_p0 = None
    
    if (gate_type == 'X') or (gate_type == 'RX') or (gate_type == 'CZ') or (gate_type == 'RY'):
        if (gate_num % 2 == 0):
            ideal_p0 = 1
        elif (gate_num % 2 == 1):
            ideal_p0 = 0
    elif gate_type == 'RZ':
        ideal_p0 = 1
    else:
        raise Exception(f"Gate Type {gate_type} not recognised, recognised gates are: {gates}")

    # We extract the gate error column (index 2)
    ep = prior_lambdas[:, 2]
    
    # Calculate noisy_p0 for all samples at once using vectorised numpy array arithmetic
    # Formula: p(cumulative flip) = (1 +/- (1-2e)^N)/2
    decay_factor = (1 - (2 * ep)) ** gate_num

    if ideal_p0 == 0:
        noisy_p0 = (1 - decay_factor) / 2
    elif ideal_p0 == 1:
        noisy_p0 = (1 + decay_factor) / 2
    else:
        raise Exception("p0 was neither a '0' value or a '1' value. There should only be native gates in the circuit.")

        
    noisy_p1 = 1 - noisy_p0

    # Stack into shape (N, 2, 1) -> [[p0], [p1]] for efficient matrix multiplication
    M_ideal = np.stack([noisy_p0, noisy_p1], axis=1)[..., np.newaxis]

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

def data_readout(qubit_index, datafile: str = '', data: np.ndarray = np.array([])):
    """
    Function to readout json files or suitable numpy arrays
    """

    if type(qubit_index) != int:
        raise Exception("Must supply an index for the interested qubit in `data_readout()`") 

    if datafile.endswith('.json'):
        with open(str(datafile), 'r') as file:
            data = json.load(file)
        qubit_props = data['oneQubitProperties']
        num_qubits = len(qubit_props)

        epsilon01 = np.zeros(num_qubits)
        epsilon10 = np.zeros(num_qubits)
        gate_epsilon = np.zeros(num_qubits)

        qub = qubit_props[qubit_index]
        
        epsilon01 = qubit_props[qub]['oneQubitFidelity'][1]['fidelity'] # Notice that even though it says "fidelity", we get error rate...
        epsilon10 = qubit_props[qub]['oneQubitFidelity'][2]['fidelity'] # Notice that even though it says "fidelity", we get error rate...
        gate_epsilon = qubit_props[qub]['oneQubitFidelity'][0]['fidelity']
        
    elif type(data) == np.ndarray:
        if data.shape[1] != 3:
            raise Exception("Warning: Data array shape does not match simulation setting!\nData shape: "
                    + str(data.shape))
        epsilon01 = data[qubit_index,0]
        epsilon10 = data[qubit_index,1]
        gate_epsilon = data[qubit_index,2]

    elif not (datafile or data):
        raise Exception("Error: No data or datafile provided for the `data_readout()`")
    else:
        raise Exception("Must either:\nsupply datafile as path string to a json data file, i.e. '../data/datafile.json'\nor supply a nx3 numpy array with 0 -> 1 errors, 1 -> 0 errors, and gate errors in each row (ONLY AVAILABLE FOR SINGLE QUBIT GATES)")

    lambdas = np.array([epsilon01,epsilon10,gate_epsilon])
    return lambdas
    

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


########################## Class for Error Filtering ########################
class SplitGateFilter:
    """ Gate and Measurement error filter.
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
                 home_dir='./',
                 meas_cal_dir=None,
                 data_file_address='./data/'):
        self.home_dir = home_dir
        os.chdir(home_dir)
        self.meas_cal_dir = meas_cal_dir if meas_cal_dir else home_dir
        self.data_file_address = data_file_address
        self.interested_qubits = interested_qubits
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
    
    def load_informed_priors(self, qubit_idx, n_samples, seed):
        """
        Loads measurement posteriors from splitmeasfilter files and resamples them.
        Returns columns for P(meas 0|prep 0) and P(meas 1|prep 1).
        """
        try:
            # Construct paths based on splitmeasfilter naming convention
            # We need the directory where splitmeasfilter saved its results
            # Assuming file_address points to that shared directory
            file_0 = f"{self.meas_cal_dir}State0_Post_Qubit{qubit_idx}.csv"
            file_1 = f"{self.meas_cal_dir}State1_Post_Qubit{qubit_idx}.csv"
            
            # Load raw samples (Using pandas as in splitmeasfilter for robustness)
            # These files contain 1D arrays of "Success Rates"
            post_0 = pd.read_csv(file_0, header=None).values.flatten()
            post_1 = pd.read_csv(file_1, header=None).values.flatten()
            
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
            print(f"Warning: Measurement calibration not found for Q{qubit_idx} at {self.meas_cal_dir}")
            return None, None

    def inference(self,
                  nPrior=40000,
                  meas_sd=0.1,
                  gate_sd=0.01,
                  seed=127,
                  shots_per_point=1024,
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
        # Ensure params are loaded (taken from existing logic)
        if self.params is None:
             # Fallback logic or error if params are needed for data shaping
             pass 

        # Main Loop over Qubits
        for q in self.interested_qubits:
            print(f'--- Inferring Gate Errors for Qubit {q} ---')
            
            # Load Observed Data
            raw_data = read_data(q, self.gate_type, self.gate_num, 
                                 file_address=self.data_file_address)
            
            # Calculate QoI (Probability of Measuring 0)
            num_points = int(len(raw_data) / shots_per_point) 
            d_obs = getData0(raw_data, num_points, q)
            
            # Construct Prior Matrix [M x 3]
            prior_lambdas = np.zeros((nPrior, 3))
            
            # --- Measurement Error Columns (p(0|0) & p(1|1)) ---
            loaded_priors = False
            if use_informed_priors:
                p0_col, p1_col = self.load_informed_priors(q, nPrior, seed)
                if p0_col is not None:
                    prior_lambdas[:, 0] = p0_col
                    prior_lambdas[:, 1] = p1_col
                    loaded_priors = True
                    print(f"-> Successfully injected informed measurement priors.")

            if not loaded_priors:
                # Fallback to Uninformed (Gaussian around calibration/default)
                print(f'-> Using uninformed Gaussian priors for measurement. 5% error assumed.')
                prior_lambdas[:, 0] = tnorm01(0.95, meas_sd, size=nPrior) # Default 5% error
                prior_lambdas[:, 1] = tnorm01(0.95, meas_sd, size=nPrior)

            # --- Gate Error {Column (2)} ---
            # We always generate this using the calibration data as an uncertain prior
            # Retrieve calibration center using existing helper
            try:
                cal_data = data_readout(q, datafile=self.meas_cal_dir)
                gate_center = cal_data[2] # 3rd element is gate error
            except:
                gate_center = 0.001 # Conservative default
                print(f"Warning: Could not read calibration data for gate error on Q{q}. Using default {gate_center}.")
            
            prior_lambdas[:, 2] = tnorm01(gate_center, gate_sd, size=nPrior)

            # Run Rejection Sampling
            # INLINING is often cleaner for "Split" filters to avoid passing 10+ args.
            
            # Calculate Densities
            d_ker = ss.gaussian_kde(d_obs)
            qs = QoI_gate(prior_lambdas, self.gate_type, self.gate_num)
            qs_ker = ss.gaussian_kde(qs)
            
            # Find Maximum Ratio (Optimization)
            # Uses existing helper findM
            max_r, max_q = findM(qs_ker, d_ker, prep_state='0') # Gate exp usually targets 0 or 1, check logic

            print('Final Accepted Posterior Lambdas')
            print('M: %.6g Maximizer: %.6g pi_obs = %.6g pi_Q(prior) = %.6g' %
            (max_r, max_q, d_ker(max_q), qs_ker(max_q)))
            
            # Rejection Sampling
            r_vals = d_ker(qs) / qs_ker(qs)
            eta = r_vals / max_r
            accept_mask = eta > np.random.uniform(0, 1, nPrior)
            post_lambdas = prior_lambdas[accept_mask]
            
            # Save and Store
            self.post['Qubit' + str(q)] = post_lambdas
            
            # Save to CSV (Compatible with existing format)
            filename = f"{self.data_file_address}Gate_Post_Qubit{q}.csv"
            np.savetxt(filename, post_lambdas, delimiter=',')
            print(f"-> Saved {len(post_lambdas)} posterior samples to {filename}")

            if plotting:
                post_qs = QoI_gate(post_lambdas, self.gate_type, self.gate_num) 
                post_ker = ss.gaussian_kde(post_qs)

                # Integration for validation (Riemann Sum)
                xs = np.linspace(0, 1, 1000)
                xsd = np.linspace(0.1, 0.9, 1000)
                
                # Vectorised integration calculation 
                q_eval = xs[:-1]
                pdf_vals = qs_ker(q_eval)
                # Avoid division by zero if pdf is 0
                valid_indices = pdf_vals > 0
                
                r_int = np.zeros_like(q_eval)
                r_int[valid_indices] = d_ker(q_eval[valid_indices]) / pdf_vals[valid_indices]
                
                delta_x = xs[1:] - xs[:-1]
                I = np.sum(r_int * pdf_vals * delta_x)

                print('Accepted Number N: %d, fraction %.3f' %
                    (post_lambdas.shape[0], post_lambdas.shape[0] / nPrior))
                print('I(pi^post_Lambda) = %.5g' % I)
                
                # Assuming closest_average and closest_mode are defined externally
                print('Posterior Lambda Mean', closest_average(post_lambdas))
                print('Posterior Lambda Mode', closest_mode(post_lambdas))

                print('0 to 1: KL-Div(pi_D^Q(post),pi_D^obs) = %6g' %
                    (ss.entropy(post_ker(xs), d_ker(xs))))
                print('0 to 1: KL-Div(pi_D^obs,pi_D^Q(post)) = %6g' %
                    (ss.entropy(d_ker(xs), post_ker(xs))))


                output_filename = f"{self.data_file_address}Gate_Post_Qubit{q}.csv"
                np.savetxt(output_filename, post_lambdas, delimiter=",")

                # Defining matplotlib width and height here as in the preamble for clarity
                width, height = 10, 6 
                
                plt.figure(figsize=(width, height), dpi=120, facecolor='white')
                plt.plot(xsd, d_ker(xsd), color='Red', linestyle='dashed', linewidth=3, label=r'$\pi^{\mathrm{obs}}_{\mathcal{D}}$')
                plt.plot(xsd, post_ker(xsd), color='Blue', label=r'$\pi_{\mathcal{D}}^{Q(\mathrm{post})}$')
                plt.xlabel('Pr(Meas. 0)')
                plt.ylabel('Density')
                plt.legend()
                plt.tight_layout()
                plt.savefig(self.data_file_address + 'QoI-Qubit%g.pdf' % q)
                plt.show()
                
                plt.figure(figsize=(width, height), dpi=100, facecolor='white')
                eps_vals = post_lambdas[:, 2]
                eps_ker = ss.gaussian_kde(eps_vals)
                eps_line = np.linspace(np.min(eps_vals), np.max(eps_vals), 1000)
                
                plt.plot(eps_line * 0.5, eps_ker(eps_line), color='Blue') 
                plt.ticklabel_format(axis="x", style="sci", scilimits=(-5, 1))
                plt.xlabel(r'$\epsilon_g$')
                plt.ylabel('Density')
                plt.tight_layout()
                plt.savefig(self.data_file_address + 'Eps-Qubit%g.pdf' % q)
                plt.show()

        # Finalize stats
        self.modes = self.mode()
        self.means = self.mean()
        
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
