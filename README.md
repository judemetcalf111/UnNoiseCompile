# UnNoiseCompile
A repository for developing _scalable_ and hardware-aware quantum compilation to minimise bitflip gate and measurement noise.

## Folder Structure

Much of this repository is built on proper management of experimental data gathered from probe circuits to be used on QPUs. As such, folder management and organistaion is crucial. The following is strongly suggested to ensure that the functions run smoothly without too much manipulation of subfolders:

```text
Project_Root/
│
│
├── setup.py                   <-- setup.py describing the version, dependencies, etc.
│
│
├── notebooks/
│   ├── notebook1.ipynb
│   └── notebook2.ipynb        <-- Any notebooks to run scripts
│
├── src/
│   ├── __init__.py
│   ├── graphconnector.py      <-- Build Graphs for 2-Qubit Gates
│   ├── splitmeasfilter.py     <-- Measurement Error & Inference
│   └── splitgatefilter.py     <-- Gate Error & Inference
│
├── data/
│   ├── *meas_cal*/             <-- Output folder for MeasFilter
│   └── *gate_exp*/             <-- Output folder for GateFilter
│
└── results/                   <-- Final plots and reports
```

## Getting Started
\
Python modules built using Python 3.11.14 in `/src`, with notebooks and samples written mostly using Jupyter notebooks in `/notebooks` and `/scripts/`. 

First, to make a .venv file as a virtual environment (use Python3.11 for Qiskit compatibility): 

`python3.11 -m venv .venv`

Activate:

Linux/OS X:\
`source .venv/bin/activate`


Windows Command Prompt:\
`.venv\Scripts\activate.bat`


Then install packages using pip:

`pip install -r requirements.txt`

To allow access to the `src/` folder for scripts, run the following to allow python to recognise the `__init__.py` and `setup.py` files already present:

`pip install -e .`

And run from there!

## The Workflow

### 1. Measurement Data

You can supply bitstring data, in the form of the `QuantumTask.result().measurements` available from the Amazon Braket API into the `SplitMeasFilter` by placing a csv file named `State0.csv` defining the a nontrivial groundstate measurement in the path defined by the `file_address` parameter. Similarly, one should place `State1.csv`, a csv file containing the bits recieved from a nontrivial flip.

The script is designed to run inference on data gathered from an expected |000...000> state and an expected |111...111> state individually. This is due to the fact that measurement errors will generally be asymmetric, according to the physics of the QPU system. `|0>` is often a ground state, and relaxes more often than exciting to the `|1>` state, for example. Thus, we run two rounds of inference to understand these asymmetric properties.

Here is a sample of how one is to use the `SplitMeasFilter` Python Class, designed to infer posterior distributions of these errors, and efficiently filter these errors from given quantum data.

```python
from src.splitmeasfilter import SplitMeasFilter

# Define measurement order of qubits
# VERY important, to ensure that the order of the data recieved is the order that AWS produced
# More often than not for complex circuits, this will not be in ascending order!
# Find through running `QuantumTask.result().measured_qubits` in the Braket API
qubits = [0, 1, 6]

# Define whichever directory contains both `State0.csv` and `State1.csv` (or only one if that alone is given)
meas_path = './data/meas_cal/'

# Instantiate Data (from `State0.csv` and `State1.csv` files)
meas_filter = SplitMeasFilter(qubit_order=qubits,    # Qubit order as above
                             file_address=meas_path, # Location of data as above
                             load_data=True,         # (optional) Load posteriors from previous inference trials
                             home_dir='.'            # (optional) Define the working directory
                             )
```
\
We can then run inference on this data:

```python
# Applying the meas_filter.inference() function
# on batches of size shots_per_point
# The 'prep_state' variable can be '0' or '1', defining either
# |000...000> or |111...111> to infer either error in measuring |0> or |1>

meas_filter.inference(nPrior=40000,          # Number of tested error rates
                        prior_sd=0.1,        # The sd of the prior distribution
                        shots_per_point=1024,# Number of shots in each batch
                        seed=28,             # Seed for the rejection sampling
                        prep_state='0')      # Prepared state (for |0> or |1>)

meas_filter.qubit_order = [1,0]

meas_filter.inference(nPrior=40000,          # Number of tested error rates
                        prior_sd=0.1,        # The sd of the prior distribution
                        shots_per_point=1024,# Number of shots in each batch
                        seed=28,             # Seed for the rejection sampling
                        prep_state='1')      # Prepared state (for |0> or |1>)
``` 
\
This will produce files in a dedicated `Posterior Data` directory located in the directory specified by the `file_address` parameter:\
`State0_Post_Qubit0.csv`,\
`State0_Post_Qubit1.csv`,\
`State0_Post_Qubit6.csv`,\
`State1_Post_Qubit1.csv`,\
`State1_Post_Qubit0.csv`,

Which are sampled arrays from posterior distribution of error rates for each qubit and each state we inferred on. We also recieve the full data in the form of an easy to load (and read) JSON file, both containing full distributions as above and summary statistics (mean, mode, and confidence interval). These are found both timestamped to retain copies of data, along with the latest run marked as:

`Post_Full_Current.json`\
`Post_MeanMode_Current.json`

Alongside the data, the psterior distributions will be saved as plots in the same `file_address` directory as 



 which contain the posterior distributions of measurement errors, in the form of rejection-sampled error rates. This will output plots of observed data against the posterior model. To reproduce these, along with the distribution statistics, and optionally save, run:
```python
meas_filter.error_distributions(plotting=True, save_plots=True)
```
And this will produce the error rate variables in the `MeasFilter` class. The following gives the `p(0|0)` error rate for Qubit0:
```python
meas_filter.post_marginals['Qubit0']['0']
```

### 2. Gate Data

#### 2a. Designing a Testing Circuit

It is not a trivial problem to design an efficient circuit to test 2-qubit gates. If the _Controlled-Z_ gate is the native 2-qubit gate (as for most superconducting QPUs at the time of writing in January of 2026), there is no direction to the gate, so _CZ[Q1-Q2]_ is the same as _CZ[Q2-Q1]_, so we need only test every connection between qubits. However, the superconducting qubits are arranged in somewhat complex arrangements on a cartesian grid, with a maximum connectivity of any qubit being 4. We need to construct 4 circuits, where each _CZ_ connection is tested in at least one of these circuits.

As this problem is equivalent to the maximal edge-colouring problem from graph theory, there are many algorithms which have been developed to efficiently generate solutions to this exact problem, 'colouring' each connection at least once, and if possible, multiple times. Using graph theory, we can also say that since the qubits are arranged on a grid, it is a bipartite graph, and thus by König's Theorem it is class 1, and only 4 circuits (and not 5) are required (for this idea, see Vizing's Theorem).

**The** `graphconnector.py` **script and its `MisraGriesSolver` class is an implementation using Kempe Chains in the Misra-Gries algorithm and graph reordering to produce valid data aand maximise data quality and efficiency.**

We use the Misra-Gries Edge-Colouring Algorithm, which is efficient of order $\mathcal{O}(N \times E) \equiv \mathcal{O}(N ^ 2)$, of order 'N' number of qubits multiply 'E' number of edges (here qubit coupling). This will be very efficient for NISQ QPUs, and can be improved if the number of circuits can be increased from 4 to 5, or shots become less expensive, wherein finding these graphs becomes easier. 

We here minimise the number of circuits to 4 (which will be either 4 or 5 in a grid-like construction as in current superconducting QPUs), and add in redundant extra CZ tests to infer the gate error even better with limited shots.

Below are 2-qubit CZ testing circuit examples from IQM Emerald and Rigetti Ankaa-3. Each have 4 circuits, with red edges being active CZ gates in the circuit, to be run some number of times to determine bit-flip error. An 'Efficiency' is given in the top left of each circuit, which gives the proportion of qubits active in a CZ gate, thus from whom error data can be gathered. Additional statistics concerning the graphs, total CZ gates tested, redundnacy, and time taken to produce them are given in the Ankaa-3 graphs:

##### **IQM Garnet:**
| | |
|:---:|:---:|
| <img src="resources/Emerald0.png" width="400" /> | <img src="resources/Emerald1.png" width="400" /> |
| <img src="resources/Emerald2.png" width="400" /> | <img src="resources/Emerald3.png" width="400" /> |
##### **Rigetti Ankaa-3:**
| | |
|:---:|:---:|
| <img src="resources/Ankaa-3_0.png" width="400" /> | <img src="resources/Ankaa-3_1.png" width="400" /> |
| <img src="resources/Ankaa-3_2.png" width="400" /> | <img src="resources/Ankaa-3_3.png" width="400" /> |
##### **Rigetti Ankaa-3 Text Statistics Readout:**
| |
|:---:|
| <img src="resources/Ankaa-3Text.png" width="400" /> |

The notebook containing the code to produce these graphs from the AWS Braket on 7 Jan 2026 can be found in `Tutorial/Producing2QubitGraphs.ipynb`

#### 2b. Inference 

After gathering data from a circuit applying the same gate to each qubit a set number of times, using the SplitGateFilter class, the gate error can be inferred. The data should be placed into the directory marked by `data_file_address`, with the files named as `Readout_{gate_num}{gate_type}Q{QubitNumber}.csv`.

```python
from src.splitgatefilter import SplitGateFilter

# 1. Setup
gate_path = './data/gate_exp/'
# "meas_path" must match the folder used in Step 1

# 2. Instantiate
# Note: data_file_address is often used for both I/O in the current script. 
# Ensure raw gate data (Readout_) is inside 'gate_path'.
gate_filter = SplitGateFilter(interested_qubits=[0],
                              gate_num=100,
                              gate_type='X',
                              work_dir='./results',
                              data_file_address=gate_path,
                              meas_cal_dir=meas_path)

```

Note the three folders. `work_dir` is the folder where the final results are placed, all plots, all posterior distribution arrays, etc. `data_file_address` contains all the measured gate data, and `meas_cal_dir` contains the calibrated informative priors.


To run inference based on the informed measurement error posteriors, we run the following

```python
gate_filter.inference(nPrior=40000, 
                        meas_sd=0.1,
                        gate_sd=0.01,
                        seed=127,
                        shots_per_point=1024,
                        use_informed_priors=True,
                        plotting=True)

```

Produces `State0_Post_Qubit0.csv` etc. Also produces with the variables:
`gate_filter.post['Qubit0']` -- The full $40000 \times 3$ posterior matrix.
`gate_filter.means['Qubit0']` -- The average error rates [Meas0, Meas1, Gate].

To output the fidelity of qubit, both towards and from the ground state, along with the gate error, return `gate_filter.mean()['Qubit0']` or similar for any number Qubit.

### Plotting

```python
import matplotlib.pyplot as plt
import scipy.stats as ss

# Access the Gate Error Column (Index 2)
gate_errors = gate_filter.post['Qubit0'][:, 2]

# Create KDE
kde = ss.gaussian_kde(gate_errors)
x_axis = np.linspace(min(gate_errors), max(gate_errors), 200)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x_axis, kde(x_axis), color='purple', label='Posterior Gate Error')
plt.fill_between(x_axis, kde(x_axis), color='purple', alpha=0.2)
plt.title("Characterized Gate Error (With Informed Measurement Priors)")
plt.xlabel("Gate Error Rate")
plt.ylabel("Density")
plt.legend()
plt.show()
```

### Denoising

It is difficult and often intractable to denoise using the gate bitflip errors, the best method would be to compile qubits to utilise their least noisy connections, and gates, thus minimising errors. 

However, a kind of efficient denoiser of **measurement error** in polynomial time is possible, by assuming that measurement error is not correlated (thus, ignoring cross-talk). Such a method is here developed for use of denoising quantum bitstring outputs to better estimate the final quantum state. To denoise a given dictionary of bitstring counts, infer the measurement error rates using the SplitMeasFilter class along with gathered data, and  run the following:

```python
from src.splitmeasfilter import SplitMeasFilter

qubits = [0,1,...]
meas_filter = SplitMeasFilter(qubit_order=qubits, file_address='./data/meas_cal')

data_example = {
    "000": 100,
    "001": 50,
    "010": 75,
    "011": 25,
    "111": 1,
}

# Here the eff_DeNoise() will take a dictionary of bitstrings and their counts, as with `data_example`, and return a denoised data dictionary
# `verbose` gives continual confirmation, GD gives an overkill gradient descent error mitigation technique,
# with lr the learning rate, and max_iter giving the maximum GD iterations
denoised_date = meas_filter.eff_DeNoise(datadict=data_example, percentage=100, verbose=True, GD = False, lr=0.1, max_iter=50)
```
