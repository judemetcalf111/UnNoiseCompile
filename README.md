# UnNoiseCompile
A repository for developing _scalable_ and hardware-aware quantum compilation to minimise bitflip gate and measurement noise.

## Folder Structure

Much of this repository is built on proper management of experimental data gathered from probe circuits to be used on QPUs. As such, folder management and organistaion is crucial. The following is strongly suggested to ensure that the functions run smoothly without too much manipulation of subfolders:

Project_Root/  
│  
├── notebooks/  
│   ├── notebook1.ipynb  
│   └── notebook2.ipynb.     <-- Any notebooks to run scripts, gather AWS data, inference, etc.  
│  
├── scripts/  
│   ├── splitmeasfilter.py   <-- Measurement Error & Inference  
│   └── splitgatefilter.py   <-- Gate Error & Inference, Incoroporating Measurement Errors from splitmeasfilter.py  
│  
├── data/  
│   ├── meas_cal/            <-- Output folder for MeasFilter  
│   └── gate_exp/            <-- Output folder for GateFilter  
│  
└── results/                 <-- Final plots and reports  

## Getting Started

Built using Python 3.11.14, mostly using Jupyter notebooks in `/notebooks` and `/scripts/` and Python modules in `/src`. To make a .venv: 

`python3.11 -m venv .venv`

Activate:

`source .venv/bin/activate`

Then install packages:

`pip install -r requirements.txt`

And run from there!

## The Workflow

### 1. Measurement Data

You can either supply data into `SplitMeasFilter`directly, using an array-like object called `data`, filled with bitstrings, or by placing a similar file called `Filter_data.csv` in the path defined by `file_address`.

The script has the capacity to run inference on data gathered from an expected |0> state or an expected |1> state, gathering bitflip error rates in both directions (both towards and away from the ground-state)

```python
from splitmeasfilter import SplitMeasFilter

# 1. Define Qubits and Data/Path
qubits = [0, 1, 6]
meas_path = './data/meas_cal/' # Pointing to a directory containing `Filter_data.csv`
meas_data = ['011110','111110','110110','010010','111111','111101','110100',...]

# 2a. Instantiate Data (file)
meas_filter = SplitMeasFilter(qubit_order=qubits, file_address=meas_path)

# 2b. Instantiate Data (array-like)
meas_filter = SplitMeasFilter(qubit_order=qubits, data=meas_data)
```

We can then run inference on this data. Ensure that you track the number of shots, as this will divide the total into several batches, each of size 'shots_per_point':

```python
# Applying the meas_filter.inference() function, on batches of size shots_per_point
# The 'prep_state' variable can be '0' or '1', and enters the prepared state, either
# 0 or 1, and thus defines the error rate tested for.
meas_filter.inference(nPrior=40000, Prior_sd=0.1, shots_per_point=1024, seed=28, prep_state='0')
```

This will produce files `State0_Post_Qubit0.csv`, `State0_Post_Qubit1.csv`, and `State0_Post_Qubit6.csv`, which contain the posterior distributions of measurement errors, in the form of rejection-sampled error rates. This will output plots of observed data against the posterior model. To reproduce these, along with the distribution statistics, and optionally save, run:
```python
meas_filter.error_distributions(plotting=True, save_plots=True)
```
And this will produce the error rate variables in the `MeasFilter` class. The following gives the `p(0|0)` error rate for Qubit0:
```python
meas_filter.post_marginals['Qubit0']['0']
```


### 2. Gate Data

After gathering data from a circuit applying the same gate to each qubit a set number of times, using the SplitGateFilter class, the gate error can be inferred. The data should be placed into the directory marked by `data_file_address`, with the files named as `Readout_{gate_num}{gate_type}Q{QubitNumber}.csv`.

```python
from splitgatefilter import SplitGateFilter

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

It is difficult and often intractable to denoise using the gate bitflip errors, the best method would be to compile the least noisy gates, to minimise errors. However, an efficient polynomial denoiser is here developed for the measurement error outputs. To denoise, run:

```python
from splitmeasfilter import SplitMeasFilter

qubits = [0,1,...]
meas_filter = SplitMeasFilter(qubit_order=qubits, file_address='./data/meas_cal')

data_example = {
    "000": 100,
    "001": 50,
    "010": 75,
    "011": 25,
    "111": 1,
}

# Here the eff_DeNoise() will take a dictionary of bitstrings and counts, as with `data_example` and return denoised data dict
# `verbose` gives continual confirmation, GD gives an overkill gradient descent error mitigation technique,
# with lr the learning rate, and max_iter giving the maximum GD iterations
denoised_date = meas_filter.eff_DeNoise(datadict=data_example, percentage=100, verbose=True, GD = False, lr=0.1, max_iter=50)
```
