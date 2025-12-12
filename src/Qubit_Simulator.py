import numpy as np
import matplotlib.pyplot as plt
import random
import json
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from MeasureMatrix import fast_interaction_multiply
from BitFlipper import add_BitFlips

def pad_counts(counts, num_qubits):
    # Generate all bitstrings
    all_bitstrings = [
        format(i, f'0{num_qubits}b') 
        for i in range(2**num_qubits)
    ]
    
    # Create a new dict with 0 counts for all bitstrings
    full_counts = {b: 0 for b in all_bitstrings}
    
    # Update 
    full_counts.update(counts)
    return full_counts


def Hadamard_Simulator(num_qubits = 5, native_gates = ['r', 'rz', 'cz'], shots = 1024, simulator = AerSimulator(), circ_seed = 312, sim_seed = 254, measure_error = False, datafile = None, data = None, plotting = True):    
	# Create GHZ circuit
	
	qc = QuantumCircuit(num_qubits)
	for i in range(num_qubits):
		qc.h(i)
	 
	qc.measure_all()
	
	# Transpile to native gates
	circuit = transpile(qc, basis_gates=native_gates, optimization_level=3)
	noisy_circuit = add_BitFlips(circuit, seed = circ_seed)
	
	# Run the circuit 
	random.seed(sim_seed)
	result = simulator.run(circuit, shots=shots, seed = sim_seed).result()
	noisy_result = simulator.run(noisy_circuit, shots=shots, seed = sim_seed).result()
	
	# 3. Get counts
	counts = result.get_counts()
	noisy_counts = noisy_result.get_counts()

	counts = pad_counts(counts, num_qubits)
	noisy_counts = pad_counts(noisy_counts, num_qubits)

	all_keys_list = [format(i, f'0{num_qubits}b') for i in range(2**num_qubits)]
	
	# Sort keys to ensure the X-axis is in binary order (00000 -> 11111)
	sorted_keys = sorted(all_keys_list)
	
	# Create lists of values aligned to the sorted keys
	# .get(key, 0) ensures we put a 0 if that bitstring didn't occur in a specific circuit
	values = [counts.get(key, 0) for key in sorted_keys]
	noisy_values = [noisy_counts.get(key, 0) for key in sorted_keys]

	# 3. Plotting
	x = np.arange(len(sorted_keys))  # The label locations
	width = 0.35  # The width of the bars

	digit_sums = [0] * num_qubits

	if (measure_error == True) and (datafile is None):
		# Default error rates if no datafile is provided
		epsilon01 = np.random.uniform(0., .02, num_qubits).tolist()
		epsilon10 = np.random.uniform(0., .02, num_qubits).tolist()
		
		noisy_values = fast_interaction_multiply(noisy_values, num_qubits, epsilon01, epsilon10)

	elif (measure_error == True):
		if datafile.endswith('.json'):
			with open("../data/" + str(datafile), 'r') as file:
				data = json.load(file)
			qubit_props = data['oneQubitProperties']
			num_qubits_in_data = len(qubit_props)
			if num_qubits_in_data != num_qubits:
				raise Exception("Warning: Number of qubits in datafile does not match simulation setting!\nDatafile qubits: "
						+ str(num_qubits_in_data) + "\nSimulation qubits: " + str(num_qubits))
			epsilon01 = np.zeros(num_qubits)
			epsilon10 = np.zeros(num_qubits)

			for qub_ind,qub in enumerate(qubit_props):
				e01 = qubit_props[qub]['oneQubitFidelity'][1]['fidelity'] # Notice that even though it says "fidelity", we get error rate...
				e10 = qubit_props[qub]['oneQubitFidelity'][2]['fidelity'] # Notice that even though it says "fidelity", we get error rate...
				epsilon01[qub_ind] = e01
				epsilon10[qub_ind] = e10
			
		elif type(data) == np.ndarray:
			if data.shape[0] != num_qubits or data.shape[1] != 2:
				raise Exception("Warning: Data array shape does not match simulation setting!\nData shape: "
						+ str(data.shape) + "\nSimulation qubits: " + str(num_qubits))
			epsilon01 = data[:,0].tolist()
			epsilon10 = data[:,1].tolist()
		elif datafile is None and data is None:
			raise Exception("Error: No data or datafile provided for error measurement!")

		noisy_values = fast_interaction_multiply(noisy_values, num_qubits, epsilon01, epsilon10)

	for b_str, count in zip(sorted_keys, noisy_values):
		for i, bit in enumerate(b_str):
			if bit == '1':
				digit_sums[i] += count / (shots)

	if plotting == True:
		fig, ax = plt.subplots(figsize=(14, 6)) # Wide figure to fit labels
		
		# Plot the two sets of bars, offset by +/- width/2
		rects1 = ax.bar(x - width/2, values, width, label='Circuit', color='#648FFF')
		rects2 = ax.bar(x + width/2, noisy_values, width, label='Noisy Circuit', color='#DC267F')
		
		# 4. Formatting
		ax.set_ylabel('Counts')
		ax.set_title('Quantum Circuit Output Comparison')
		ax.set_xticks(x)
		ax.set_xticklabels(sorted_keys, rotation=90) # Rotate labels for readability
		ax.legend()
		
		# Optional: Add grid for easier reading of height
		ax.grid(axis='y', linestyle='--', alpha=0.7)
		
		plt.tight_layout()
		plt.show()
	# Output the results
	return digit_sums



# def OTHER_Simulator(num_qubits = 5, native_gates = ['r', 'rz', 'cz'], shots = 1024, simulator = AerSimulator(), circ_seed = 312, sim_seed = 254):    
# 	# Create GHZ circuit
	
# 	qc = QuantumCircuit(num_qubits)
# 	for i in range(num_qubits):
# 		qc.h(i)
	 
# 	qc.measure_all()
	
# 	# Transpile to native gates
# 	circuit = transpile(qc, basis_gates=native_gates, optimization_level=3)
# 	noisy_circuit = add_BitFlips(circuit, seed = circ_seed)
	
# 	# Run the circuit 
# 	random.seed(sim_seed)
# 	result = simulator.run(circuit, shots=shots, seed = sim_seed).result()
# 	noisy_result = simulator.run(noisy_circuit, shots=shots, seed = sim_seed).result()
	
# 	# 3. Get counts
# 	counts = result.get_counts()
# 	noisy_counts = noisy_result.get_counts()
	
# 	all_keys = set(counts.keys()) | set(noisy_counts.keys())
	
# 	# Sort keys to ensure the X-axis is in binary order (00000 -> 11111)
# 	sorted_keys = sorted(list(all_keys))
	
# 	# Create lists of values aligned to the sorted keys
# 	# .get(key, 0) ensures we put a 0 if that bitstring didn't occur in a specific circuit
# 	values = [counts.get(key, 0) for key in sorted_keys]
# 	noisy_values = [noisy_counts.get(key, 0) for key in sorted_keys]
	
# 	# 3. Plotting
# 	x = np.arange(len(sorted_keys))  # The label locations
# 	width = 0.35  # The width of the bars
	
# 	fig, ax = plt.subplots(figsize=(14, 6)) # Wide figure to fit labels
	
# 	# Plot the two sets of bars, offset by +/- width/2
# 	rects1 = ax.bar(x - width/2, values, width, label='Circuit', color='#648FFF')
# 	rects2 = ax.bar(x + width/2, noisy_values, width, label='Noisy Circuit', color='#DC267F')
	
# 	# 4. Formatting
# 	ax.set_ylabel('Counts')
# 	ax.set_title('Quantum Circuit Output Comparison')
# 	ax.set_xticks(x)
# 	ax.set_xticklabels(sorted_keys, rotation=90) # Rotate labels for readability
# 	ax.legend()
	
# 	# Optional: Add grid for easier reading of height
# 	ax.grid(axis='y', linestyle='--', alpha=0.7)
	
# 	plt.tight_layout()
# 	plt.show()

# 	return 
