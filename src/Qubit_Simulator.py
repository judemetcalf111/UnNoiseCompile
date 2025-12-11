from qiskit import QuantumCircuit, transpile
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator
import numpy as np
from BitFlipper import add_BitFlips
import random

def Hadamard_Simulator(num_qubits = 5, native_gates = ['r', 'rz', 'cz'], shots = 1024, simulator = AerSimulator(), circ_seed = 312, sim_seed = 254):    
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

	all_keys = set(counts.keys()) | set(noisy_counts.keys())
	
	# Sort keys to ensure the X-axis is in binary order (00000 -> 11111)
	sorted_keys = sorted(list(all_keys))
	
	# Create lists of values aligned to the sorted keys
	# .get(key, 0) ensures we put a 0 if that bitstring didn't occur in a specific circuit
	values = [counts.get(key, 0) for key in sorted_keys]
	noisy_values = [noisy_counts.get(key, 0) for key in sorted_keys]

	# 3. Plotting
	x = np.arange(len(sorted_keys))  # The label locations
	width = 0.35  # The width of the bars
	
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

	digit_sums = [0] * num_qubits

	for b_str, count in zip(sorted_keys, values):
	    for i, bit in enumerate(b_str):
	        if bit == '1':
	            digit_sums[i] += count / (shots * num_qubits)
	
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