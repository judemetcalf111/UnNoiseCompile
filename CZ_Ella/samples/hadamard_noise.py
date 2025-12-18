# Create path to src directory
# use specified path if running on e.g. a HPC system 
import sys
sys.path.insert(1, '../src')

import numpy as np
from Qubit_Simulator import Hadamard_Simulator

if __name__ == "__main__":
	if len(sys.argv) == 2: 
		results = Hadamard_Simulator(num_qubits = int(sys.argv[1]), shots = 2**10, measure_error = True)#, datafile = "qubit_data")
		print("\nThe average value of each qubit, from 0 to " + f"{sys.argv[1]} is: " + str(results))
	elif len(sys.argv) == 3:
		results = Hadamard_Simulator(num_qubits = int(sys.argv[1]), shots = int(sys.argv[2]), measure_error = True)#, datafile = "qubit_data")
		print("\nThe average value of each qubit, from 0 to " + f"{sys.argv[1]} is: " + str(results))
	elif len(sys.argv) == 4:
		noise_data = np.loadtxt(sys.argv[3], delimiter=',')
		if len(noise_data) != int(sys.argv[1]):
			raise Exception("Error: The number of qubits does not match the length of the noise data provided!")
		results = Hadamard_Simulator(num_qubits = int(sys.argv[1]), shots = int(sys.argv[2]), measure_error = True, data = noise_data)#, datafile = "qubit_data")
		print("\nThe average value of each qubit, from 0 to " + f"{sys.argv[1]} is: " + str(results))
	else:
		print("Too many arguments!!")
