# Add path
import sys
sys.path.insert(1, '../src')

from BitFlipper import add_BitFlips
from Qubit_Simulator import pad_counts
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np
import random
from tqdm import tqdm

def CZ_errors(num_gates):
    qc = QuantumCircuit(2)

    for i in range(num_gates):
        qc.cz(0,1)
    qc.measure_all()

    return qc


# sim_seeds = np.random.randint(0,1073741824,1048576,dtype=int)
sim_seeds = np.load('../data/sim_seeds.npy')
sim_seeds = np.array(sim_seeds,dtype = int)/10000


num_qubits = 2
shots_per_circuit = 32
e_g1 = 0.0035
e_g2 = 0.0027
num_gates = round(0.2/(max(e_g1,e_g2)))
test_circuit = CZ_errors(num_gates)

simulator = AerSimulator()
flipped_counts = np.zeros(4)
bitstring_list = [
        format(i, f'0{num_qubits}b') 
        for i in range(2**num_qubits)
    ]


for sim_seed in tqdm(sim_seeds):
    flipped_circuit = add_BitFlips(test_circuit, seed = sim_seed, gate_errors = [e_g1,e_g2])
    flipped_result = simulator.run(flipped_circuit, shots=shots_per_circuit).result()
    counts = pad_counts(flipped_result.get_counts(),2)
    bitstring_counts = [counts.get(key, 0) for key in bitstring_list]
    flipped_counts += bitstring_counts

print(bitstring_list)
flipped_counts

raw_output = []

for i, bitstr in enumerate(bitstring_list):
    raw_output.extend([bitstr] * int(flipped_counts[i]))

np.savetxt('../data/Filter_data_CZ.csv', np.array(raw_output).reshape(1, -1), delimiter=',', fmt='%s')
