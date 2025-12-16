from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate
import random
from math import pi
from Infidelity import get_Infidelity

def add_BitFlips(original_qc: QuantumCircuit, seed = 27, gate_errors = None) -> QuantumCircuit:
    # 1. Create a new empty circuit
    random.seed(seed)
    noisy_qc = original_qc.copy_empty_like()
    ignore_ops = {'measure', 'barrier', 'reset', 'snapshot'}
    # 3. Iterate through the original instructions
    for instruction in original_qc.data:
    
        # A. Append the original gate
        noisy_qc.append(instruction)
        
        op_name = instruction.operation.name
        
        # B. Check if we should inject noise
        if op_name == 'cz':
            control_switch = 0
            for q in instruction.qubits:
                q_idx = original_qc.find_bit(q).index
                
                if gate_errors == None:
                    current_infidelity = get_Infidelity(q_idx, op_name + str(control_switch))
                else:
                    try:
                        current_infidelity = gate_errors[q_idx]
                    except TypeError:
                        print("'gate_index' should be a list or array")
                if current_infidelity > random.random():
                    bit_flip = RXGate(pi)
                    noisy_qc.append(bit_flip, [q])
                    
                    control_switch = 1
        
        elif op_name not in ignore_ops:
        
            q = instruction.qubits[0]
            q_idx = original_qc.find_bit(q).index
            
            if gate_errors == None:
                current_infidelity = get_Infidelity(q_idx, op_name)
            else:
                try:
                    current_infidelity = gate_errors[q_idx]
                except TypeError:
                    print("'gate_index' should be a list or array")                
            if current_infidelity > random.random():
                bit_flip = RXGate(pi)
                noisy_qc.append(bit_flip, [q])
    
    return noisy_qc