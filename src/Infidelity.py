def get_Infidelity(qubit_index, gate_name):
    """
    Returns the angle for the RX error gate.
    Logic: 
    - Faster gates (R) might have small over-rotations.
    - Two-qubit gates (CX) usually induce larger errors on both qubits.
    - specific qubits (e.g., q[4]) might be 'bad' qubits with higher error.
    """
    
    # Base error rates
    ## Replace with learned matrix later...
    if gate_name == 'cz0':
        base_error = 0.01 # Larger error for entangling gates
    elif gate_name == 'cz1':
        base_error = 0.1  # Larger error for entangling gates
    elif gate_name == 'r':
        base_error = 0.1  # Smaller error for single rotations
    elif gate_name == 'rz':
        base_error = 0 # Z-rotations are usually virtual and precise
    else:
        print(gate_name + ": Not a registered native gate...")
        return 0.0 # No error for other gates, flag if outside native gate set

    # Qubit specific multiplier (e.g., Qubit 4 is noisy)
    ## Make a matrix when have data
    qubit_multiplier = 1.5 if qubit_index == 4 else 1.0
    
    return base_error * qubit_multiplier