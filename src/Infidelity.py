def get_Infidelity(qubit_index, gate_name):
    """
    Returns the Infidelity for specified gates
    Logic: 
    - Clearer gates might have small bitflip error rates. (R_Z should have zero!)
    - Two-qubit gates (CZ) usually induce larger bitflip rates on both qubits. 
        I have encoded this as one rate for the controller qubit, and another for the controlled.
    - Specific qubits (e.g., q[4] here) might be bad qubits with high error.


    NOTE:
    This function should be replaced with a matrix, or dictionary, where we access very specific information for each qubit 
    and it's corresponding gate. For now, if-then statements will do!
    """
    
    # Base error rates
    ## Replace with learned matrix later...
    if gate_name == 'cz0':
        base_error = 0.01 # Larger error for entangling gates
    elif gate_name == 'cz1':
        base_error = 0.1  # Larger error for entangling gates
    elif gate_name == 'r':
        base_error = 0.1  # Smaller error for single gates
    elif gate_name == 'rz':
        base_error = 0 # Z-rotations should be virtual and very precise
    else:
        print(gate_name + ": Not a registered native gate...")
        return 0.0 # No error for other gates, flag and print if outside native gate set

    # Qubit specific multiplier (e.g., Qubit 4 is noisy)
    ## Shouldn't be here when this is rigorous as a dict or a matrix
    qubit_multiplier = 1.5 if qubit_index == 4 else 1.0
    
    return base_error * qubit_multiplier