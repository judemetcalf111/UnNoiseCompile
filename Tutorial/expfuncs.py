# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 12:52:17 2020
Updated for Qiskit 2.2.3 & qiskit-aer
"""

import csv
import numpy as np

# Qiskit 1.0+ Imports
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

def QAOAexp(backend, file_address=''):
    """
        QAOA from https://arxiv.org/abs/1804.03719

    Parameters
    ----------
    backend : Backend
        Qiskit backend (Simulator or Real Device).
    file_address : String, optional
        address for save data. The default is ''. Ends with "/" if not empty.

    Returns
    -------
    None.

    """
    pi = np.pi
    g1 = 0.2 * pi
    g2 = 0.4 * pi
    b1 = 0.15 * pi
    b2 = 0.05 * pi

    num = 5
    QAOA = QuantumCircuit(num, num)

    for i in range(1, 5):
        QAOA.h(i)
    QAOA.barrier()

    # k = 1
    QAOA.cx(3, 2)
    QAOA.p(-g1, 2)
    QAOA.cx(3, 2)
    QAOA.barrier()

    QAOA.cx(4, 2)
    QAOA.p(-g1, 2)
    QAOA.cx(4, 2)
    QAOA.barrier()

    QAOA.cx(1, 2)
    QAOA.p(-g1, 2)
    QAOA.cx(1, 2)
    QAOA.cx(4, 3)
    QAOA.p(-g1, 3)
    QAOA.cx(4, 3)
    QAOA.barrier()

    for i in range(1, 5):
        # QAOA.u3 -> Replaced with u
        QAOA.u(2 * b1, -pi / 2, pi / 2, i)
    QAOA.barrier()

    # k = 2
    QAOA.cx(3, 2)
    QAOA.p(-g2, 2)
    QAOA.cx(3, 2)
    QAOA.barrier()

    QAOA.cx(4, 2)
    QAOA.p(-g2, 2)
    QAOA.cx(4, 2)
    QAOA.barrier()

    QAOA.cx(1, 2)
    QAOA.p(-g2, 2)
    QAOA.cx(1, 2)
    QAOA.cx(4, 3)
    QAOA.p(-g2, 3)
    QAOA.cx(4, 3)
    QAOA.barrier()

    for i in range(1, 5):
        QAOA.u(2 * b2, -pi / 2, pi / 2, i)
    QAOA.barrier()

    QAOA.barrier()
    QAOA.measure([1, 2, 3, 4], [1, 2, 3, 4])
    
    # Transpile with optimization_level=0 to preserve structure
    QAOA_trans = transpile(QAOA, backend, initial_layout=range(num), optimization_level=0)
    print('QAOA circuit depth is ', QAOA_trans.depth())

    # --- Run on simulator ---
    simulator = AerSimulator()
    simu_shots = 100000
    # Transpile for simulator
    sim_circ = transpile(QAOA, simulator)
    sim_job = simulator.run(sim_circ, shots=simu_shots)
    QAOA_results = sim_job.result()
    
    with open(file_address + 'Count_QAOA_Simulator.csv', mode='w',
              newline='') as sgm:
        count_writer = csv.writer(sgm,
                                  delimiter=',',
                                  quotechar='"',
                                  quoting=csv.QUOTE_MINIMAL)
        for key, val in QAOA_results.get_counts().items():
            count_writer.writerow([key, val])

    # --- Run on real device ---
    shots = 8192
    # Use backend.run instead of execute
    job_exp = backend.run(QAOA_trans, shots=shots)
    
    print("Job id:", job_exp.job_id())
    # job_monitor removed for compatibility
    
    try:
        exp_results = job_exp.result()
        with open(file_address + 'Count_QAOA.csv', mode='w', newline='') as sgm:
            count_writer = csv.writer(sgm,
                                      delimiter=',',
                                      quotechar='"',
                                      quoting=csv.QUOTE_MINIMAL)
            for key, val in exp_results.get_counts().items():
                count_writer.writerow([key, val])
    except Exception as e:
        print(f"Error retrieving/saving results: {e}")


def Groverexp(backend, file_address=''):
    """
        Gorver's search from https://arxiv.org/abs/1804.03719

    Parameters
    ----------
    backend : Backend
        backend.
    file_address : String, optional
        address for save data. The default is ''. Ends with "/" if not empty.

    Returns
    -------
    None.

    """
    num = 3
    Grover = QuantumCircuit(num, num)

    Grover.x(0)
    Grover.h(1)
    Grover.h(2)
    Grover.barrier()

    Grover.h(0)
    Grover.barrier()

    Grover.h(0)

    Grover.cx(1, 0)
    Grover.tdg(0)
    Grover.cx(2, 0)
    Grover.t(0)

    Grover.cx(1, 0)
    Grover.tdg(0)
    Grover.cx(2, 0)
    Grover.barrier()
    Grover.t(0)
    Grover.tdg(1)
    Grover.barrier()

    Grover.h(0)
    Grover.cx(2, 1)
    Grover.tdg(1)
    Grover.cx(2, 1)
    Grover.s(1)
    Grover.t(2)
    Grover.barrier()

    Grover.h(1)
    Grover.h(2)
    Grover.barrier()
    Grover.x(1)
    Grover.x(2)
    Grover.barrier()
    Grover.h(1)
    Grover.cx(2, 1)
    Grover.h(1)
    Grover.x(2)
    Grover.barrier()
    Grover.x(1)
    Grover.h(2)
    Grover.barrier()
    Grover.h(1)

    Grover.barrier()
    Grover.measure([1, 2], [1, 2])
    
    # Transpile with optimization_level=0 to preserve structure
    Grover_trans = transpile(Grover, backend, initial_layout=[0, 1, 2], optimization_level=0)
    print('Grover circuit depth is ', Grover_trans.depth())

    # Run on real device
    shots = 8192
    # Use backend.run instead of execute
    job_exp = backend.run(Grover_trans, shots=shots)
    
    print("Job id:", job_exp.job_id())
    # job_monitor removed for compatibility
    
    try:
        exp_results = job_exp.result()
        with open(file_address + 'Count_Grover.csv', mode='w', newline='') as sgm:
            count_writer = csv.writer(sgm,
                                      delimiter=',',
                                      quotechar='"',
                                      quoting=csv.QUOTE_MINIMAL)
            for key, val in exp_results.get_counts().items():
                count_writer.writerow([key, val])
    except Exception as e:
        print(f"Error retrieving/saving results: {e}")
