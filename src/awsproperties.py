# general imports
import json
import numpy as np

from braket.aws import AwsDevice
from braket.devices import Devices, LocalSimulator

def get_braket_calibration_dict(device_arn, n_qubits=None):
    """
    Returns the params list in the exact format required by MeasFilter.
    """
    device = AwsDevice(device_arn)
    properties = device.properties
    standardized = getattr(properties, "standardized", None)
    if standardized and hasattr(standardized, "dict"):
        qubit_props = standardized.dict()
    else:
        try:
            qubit_props = properties.dict()
        except:
            qubit_props = {}
    
    one_props = qubit_props["oneQubitProperties"]
    two_props = qubit_props["twoQubitProperties"]
    one_props_len = len(one_props)

    if n_qubits is None:
        n_qubits = one_props_len
    else:
        if one_props_len != n_qubits:
            raise ValueError(f"Warning: Provided n_qubits={n_qubits} does not match device oneQubitProperties length={one_props_len}. Using device length.")
            
    properties = {}

    for q in range(n_qubits):
        q_one_prop = one_props[str(q)]["oneQubitFidelity"]
        q_two_props = {k: v["twoQubitGateFidelity"][0]["fidelity"]
                        for k, v in two_props.items() if str(q) in k.split("-")}

        pm1p0 = q_one_prop[1]["fidelity"]
        pm0p1 = q_one_prop[2]["fidelity"]
        properties[str(q)] = {"pm1p0": pm1p0, "pm0p1": pm0p1, "twoQubitProperties": q_two_props}