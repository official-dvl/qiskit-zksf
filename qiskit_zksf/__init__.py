"""Qiskit provider for ZKSF (Zero Kelvin Simulation Foundry).

Point existing Qiskit code at classical simulators, GPU accelerators or real
quantum processors, and get a documented accuracy statement back with every
approximate result.

    from qiskit_zksf import ZKSFProvider

    backend = ZKSFProvider().backend("zksf_auto")
    job = backend.run(circuit, shots=1000)

    job.result().get_counts()   # the answer
    job.error_info()            # how far it may be from the truth
"""
from qiskit_zksf.backend import ENGINES, EngineSpec, ZKSFBackend
from qiskit_zksf.job import JobRejected, ZKSFJob
from qiskit_zksf.primitives import TooManyJobs, ZKSFEstimator, ZKSFSampler
from qiskit_zksf.provider import ZKSFProvider

__version__ = "0.2.1"
__all__ = [
    "ZKSFProvider",
    "ZKSFBackend",
    "ZKSFJob",
    "ZKSFSampler",
    "ZKSFEstimator",
    "JobRejected",
    "TooManyJobs",
    "EngineSpec",
    "ENGINES",
]
