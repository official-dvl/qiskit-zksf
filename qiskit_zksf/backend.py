"""ZKSF backends exposed through Qiskit's BackendV2 interface.

One backend per engine, because the engines differ in the two things a Target
has to state honestly: how many qubits they reach, and which gates they accept.
A single backend advertising the union of all of them would let the transpiler
produce circuits the chosen engine then rejects.
"""
from __future__ import annotations

from dataclasses import dataclass

from qiskit.circuit import Measure, Parameter, QuantumCircuit, Reset
from qiskit.circuit.library import standard_gates as sg
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target

# A universal set covering what the service accepts as OpenQASM 2, which is
# qelib1 plus the rzz/rxx/ryy rotations that QAOA circuits are written with.
_UNIVERSAL = [
    sg.HGate(), sg.XGate(), sg.YGate(), sg.ZGate(),
    sg.SGate(), sg.SdgGate(), sg.TGate(), sg.TdgGate(),
    sg.SXGate(), sg.SXdgGate(), sg.IGate(),
    sg.RXGate(Parameter("θ")), sg.RYGate(Parameter("θ")), sg.RZGate(Parameter("θ")),
    sg.PhaseGate(Parameter("θ")),
    sg.UGate(Parameter("θ"), Parameter("φ"), Parameter("λ")),
    sg.CXGate(), sg.CYGate(), sg.CZGate(), sg.CHGate(), sg.SwapGate(),
    sg.CRZGate(Parameter("θ")), sg.CPhaseGate(Parameter("θ")),
    sg.CUGate(Parameter("θ"), Parameter("φ"), Parameter("λ"), Parameter("γ")),
    sg.CCXGate(), sg.CSwapGate(),
    sg.RZZGate(Parameter("θ")), sg.RXXGate(Parameter("θ")), sg.RYYGate(Parameter("θ")),
]

# The stabilizer engine is exact at any scale but only for Clifford circuits.
# Advertising T or arbitrary rotations here would invite the transpiler to emit
# something Stim cannot represent, and the rejection would arrive from the
# server rather than from the transpiler where it belongs.
_CLIFFORD = [
    sg.HGate(), sg.XGate(), sg.YGate(), sg.ZGate(), sg.SGate(), sg.SdgGate(),
    sg.CXGate(), sg.CYGate(), sg.CZGate(), sg.SwapGate(), sg.IGate(),
]


@dataclass(frozen=True)
class EngineSpec:
    backend_name: str
    engine: str | None       # None lets the service's router choose
    num_qubits: int
    description: str
    clifford_only: bool = False
    hardware: bool = False


# Qubit counts mirror the limits the service actually enforces, so a circuit
# that would be rejected fails locally at transpile time instead of after a
# round trip. Where a method has no hard ceiling (stabilizer, Pauli
# propagation) the figure is a practical one, and the service remains the
# authority.
ENGINES: tuple[EngineSpec, ...] = (
    EngineSpec("zksf_auto", None, 128,
               "Router picks the cheapest adequate engine for the circuit"),
    EngineSpec("zksf_exact_cpu", "exact.cpu", 30,
               "Exact statevector on CPU"),
    EngineSpec("zksf_exact_gpu", "exact.gpu", 29,
               "Exact statevector on GPU"),
    EngineSpec("zksf_mps", "mps.quimb.cpu", 128,
               "Matrix product state (quimb), supports certified=True"),
    EngineSpec("zksf_mps_aer", "mps.aer.cpu", 128,
               "Matrix product state (Aer)"),
    EngineSpec("zksf_clifford", "clifford", 5000,
               "Stabilizer (Stim), exact for Clifford circuits at any scale",
               clifford_only=True),
    EngineSpec("zksf_pauli", "pauli.cpu", 1024,
               "Pauli propagation in the Heisenberg picture, needs an observable"),
    EngineSpec("zksf_noisy", "noisy.cpu", 30,
               "Device-class noise model, supports mitigate=True"),
    EngineSpec("zksf_rigetti", "qpu.rigetti", 108,
               "Rigetti Cepheus-1 superconducting hardware", hardware=True),
    EngineSpec("zksf_ionq", "qpu.ionq", 36,
               "IonQ Forte-1 trapped-ion hardware", hardware=True),
)


class ZKSFBackend(BackendV2):
    """A single ZKSF engine, addressable as a Qiskit backend."""

    def __init__(self, spec: EngineSpec, client, provider=None):
        super().__init__(
            provider=provider,
            name=spec.backend_name,
            description=spec.description,
            backend_version="0.1.0",
        )
        self._spec = spec
        self._client = client
        self._target: Target | None = None

    @property
    def spec(self) -> EngineSpec:
        return self._spec

    @property
    def target(self) -> Target:
        # Built lazily and without per-qubit properties: these are simulators
        # and cloud hardware with no coupling map published here, so every
        # instruction applies to every qubit. Attaching properties per qubit
        # would make constructing a 5000-qubit target pointlessly expensive.
        if self._target is None:
            target = Target(num_qubits=self._spec.num_qubits)
            gates = _CLIFFORD if self._spec.clifford_only else _UNIVERSAL
            for gate in gates:
                target.add_instruction(gate, name=gate.name)
            target.add_instruction(Measure(), name="measure")
            target.add_instruction(Reset(), name="reset")
            self._target = target
        return self._target

    @property
    def max_circuits(self) -> None:
        # The service has no batch endpoint yet, so several circuits are
        # submitted as several jobs. There is no limit worth declaring, but
        # each circuit is a separate billed job: see the README.
        return None

    @classmethod
    def _default_options(cls) -> Options:
        return Options(shots=1024, observable=None, certified=False, mitigate=False)

    def run(self, run_input, **options):
        """Submit one or more circuits. Returns a ZKSFJob."""
        from qiskit_zksf.job import ZKSFJob

        circuits = [run_input] if isinstance(run_input, QuantumCircuit) else list(run_input)
        if not circuits:
            raise ValueError("no circuits to run")

        opts = {**self.options.__dict__, **options}
        shots = int(opts.pop("shots", 1024))
        observable = opts.pop("observable", None)
        # Anything left over is passed through to the engine as params, which
        # is how certified=True and mitigate=True reach it.
        params = {k: v for k, v in opts.items() if v not in (None, False)}

        job_ids = [
            self._client.submit(
                circuit,
                shots=shots,
                engine=self._spec.engine,
                observable=observable,
                **params,
            )
            for circuit in circuits
        ]
        return ZKSFJob(self, job_ids, circuits, shots)
