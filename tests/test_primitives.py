"""V2 primitives, against a stub client so the suite needs no token.

The load-bearing checks are the ones where a mistake is silent: splitting one
counts dict across classical registers, and refusing a parameter sweep that
would fan out into a pile of billed jobs.
"""
from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

from qiskit_zksf import ENGINES, TooManyJobs, ZKSFEstimator, ZKSFSampler
from qiskit_zksf.backend import ZKSFBackend


class StubClient:
    def __init__(self, payload=None):
        self.submitted = []
        self._payload = payload or {
            "id": "j", "status": "done", "engine": "clifford",
            "result": {"counts": {"000": 512, "111": 512},
                       "error_info": {"protocol": "ZCC-v0.1", "truncation_error": 0.0}},
        }

    def submit(self, circuit, shots=1024, engine=None, observable=None, **params):
        self.submitted.append({"shots": shots, "engine": engine, "observable": observable})
        return f"j{len(self.submitted)}"

    def job(self, job_id):
        return dict(self._payload, id=job_id)


def backend_with(payload=None, name="zksf_auto"):
    stub = StubClient(payload)
    spec = next(s for s in ENGINES if s.backend_name == name)
    return ZKSFBackend(spec, stub), stub


def ghz(n=3):
    qc = QuantumCircuit(n, n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure(range(n), range(n))
    return qc


# ----------------------------------------------------------------- sampler

def test_sampler_returns_counts_under_the_register_name():
    b, _ = backend_with()
    res = ZKSFSampler(b).run([ghz(3)], shots=1024).result()
    assert len(res) == 1
    counts = res[0].data.c.get_counts()
    assert counts == {"000": 512, "111": 512}


def test_sampler_names_the_field_after_the_register():
    """measure_all() creates a register called 'meas', and a DataBin field has
    to match it or downstream code cannot find its own results."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()
    b, _ = backend_with()
    res = ZKSFSampler(b).run([qc]).result()
    assert hasattr(res[0].data, "meas")
    assert res[0].data.meas.get_counts() == {"000": 512, "111": 512}


def test_counts_split_across_two_registers():
    """One service-side bitstring has to be cut into the right per-register
    slices. Getting the offset wrong silently returns plausible wrong labels."""
    from qiskit.circuit import ClassicalRegister, QuantumRegister

    qr = QuantumRegister(3)
    a, b_reg = ClassicalRegister(1, "a"), ClassicalRegister(2, "b")
    qc = QuantumCircuit(qr, a, b_reg)
    qc.measure(0, a[0])
    qc.measure(1, b_reg[0])
    qc.measure(2, b_reg[1])

    # clbit order is a[0], b[0], b[1]; the service string is MSB-first, so
    # "101" means b[1]=1, b[0]=0, a[0]=1.
    payload = {"id": "j", "status": "done", "engine": "exact.cpu",
               "result": {"counts": {"101": 10}, "error_info": {}}}
    backend, _ = backend_with(payload)
    res = ZKSFSampler(backend).run([qc]).result()
    assert res[0].data.a.get_counts() == {"1": 10}
    assert res[0].data.b.get_counts() == {"10": 10}


def test_sampler_carries_the_error_statement():
    b, _ = backend_with()
    res = ZKSFSampler(b).run([ghz(3)]).result()
    assert res[0].metadata["error_info"]["protocol"] == "ZCC-v0.1"


def test_sampler_passes_shots_through():
    b, stub = backend_with()
    ZKSFSampler(b).run([ghz(3)], shots=333).result()
    assert stub.submitted[0]["shots"] == 333


def test_sampler_pub_shots_beat_the_run_default():
    b, stub = backend_with()
    ZKSFSampler(b, default_shots=10).run([(ghz(3), None, 777)]).result()
    assert stub.submitted[0]["shots"] == 777


def test_several_pubs_give_several_results():
    b, _ = backend_with()
    res = ZKSFSampler(b).run([ghz(3), ghz(3)]).result()
    assert len(res) == 2


# --------------------------------------------------------------- estimator

def test_estimator_returns_an_expectation_value():
    payload = {"id": "j", "status": "done", "engine": "pauli.cpu",
               "result": {"counts": {}, "expectation": -1.137,
                          "error_info": {"error_bound": 2.5e-4}}}
    b, _ = backend_with(payload, name="zksf_pauli")
    qc = QuantumCircuit(2)
    qc.h(0)
    res = ZKSFEstimator(b).run([(qc, SparsePauliOp("ZZ"))]).result()
    assert res[0].data.evs == pytest.approx(-1.137)


def test_estimator_reports_the_bound_as_the_standard_error():
    """The protocol already produces an uncertainty on the value, so it carries
    through rather than being zeroed or invented."""
    payload = {"id": "j", "status": "done", "engine": "pauli.cpu",
               "result": {"expectation": 1.0, "error_info": {"error_bound": 3.2e-8}}}
    b, _ = backend_with(payload, name="zksf_pauli")
    qc = QuantumCircuit(2)
    res = ZKSFEstimator(b).run([(qc, SparsePauliOp("ZI"))]).result()
    assert res[0].data.stds == pytest.approx(3.2e-8)


def test_estimator_sends_the_observable_in_service_form():
    payload = {"id": "j", "status": "done", "engine": "pauli.cpu",
               "result": {"expectation": 0.5, "error_info": {}}}
    b, stub = backend_with(payload, name="zksf_pauli")
    qc = QuantumCircuit(2)
    ZKSFEstimator(b).run([(qc, SparsePauliOp.from_list([("ZZ", 0.7)]))]).result()
    assert stub.submitted[0]["observable"] == [[0.7, "ZZ"]]


# ------------------------------------------------------------ cost guards

def test_a_parameter_sweep_is_refused_rather_than_billed():
    """Each binding is a separate billed job. A local simulator would run a
    thousand without comment; here that is somebody's money."""
    theta = Parameter("t")
    qc = QuantumCircuit(1, 1)
    qc.rx(theta, 0)
    qc.measure(0, 0)

    b, stub = backend_with()
    sweep = np.linspace(0, np.pi, 50).reshape(50, 1)
    with pytest.raises(TooManyJobs, match="50 separate billed jobs"):
        ZKSFSampler(b).run([(qc, sweep)])
    assert stub.submitted == [], "nothing should have been submitted"


def test_the_sweep_limit_can_be_raised_deliberately():
    """A PUB always yields exactly one result. Bindings are carried inside the
    array shape, not as extra results, which is what Qiskit's own primitives
    do and what downstream code indexes against."""
    theta = Parameter("t")
    qc = QuantumCircuit(1, 1)
    qc.rx(theta, 0)
    qc.measure(0, 0)
    b, stub = backend_with({"id": "j", "status": "done", "engine": "exact.cpu",
                            "result": {"counts": {"0": 4}, "error_info": {}}})
    res = ZKSFSampler(b, max_jobs_per_run=4).run(
        [(qc, np.linspace(0, np.pi, 3).reshape(3, 1))]
    ).result()
    assert len(res) == 1, "one PUB, one result"
    assert res[0].data.c.shape == (3,), "bindings belong in the shape"
    assert len(stub.submitted) == 3, "still three billed jobs"


def test_scalar_pub_gives_scalar_shape():
    """A single circuit at a single point has shape (), like the reference
    primitives. Returning (1,) instead makes float(evs) a deprecation warning
    today and an error later."""
    b, _ = backend_with()
    res = ZKSFSampler(b).run([ghz(3)]).result()
    assert res[0].data.c.shape == ()


def test_estimator_scalar_observable_gives_scalar_shape():
    payload = {"id": "j", "status": "done", "engine": "pauli.cpu",
               "result": {"expectation": -1.0, "error_info": {"error_bound": 0.0}}}
    b, _ = backend_with(payload, name="zksf_pauli")
    res = ZKSFEstimator(b).run([(QuantumCircuit(2), SparsePauliOp("ZZ"))]).result()
    assert np.asarray(res[0].data.evs).shape == ()
    assert float(res[0].data.evs) == pytest.approx(-1.0)


def test_estimator_observable_array_keeps_its_shape():
    payload = {"id": "j", "status": "done", "engine": "pauli.cpu",
               "result": {"expectation": 0.25, "error_info": {"error_bound": 1e-9}}}
    b, _ = backend_with(payload, name="zksf_pauli")
    obs = [SparsePauliOp("ZZ"), SparsePauliOp("XX"), SparsePauliOp("YY")]
    res = ZKSFEstimator(b).run([(QuantumCircuit(2), obs)]).result()
    assert np.asarray(res[0].data.evs).shape == (3,)


def test_observable_array_with_a_sweep_is_refused_not_guessed():
    """Broadcasting the two together is not implemented, and inventing an
    ordering would mislabel every value returned."""
    theta = Parameter("t")
    qc = QuantumCircuit(2, 2)
    qc.rx(theta, 0)
    b, _ = backend_with({"id": "j", "status": "done", "engine": "pauli.cpu",
                         "result": {"expectation": 0.0, "error_info": {}}},
                        name="zksf_pauli")
    with pytest.raises(NotImplementedError, match="parameter sweep"):
        ZKSFEstimator(b).run(
            [(qc, [SparsePauliOp("ZZ"), SparsePauliOp("XX")],
              np.linspace(0, 1, 2).reshape(2, 1))]
        )


def test_estimator_refuses_a_large_observable_array():
    payload = {"id": "j", "status": "done", "engine": "pauli.cpu",
               "result": {"expectation": 0.0, "error_info": {}}}
    b, stub = backend_with(payload, name="zksf_pauli")
    qc = QuantumCircuit(2)
    obs = [SparsePauliOp(p) for p in ("II", "IZ", "ZI", "ZZ", "XX", "YY",
                                      "XI", "IX", "YI", "IY", "XY", "YX",
                                      "XZ", "ZX", "YZ", "ZY", "II")]
    with pytest.raises(TooManyJobs):
        ZKSFEstimator(b).run([(qc, obs)])
    assert stub.submitted == []
