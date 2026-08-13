"""Offline tests against a stub client, so the suite needs no token or network.

The load-bearing one is the counts round trip: the service answers in
bitstrings and Qiskit formats counts out of hex using the experiment header, so
a mistake there silently returns the wrong outcome labels rather than failing.
"""
from __future__ import annotations

import pytest
from qiskit import QuantumCircuit, transpile

from qiskit_zksf import ENGINES, JobRejected, ZKSFProvider


class StubClient:
    """Stands in for qsim_sdk.Client, recording what it was asked to do."""

    def __init__(self, payload=None):
        self.submitted = []
        self.estimated = []
        self._payload = payload or {
            "id": "job1",
            "status": "done",
            "engine": "clifford",
            "result": {
                "counts": {"000": 531, "111": 469},
                "error_info": {"protocol": "ZCC-v0.1", "truncation_error": 0.0},
            },
        }

    def submit(self, circuit, shots=1024, engine=None, observable=None, **params):
        self.submitted.append(
            {"circuit": circuit, "shots": shots, "engine": engine,
             "observable": observable, "params": params}
        )
        return f"job{len(self.submitted)}"

    def job(self, job_id):
        return dict(self._payload, id=job_id)

    def estimate(self, circuit, shots=1024, engine=None):
        self.estimated.append({"shots": shots, "engine": engine})
        return {"engine": engine or "clifford", "predicted_cost_usd": 0.001}


@pytest.fixture
def provider(monkeypatch):
    def make(payload=None):
        stub = StubClient(payload)
        p = ZKSFProvider.__new__(ZKSFProvider)
        p._client = stub
        from qiskit_zksf.backend import ZKSFBackend
        p._backends = {
            s.backend_name: ZKSFBackend(s, stub, provider=p) for s in ENGINES
        }
        return p, stub
    return make


def ghz(n=3):
    qc = QuantumCircuit(n, n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure(range(n), range(n))
    return qc


# ------------------------------------------------------------------- provider

def test_every_engine_is_exposed(provider):
    p, _ = provider()
    assert len(p.backends()) == len(ENGINES)
    assert {b.name for b in p.backends()} == {s.backend_name for s in ENGINES}


def test_default_backend_is_the_router(provider):
    p, _ = provider()
    assert p.backend().name == "zksf_auto"
    assert p.backend().spec.engine is None


def test_unknown_backend_lists_the_alternatives(provider):
    p, _ = provider()
    with pytest.raises(KeyError, match="zksf_auto"):
        p.backend("nope")


def test_filter_by_qubit_count_and_hardware(provider):
    p, _ = provider()
    assert all(b.num_qubits >= 100 for b in p.backends(min_num_qubits=100))
    names = {b.name for b in p.backends(hardware=True)}
    assert names == {"zksf_rigetti", "zksf_ionq"}


# --------------------------------------------------------------------- target

def test_qubit_limits_match_the_service(provider):
    p, _ = provider()
    assert p.backend("zksf_exact_cpu").num_qubits == 30
    assert p.backend("zksf_rigetti").num_qubits == 108
    assert p.backend("zksf_ionq").num_qubits == 36


def test_stabilizer_backend_advertises_no_non_clifford_gate(provider):
    """Otherwise the transpiler would happily emit a T gate for an engine that
    cannot represent one, and the rejection would arrive from the server."""
    p, _ = provider()
    ops = p.backend("zksf_clifford").target.operation_names
    assert "t" not in ops and "rz" not in ops
    assert {"h", "cx", "s", "measure"} <= set(ops)


def test_universal_backend_advertises_rotations_and_qaoa_gates(provider):
    p, _ = provider()
    ops = p.backend("zksf_mps").target.operation_names
    assert {"t", "rz", "rzz", "ccx", "measure"} <= set(ops)


def test_circuits_transpile_against_the_target(provider):
    p, _ = provider()
    out = transpile(ghz(3), p.backend("zksf_mps"))
    assert out.num_qubits <= 128


# ------------------------------------------------------------------- dispatch

def test_engine_is_passed_through_on_submit(provider):
    p, stub = provider()
    p.backend("zksf_pauli").run(ghz(3), shots=500)
    assert stub.submitted[0]["engine"] == "pauli.cpu"
    assert stub.submitted[0]["shots"] == 500


def test_router_backend_sends_no_engine(provider):
    p, stub = provider()
    p.backend().run(ghz(3))
    assert stub.submitted[0]["engine"] is None


def test_extra_options_reach_the_engine_as_params(provider):
    p, stub = provider()
    p.backend("zksf_mps").run(ghz(3), shots=100, certified=True)
    assert stub.submitted[0]["params"] == {"certified": True}


def test_unset_flags_are_not_sent(provider):
    """certified defaults to False; sending it would make every job look like
    an explicit opt-out rather than a default."""
    p, stub = provider()
    p.backend("zksf_mps").run(ghz(3))
    assert stub.submitted[0]["params"] == {}


def test_several_circuits_become_several_jobs(provider):
    p, stub = provider()
    job = p.backend("zksf_mps").run([ghz(3), ghz(3)])
    assert len(stub.submitted) == 2
    assert len(job.job_ids) == 2


def test_empty_input_is_rejected_locally(provider):
    p, _ = provider()
    with pytest.raises(ValueError):
        p.backend().run([])


# -------------------------------------------------------------------- results

def test_counts_survive_the_round_trip(provider):
    """The service answers in bitstrings, Qiskit formats counts out of hex via
    the header. Getting this wrong returns plausible but wrong labels."""
    p, _ = provider()
    counts = p.backend("zksf_clifford").run(ghz(3)).result().get_counts()
    assert counts == {"000": 531, "111": 469}


def test_wider_registers_keep_their_leading_zeros(provider):
    """hex() drops leading zeros, so a 5-bit outcome of 00011 must come back
    as 00011 and not 11."""
    payload = {
        "id": "j", "status": "done", "engine": "exact.cpu",
        "result": {"counts": {"00011": 7, "00000": 3}, "error_info": {}},
    }
    p, _ = provider(payload)
    qc = QuantumCircuit(5, 5)
    qc.measure(range(5), range(5))
    assert p.backend("zksf_exact_cpu").run(qc).result().get_counts() == {
        "00011": 7, "00000": 3
    }


def test_error_info_is_reachable(provider):
    p, _ = provider()
    job = p.backend("zksf_clifford").run(ghz(3))
    assert job.error_info(0)["protocol"] == "ZCC-v0.1"
    assert job.error_info()[0]["truncation_error"] == 0.0


def test_error_info_also_rides_on_the_result(provider):
    p, _ = provider()
    result = p.backend("zksf_clifford").run(ghz(3)).result()
    assert result.results[0].metadata["error_info"]["protocol"] == "ZCC-v0.1"


def test_expectation_value_is_preserved(provider):
    payload = {
        "id": "j", "status": "done", "engine": "pauli.cpu",
        "result": {"counts": {}, "expectation": 1.0,
                   "error_info": {"error_bound": 0.0}},
    }
    p, _ = provider(payload)
    job = p.backend("zksf_pauli").run(ghz(3), observable=[[1.0, "ZZZ"]])
    assert job.result().results[0].data.expectation == 1.0


def test_rejection_explains_itself(provider):
    payload = {"id": "j", "status": "rejected",
               "reason": "intractable classically at this structure"}
    p, _ = provider(payload)
    job = p.backend("zksf_mps").run(ghz(3))
    with pytest.raises(JobRejected, match="intractable"):
        job.result()


def test_engine_failure_surfaces_as_an_error(provider):
    payload = {"id": "j", "status": "error", "error": "worker died"}
    p, _ = provider(payload)
    with pytest.raises(RuntimeError, match="worker died"):
        p.backend("zksf_mps").run(ghz(3)).result()


def test_raw_records_are_available(provider):
    p, _ = provider()
    assert p.backend("zksf_clifford").run(ghz(3)).raw()[0]["engine"] == "clifford"


def test_estimate_uses_the_named_backend(provider):
    p, stub = provider()
    p.estimate(ghz(3), shots=10, backend="zksf_ionq")
    assert stub.estimated[0]["engine"] == "qpu.ionq"
