"""Job handle for circuits submitted to ZKSF.

The service returns an accuracy statement alongside every approximate result,
and Qiskit's Result has nowhere obvious to put one. It is therefore carried in
two places: on each experiment's metadata, so it survives `Result.to_dict()`,
and on the job itself via ``job.error_info()``, which is where anyone is
actually going to look for it.
"""
from __future__ import annotations

import time

from qiskit.providers import JobStatus, JobV1
from qiskit.result import Result

_TERMINAL = {"done", "rejected", "error"}

_STATUS = {
    "queued": JobStatus.QUEUED,
    "running": JobStatus.RUNNING,
    "done": JobStatus.DONE,
    "rejected": JobStatus.CANCELLED,
    "error": JobStatus.ERROR,
}


class JobRejected(RuntimeError):
    """The service declined to run the circuit, and said why.

    A rejection is a deliberate outcome rather than a failure: an approximate
    method whose own error bound would be vacuous is refused instead of
    returning a number nobody should trust.
    """


class ZKSFJob(JobV1):
    def __init__(self, backend, job_ids: list[str], circuits, shots: int):
        super().__init__(backend=backend, job_id=job_ids[0])
        self._job_ids = job_ids
        self._circuits = circuits
        self._shots = shots
        self._payloads: list[dict] | None = None

    # JobV1 requires this; submission already happened in Backend.run.
    def submit(self):
        raise NotImplementedError("circuits are submitted by ZKSFBackend.run()")

    @property
    def job_ids(self) -> list[str]:
        return list(self._job_ids)

    def _poll_once(self) -> list[dict]:
        client = self.backend()._client
        return [client.job(jid) for jid in self._job_ids]

    def status(self):
        payloads = self._poll_once()
        # A batch is only as finished as its least finished member, and only as
        # healthy as its unhealthiest.
        states = [p.get("status", "queued") for p in payloads]
        for bad in ("error", "rejected"):
            if bad in states:
                return _STATUS[bad]
        if all(s == "done" for s in states):
            return JobStatus.DONE
        if any(s == "running" for s in states):
            return JobStatus.RUNNING
        return JobStatus.QUEUED

    def wait_for_final_state(self, timeout=None, wait=0.5, callback=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            payloads = self._poll_once()
            if all(p.get("status") in _TERMINAL for p in payloads):
                self._payloads = payloads
                return
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(f"jobs {self._job_ids} still running after {timeout}s")
            if callback is not None:
                callback(self.job_id(), self.status(), self)
            time.sleep(wait)

    def result(self, timeout=None, wait=0.5) -> Result:
        if self._payloads is None or any(
            p.get("status") not in _TERMINAL for p in self._payloads
        ):
            self.wait_for_final_state(timeout=timeout, wait=wait)
        payloads = self._payloads or []

        for payload in payloads:
            if payload.get("status") == "rejected":
                raise JobRejected(payload.get("reason", "no reason given"))
            if payload.get("status") == "error":
                raise RuntimeError(payload.get("error", "job failed"))

        experiments = [
            _experiment(payload, circuit, self._shots)
            for payload, circuit in zip(payloads, self._circuits)
        ]
        return Result.from_dict(
            {
                "backend_name": self.backend().name,
                "backend_version": self.backend().backend_version,
                "job_id": self.job_id(),
                "qobj_id": self.job_id(),
                "success": True,
                "results": experiments,
            }
        )

    def error_info(self, index: int | None = None):
        """The accuracy statement for each circuit: what the result is worth.

        This is the reason to use this backend rather than a local simulator,
        so it gets a first-class accessor instead of being buried in metadata.
        """
        if self._payloads is None:
            self.wait_for_final_state()
        infos = [
            (p.get("result") or {}).get("error_info", {}) for p in (self._payloads or [])
        ]
        return infos if index is None else infos[index]

    def raw(self) -> list[dict]:
        """The service's own job records, unmodified."""
        if self._payloads is None:
            self.wait_for_final_state()
        return list(self._payloads or [])


def _experiment(payload: dict, circuit, shots: int) -> dict:
    result = payload.get("result") or {}
    counts = result.get("counts") or {}
    n_clbits = getattr(circuit, "num_clbits", 0) or _width(counts)

    # Qiskit formats counts back into bitstrings from hex using the header, so
    # the keys go out as hex and the creg layout comes with them. Emitting
    # bitstrings directly would be re-interpreted and mangled.
    hex_counts = {hex(int(bits, 2)): n for bits, n in counts.items()} if counts else {}

    experiment: dict = {
        "shots": shots,
        "success": True,
        "status": payload.get("status", "done"),
        "data": {"counts": hex_counts},
        "header": {
            "memory_slots": n_clbits,
            "creg_sizes": [["c", n_clbits]] if n_clbits else [],
            "name": getattr(circuit, "name", "circuit"),
        },
        # Carried on the experiment so it survives Result.to_dict(); the
        # discoverable path is job.error_info().
        "metadata": {
            "error_info": result.get("error_info", {}),
            "engine": payload.get("engine"),
            "zksf_job_id": payload.get("id"),
        },
    }
    if result.get("expectation") is not None:
        # Pauli propagation and the mitigation paths answer with an expectation
        # value rather than counts, and it would otherwise be dropped.
        experiment["data"]["expectation"] = result["expectation"]
    return experiment


def _width(counts: dict) -> int:
    return len(next(iter(counts))) if counts else 0
