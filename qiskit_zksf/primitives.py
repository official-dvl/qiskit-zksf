"""SamplerV2 and EstimatorV2 over the ZKSF API.

Modern Qiskit code does not call ``backend.run()``. It builds a primitive and
hands it PUBs, so a provider that only exposes a backend cannot be driven by
code written the current way. These are that interface.

One thing behaves differently here to a local primitive, and it is deliberate.
A PUB can carry an array of parameter bindings, and a local simulator will
happily evaluate a thousand of them. Every binding here is a separate billed
job, because the service has no batch endpoint yet, so a careless sweep would
quietly cost real money. Fan-out past ``max_jobs_per_run`` is refused with a
message saying what it would have submitted, rather than run.
"""
from __future__ import annotations

import numpy as np
from qiskit.primitives import (
    BaseEstimatorV2,
    BasePrimitiveJob,
    BaseSamplerV2,
    PrimitiveResult,
    PubResult,
    SamplerPubResult,
)
from qiskit.primitives.containers import EstimatorPub, SamplerPub
from qiskit.primitives.containers.bit_array import BitArray
from qiskit.primitives.containers.data_bin import DataBin

# Submitting more than this many jobs from a single run() is almost always a
# parameter sweep the caller did not realise would be billed per point.
DEFAULT_MAX_JOBS = 16


class TooManyJobs(RuntimeError):
    """A single run() would have fanned out into more billed jobs than allowed."""


class _CompletedJob(BasePrimitiveJob):
    """Jobs are awaited inside run(), so this only carries the finished result.

    The primitive interface is asynchronous, but the underlying submissions are
    already resolved by the time this is constructed, so every state query
    answers from a terminal state rather than pretending to poll.
    """

    def __init__(self, job_id: str, result: PrimitiveResult):
        super().__init__(job_id)
        self._result = result

    def result(self) -> PrimitiveResult:
        return self._result

    def status(self) -> str:
        return "DONE"

    def done(self) -> bool:
        return True

    def running(self) -> bool:
        return False

    def cancelled(self) -> bool:
        return False

    def in_final_state(self) -> bool:
        return True

    def cancel(self):
        raise NotImplementedError("jobs complete before the primitive returns")


def _bindings(pub) -> list[dict]:
    """Every parameter binding in a PUB, flattened to a list of dicts.

    Returns a single empty binding for a circuit with no free parameters, which
    is the ordinary case.
    """
    pv = pub.parameter_values
    if pv is None or pv.size == 0 or not pub.circuit.parameters:
        return [{}]
    flat = pv.ravel()
    return [flat.bind(pub.circuit, (i,)) for i in range(flat.size)]


def _split_by_register(circuit, counts_per_binding: list[dict], shape) -> dict[str, BitArray]:
    """Split bitstring-keyed counts into a BitArray per classical register,
    which is how a DataBin names its fields.

    The service returns one string across every classical bit, most significant
    bit leftmost, so a register occupying bits [start, start+n) is a slice
    counted from the right.

    ``shape`` is the PUB's shape. A PUB always yields exactly one result, and
    parameter bindings are carried inside the array rather than as extra
    results, which is what Qiskit's own primitives do.
    """
    total = circuit.num_clbits
    out: dict[str, BitArray] = {}
    start = 0
    for creg in circuit.cregs:
        n = creg.size
        lo = total - start - n
        per_binding = []
        for counts in counts_per_binding:
            sliced: dict[str, int] = {}
            for bits, freq in counts.items():
                key = bits.zfill(total)[lo : lo + n]
                sliced[key] = sliced.get(key, 0) + freq
            per_binding.append(sliced)
        # A scalar PUB takes a single mapping and yields shape (); an array of
        # mappings yields a leading axis.
        source = per_binding[0] if shape == () else per_binding
        out[creg.name] = BitArray.from_counts(source, num_bits=n)
        start += n
    return out


def _observable_to_service(obs) -> list[list]:
    """An ObservablesArray element is a mapping of Pauli string to coefficient.
    The service takes the same information as [[coeff, pauli], ...]."""
    return [[float(coeff), str(pauli)] for pauli, coeff in dict(obs).items()]


class ZKSFSampler(BaseSamplerV2):
    """SamplerV2 backed by a ZKSF engine.

        sampler = ZKSFSampler(backend)
        result = sampler.run([circuit], shots=1000).result()
        result[0].data.meas.get_counts()
    """

    def __init__(self, backend, *, default_shots: int = 1024,
                 max_jobs_per_run: int = DEFAULT_MAX_JOBS, **run_options):
        self._backend = backend
        self._default_shots = default_shots
        self._max_jobs = max_jobs_per_run
        self._run_options = run_options

    @property
    def backend(self):
        return self._backend

    def run(self, pubs, *, shots: int | None = None) -> BasePrimitiveJob:
        coerced = [SamplerPub.coerce(p, shots or self._default_shots) for p in pubs]

        planned = sum(len(_bindings(p)) for p in coerced)
        if planned > self._max_jobs:
            raise TooManyJobs(
                f"this run would submit {planned} separate billed jobs "
                f"({len(coerced)} pub(s) with parameter bindings), above the "
                f"limit of {self._max_jobs}. The service has no batch endpoint "
                f"yet, so each binding costs a job. Raise max_jobs_per_run if "
                f"that is intended."
            )

        results = []
        for pub in coerced:
            counts_per_binding, payloads = [], []
            for binding in _bindings(pub):
                circuit = binding if binding else pub.circuit
                job = self._backend.run(circuit, shots=pub.shots, **self._run_options)
                payload = job.raw()[0]
                payloads.append(payload)
                counts_per_binding.append((payload.get("result") or {}).get("counts") or {})

            infos = [(p.get("result") or {}).get("error_info", {}) for p in payloads]
            results.append(
                SamplerPubResult(
                    DataBin(
                        **_split_by_register(pub.circuit, counts_per_binding, pub.shape),
                        shape=pub.shape,
                    ),
                    metadata={
                        "shots": pub.shots,
                        "engine": payloads[0].get("engine"),
                        # The reason to run here rather than locally.
                        "error_info": infos[0] if len(infos) == 1 else infos,
                        "zksf_job_ids": [p.get("id") for p in payloads],
                    },
                )
            )
        return _CompletedJob("zksf-sampler", PrimitiveResult(results))


class ZKSFEstimator(BaseEstimatorV2):
    """EstimatorV2 backed by a ZKSF engine.

        estimator = ZKSFEstimator(backend)
        result = estimator.run([(circuit, observable)]).result()
        result[0].data.evs

    Defaults to the Pauli propagation engine, which answers with an expectation
    value directly and reports the discarded coefficient mass as a bound on it.
    """

    def __init__(self, backend, *, default_shots: int = 1024,
                 max_jobs_per_run: int = DEFAULT_MAX_JOBS, **run_options):
        self._backend = backend
        self._default_shots = default_shots
        self._max_jobs = max_jobs_per_run
        self._run_options = run_options

    @property
    def backend(self):
        return self._backend

    def run(self, pubs, *, precision: float | None = None) -> BasePrimitiveJob:
        coerced = [EstimatorPub.coerce(p, precision) for p in pubs]

        planned = sum(p.observables.size * len(_bindings(p)) for p in coerced)
        if planned > self._max_jobs:
            raise TooManyJobs(
                f"this run would submit {planned} separate billed jobs "
                f"(observables times parameter bindings), above the limit of "
                f"{self._max_jobs}. Each one costs a job because the service "
                f"has no batch endpoint yet. Raise max_jobs_per_run if that is "
                f"intended."
            )

        results = []
        for pub in coerced:
            bindings = _bindings(pub)
            obs_flat = pub.observables.ravel()

            # Sweeping observables and parameters at once needs the two arrays
            # broadcast against each other. That is not implemented, and
            # guessing an order would silently mislabel every value, so it is
            # refused instead.
            if len(bindings) > 1 and obs_flat.size > 1:
                raise NotImplementedError(
                    "an observable array combined with a parameter sweep is not "
                    "supported. Run one observable across the sweep, or several "
                    "observables at a single parameter point."
                )

            evs, stds, infos, payloads = [], [], [], []
            for binding in bindings:
                circuit = binding if binding else pub.circuit
                for i in range(obs_flat.size):
                    job = self._backend.run(
                        circuit,
                        shots=self._default_shots,
                        observable=_observable_to_service(obs_flat[i]),
                        **self._run_options,
                    )
                    payload = job.raw()[0]
                    payloads.append(payload)
                    result = payload.get("result") or {}
                    info = result.get("error_info") or {}
                    evs.append(result.get("expectation"))
                    # An EstimatorV2 std is an uncertainty on the value, which
                    # is exactly what the protocol already reports, so it
                    # carries through rather than being invented or zeroed.
                    stds.append(info.get("error_bound", 0.0))
                    infos.append(info)

            # One result per PUB, shaped by the PUB. A single observable at a
            # single point gives shape (), matching Qiskit's own primitives.
            shape = pub.shape
            results.append(
                PubResult(
                    DataBin(
                        evs=np.asarray(evs, dtype=float).reshape(shape),
                        stds=np.asarray(stds, dtype=float).reshape(shape),
                        shape=shape,
                    ),
                    metadata={
                        "target_precision": pub.precision,
                        "engine": payloads[0].get("engine"),
                        "error_info": infos[0] if len(infos) == 1 else infos,
                        "zksf_job_ids": [p.get("id") for p in payloads],
                    },
                )
            )
        return _CompletedJob("zksf-estimator", PrimitiveResult(results))
