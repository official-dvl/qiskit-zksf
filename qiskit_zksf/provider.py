"""Provider entry point: pick a backend, run your existing Qiskit code."""
from __future__ import annotations

import os

from qiskit_zksf.backend import ENGINES, ZKSFBackend


class ZKSFProvider:
    """Backends for the Zero Kelvin Simulation Foundry.

        from qiskit_zksf import ZKSFProvider

        backend = ZKSFProvider(token="...").backend("zksf_auto")
        job = backend.run(circuit, shots=1000)
        job.result().get_counts()
        job.error_info()      # how far the answer may be from the truth

    The token is read from the ``ZKSF_TOKEN`` environment variable when not
    passed. Get one from the console at https://app.zksf.org.
    """

    def __init__(self, token: str | None = None, base_url: str | None = None):
        import qsim_sdk

        token = token or os.environ.get("ZKSF_TOKEN")
        kwargs = {"token": token} if token else {}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = qsim_sdk.Client(**kwargs)
        self._backends = {
            spec.backend_name: ZKSFBackend(spec, self._client, provider=self)
            for spec in ENGINES
        }

    def backends(self, name: str | None = None, **kwargs) -> list[ZKSFBackend]:
        found = list(self._backends.values())
        if name is not None:
            found = [b for b in found if b.name == name]
        if kwargs.get("min_num_qubits") is not None:
            found = [b for b in found if b.num_qubits >= kwargs["min_num_qubits"]]
        if kwargs.get("hardware") is not None:
            found = [b for b in found if b.spec.hardware is kwargs["hardware"]]
        return found

    def backend(self, name: str = "zksf_auto") -> ZKSFBackend:
        try:
            return self._backends[name]
        except KeyError:
            raise KeyError(
                f"unknown backend {name!r}. Available: "
                f"{', '.join(sorted(self._backends))}"
            ) from None

    def sampler(self, backend: str = "zksf_auto", **options):
        """A SamplerV2 bound to one engine, for outcome distributions."""
        from qiskit_zksf.primitives import ZKSFSampler

        return ZKSFSampler(self.backend(backend), **options)

    def estimator(self, backend: str = "zksf_pauli", **options):
        """An EstimatorV2 bound to one engine, for expectation values.

        Defaults to Pauli propagation, which answers with an expectation value
        directly and bounds it by the discarded coefficient mass."""
        from qiskit_zksf.primitives import ZKSFEstimator

        return ZKSFEstimator(self.backend(backend), **options)

    def estimate(self, circuit, shots: int = 1024, backend: str = "zksf_auto") -> dict:
        """Free pre-run check: engine, predicted runtime and cost, or why not.

        Qiskit has no equivalent concept, so it is exposed here rather than on
        the backend. Calling it before run() is the only way to learn that a
        circuit would be rejected without submitting it.
        """
        return self._client.estimate(
            circuit, shots=shots, engine=self._backends[backend].spec.engine
        )

    def __repr__(self) -> str:
        return f"<ZKSFProvider: {len(self._backends)} backends>"
