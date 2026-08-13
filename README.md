# qiskit-zksf

[![CI](https://github.com/official-dvl/qiskit-zksf/actions/workflows/ci.yml/badge.svg)](https://github.com/official-dvl/qiskit-zksf/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/qiskit-zksf.svg?v=2)](https://pypi.org/project/qiskit-zksf/)
[![Python](https://img.shields.io/pypi/pyversions/qiskit-zksf.svg?v=2)](https://pypi.org/project/qiskit-zksf/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21851381-blue.svg)](https://doi.org/10.5281/zenodo.21851381)

Qiskit provider for **ZKSF** (Zero Kelvin Simulation Foundry). Point circuits you have
already written at classical simulators, GPU accelerators, or real quantum processors,
and get a documented accuracy statement back with every approximate result.

```bash
pip install qiskit-zksf
```

```python
from qiskit import QuantumCircuit
from qiskit_zksf import ZKSFProvider

qc = QuantumCircuit(40, 40)
qc.h(0)
for i in range(39):
    qc.cx(i, i + 1)
qc.measure(range(40), range(40))

backend = ZKSFProvider(token="...").backend("zksf_auto")
job = backend.run(qc, shots=1000)

print(job.result().get_counts())
print(job.error_info())     # how far that answer may be from the truth
```

The token comes from the console at [app.zksf.org](https://app.zksf.org), or from the
`ZKSF_TOKEN` environment variable.

## Why this exists

Exact statevector simulation stops near 30 to 32 qubits, because state size grows as
`2^n`. Past that, every practical method is approximate: tensor networks truncate the
bond dimension, Pauli propagation truncates operator weight, real hardware substitutes
device noise for the ideal distribution.

Simulators do not normally tell you how much of the answer that cost you, even though
the error quantities exist inside the simulation. `job.error_info()` is that number.

```python
{'protocol': 'ZCC-v0.1',
 'method': 'MPS (quimb), measured discarded-weight bound',
 'truncation_weight': 2.220446049250313e-16,
 'error_bound': 2.1073424255447017e-08,
 'certified': True,
 'converged': True}
```

Any finished job can be minted into a public certificate that anyone can check without
an account, using [`zcc-verify`](https://pypi.org/project/zcc-verify/). The protocols are
specified in a citable paper: [doi.org/10.5281/zenodo.21851381](https://doi.org/10.5281/zenodo.21851381).

## Backends

| Backend | Engine | Qubits | Notes |
|---|---|---|---|
| `zksf_auto` | router picks | 128 | Default. Chooses the cheapest adequate engine |
| `zksf_exact_cpu` | `exact.cpu` | 30 | Exact statevector |
| `zksf_exact_gpu` | `exact.gpu` | 29 | Exact statevector on GPU |
| `zksf_mps` | `mps.quimb.cpu` | 128 | Tensor network, supports `certified=True` |
| `zksf_mps_aer` | `mps.aer.cpu` | 128 | Tensor network (Aer) |
| `zksf_clifford` | `clifford` | 5000 | Stabilizer, exact for Clifford circuits |
| `zksf_pauli` | `pauli.cpu` | 1024 | Heisenberg picture, needs an observable |
| `zksf_noisy` | `noisy.cpu` | 30 | Device noise model, supports `mitigate=True` |
| `zksf_rigetti` | `qpu.rigetti` | 108 | Rigetti Cepheus-1, real hardware |
| `zksf_ionq` | `qpu.ionq` | 36 | IonQ Forte-1, real hardware |

```python
provider = ZKSFProvider()
provider.backends()                      # all of them
provider.backends(min_num_qubits=100)    # only the ones that reach 100 qubits
provider.backends(hardware=True)         # only real quantum processors
```

Qubit counts mirror the limits the service enforces, so a circuit too large for an
engine fails at transpile time instead of after a round trip. The stabilizer backend
advertises only Clifford gates, so the transpiler will not hand it a `T` gate that Stim
cannot represent.

## Estimate before you spend

Qiskit has no equivalent concept, so this lives on the provider. It is free, instant, and
the only way to learn that a circuit would be **rejected** without submitting it.

```python
est = provider.estimate(qc, shots=1000)
print(est["engine"], est["predicted_cost_usd"], est["reason"])
```

## Asking for a measured bound

By default an approximate run is checked by convergence: the circuit is simulated again
at double the resource budget and the shift in outcome probabilities is reported. That is
evidence of accuracy, not a bound.

`certified=True` asks the tensor-network engine for a stronger statement. It runs with
state renormalization disabled, so the final state's norm deficit equals the total weight
discarded across every truncation, read directly off the result rather than estimated.

```python
job = provider.backend("zksf_mps").run(qc, shots=1000, certified=True)
job.error_info()["error_bound"]
```

## Rejections are a feature

A simulation whose own error bound would be vacuous is refused rather than returned, and
the refusal says what would make the circuit tractable.

```python
from qiskit_zksf import JobRejected

try:
    job.result()
except JobRejected as exc:
    print(exc)   # "intractable classically at this structure: ... Options: ..."
```

## Notes and limits

- **Several circuits become several jobs.** The service has no batch endpoint yet, so
  `backend.run([qc1, qc2])` submits them individually and each is billed separately.
- **No gradients.** This is a cloud job queue with per-job billing, so parameter-shift
  differentiation through it would be expensive and slow. Use a local simulator for
  optimization loops and this for the runs whose accuracy you need to state.
- **Hardware costs real money** and queues in hours, not seconds. Call `estimate()`
  first.

## Links

- Documentation: [zksf.org/docs](https://zksf.org/docs)
- Python SDK: [`qsim-sdk`](https://pypi.org/project/qsim-sdk/)
- Certificate checker: [`zcc-verify`](https://pypi.org/project/zcc-verify/)
- Android app: [Google Play](https://play.google.com/store/apps/details?id=com.quantumcomputing.app)
- Protocol paper: [10.5281/zenodo.21851381](https://doi.org/10.5281/zenodo.21851381)

## Licence

MIT.
