#!/usr/bin/env python3
"""
pycliff_mit.py

driver for expectation value minimization over superpositions of stabilizer states

Author: Martin-Isbjörn Trappe
Email: martin.trappe@quantumlah.org
Date: 2025-09-12

Usage:
    bash run_pycliff_mit.sh

See the “BEGIN USER INPUT” / “END USER INPUT” section below to configure calculation
"""

import os

# ============================
# ===== BEGIN USER INPUT =====
# ============================

PROJECT = 'TFIM_Clifford_Optimization' # Project name

# Computation variables
MIDDLEWARE = 'qiskit' # 'qiskit' or 'qibo'
USE_RUNTIME = False  # keep False for local EstimatorV2 runs
USE_QIBOJIT = False  # default: False for numpy backend
USE_GPU = False # default: False for CPU
PRINTALL = True # False -- True
THREADS = 20  # Parallel processes
SEED = 42           # base RNG seed for per-run reproducibility
OPTIMIZER = "BFGS"   # Classical (scipy) optimizer: BFGS -- CG -- COBYLA -- Nelder-Mead -- Powell -- etc
R_LANCZOS = 256  # switch to Lanczos when rank(S) > this

# Task definition
TASK = 'Energy_vs_NumStabStates'  # 'Minimal_Example' for consistency checks or 'Energy_vs_NumStabStates' for pure lottery plot or 'Whole_Shebang' for lottery(log2)_+_GAO_result_+_CMA_result
DRAWS = 100*THREADS          # number of draws of stabilizer sets of size k (parallelized)
INCLUDE_FIXED_STATES = False # choose True to add stabilizer states |0>^n |1>^n |+>^n |->^n |Néel_10> |Néel_01> |GHZ>
STABILIZER_MAPPING = 'GrayCoded' # 'Random' for Lottery and GAO or 'GrayCoded' for CMA
LAYER_SCHEDULE = 'sched_fine' # for GrayCoded STABILIZER_MAPPING: pick from schedule library ('sched5' -- 'sched11' -- 'sched_fine')
K_MODE = 'auto'          # "log2" or "linear" or "list" or "auto"
K_LIST = None            # List of number k of random stabilizer states in each stabilizer set, e.g. [1,2,4,12,16]; default: None; explicit list is used if K_MODE == "list"
K_STEP = 1               # step between consequtive ks; only used if K_MODE == "linear"
KMAX = None              # maximum k; set None to use 2**NUM_QUBITS
TOL = 1e-6               # absolute tolerance for hitting target energy

# System variables
SYSTEM = 'TFIM'
PERIODIC = False
NUM_QUBITS = 6
PARAM_J = 1.0
PARAM_h = 0.6 # 1.3

# ==========================
# ===== END USER INPUT =====
# ==========================




# =========================
# ===== BEGIN IMPORTS =====
# =========================

# Force all BLAS/MKL/OpenBLAS backends to a single thread to avoid oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import json
import shutil
import sys
import time
from datetime import datetime
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

os.environ["QIBO_LOG_LEVEL"] = "4"  # ERROR
qlog = logging.getLogger("qibo")
qlog.setLevel(logging.ERROR)

# ---- Numerics ----
import struct, math
import numpy as np
import hashlib
from numpy.random import default_rng, SeedSequence
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from pyscf import gto, scf, fci ,cc, ao2mo, mcscf

# ---- Plotting ----
import matplotlib
# Set the matplotlib backend to a non-GUI backend ('Agg'), BEFORE pyplot is imported.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---- Qibo ----
from qibochem.driver import Molecule  # Quantum chemistry driver: builds molecule & integrals via PySCF
from qibo import set_backend, get_backend, gates, Circuit # Qibo: quantum computing framework
from qibochem.ansatz.ucc import ucc_ansatz, hf_circuit, ucc_circuit  # Unitary Coupled Cluster ansatz builder, etc.
from qibochem.measurement.result import expectation

# ---- OpenFermion ----
from openfermion.circuits import uccsd_singlet_paramsize, uccsd_singlet_generator
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.transforms import jordan_wigner, get_fermion_operator
from openfermion.ops import InteractionOperator

# ---- Qiskit ----
import qiskit
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Pauli, StabilizerState, Statevector, random_clifford

# Prefer Aer EstimatorV2 if present, else SDK's reference EstimatorV2 (ADD)
try:
    from qiskit_aer.primitives import EstimatorV2 as LocalEstimator
except Exception:
    from qiskit.primitives import EstimatorV2 as LocalEstimator  # same V2 interface

if USE_RUNTIME:
    # 1) Import the Runtime client
    from qiskit_ibm_runtime import QiskitRuntimeService

    # 2) Load your IBM creds from the JSON next to this script
    cfg_path = os.path.join(os.path.dirname(__file__), "IBM_apikey.json")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    # 3) Save (once) so that QiskitRuntimeService() can auto-discover them
    home_cfg = os.path.expanduser("~/.qiskit/qiskit-ibm.json")
    if not os.path.exists(home_cfg):
        QiskitRuntimeService.save_account(
            channel="ibm_quantum_platform",
            token=cfg["apikey"],
            instance=cfg["crn"],
            overwrite=False
        )

    # 4) Instantiate the service (reads from ~/.qiskit/qiskit-ibm.json)
    service = QiskitRuntimeService()

    # 5) Simple check: list available backends
    all_backends = service.backends()
    hardware = [b.name for b in all_backends if not b.configuration().simulator]
    simulators = [b.name for b in all_backends if b.configuration().simulator]

    print("Operational hardware backends:", hardware)
    print("Available simulators:", simulators,"\n")

# =======================
# ===== END IMPORTS =====
# =======================




# ===============================
# ===== BEGIN FILE HANDLING =====
# ===============================

# === Output directory and backup setup ===
timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
script_path = os.path.realpath(__file__)
script_dir  = os.path.dirname(script_path)

BASE_NAME = f"pycliff_mit_{PROJECT}_{timestamp}"
output_dir = os.path.join(script_dir, "data", BASE_NAME)
os.makedirs(output_dir, exist_ok=True)  # creates parents too

# Copy this script into output folder for reproducibility
backup_name = f"{os.path.splitext(os.path.basename(script_path))[0]}_{PROJECT}_{timestamp}_backup.py"
shutil.copy2(script_path, os.path.join(output_dir, backup_name))

log_path = os.path.join(output_dir, BASE_NAME + ".log")
log_file = open(log_path, "a", buffering=1, encoding="utf-8")  # line-buffered

class Tee:
    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(sys.__stdout__, "encoding", "utf-8")
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()  # critical: flush on every write
    def flush(self):
        for s in self.streams:
            s.flush()
    def isatty(self):
        return False

sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = sys.stdout

# =============================
# ===== END FILE HANDLING =====
# =============================





# ============================================
# ===== BEGIN POST-PROCESSING USER INPUT =====
# ============================================

if TASK == 'Minimal_Example':
    THREADS = 1

if MIDDLEWARE == 'qiskit':
    USE_QIBOJIT = False
    if TASK == 'Minimal_Example':
        print("Test local qiskit simulator...")
        # 1) Build your circuit (e.g. Bell)
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        # 2) Add measurements
        qc.measure_all()
        # 3) Pick the local simulator backend
        sim = AerSimulator()
        # 4) Transpile to match the simulator’s expected instruction set
        qc2 = transpile(qc, sim)
        # 5) Run and fetch results
        job = sim.run(qc2, shots=1024)
        result = job.result()
        print("Counts:", result.get_counts())
        print("...local qiskit simulator works\n")

if USE_QIBOJIT:
    if USE_GPU:
        set_backend("qibojit", platform="gpu")  # if using CuPy/CUDA
    else:
        set_backend("qibojit") # for CPU
else:
    USE_GPU = False
    set_backend("numpy")
backend = get_backend()
if MIDDLEWARE == 'qibo':
    print(f"\n → Qibo backend = {backend.name}, platform = {backend.platform}\n")
    logging.getLogger("qibo").setLevel(logging.WARNING)
else:
    logging.getLogger('qiskit').setLevel(logging.WARNING)

if str(K_MODE).lower() == "auto":
    KMAX = None
    if NUM_QUBITS < 6:
        K_MODE = "log2"
        K_LIST = None
        KMAX = 2**NUM_QUBITS
        print(f"\n !!! WARNING !!! K_MODE → log2\n")


# --- schedule library ---
SCHEDULES = {
    "sched5":   ['H', 'P', ('C','fwd'), 'P', 'H'],
    "sched11":  ['H', ('C','fwd'), 'P', ('C','fwd'), 'P', ('C','fwd'),
                 'H', 'P', ('C','fwd'), 'P', ('C','fwd')],
    "sched_fine": ['H',
                   ('C','odd-even'), 'P',
                   ('C','even-odd'), 'P',
                   ('C','rev'),
                   'H', 'P',
                   ('C','fwd'), 'P',
                   ('C','fwd')],
}

# ==========================================
# ===== END POST-PROCESSING USER INPUT =====
# ==========================================





# ==================================
# ===== BEGIN HELPER FUNCTIONS =====
# ==================================

def SleepForever():
    try:
        print("Sleeping forever. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)  # sleep 1 second per loop
    except KeyboardInterrupt:
        print("\nInterrupted by user, exiting.")


# Build V: columns are |psi_j>
def build_V(stab_circuits):
    cols = []
    for qc in stab_circuits:
        cliff = StabilizerState(qc).clifford.to_instruction()
        cols.append(Statevector.from_instruction(cliff).data)
    return np.column_stack(cols)  # shape (2**n, S)


def ghz_circuit(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(0)
    for k in range(1, n):
        qc.cx(0, k)
    return qc  # stabilizer state


def random_stabilizer_circuit(n: int, seed=None) -> QuantumCircuit:
    cliff = random_clifford(n, seed=seed)
    qc = QuantumCircuit(n)
    qc.append(cliff.to_instruction(), range(n))
    return qc  # Clifford|0…0>, hence stabilizer state


def gram_matrix(V: np.ndarray) -> np.ndarray:
    """S_ij = <psi_i|psi_j>."""
    return V.conj().T @ V


def _as_1d(val, length, name, periodic_flag):
    if np.isscalar(val):
        return np.full(length, float(val), dtype=float)
    arr = np.asarray(val, dtype=float)
    if arr.shape != (length,):
        raise ValueError(f"{name} must be scalar or length {length} when periodic={periodic_flag}")
    return arr


def _span_overlap_term(V, op, idx):
    d = V.shape[0]
    rows = np.arange(d, dtype=np.int64)

    if op == "Z":
        i = idx[0]
        phase = 1.0 - 2.0 * ((rows >> i) & 1)
        W = V * phase[:, None]

    elif op == "X":
        i = idx[0]
        W = V[rows ^ (1 << i), :]

    elif op == "XX":
        i, j = idx
        W = V[rows ^ ((1 << i) | (1 << j)), :]

    elif op == "ZZ":
        i, j = idx
        phase_i = 1.0 - 2.0 * ((rows >> i) & 1)
        phase_j = 1.0 - 2.0 * ((rows >> j) & 1)
        W = V * (phase_i * phase_j)[:, None]

    else:
        raise ValueError(f"unknown op {op}")

    return V.conj().T @ W


def _make_k_list(n: int) -> list[int]:
    Kmax_default = 2**n
    Kmax = Kmax_default if (KMAX is None) else int(min(KMAX, Kmax_default))
    kmode = str(K_MODE).lower()

    if kmode == "auto":
        ks = []
        k = 1
        while k <= Kmax // 2:
            ks.append(k)
            k <<= 1
        M = int(math.log2(Kmax)) + 1
        tail = [Kmax - Kmax // (1 << m) for m in range(2, M + 1)] + [Kmax]
        ks = sorted(set(ks + tail))
        return ks

    if isinstance(K_LIST, (list, tuple)) and len(K_LIST) > 0:
        ks = sorted({int(k) for k in K_LIST if 1 <= int(k) <= Kmax})
        if ks[-1] != Kmax:
            ks.append(Kmax)
        return ks

    if kmode == "linear":
        step = max(1, int(K_STEP))
        ks = list(range(1, Kmax + 1, step))
        if ks[-1] != Kmax:
            ks.append(Kmax)
        return ks

    ks = [1]
    k = 1
    while k < Kmax:
        k *= 2
        ks.append(min(k, Kmax))
    return sorted(set(ks))


def _float64_bits(x: float) -> int:
    # Canonicalize -0.0 and all NaNs for stability
    if x == 0.0:
        x = 0.0
    if math.isnan(x):
        return 0x7ff8000000000000  # quiet NaN canonical
    if math.isinf(x):
        return 0x7ff0000000000000 if x > 0 else 0xfff0000000000000
    return int.from_bytes(struct.pack(">d", float(x)), "big", signed=False)


def _splitmix64(z: int) -> int:
    z = (z + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    z ^= (z >> 30)
    z = (z * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    z ^= (z >> 27)
    z = (z * 0x94D049BB133111EB) & ((1 << 64) - 1)
    z ^= (z >> 31)
    return z & ((1 << 64) - 1)


def floats_to_seeds(vals, salt: int = 0x243F6A8885A308D3, out_bits: int = 31):
    """Map floats -> list of uint seeds in [0, 2^out_bits-1]. Deterministic."""
    mask = (1 << 64) - 1
    seeds = []
    s = salt & mask
    for i, v in enumerate(vals):
        b = _float64_bits(float(v))
        s = _splitmix64((b ^ s ^ ((i * 0x9E3779B97F4A7C15) & mask)) & mask)
        seeds.append(int(s & ((1 << out_bits) - 1)))
    return seeds


def seed32(s: int) -> int:
    h = hashlib.blake2b(str(int(s)).encode(), digest_size=4).digest()
    return int.from_bytes(h, 'big', signed=False)


def random_clifford_safe(n: int, seed: int):
    return random_clifford(n, seed=seed32(seed))


def cliff_sig(C):
    """
    Robust short fingerprint of a Clifford.
    Tries tableau first; falls back to full matrix if needed.
    """
    try:
        T = C.tableau
        b = T.x.tobytes() + T.z.tobytes() + T.phase.tobytes()
    except Exception:
        # Fallback: safe but heavier
        b = C.to_matrix().tobytes()
    return hashlib.blake2b(b, digest_size=8).hexdigest()

def tfim_energy_sv(sv: Statevector, J: float, h: float, periodic: bool) -> float:
    n = int(np.log2(len(sv)))
    E = 0.0
    for j in range(n - 1):
        P = Pauli('I' * j + 'ZZ' + 'I' * (n - j - 2))
        E += J * sv.expectation_value(P).real
    if periodic and n > 2:
        P = Pauli('Z' + 'I' * (n - 2) + 'Z')
        E += J * sv.expectation_value(P).real
    for j in range(n):
        P = Pauli('I' * j + 'X' + 'I' * (n - j - 1))
        E += h * sv.expectation_value(P).real
    return -E

def probe_two_states(num_qubits: int, J: float, h: float, periodic: bool) -> None:
    sv0 = Statevector.from_label('0' * num_qubits)
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    svH = sv0.evolve(qc)
    e0 = tfim_energy_sv(sv0, J, h, periodic)
    eH = tfim_energy_sv(svH, J, h, periodic)
    print(f"[probe] E(|0..0>)={e0:.8f}  E(H0|0..0>)={eH:.8f}")


# --- Gray-coded mapping from R -> stabilizer state ---

def _edges_line(n: int, periodic: bool):
    return [(i, (i + 1) % n) for i in range(n)] if periodic else [(i, i + 1) for i in range(n - 1)]


def _edges_line_rev(n: int, periodic: bool):
    e = _edges_line(n, periodic)
    return list(reversed(e))


def _edges_odd_even(n: int, periodic: bool):
    base = _edges_line(n, periodic)
    odd  = [p for i, p in enumerate(base) if i % 2 == 1]
    even = [p for i, p in enumerate(base) if i % 2 == 0]
    return odd + even


def _edges_even_odd(n: int, periodic: bool):
    base = _edges_line(n, periodic)
    odd  = [p for i, p in enumerate(base) if i % 2 == 1]
    even = [p for i, p in enumerate(base) if i % 2 == 0]
    return even + odd


def build_layers(n: int, periodic: bool, schedule):
    """
    schedule: list with entries:
      'H' or 'P'                      -> per-qubit layer
      ('C','fwd'|'rev'|'odd-even'|'even-odd'|'explicit', edges?)
        - 'explicit' requires a 2nd item: list[(c,t),...]
    """
    layers = []
    for item in schedule:
        if item == 'H':
            layers.append(('H', list(range(n))))
        elif item == 'P':
            layers.append(('P', list(range(n))))
        elif isinstance(item, tuple) and item[0] == 'C':
            mode = item[1]
            if mode == 'fwd':
                edges = _edges_line(n, periodic)
            elif mode == 'rev':
                edges = _edges_line_rev(n, periodic)
            elif mode == 'odd-even':
                edges = _edges_odd_even(n, periodic)
            elif mode == 'even-odd':
                edges = _edges_even_odd(n, periodic)
            elif mode == 'explicit':
                edges = item[2]
            else:
                raise ValueError(f"unknown C mode {mode}")
            layers.append(('C', edges))
        else:
            raise ValueError(f"bad schedule item {item}")
    return layers


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x); return 1.0 / (1.0 + z)
    z = math.exp(x); return z / (1.0 + z)


def _int_to_gray(t: int) -> int:
    return t ^ (t >> 1)


def _bit(g: int, i: int) -> int:
    return (g >> i) & 1


def real_to_stabilizer_qc(n: int, x: float, periodic: bool, schedule, beta: float | None = None) -> QuantumCircuit:
    layers = build_layers(n, periodic, schedule)
    m = sum(len(items) for kind, items in layers)  # bit budget
    if beta is None:
        beta = (1 << m) / 4.0  # 1 unit x ≈ 1 bin near center
    u = _sigmoid(beta * float(x))
    t = int(min(max(math.floor(u * (1 << m)), 0), (1 << m) - 1))
    g = _int_to_gray(t)

    qc = QuantumCircuit(n)
    b = 0
    for kind, items in layers:
        if kind == 'H':
            for q in items:
                if _bit(g, b): qc.h(q)
                b += 1
        elif kind == 'P':
            for q in items:
                if _bit(g, b): qc.s(q)
                b += 1
        else:  # 'C'
            for c, t_ in items:
                if _bit(g, b): qc.cx(c, t_)
                b += 1
    return qc


def pick_schedule(layer_schedule) -> list:
    """layer_schedule may be a name (str) or a schedule list."""
    if isinstance(layer_schedule, str):
        try:
            return SCHEDULES[layer_schedule]
        except KeyError as e:
            raise ValueError(f"unknown LAYER_SCHEDULE '{layer_schedule}'") from e
    return layer_schedule  # assume already a schedule list


def reals_to_stabilizers(n: int, xs: list[float], periodic: bool,
                         layer_schedule=LAYER_SCHEDULE) -> list[QuantumCircuit]:
    sched = pick_schedule(layer_schedule)
    return [real_to_stabilizer_qc(n, float(x), periodic, sched, beta=None) for x in xs]

# ================================
# ===== END HELPER FUNCTIONS =====
# ================================





# =============================
# ===== BEGIN CORE MODULES =====
# =============================

def tfim_E0_periodic(N: int, J: float, h: float) -> float:
    ks_even = 2*np.pi*(np.arange(N)+0.5)/N
    ks_odd  = 2*np.pi*np.arange(N)/N
    eps_even = 2*np.sqrt(J*J + h*h - 2*J*h*np.cos(ks_even))
    eps_odd  = 2*np.sqrt(J*J + h*h - 2*J*h*np.cos(ks_odd))
    E_even = -0.5 * np.sum(eps_even)
    E_odd  = -0.5 * np.sum(eps_odd)
    return min(E_even, E_odd)


def tfim_E0_open(n: int, J: float, h: float, sign: float = -1.0) -> float:
    """
    Exact GS energy for TFIM with open boundaries (linear chain).
    H = sign * ( J * sum_{i=0}^{n-2} Z_i Z_{i+1} + h * sum_{i=0}^{n-1} X_i )
    Dense ED; safe up to n≈13.
    """
    dim = 1 << n
    if dim > 8192:  # n > 13
        raise ValueError(f"tfim_E0_open: n={n} too large for dense ED")

    H = np.zeros((dim, dim), dtype=np.float64)
    s = np.arange(dim, dtype=np.uint64)

    # Z Z diagonal part
    bits = ((s[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(np.int8)
    z = 1 - 2 * bits  # +1 / -1
    zz = np.zeros(dim, dtype=np.float64)
    for i in range(n - 1):
        zz += J * (z[:, i] * z[:, i + 1])
    H[np.arange(dim), np.arange(dim)] += sign * zz

    # X field off-diagonal part
    for i in range(n):
        t = s ^ (1 << i)
        H[s, t] += sign * h

    w = np.linalg.eigvalsh(H)
    return float(w[0])


def GetRandomStabilizerStates(n: int, num_stab: int = 8, seed: int = 1):
    """Structured stabilizer set: |0…0>, |+…+>, two Néel Z patterns, GHZ, then random Cliffords."""
    qcs = []

    if INCLUDE_FIXED_STATES:
        if len(qcs) < num_stab:
            # |0...0>
            qc0 = QuantumCircuit(n)
            qcs.append(qc0)

        if len(qcs) < num_stab:
            # |1...1>
            qc1 = QuantumCircuit(n)
            for q in range(n):
                qc1.x(q)
            qcs.append(qc1)

        if len(qcs) < num_stab:
            # |+...+>
            qcp = QuantumCircuit(n)
            for q in range(n):
                qcp.h(q)
            qcs.append(qcp)

        if len(qcs) < num_stab:
            # |-...->
            qc_minus = QuantumCircuit(n)
            for q in range(n):
                qc_minus.h(q)   # |+>
                qc_minus.z(q)   # Z|+> = |->
            qcs.append(qc_minus)

        if len(qcs) < num_stab:
            # Néel pattern in Z (1010...)
            qcz1 = QuantumCircuit(n)
            for q in range(0, n, 2):
                qcz1.x(q)
            qcs.append(qcz1)

        if len(qcs) < num_stab:
            # Néel pattern in Z (0101...)
            qcz2 = QuantumCircuit(n)
            for q in range(1, n, 2):
                qcz2.x(q)
            qcs.append(qcz2)

        if len(qcs) < num_stab:
            # GHZ
            qcs.append(ghz_circuit(n))

    if len(qcs) < num_stab:
        # Fill up to num_stab with random Clifford stabilizers in a reproducible way
        for r in range(num_stab - len(qcs)):
            qcs.append(random_stabilizer_circuit(n, seed=seed + r))

    return qcs


def tfim_Qs_and_M(V, J=1.0, h=1.3, periodic=False, sign=-1.0):
    """
    H = sign * (Σ_bonds J_b Z_i Z_j + Σ_i h_i X_i)
    Use sign = -1.0 to match the standard TFIM convention in the notes.
    OBC: len(J)=n-1. PBC: len(J)=n. Scalars also accepted.
    """

    V = np.asarray(V); d, m = V.shape
    n = int(round(np.log2(d)))
    if 2**n != d:
        raise ValueError("V.shape[0] must be a power of two.")

    bonds = [(i, i + 1) for i in range(n - 1)] if not periodic else [(i, (i + 1) % n) for i in range(n)]

    Jv = _as_1d(J, len(bonds), "J", periodic)
    hv = _as_1d(h, n, "h", periodic)

    rows = np.arange(d, dtype=np.int64)

    def apply_ZiZj(mat, i, j):
        phase_i = 1.0 - 2.0 * ((rows >> i) & 1)
        phase_j = 1.0 - 2.0 * ((rows >> j) & 1)
        return mat * (phase_i * phase_j)[:, None]

    def apply_Xi(mat, i):
        return mat[rows ^ (1 << i), :]

    M = np.zeros((m, m), dtype=np.complex128)
    Qs = []

    # ZZ bonds
    for b, (i, j) in enumerate(bonds):
        W = apply_ZiZj(V, i, j)
        M += sign * Jv[b] * (V.conj().T @ W)
        Qs.append(("ZZ", (i, j), float(Jv[b])))

    # X field
    for i in range(n):
        W = apply_Xi(V, i)
        M += sign * hv[i] * (V.conj().T @ W)
        Qs.append(("X", (i,), float(hv[i])))

    M = 0.5 * (M + M.conj().T)
    return Qs, M


def generalized_min_eig(M: np.ndarray, S: np.ndarray, tol: float = 1e-12):
    """
    Solve min α†Mα s.t. α†Sα=1. Projects to range(S) if S is singular.
    Returns (λ_min, α_normalized).
    """
    # eigen-decompose S
    w, U = np.linalg.eigh(0.5 * (S + S.conj().T))
    mask = w > tol
    if mask.sum() == 0:
        raise ValueError("S is numerically zero; stabilizer set is linearly dependent.")
    Ur = U[:, mask]
    dr = w[mask]
    Sinvhalf = Ur @ np.diag(1.0 / np.sqrt(dr))
    # reduce to standard Hermitian eigenproblem
    A = Sinvhalf.conj().T @ M @ Sinvhalf
    A = 0.5 * (A + A.conj().T)
    wA, vA = np.linalg.eigh(A)
    idx = int(np.argmin(wA))
    v = vA[:, idx]
    alpha = Sinvhalf @ v
    # normalize so α†Sα=1
    alpha /= np.sqrt(alpha.conj().T @ S @ alpha)
    return float(np.real(wA[idx])), alpha





def _project_to_range_S(M: np.ndarray, S: np.ndarray, tol: float = 1e-12):
    M = (M + M.conj().T) / 2
    S = (S + S.conj().T) / 2
    wS, US = eigh(S)                       # S = US diag(wS) US^†
    mask = wS > tol
    if mask.sum() == 0:
        raise ValueError("rank(S)=0")
    Ur = US[:, mask]                       # N×r
    sr = wS[mask]                          # r
    Dr_is = np.diag(1.0 / np.sqrt(sr))     # r×r
    B = Ur.conj().T @ M @ Ur               # r×r
    Ar = Dr_is @ B @ Dr_is                 # r×r  (Hermitian)
    return Ar, Ur, Dr_is

def smallest_gen_eig_dense(M: np.ndarray, S: np.ndarray, tol: float = 1e-12):
    Ar, Ur, Dr_is = _project_to_range_S(M, S, tol)
    w, v = eigh(Ar)                        # dense, all eigenpairs
    lam = float(w[0].real)
    alpha = Ur @ (Dr_is @ v[:, 0])
    alpha /= np.linalg.norm(alpha)
    return lam, alpha


def generalized_min_eig_lanczos(M: np.ndarray, S: np.ndarray, tol: float = 1e-12):
    """
    Lanczos/ARPACK solve of min α^† M α s.t. α^† S α = 1.
    Projects to range(S) if S is singular.
    Returns (λ_min, α_normalized).
    """
    # Hermitize
    M = 0.5 * (M + M.conj().T)
    S = 0.5 * (S + S.conj().T)

    # Project to well-conditioned subspace of S
    w, U = np.linalg.eigh(S)
    mask = w > tol
    if mask.sum() == 0:
        raise ValueError("S is numerically zero; stabilizer set is linearly dependent.")
    Ur = U[:, mask]
    Sr = np.diag(w[mask])
    Mr = Ur.conj().T @ M @ Ur

    # Lanczos generalized smallest algebraic eigenpair
    vals, vecs = eigsh(Mr, k=1, M=Sr, which="SA", tol=1e-10)
    lam = float(vals[0])
    alpha_r = vecs[:, 0]

    # Lift back and S-normalize
    alpha = Ur @ alpha_r
    alpha = alpha / np.sqrt(alpha.conj().T @ S @ alpha)

    return lam, alpha

def min_energy(M: np.ndarray, S: np.ndarray, method: str = "auto", tol: float = 1e-12):
    r = np.linalg.matrix_rank((S + S.conj().T) / 2, tol=tol)
    if method == "dense" or (method == "auto" and r <= R_LANCZOS):
        return smallest_gen_eig_dense(M, S, tol)
    if method == "lanczos" or method == "auto":
        return generalized_min_eig_lanczos(M, S, tol)
    raise ValueError(f"unknown method {method}")


def _single_energy_for_k(args):
    """
    Worker: compute λ_min for a single random stabilizer set of size k.
    Returns float energy.
    """
    (seed, n, k, J, h, periodic, sign) = args

    if STABILIZER_MAPPING == 'Random':
        qcs = GetRandomStabilizerStates(n, num_stab=k, seed=seed)
    elif STABILIZER_MAPPING == 'GrayCoded':
        qcs = reals_to_stabilizers(n, xs, periodic, layer_schedule=LAYER_SCHEDULE)

    V = build_V(qcs)
    Qs, M = tfim_Qs_and_M(V, J=J, h=h, periodic=periodic, sign=sign)
    S = gram_matrix(V)
    #lam, _ = generalized_min_eig_lanczos(M, S, tol=1e-12)
    lam, alpha = min_energy(M, S, method="auto", tol=1e-12)

    return float(lam)


def run_energy_vs_k_fixedD(num_qubits: int,
                           J: float,
                           h: float,
                           periodic: bool,
                           D: int,
                           seed_base: int,
                           threads: int):
    """
    For each k in K-list, launch D parallel workers, each computing λ_min for
    an independent random stabilizer set of size k (seeded).
    Save E matrix of shape (D, K) and plot all D curves in gray with best in color.
    """

    t_total0 = time.perf_counter()
    t_prev = t_total0
    times_sec = []
    t0 = time.perf_counter()

    print(f"[E vs #StabStates] submitting D = {D} jobs:")
    print(f"                         SYSTEM = {SYSTEM}")
    print(f"                       periodic = {periodic}")
    print(f"                        #Qubits = {num_qubits}")
    print(f"                              J = {J}")
    print(f"                              h = {h}")
    print(f"               #ParallelThreads = {threads}\n")

    ks = _make_k_list(num_qubits)
    print(f"                   iterate over #StabStates = {ks}\n")
    K = len(ks)
    Kmax_DEFAULT = 2**NUM_QUBITS
    seeds = np.array([seed_base + 1000 * r for r in range(D)], dtype=np.int64)

    E = np.empty((D, K), dtype=np.float64)

    # Reference energy
    E0_ref = None
    if SYSTEM == "TFIM":
        E0_ref = tfim_E0_periodic(num_qubits, J, h) if periodic else tfim_E0_open(num_qubits, J, h)
        print(f"                   reference energy = {E0_ref:.14f}\n")

    for j, k in enumerate(ks):
        FixedStates = ""
        if INCLUDE_FIXED_STATES:
            FixedStates += " [ fixed states: |0>^n"
            if k>1:
                FixedStates += " |1>^n"
            if k>2:
                FixedStates += " |+>^n"
            if k>3:
                FixedStates += " |->^n"
            if k>4:
                FixedStates += " |Néel_10>"
            if k>5:
                FixedStates += " |Néel_01>"
            if k>6:
                FixedStates += " |GHZ>"
            FixedStates += " ]"
        print(f"                   k={k}  {FixedStates}")
        args = [(int(seeds[r]), num_qubits, int(k), float(J), float(h), bool(periodic), -1.0)
                for r in range(D)]
        vals = [None] * D
        with ProcessPoolExecutor(max_workers=threads) as ex:
            futs = {ex.submit(_single_energy_for_k, a): idx for idx, a in enumerate(args)}
            for fut in as_completed(futs):
                r = futs[fut]
                vals[r] = fut.result()
        E[:, j] = np.array(vals, dtype=np.float64)
        dt = time.perf_counter() - t0
        times_sec.append(dt)
        now = time.perf_counter()
        delta = now - t_prev
        t_prev = now
        print(f"                         timing={delta:.2f}s")
        print(f"                   --->  min={E[:, j].min():.8f}  mean={E[:, j].mean():.8f}  max={E[:, j].max():.8f}")
        print(f"                   _____________________________________________________________________________________")

    # Save artifacts beside the log/backup
    klabel = f"{K_MODE}" if (K_LIST is None) else "list"
    base = BASE_NAME + f"_energy_vs_k_n{num_qubits}_D{D}_{klabel}"
    csv_path = os.path.join(output_dir, base + ".csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        header = ["k"] + [f"run_{r}" for r in range(D)]
        f.write(",".join(header) + "\n")
        for j, k in enumerate(ks):
            row = [str(k)] + [f"{E[r, j]:.12f}" for r in range(D)]
            f.write(",".join(row) + "\n")
    print(f"\n                    saved data: {os.path.basename(csv_path)}")

    # Plot: all curves gray, best in color
    xs = np.array(ks, dtype=np.int64)
    # Get figure and axis objects for more control
    fig, ax = plt.subplots(figsize=(9, 6))

    for r in range(D):
        # Use ax.plot instead of plt.plot
        ax.plot(xs, E[r, :], color="#bbbbbb", linewidth=1.0, alpha=0.7)

    # per-k minima envelope (mark best value at each k)
    best_per_k = E.min(axis=0)
    best_run_idx = E.argmin(axis=0)
    ax.plot(xs, best_per_k, linewidth=2.5, marker='o', markersize=3, color="red", label="best per k")

    # Set the x-axis to a log scale with base 2
    ax.set_xscale('log', base=2)
    # Format x-axis ticks to be integers (e.g., "2" instead of "2.0")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    ref_txt = "N/A"
    if E0_ref is not None:
        # count runs that reach the reference at the final k
        E0_hits = int(np.count_nonzero(np.isclose(E[:, -1], E0_ref, rtol=0.0, atol=TOL)))
        ref_txt = f"{E0_ref:.14f}"
        # Use ax.axhline
        ax.axhline(E0_ref, linestyle="--", linewidth=1.5, color="black", label=rf"$E_\mathrm{{ex}}$ = {E0_ref:.6f}")

    #if TASK == 'Whole_Shebang':
        #ToDo: add the GAO and CMA results!!!

    # Use ax.set_xlabel, etc. for labels and title
    ax.set_xlabel("k (number of stabilizer states)")
    ax.set_ylabel("energy")
    ax.set_title(
        f"{SYSTEM} {'PBC' if periodic else 'OBC'} -- J={J:g} -- h={h:g}\n"
        rf"qubits={num_qubits} $\,\to\; k_\mathrm{{max}}=2^{{{num_qubits}}}={Kmax_DEFAULT}$"
        "\n"
        rf"target hits / draws = {E0_hits} / {D} @ $k_\mathrm{{max}}$ (abstol={TOL})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    pdf_path = os.path.join(output_dir, base + ".pdf")
    plt.savefig(pdf_path)
    plt.close()

    print(f"                   saved plots: {os.path.basename(pdf_path)}\n")
    # final energy at last k (best across D runs)
    E0_final = float(E[:, -1].min())
    print(f"                   reference energy = {ref_txt}")
    print(f"                       final energy = {E0_final:.14f}")
    if E0_ref is not None:
        print(f"                      # target hits = {E0_hits}/{D}")
    else:
        print(f"                      # target hits = N/A")
    total_sec = time.perf_counter() - t_total0
    print(f"                    total wall time = {total_sec:.2f}s\n")


def GetEnergy(stab_seeds: list[int],
              num_qubits: int,
              J: float,
              h: float,
              periodic: bool) -> float:
    """
    Calculates the minimum energy for a given set of stabilizer states,
    identified by their integer seeds. This is the fitness function for the external GA.

    Args:
        stab_seeds (list[int]): A list of k integers, each used as a seed
                                to generate a unique Clifford operator.
        num_qubits (int): Number of qubits in the system.
        J (float): The ZZ coupling strength.
        h (float): The transverse field strength.
        periodic (bool): System boundary conditions.

    Returns:
        float: The calculated minimum energy (generalized eigenvalue).
    """
    if not stab_seeds:
        # Return a high energy for invalid input to penalize in GA
        return float('inf')

    # 1. Generate the k Clifford operators from the seeds, each s gives a unique deterministic Clifford
    #cliffords = [random_clifford(num_qubits, seed=s) for s in stab_seeds]
    cliffords = [random_clifford_safe(num_qubits, seed=s) for s in stab_seeds]

    # 2. Diagnostics block — BEFORE V, S, M are built
    #    Leave always-on while debugging; later gate by an env flag if desired.
    sigs = [cliff_sig(c) for c in cliffords]
    uniq = len(set(sigs))

    cols = [Statevector.from_instruction(c.to_instruction()).data for c in cliffords]
    V = np.column_stack(cols)
    S = V.conj().T @ V
    rank_S = np.linalg.matrix_rank(S, tol=1e-12)
    max_off = np.max(np.abs(S - np.diag(np.diag(S)))) if S.size else 0.0
    print(f"#uniq={uniq}/{len(sigs)}  rank(S)={rank_S}  max|S_ij| offdiag={max_off:.2e}")

    # 3. Build the Hamiltonian overlap matrix M and Gram matrix S
    _, M = tfim_Qs_and_M(V, J=J, h=h, periodic=periodic, sign=-1.0)
    S = gram_matrix(V)

    try:
        # 4. Solve the generalized eigenvalue problem
        #lam, _ = generalized_min_eig_lanczos(M, S, tol=1e-12)
        lam, alpha = min_energy(M, S, method="auto", tol=1e-12)
        return float(lam)
    except ValueError:
        # This can happen if the stabilizer set is linearly dependent (S is singular)
        # The GA should penalize such solutions.
        return float('inf')


def main_mpDPFT_interface(args=None):
    """
    This function is called when the script is run from the command line
    by the external C++ Genetic Algorithm.
    It parses seeds from command-line arguments, calculates energy, and prints it to stdout.
    """
    # Example usage from command line:
    # python pycliff_mit.py 123 456 789 1011

    # In C++ we can generate uint64_t as seeds without any concern. The full range of 0 to 18,446,744,073,709,551,615 is perfectly safe to pass to the Python script.
    """
    Called by external GA or CLI.
    """

    if args is None:
        args = sys.argv[1:]
    if not args:
        print("Error: no arguments", file=sys.stderr)
        sys.exit(1)

    # Try ints first; fall back to floats->seeds
    try:
        seeds = [int(a, 0) for a in args]  # supports "10", "0xA"
    except ValueError:
        floats = [float(a) for a in args]  # supports "1.2", "3e-4"
        seeds = floats_to_seeds(floats)

    # --- Using the global configuration from the USER INPUT of pycliff_mit.py ---
    energy = GetEnergy(
        stab_seeds=seeds,
        num_qubits=NUM_QUBITS,
        J=PARAM_J,
        h=PARAM_h,
        periodic=PERIODIC
    )

    # CRITICAL: Print only the final float value to stdout
    # The C++ program will read this single line.
    print(f"{energy:.14f}")

# ============================
# ===== END CORE MODULES =====
# ============================




# ==================================
# ===== BEGIN MAIN COMPUTATION =====
# ==================================

def main():
    # GAO fast path: if CLI carries only integers, run GA interface and exit
    if len(sys.argv) > 1 and all(a.lstrip("-").isdigit() for a in sys.argv[1:]):
        return main_mpDPFT_interface(sys.argv[1:])

    print("\nQiskit version:", qiskit.__version__, "\n")

    np.random.seed(0)
    np.set_printoptions(precision=4, suppress=True)

    if TASK == 'Minimal_Example':
        num_qubits = NUM_QUBITS
        num_stab = 1
        qcs = GetRandomStabilizerStates(num_qubits, num_stab=num_stab, seed=1)
        V = build_V(qcs)

        # Build M for TFIM (ZZ bonds, X field)
        Qs, M = tfim_Qs_and_M(V, J=PARAM_J, h=PARAM_h, periodic=PERIODIC, sign=-1.0)
        # print a few terms and their overlap matrices
        for k, (op, idx, coeff) in enumerate(Qs[:3]):
            Q = coeff * _span_overlap_term(V, op, idx)
            print(f"\n[{k:02d}] term={op}{idx}  coeff={coeff:+.6g}  Q.shape={Q.shape}")
            print(Q)

        # Overlap matrix S and Rayleigh-quotient minimization
        S = gram_matrix(V)
        lam_min, alpha = generalized_min_eig(M, S, tol=1e-12)

        print("\n ***** Consistency Checks ***** \n")
        n = V.shape[0].bit_length() - 1
        print("sum |J| bonds =", sum(1 for op, idx, _ in Qs if len(idx) == 2))
        print("periodic used =", PERIODIC, "sign =", -1.0)
        print("min diag(M) =", np.real(np.diag(M)).min())
        print(f"M shape               = {M.shape}")
        diagM = np.real(np.diag(M))
        diagS = np.real(np.diag(S))
        print("min diag(M)            =", diagM.min())
        print("argmin diag(M) [index] =", diagM.argmin())
        print("Sii                    =", diagS[diagM.argmin()])
        print("Mii/Sii                =", diagM.argmin(), diagM.min()/diagS[diagM.argmin()])

        print("\n ***** Final Results ***** \n")
        if PERIODIC:
            E0_ref = tfim_E0_periodic(n, 1.0, 1.3) if PERIODIC else tfim_E0_open(n, 1.0, 1.3)
            print(f"E0 = {E0_ref:.8f}   E0/num_qubits = {E0_ref/num_qubits:.8f}")
        else:
            print("exact non-periodic energy to be implemented")
        print(f"lambda_min (Rayleigh) = {lam_min:.8f}")
        print(f"alpha^† S alpha       = {float(np.real(alpha.conj().T @ S @ alpha)):.6f}")
        print(f"E(alpha) check        = {float(np.real(alpha.conj().T @ M @ alpha)):.8f}")

        print("\n ***** Lanczos double-check ***** \n")
        # lam_lanczos, _ = generalized_min_eig_lanczos(M, S, tol=1e-12)
        lam_lanczos, _ = min_energy(M, S, method="lanczos", tol=1e-12)
        print(f"lambda_min (Lanczos) = {lam_lanczos:.8f}")
        print(f"alpha^† S alpha (Lanczos)  = {float(np.real(alpha_lz.conj().T @ S @ alpha_lz)):.6f}")
        print(f"E(alpha_Lanczos) check     = {float(np.real(alpha_lz.conj().T @ M @ alpha_lz)):.8f}")

        print("\n ***** optimized α coefficients (Lanczos) ***** \n")
        np.set_printoptions(precision=8, suppress=True, linewidth=200)
        print(f"{alpha_lz}")

    # Lottery-based experiment
    if TASK == 'Energy_vs_NumStabStates' or TASK == 'Whole_Shebang':
        run_energy_vs_k_fixedD(
            num_qubits=NUM_QUBITS,
            J=PARAM_J,
            h=PARAM_h,
            periodic=PERIODIC,
            D=DRAWS,
            seed_base=SEED,
            threads=THREADS
        )



# ================================
# ===== END MAIN COMPUTATION =====
# ================================




# ==============================
# ===== BEGIN MAIN PROGRAM =====
# ==============================

if __name__ == "__main__":
    # Determine the type of script execution

    # We skip the first argument, which is the script name itself
    args = sys.argv[1:]

    # === Probe entry =========================================================
    if args and args[0] == "--probe":
        # Usage: python pycliff_mit.py --probe [N [J [h [periodic]]]]
        # Example: python pycliff_mit.py --probe 6 1.0 0.6 False
        n = int(args[1]) if len(args) > 1 else 6
        J = float(args[2]) if len(args) > 2 else 1.0
        h = float(args[3]) if len(args) > 3 else 0.6
        periodic = (args[4].lower() in {"1", "true", "yes", "y"}) if len(args) > 4 else False
        probe_two_states(n, J, h, periodic)
        sys.exit(0)
    # ========================================================================

    if not args:
        # Search for ground state energy via lottery of Clifford sets
        main()
    else:
        # The script will act as an interface for Optimimze() [genetic algorithm, CMA-ES, particle swarm, etc. in mpDPFT]
        THREADS = 1 # parallelization is handled within mpDPFT
        main_mpDPFT_interface(args)


# ============================
# ===== END MAIN PROGRAM =====
# ============================
