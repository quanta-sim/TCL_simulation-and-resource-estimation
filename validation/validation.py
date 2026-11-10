"""Consolidated TCL validation implementation.

Static, readable definitions; no embedded modules or parent-bundle imports.
Internal prefixes distinguish shared definitions merged from the development code.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"resource_estimation"))
import block_encoding as be
from qualtran.bloqs.bookkeeping._bookkeeping_bloq import _BookkeepingBloq as enc__BookkeepingBloq
from qualtran.bloqs.bookkeeping import Allocate as tape_Allocate
from qualtran.bloqs.bookkeeping import Free as tape_Free
from qualtran._infra.quantum_graph import LeftDangle as tape_LeftDangle
from dataclasses import dataclass as tape_dataclass
from functools import lru_cache as tape_lru_cache
import numpy as tape_np
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState as check_PhaseGradientState
import math as check_math
import numpy as check_np
from pathlib import Path as open_physics_Path
import csv as open_physics_csv
from scipy.linalg import expm as open_physics_expm
from scipy.sparse.linalg import expm_multiply as open_physics_expm_multiply
import importlib.metadata
import importlib as open_physics_importlib
import itertools as open_physics_itertools
import json as open_physics_json
import math as open_physics_math
import numpy as open_physics_np
import platform as open_physics_platform
import scipy as open_physics_scipy
from scipy import sparse as open_physics_sparse


# tcl_direct_validation.py -- retained active definitions

@tape_dataclass
class tape_Tape:
    inputs: dict
    outputs: dict
    operations: list
    width: int

def tape_gate_tape(bloq):
    """Expand the actual coherent hierarchy into a reusable physical-wire tape."""
    next_wire, free, ops = ([0], [], [])

    def alloc(n):
        out = []
        for _ in range(n):
            if free:
                out.append(free.pop())
            else:
                out.append(next_wire[0])
                next_wire[0] += 1
        return tape_np.asarray(out, dtype=int)

    def assign(mapping, soqs, reg, wire_array):
        if isinstance(soqs, tape_np.ndarray):
            for s, w in zip(soqs.reshape(-1), wire_array.reshape(-1, int(reg.bitsize))):
                mapping[s] = w
        else:
            mapping[soqs] = wire_array.reshape(-1)

    def emit(b, layout):
        b = be.enc_coherent(b)
        if isinstance(b, tape_Allocate):
            if b.dirty:
                raise ValueError('Validation requires clean allocation.')
            r = next(b.signature.rights())
            return {r.name: alloc(int(r.total_bits())).reshape((*r.shape, int(r.bitsize)))}
        if isinstance(b, tape_Free):
            if b.dirty:
                raise ValueError('Unexpected dirty free.')
            wires = tape_np.concatenate([layout[r.name].reshape(-1) for r in b.signature.lefts()])
            ops.append(('zero', tuple(wires)))
            free.extend(wires.tolist())
            return {}
        if isinstance(b, enc__BookkeepingBloq):
            wires = tape_np.concatenate([layout[r.name].reshape(-1) for r in b.signature.lefts()])
            out, offset = ({}, 0)
            for r in b.signature.rights():
                n = int(r.total_bits())
                out[r.name] = wires[offset:offset + n].reshape((*r.shape, int(r.bitsize)))
                offset += n
            if offset != len(wires):
                raise AssertionError('Bookkeeping width mismatch.')
            return out
        if be.enc__is_terminal(b):
            wires = tuple((int(x) for r in b.signature.lefts() for x in layout[r.name].reshape(-1)))
            ops.append((b, wires))
            return layout.copy()
        cb = b.body if isinstance(b, be.enc_Coherent) else b.decompose_bloq()
        mapping = {}
        for inst, ins, old_out in cb.iter_bloqsoqs():
            child_layout = {}
            for reg in inst.bloq.signature.lefts():
                soqs = ins[reg.name]
                flat = tape_np.asarray(soqs, dtype=object).reshape(-1)
                pieces = []
                for s in flat:
                    if s.binst is tape_LeftDangle:
                        w = layout[s.reg.name]
                        if s.reg.shape:
                            w = w[s.idx]
                    else:
                        w = mapping[s]
                    pieces.extend(tape_np.asarray(w).reshape(-1))
                child_layout[reg.name] = tape_np.asarray(pieces, dtype=int).reshape((*reg.shape, int(reg.bitsize)))
            outputs = emit(inst.bloq, child_layout)
            for reg, soqs in zip(inst.bloq.signature.rights(), old_out):
                assign(mapping, soqs, reg, outputs[reg.name])
        result = {}
        for reg in cb.signature.rights():
            pieces = []
            for s in tape_np.asarray(cb.final_soqs()[reg.name], dtype=object).reshape(-1):
                if s.binst is tape_LeftDangle:
                    w = layout[s.reg.name]
                    if s.reg.shape:
                        w = w[s.idx]
                else:
                    w = mapping[s]
                pieces.extend(tape_np.asarray(w).reshape(-1))
            result[reg.name] = tape_np.asarray(pieces, dtype=int).reshape((*reg.shape, int(reg.bitsize)))
        return result
    inputs = {r.name: alloc(int(r.total_bits())).reshape((*r.shape, int(r.bitsize))) for r in bloq.signature.lefts()}
    outputs = emit(bloq, inputs)
    return tape_Tape(inputs, outputs, ops, next_wire[0])

def tape_value_mask(wires, value):
    mask = 0
    for k, w in enumerate(tape_np.asarray(wires).reshape(-1)):
        if int(value) >> len(tape_np.asarray(wires).reshape(-1)) - k - 1 & 1:
            mask |= 1 << int(w)
    return mask

def tape__merge(keys, amps, tol=1e-13):
    unique, inverse = tape_np.unique(keys, return_inverse=True)
    vals = tape_np.zeros(len(unique), complex)
    tape_np.add.at(vals, inverse, amps)
    keep = tape_np.abs(vals) > tol
    return (unique[keep], vals[keep])

@tape_lru_cache(None)
def tape__matrix(b):
    return tape_np.asarray(b.tensor_contract(), complex)


# tcl_manuscript_validation.py -- retained active definitions

def check_extract(keys, wires):
    out = check_np.zeros(len(keys), check_np.uint64)
    for w in check_np.asarray(wires).reshape(-1):
        out = out << check_np.uint64(1) | check_np.asarray(keys >> int(w) & 1, dtype=check_np.uint64)
    return out

def check_run(tape, inputs=None, *, state=None, max_states=2000000, tolerance=2e-10):
    dtype = check_np.uint64 if tape.width <= 64 else object
    scalar = (lambda x: check_np.uint64(x)) if dtype == check_np.uint64 else int
    if state is None:
        key = 0
        for name, value in (inputs or {}).items():
            key |= tape_value_mask(tape.inputs[name], value)
        keys, amps = (check_np.asarray([key], dtype=dtype), check_np.asarray([1.0 + 0j]))
    else:
        keys, amps = (check_np.asarray(state[0], dtype=dtype).copy(), check_np.asarray(state[1], complex).copy())
    peak = 0
    for gate, w in tape.operations:
        if gate == 'zero':
            mask = scalar(sum((1 << int(k) for k in w)))
            leak = check_np.linalg.norm(amps[keys & mask != 0])
            if leak > tolerance:
                raise AssertionError(f'Unclean Free, residual norm {leak:.3g}')
            continue
        name = type(gate).__name__
        if name in ('CNOT', 'Toffoli'):
            cond = check_np.ones(len(keys), bool)
            for c in w[:-1]:
                cond &= keys >> int(c) & scalar(1) != 0
            keys ^= check_np.where(cond, check_np.asarray(1 << w[-1], dtype=dtype), check_np.asarray(0, dtype=dtype))
        elif name == 'XGate':
            keys ^= scalar(1 << w[0])
        elif name in ('CZ', 'ZGate'):
            cond = check_np.ones(len(keys), bool)
            for c in w:
                cond &= keys >> int(c) & scalar(1) != 0
            amps[cond] *= -1
        elif name == 'Identity':
            pass
        elif len(w) == 1:
            U = tape__matrix(gate).reshape(2, 2)
            bit = check_np.asarray(keys >> int(w[0]) & scalar(1), dtype=int)
            base = keys & ~scalar(1 << w[0])
            keys, amps = tape__merge(check_np.concatenate([base, base | scalar(1 << w[0])]), check_np.concatenate([amps * U[0, bit], amps * U[1, bit]]))
        else:
            raise NotImplementedError(f'Unexpected leaf {gate}')
        peak = max(peak, len(keys))
        if len(keys) > max_states:
            raise MemoryError('Sparse state size exceeded limit.')
    return (keys, amps, dict(peak_sparse_states=peak, physical_wires=tape.width))

def check_catalyst_state(tape, inputs=None, resource='resource'):
    wires = tape.inputs[resource].reshape(-1)
    pg = check_np.asarray(check_PhaseGradientState(len(wires), exponent=1).tensor_contract(), complex).reshape(-1)
    key = 0
    for name, value in (inputs or {}).items():
        key |= tape_value_mask(tape.inputs[name], value)
    dtype = check_np.uint64 if tape.width <= 64 else object
    return (check_np.asarray([key | tape_value_mask(wires, k) for k in range(len(pg))], dtype=dtype), pg)

def check_state_distance(tape, state, initial, resource='resource'):
    expected = {}
    keys, amps = check_catalyst_state(tape, initial, resource)
    pg = check_np.asarray(check_PhaseGradientState(len(tape.outputs[resource].reshape(-1)), exponent=1).tensor_contract(), complex).reshape(-1)
    for k, a in enumerate(pg):
        key = tape_value_mask(tape.outputs[resource], k)
        for name, value in initial.items():
            key |= tape_value_mask(tape.outputs[name], value)
        expected[key] = a
    got = dict(zip(map(int, state[0]), state[1]))
    union = expected.keys() | got.keys()
    return float(check_np.linalg.norm([expected.get(k, 0) - got.get(k, 0) for k in union]))

def check_project(tape, state, system='system', resource='resource', zero_registers=None):
    keys, amps = state[:2]
    w = tape.outputs[resource].reshape(-1)
    pg = check_np.asarray(check_PhaseGradientState(len(w), exponent=1).tensor_contract(), complex).reshape(-1)
    if zero_registers is None:
        zero_registers = [n for n in tape.outputs if n not in (system, resource)]
    mask = sum((1 << int(k) for n in zero_registers for k in tape.outputs[n].reshape(-1)))
    keep = keys & (check_np.uint64(mask) if tape.width <= 64 else mask) == 0
    out = check_np.zeros(2 ** len(tape.outputs[system].reshape(-1)), complex)
    check_np.add.at(out, check_extract(keys[keep], tape.outputs[system]).astype(int), amps[keep] * check_np.conj(pg[check_extract(keys[keep], w).astype(int)]))
    gmask = sum((1 << int(k) for k in w))
    other = keys & ~(check_np.uint64(gmask) if tape.width <= 64 else gmask)
    _, inv = check_np.unique(other, return_inverse=True)
    g = check_extract(keys, w).astype(int)
    reduced = check_np.zeros(int(inv.max()) + 1, complex)
    check_np.add.at(reduced, inv, amps * check_np.conj(pg[g]))
    residual = amps - reduced[inv] * pg[g]
    present = check_np.bincount(inv, minlength=len(reduced))
    missing = check_np.sum(abs(reduced) ** 2 * check_np.maximum(0.0, 1 - present / len(pg)))
    return (out, check_math.sqrt(float(check_np.vdot(residual, residual).real + missing)))

def check_projected_columns(bloq, columns):
    tape = tape_gate_tape(bloq)
    blocks = []
    leaks = []
    peak = 0
    for col in columns:
        state = check_run(tape, state=check_catalyst_state(tape, dict(system=int(col))))
        out, leak = check_project(tape, state)
        blocks.append(out)
        leaks.append(leak)
        peak = max(peak, state[2]['peak_sparse_states'])
    return (check_np.stack(blocks, axis=1), dict(columns=len(columns), catalyst_leakage=max(leaks), physical_wires=tape.width, peak_sparse_states=peak))

def check_rounded_tcl_columns(pair, columns):
    """Independent classical action with rounded probabilities and cosines.

    No circuit simulation or resource formulas are used to form this reference.
    It predicts even invalid binary labels and signed directed bonds.
    """
    full = pair['direct']['full']
    loc = full.local
    L = loc.L
    N = loc.N
    R = 1 << loc.q
    dims = [v for _ in range(L) for v in [R, *[2] * N]]
    out = check_np.zeros((int(check_np.prod(dims)), len(columns)), complex)
    A, B = be.model_factor_tables(loc.n_max, loc.b)
    ca = check_np.cos(check_np.pi * check_np.asarray(A) / (1 << loc.b))
    cb = check_np.cos(check_np.pi * check_np.asarray(B) / (1 << loc.b))

    def add(k, state, amplitude):
        out[check_np.ravel_multi_index(tuple(state), dims), k] += amplitude
    for k, col in enumerate(columns):
        state = list(check_np.unravel_index(col, dims))
        for label, coeff, prob in zip(loc.labels, loc.coefficients, full.prep_local.data['approx_probs']):
            weight = full.alpha_local / full.alpha * prob * (-1 if coeff < 0 else 1)
            if not weight:
                continue
            if label[0] == 'int':
                _, x, y, a = label
                j = y * loc.Lx + x
                pos = j * (N + 1)
                n = state[pos]
                atom = state[pos + 1 + a]
                mid = (n + atom) % R
                new = state.copy()
                new[pos] = (mid - (1 - atom)) % R
                new[pos + 1 + a] ^= 1
                add(k, new, weight * ca[mid])
            elif label[0] in ('ph', 'at'):
                _, x, y, a = label
                j = y * loc.Lx + x
                pos = j * (N + 1)
                bit = state[pos] >> a & 1 if label[0] == 'ph' else state[pos + 1 + a]
                add(k, state, weight * (-1) ** bit)
            else:
                add(k, state, weight)
        for label, (t, prob) in enumerate(zip(full.hop.t_table, full.prep_hop.data['approx_probs'])):
            j = label >> 2
            axis = label >> 1 & 1
            sign = label & 1
            x, y = (j % loc.Lx, j // loc.Lx)
            step = 1 if sign == 0 else -1
            i = y * loc.Lx + (x + step) % loc.Lx if axis == 0 else (y + step) % loc.Ly * loc.Lx + x
            pos = j * (N + 1)
            dst = i * (N + 1)
            new = state.copy()
            new[pos] = (new[pos] - 1) % R
            new[dst] = (new[dst] + 1) % R
            weight = full.alpha_hop / full.alpha * prob * (-1 if t > 0 else 1) * ca[state[pos]] * cb[state[dst]]
            add(k, new, weight)
    return out


# Codes/validate_open.py -- retained active definitions

open_physics_HERE = open_physics_Path(__file__).resolve().parent / 'results'
open_physics_HERE.mkdir(parents=True,exist_ok=True)

open_physics_FIGURE = open_physics_HERE / 'open_validation.pdf'

open_physics_FINAL_TIME = 4.0

open_physics_TIME_POINTS = 257

open_physics_STEP_COUNTS = (32, 64, 128, 256, 512, 1024)

open_physics_PARAMETERS = {'cavities': 2, 'emitters_per_cavity': 2, 'photon_cutoff': 2, 'maximum_total_excitation': 2, 'coupling_g': 1.0, 'hopping_t': 0.2, 'detuning': 0.0, 'polariton_potential': 0.0, 'photon_loss': 0.1, 'unequal_relaxation_per_cavity': [0.02, 0.2], 'equal_relaxation_per_cavity': [0.11, 0.11], 'dephasing': 0.0, 'initial_state': 'photons |0,0>; emitters |ee;gg>', 'final_time_in_inverse_g': open_physics_FINAL_TIME}

def open_physics_tensor(operators):
    result = open_physics_sparse.csr_matrix([[1.0]])
    for op in operators:
        result = open_physics_sparse.kron(result, open_physics_sparse.csr_matrix(op), format='csr')
    return result

def open_physics_local_operator(op, position):
    dims = (3, 3, 2, 2, 2, 2)
    return open_physics_tensor([op if i == position else open_physics_sparse.eye(d) for i, d in enumerate(dims)])

def open_physics_full_operators():
    """Independent tensor-product construction in the 144-dimensional space."""
    a = open_physics_np.diag(open_physics_np.sqrt([1.0, 2.0]), 1)
    sigma = open_physics_np.array([[0.0, 1.0], [0.0, 0.0]])
    photons = [open_physics_local_operator(a, i) for i in range(2)]
    emitters = [open_physics_local_operator(sigma, i + 2) for i in range(4)]
    h = -0.2 * (photons[0].T @ photons[1] + photons[1].T @ photons[0])
    for i in range(2):
        for r in range(2):
            exchange = photons[i].T @ emitters[2 * i + r]
            h = h + exchange + exchange.T
    return (h.tocsr(), photons, emitters)

def open_physics_basis_and_operators():
    full_basis = list(open_physics_itertools.product(range(3), range(3), *[range(2)] * 4))
    selected = [i for i, state in enumerate(full_basis) if sum(state) <= 2]
    basis = [full_basis[i] for i in selected]
    index = {state: i for i, state in enumerate(basis)}
    d = len(basis)
    h = open_physics_np.zeros((d, d), dtype=complex)
    photons = [open_physics_np.zeros_like(h) for _ in range(2)]
    emitters = [open_physics_np.zeros_like(h) for _ in range(4)]
    for column, state in enumerate(basis):
        for i in range(2):
            if state[i]:
                target = list(state)
                target[i] -= 1
                photons[i][index[tuple(target)], column] = open_physics_math.sqrt(state[i])
        for r in range(4):
            if state[r + 2]:
                target = list(state)
                target[r + 2] = 0
                emitters[r][index[tuple(target)], column] = 1
        for destination, source in ((0, 1), (1, 0)):
            if state[source] and state[destination] < 2:
                target = list(state)
                target[source] -= 1
                target[destination] += 1
                h[index[tuple(target)], column] -= 0.2 * open_physics_math.sqrt(state[source] * (state[destination] + 1))
        for i in range(2):
            for r in range(2):
                emitter = 2 + 2 * i + r
                if state[emitter] and state[i] < 2:
                    target = list(state)
                    target[emitter] = 0
                    target[i] += 1
                    row = index[tuple(target)]
                    h[row, column] += open_physics_math.sqrt(state[i] + 1)
                    h[column, row] += open_physics_math.sqrt(state[i] + 1)
    full_h, full_photons, full_emitters = open_physics_full_operators()
    restriction = open_physics_np.ix_(selected, selected)
    assert open_physics_np.max(open_physics_np.abs(h - full_h.toarray()[restriction])) < 1e-14
    for small, full in zip(photons + emitters, full_photons + full_emitters):
        assert open_physics_np.max(open_physics_np.abs(small - full.toarray()[restriction])) < 1e-14
    assert open_physics_np.linalg.norm(h - h.conj().T) < 1e-14
    numbers = open_physics_np.array([sum(state) for state in basis])
    active = open_physics_np.flatnonzero((numbers[:, None] == numbers[None, :]).reshape(-1, order='F'))
    rho0 = open_physics_np.zeros_like(h)
    rho0[index[0, 0, 1, 1, 0, 0], index[0, 0, 1, 1, 0, 0]] = 1
    observables = {}
    for i in range(2):
        occupations = open_physics_np.array([state[i] for state in basis])
        observables[f'n{i + 1}'] = open_physics_np.diag(occupations)
        observables[f'factorial{i + 1}'] = open_physics_np.diag(occupations * (occupations - 1))
        observables[f'imbalance{i + 1}'] = open_physics_np.diag([state[2 + 2 * i] - state[3 + 2 * i] for state in basis])
        singlet = open_physics_np.zeros_like(h)
        for column, state in enumerate(basis):
            left, right = (2 + 2 * i, 3 + 2 * i)
            if state[left] != state[right]:
                target = list(state)
                target[left], target[right] = (target[right], target[left])
                singlet[column, column] = 0.5
                singlet[index[tuple(target)], column] = -0.5
        observables[f'singlet{i + 1}'] = singlet
    return (basis, selected, active, h, photons, emitters, rho0, observables)

def open_physics_generator(h, jumps):
    h = open_physics_sparse.csr_matrix(h)
    identity = open_physics_sparse.eye(h.shape[0], format='csr')
    result = -1j * (open_physics_sparse.kron(identity, h) - open_physics_sparse.kron(h.T, identity))
    for jump in jumps:
        jump = open_physics_sparse.csr_matrix(jump)
        norm_square = jump.conj().T @ jump
        result = result + open_physics_sparse.kron(jump.conj(), jump) - 0.5 * (open_physics_sparse.kron(identity, norm_square) + open_physics_sparse.kron(norm_square.T, identity))
    return result.tocsr()

def open_physics_jumps_for(photons, emitters, rates):
    return [open_physics_math.sqrt(0.1) * op for op in photons] + [open_physics_math.sqrt(rates[r % 2]) * op for r, op in enumerate(emitters)]

def open_physics_pack(rho, active):
    return rho.reshape(-1, order='F')[active]

def open_physics_unpack(vector, active, d):
    result = open_physics_np.zeros(d * d, dtype=complex)
    result[active] = vector
    return result.reshape((d, d), order='F')

def open_physics_values(rho, observables):
    return {name: float(open_physics_np.trace(op @ rho).real) for name, op in observables.items()}

def open_physics_trace_norm(matrix):
    assert open_physics_np.linalg.norm(matrix - matrix.conj().T) < 1e-10
    return float(open_physics_np.sum(open_physics_np.abs(open_physics_np.linalg.eigvalsh((matrix + matrix.conj().T) / 2))))

def open_physics_kraus_map(kraus, active):
    d = kraus[0].shape[0]
    completeness = sum((k.conj().T @ k for k in kraus))
    residual = float(open_physics_np.linalg.norm(completeness - open_physics_np.eye(d), ord=2))
    assert residual < 2e-12, residual
    result = sum((open_physics_np.kron(k.conj(), k) for k in kraus))
    return (result[open_physics_np.ix_(active, active)], residual)

def open_physics_sampled_map(h, jumps, dt, active):
    squared_norms = open_physics_np.array([open_physics_np.linalg.norm(j, ord=2) ** 2 for j in jumps])
    probabilities = squared_norms / squared_norms.sum()
    d = len(h)
    result = open_physics_np.zeros((len(active), len(active)), dtype=complex)
    max_tp_error = 0.0
    for probability, jump in zip(probabilities, jumps):
        scaled = jump / open_physics_math.sqrt(probability)
        dilation = open_physics_np.block([[open_physics_math.sqrt(dt) * h, scaled.conj().T], [scaled, open_physics_np.zeros_like(h)]])
        unitary = open_physics_expm(-1j * open_physics_math.sqrt(dt) * dilation)
        channel, residual = open_physics_kraus_map([unitary[:d, :d], unitary[d:, :d]], active)
        result += probability * channel
        max_tp_error = max(max_tp_error, residual)
    return (result, max_tp_error)

def open_physics_loss_kraus(basis, site, dt):
    """Exact truncated pure-loss channel, including all loss multiplicities."""
    eta = open_physics_math.exp(-0.1 * dt)
    index = {state: i for i, state in enumerate(basis)}
    matrices = []
    for lost in range(3):
        matrix = open_physics_np.zeros((len(basis), len(basis)), dtype=complex)
        for column, state in enumerate(basis):
            n = state[site]
            if n >= lost:
                target = list(state)
                target[site] -= lost
                matrix[index[tuple(target)], column] = open_physics_math.sqrt(open_physics_math.comb(n, lost) * (1 - eta) ** lost * eta ** (n - lost))
        matrices.append(matrix)
    return matrices

def open_physics_symmetric_split_map(basis, h, emitters, rates, dt, active):
    """Strang split: exact H/2, exact independent local channels, exact H/2."""
    half_h, residual = open_physics_kraus_map([open_physics_expm(-0.5j * dt * h)], active)
    dissipative = open_physics_np.eye(len(active), dtype=complex)
    for i in range(2):
        channel, error = open_physics_kraus_map(open_physics_loss_kraus(basis, i, dt), active)
        dissipative = channel @ dissipative
        residual = max(residual, error)
    for r, lowering in enumerate(emitters):
        eta = open_physics_math.exp(-rates[r % 2] * dt)
        k0 = open_physics_np.diag([open_physics_math.sqrt(eta) if state[2 + r] else 1.0 for state in basis])
        k1 = open_physics_math.sqrt(1 - eta) * lowering
        channel, error = open_physics_kraus_map([k0, k1], active)
        dissipative = channel @ dissipative
        residual = max(residual, error)
    return (half_h @ dissipative @ half_h, residual, dissipative)

def open_physics_evolve_map(channel, rho0, steps, active, observables, final_time, stride=1):
    vector = open_physics_pack(rho0, active)
    d = len(rho0)
    records = []
    max_trace_error = max_hermiticity_error = 0.0
    minimum_eigenvalue = 1.0
    for step in range(steps + 1):
        rho = open_physics_unpack(vector, active, d)
        max_trace_error = max(max_trace_error, abs(open_physics_np.trace(rho) - 1))
        max_hermiticity_error = max(max_hermiticity_error, open_physics_np.linalg.norm(rho - rho.conj().T))
        minimum_eigenvalue = min(minimum_eigenvalue, float(open_physics_np.linalg.eigvalsh((rho + rho.conj().T) / 2)[0]))
        if step % stride == 0:
            records.append({'time': step * final_time / steps, **open_physics_values(rho, observables)})
        if step != steps:
            vector = channel @ vector
    checks = {'maximum_trace_error': float(max_trace_error), 'maximum_hermiticity_error': float(max_hermiticity_error), 'minimum_state_eigenvalue': minimum_eigenvalue}
    assert max_trace_error < 2e-11, checks
    assert max_hermiticity_error < 2e-11, checks
    assert minimum_eigenvalue > -2e-11, checks
    return (rho, records, checks)

def open_physics_write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = open_physics_csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def open_physics_make_plot(trajectories, convergence):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.7), constrained_layout=True)
    colors = ['#1565A3', '#CF6B18']
    for ax, prefix, title, ylabel in ((axes[0, 0], 'n', '(a) Photon populations', '$\\langle n_i\\rangle$'), (axes[0, 1], 'factorial', '(b) Photon factorial moments', '$\\langle n_i(n_i-1)\\rangle$')):
        exact = [row for row in trajectories if row['method'] == 'exact_unequal']
        sampled = [row for row in trajectories if row['method'] == 'sampled_unequal']
        for i, color in enumerate(colors, 1):
            ax.plot([r['time'] for r in exact], [r[f'{prefix}{i}'] for r in exact], color=color, linewidth=1.25, label=f'cavity {i}')
            ax.plot([r['time'] for r in sampled[::16]], [r[f'{prefix}{i}'] for r in sampled[::16]], linestyle='none', marker='o', markersize=2.4, markerfacecolor='white', markeredgewidth=0.65, color=color)
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([], [], linestyle='none', marker='o', markersize=3, markerfacecolor='white', color='0.2'))
        labels.append('averaged dilation')
        ax.legend(handles, labels, loc='upper right', frameon=False)
        ax.set(xlabel='$g t$', ylabel=ylabel, title=title, xlim=(0, open_physics_FINAL_TIME))
    ax = axes[1, 0]
    for method, linestyle, rate_label in (('exact_unequal', '-', 'unequal'), ('exact_equal', '--', 'equal mean')):
        rows = [r for r in trajectories if r['method'] == method]
        for name, color, label in (('singlet1', colors[0], '$P_{\\mathrm{s},1}$'), ('imbalance1', colors[1], '$I_1$')):
            ax.plot([r['time'] for r in rows], [r[name] for r in rows], color=color, linestyle=linestyle, linewidth=1.25, label=f'{label}, {rate_label}')
    ax.legend(frameon=False, loc='upper left', ncol=2)
    ax.set(xlabel='$g t$', ylabel='Singlet probability / imbalance', title='(c) Independent emitter relaxation', xlim=(0, open_physics_FINAL_TIME), ylim=(-0.085, 0.165))
    ax = axes[1, 1]
    sampled = [r for r in convergence if r['method'] == 'sampled']
    split = [r for r in convergence if r['method'] == 'symmetric_split']
    for rows, key, color, marker, label in ((sampled, 'trace_norm_error', '0.15', 'o', 'sampled: state'), (split, 'trace_norm_error', '#8856A7', 'D', 'symmetric split: state')):
        ax.loglog([r['dt'] for r in rows], [r[key] for r in rows], color=color, marker=marker, markersize=3, linewidth=1, linestyle='--' if rows is split else '-', label=label)
    ax.legend(frameon=False, loc='lower right')
    ax.set(xlabel='$g\\,\\Delta t$', ylabel='Error at $gT=4$', title='(d) Timestep convergence')
    for ax in axes.flat:
        ax.grid(alpha=0.15, linewidth=0.5)
    open_physics_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(open_physics_FIGURE)
    fig.savefig(open_physics_HERE / 'open_validation_preview.png', dpi=170)
    plt.close(fig)

def open_physics_main():
    basis, selected, active, h, photons, emitters, rho0, observables = open_physics_basis_and_operators()
    assert len(basis) == 24 and len(active) == 326
    times = open_physics_np.linspace(0, open_physics_FINAL_TIME, open_physics_TIME_POINTS)
    trajectories, convergence, checks, slopes = ([], [], {}, {})
    exact_final = {}
    for case, rates in (('unequal', (0.02, 0.2)), ('equal', (0.11, 0.11))):
        jumps = open_physics_jumps_for(photons, emitters, rates)
        liouvillian = open_physics_generator(h, jumps)[active][:, active]
        exact_vectors = open_physics_expm_multiply(liouvillian, open_physics_pack(rho0, active), start=0, stop=open_physics_FINAL_TIME, num=open_physics_TIME_POINTS)
        min_eig, trace_error = (1.0, 0.0)
        for time, vector in zip(times, exact_vectors):
            rho = open_physics_unpack(vector, active, len(basis))
            min_eig = min(min_eig, float(open_physics_np.linalg.eigvalsh((rho + rho.conj().T) / 2)[0]))
            trace_error = max(trace_error, float(abs(open_physics_np.trace(rho) - 1)))
            trajectories.append({'method': f'exact_{case}', 'time': float(time), **open_physics_values(rho, observables)})
        exact_final[case] = rho
        assert min_eig > -2e-11 and trace_error < 2e-11
        checks[f'exact_{case}'] = {'minimum_state_eigenvalue': min_eig, 'maximum_trace_error': trace_error}
        derivative = open_physics_unpack(liouvillian @ open_physics_pack(rho0, active), active, len(basis))
        actual = open_physics_values(derivative, observables)
        singlet_expected = sum(rates) / 2
        imbalance_expected = rates[1] - rates[0]
        assert abs(actual['singlet1'] - singlet_expected) < 1e-13
        assert abs(actual['imbalance1'] - imbalance_expected) < 1e-13
        slopes[case] = {'singlet1': actual['singlet1'], 'singlet1_expected': singlet_expected, 'imbalance1': actual['imbalance1'], 'imbalance1_expected': imbalance_expected}
    full_h, full_photons, full_emitters = open_physics_full_operators()
    full_rho0 = open_physics_np.zeros((144, 144), dtype=complex)
    full_rho0[open_physics_np.ix_(selected, selected)] = rho0
    full_l = open_physics_generator(full_h, open_physics_jumps_for(full_photons, full_emitters, (0.02, 0.2)))
    full_vector = open_physics_expm_multiply(open_physics_FINAL_TIME * full_l, full_rho0.reshape(-1, order='F'))
    full_final = full_vector.reshape((144, 144), order='F')
    embedded = open_physics_np.zeros_like(full_final)
    embedded[open_physics_np.ix_(selected, selected)] = exact_final['unequal']
    full_residual = float(open_physics_np.linalg.norm(full_final - embedded, ord='fro'))
    assert full_residual < 1e-11, full_residual
    checks['full_tensor_product_reference_frobenius_residual'] = full_residual
    jumps = open_physics_jumps_for(photons, emitters, (0.02, 0.2))
    reference_values = open_physics_values(exact_final['unequal'], observables)
    reference_trajectory = {r['time']: r for r in trajectories if r['method'] == 'exact_unequal'}
    for steps in open_physics_STEP_COUNTS:
        dt = open_physics_FINAL_TIME / steps
        for method in ('sampled', 'symmetric_split'):
            if method == 'sampled':
                channel, tp_error = open_physics_sampled_map(h, jumps, dt, active)
            else:
                channel, tp_error, dissipative = open_physics_symmetric_split_map(basis, h, emitters, (0.02, 0.2), dt, active)
                if steps == open_physics_STEP_COUNTS[0]:
                    exact_d = open_physics_expm(dt * open_physics_generator(open_physics_np.zeros_like(h), jumps)[active][:, active].toarray())
                    error = float(open_physics_np.linalg.norm(dissipative - exact_d, ord='fro'))
                    assert error < 1e-11
                    checks['local_channels_vs_exact_dissipator_frobenius_residual'] = error
            final, rows, state_checks = open_physics_evolve_map(channel, rho0, steps, active, observables, open_physics_FINAL_TIME, stride=max(1, steps // 256))
            if steps == open_physics_STEP_COUNTS[-1]:
                trajectories += [{'method': f'{method}_unequal', **row} for row in rows]
            final_values = open_physics_values(final, observables)
            convergence.append({'method': method, 'steps': steps, 'dt': dt, 'trace_norm_error': open_physics_trace_norm(final - exact_final['unequal']), 'maximum_photon_error': max((abs(final_values[f'n{i}'] - reference_values[f'n{i}']) for i in (1, 2))), 'maximum_factorial_error': max((abs(final_values[f'factorial{i}'] - reference_values[f'factorial{i}']) for i in (1, 2))), 'maximum_photon_error_over_time': max((abs(r[f'n{i}'] - reference_trajectory[r['time']][f'n{i}']) for r in rows for i in (1, 2))), 'maximum_factorial_error_over_time': max((abs(r[f'factorial{i}'] - reference_trajectory[r['time']][f'factorial{i}']) for r in rows for i in (1, 2))), 'maximum_kraus_completeness_error': tp_error, **state_checks})
            print(f'{method:16s} steps={steps:4d} trace norm error={convergence[-1]['trace_norm_error']:.8g}', flush=True)
    orders = {}
    for method in ('sampled', 'symmetric_split'):
        rows = [r for r in convergence if r['method'] == method]
        orders[method] = {}
        for key in ('trace_norm_error', 'maximum_photon_error', 'maximum_factorial_error'):
            ratios = [rows[i][key] / rows[i + 1][key] for i in range(len(rows) - 1)]
            orders[method][key] = {'halving_ratios': ratios, 'last_pair_order': open_physics_math.log2(ratios[-1])}
        expected_order = 1 if method == 'sampled' else 2
        assert abs(orders[method]['trace_norm_error']['last_pair_order'] - expected_order) < 0.08
    unequal_rows = [r for r in trajectories if r['method'] == 'exact_unequal']
    equal_rows = [r for r in trajectories if r['method'] == 'exact_equal']
    assert max((r['singlet1'] for r in equal_rows)) > 0.001
    assert max((abs(r['imbalance1']) for r in equal_rows)) < 1e-12
    summary = {'status': 'PASS', 'parameters': open_physics_PARAMETERS, 'scope': 'Ideal averaged channel and small-graph physical validation; not gate-level synthesis, a diamond-norm certificate, a 2D resource benchmark, or a quantum-advantage demonstration.', 'sector_dimensions': [1, 6, 17], 'hilbert_dimension': len(basis), 'density_block_entries': len(active), 'full_cutoff_hilbert_dimension': 144, 'photon_cutoff_exact_for_this_initial_state_and_transient_observables': True, 'nonzero_jump_channels': 6, 'dissipative_normalization': 0.84, 'time_points': open_physics_TIME_POINTS, 'step_counts': list(open_physics_STEP_COUNTS), 'trace_norm_convention': '||rho_approx - rho_exact||_1, with no factor 1/2', 'monte_carlo_sampling': False, 'method': 'Expected independently resampled channel at each step, iterated deterministically.', 'symmetric_split_reference': 'Exact Hamiltonian half steps and exact independent local amplitude-damping/pure-loss channels. Equal timestep is not equal gate cost.', 'analytic_initial_slopes': slopes, 'checks': checks, 'convergence': convergence, 'observed_orders': orders, 'final_exact_observables': {case: open_physics_values(rho, observables) for case, rho in exact_final.items()}, 'maximum_equal_rate_imbalance': max((abs(r['imbalance1']) for r in equal_rows)), 'maximum_equal_rate_singlet_probability': max((r['singlet1'] for r in equal_rows)), 'maximum_unequal_rate_singlet_probability': max((r['singlet1'] for r in unequal_rows)), 'maximum_unequal_rate_imbalance': max((r['imbalance1'] for r in unequal_rows)), 'maximum_unequal_vs_equal_photon_difference': max((abs(u[f'n{i}'] - e[f'n{i}']) for u, e in zip(unequal_rows, equal_rows) for i in (1, 2))), 'versions': {'python': open_physics_platform.python_version(), 'numpy': open_physics_np.__version__, 'scipy': open_physics_scipy.__version__, 'matplotlib': open_physics_importlib.metadata.version('matplotlib')}}
    open_physics_write_csv(open_physics_HERE / 'open_trajectories.csv', trajectories)
    open_physics_write_csv(open_physics_HERE / 'open_convergence.csv', convergence)
    (open_physics_HERE / 'open_validation_summary.json').write_text(open_physics_json.dumps(summary, indent=2) + '\n')
    open_physics_make_plot(trajectories, convergence)
    print(open_physics_json.dumps({'status': 'PASS', 'figure': str(open_physics_FIGURE), 'sampled_order': orders['sampled']['trace_norm_error']['last_pair_order'], 'split_order': orders['symmetric_split']['trace_norm_error']['last_pair_order']}, indent=2))

# Notebook API and independent references for consolidated circuit tests.
import math
import hashlib

import json
import numpy as np
import pandas as pd
import pytest
import walk as wk
import gqsp as gq
from types import SimpleNamespace
from qualtran import BloqBuilder
from qualtran.bloqs.basic_gates import Hadamard
from qualtran.bloqs.rotations.phase_gradient import PhaseGradientState


def reference_jump(Lx,Ly,N,n_max,b,kind,site=0,atom=0):
    q=(n_max).bit_length()
    return SimpleNamespace(Lx=Lx,Ly=Ly,N=N,n_max=n_max,b=b,kind=kind,site=site,atom=atom,q=q,system_bitsize=Lx*Ly*(q+N))


def reference_jump_matrix(jump,columns):
    R=1<<jump.q;dims=[v for _ in range(jump.Lx*jump.Ly) for v in [R,*([2]*jump.N)]]
    out=np.zeros((1<<jump.system_bitsize,len(columns)),complex)
    for k,col in enumerate(columns):
        state=list(np.unravel_index(col,dims));pos=jump.site*(jump.N+1)
        if jump.kind=='photon_loss':
            word=be.model_factor_tables(jump.n_max,jump.b)[0][state[pos]]
            amp=math.cos(math.pi*word/(1<<jump.b));state[pos]=(state[pos]-1)%R
        elif jump.kind=='atomic_loss':
            pos+=1+jump.atom;amp=float(state[pos]);state[pos]=0
        else:
            pos+=1+jump.atom;amp=(-1.)**state[pos]
        out[np.ravel_multi_index(tuple(state),dims),k]=amp
    return out


def logical_state(tape,state,names):
    result={}
    for key,amp in zip(state[0],state[1]):
        label=tuple(int(check_extract(np.asarray([key],dtype=object),tape.outputs[n])[0]) for n in names)
        result[label]=result.get(label,0)+amp
    return result


def state_distance(a,b):return float(np.linalg.norm([a.get(k,0)-b.get(k,0) for k in a.keys()|b.keys()]))


def small_full(nmax=1, b=1):
    return be.model_build_pair(Lx=2, Ly=2, N=1, n_max=nmax, b=b, B=1, mu=np.zeros((2, 2)), modes=('direct',))['direct']['full']


def test_plan_and_sampling_allowance():
    p = gq.opened_plan()
    assert tuple((p[k] for k in ('b', 'B', 'K', 'm', 'N_step', 'N_shot'))) == (33, 24, 6, 48, 251658240, 1794879)
    assert float(p['state_tolerance']) == 0.0015625
    assert float(p['sampling_error_bound']) <= float(p['sampling_budget']) * (1 + 1e-14)
    assert float(p['two_eta_bound']) <= float(p['polynomial_budget'])
    assert float(p['previous_degree_bound']) > float(p['polynomial_budget'])
    assert not p['full_channel_error_certified']
    q = gq.opened_plan(dps=120)
    assert all((p[k] == q[k] for k in ('b', 'B', 'K', 'm', 'N_step', 'N_shot')))
    assert gq.opened_plan(epsilon_g='.05')['N_shot'] > p['N_shot']


@pytest.mark.parametrize('kind', wk.opened_KINDS)
@pytest.mark.parametrize('nmax', [1, 2, 3])
def test_jump_projected_matrix_including_vacuum_and_padding(kind, nmax):
    full = small_full(nmax, 3)
    jump = wk.opened_Jump(full, kind, site=1)
    bb = BloqBuilder()
    system = bb.add_register('system', be.enc_QAny(full.system_bitsize))
    eta = bb.add_register('ancilla', be.enc_QBit())
    resource = bb.add_register('resource', be.enc_QAny(full.resource_bitsize))
    e = bb.add(be.enc_XGate(), q=bb.allocate(dtype=be.enc_QBit()))
    o = bb.add_d(jump, enable=e, system=system, eta=eta, resource=resource)
    bb.free(bb.add(be.enc_XGate(), q=o.pop('enable')))
    block = bb.finalize(system=o['system'], ancilla=o['eta'], resource=o['resource'])
    stride = full.local.q + 1
    shift = full.system_bitsize - 2 * stride
    cols = [n * 2 + a << shift for n in range(1 << full.local.q) for a in (0, 1)]
    got, diag = check_projected_columns(block, cols)
    reference = reference_jump(2, 2, 1, nmax, 3, kind, site=1)
    target = reference_jump_matrix(reference, cols)
    assert np.linalg.norm(got - target) < 1e-10
    assert diag['catalyst_leakage'] < 1e-10


def projected_dilation(walk):
    """Undo only R and prepare/project h=+, exposing the encoded dilation."""
    full = walk.full
    bb = BloqBuilder()
    system = bb.add_register('system', be.enc_QAny(full.system_bitsize + 1))
    A = be.enc__split(bb, bb.add_register('ancilla', be.enc_QAny(full.ancilla_bitsize + 3)), full.ancilla_bitsize + 3)
    resource = bb.add_register('resource', be.enc_QAny(full.resource_bitsize))
    A[0] = bb.add(Hadamard(), q=A[0])
    ctrl = bb.add(be.enc_XGate(), q=bb.allocate(dtype=be.enc_QBit()))
    o = bb.add_d(walk, ctrl=ctrl, selection=be.enc__join(bb, A), system=system, resource=resource)
    A = be.enc__split(bb, o['selection'], full.ancilla_bitsize + 3)
    r = bb.add_d(be.model_reflection(full.ancilla_bitsize + 2), ctrl=o['ctrl'], h=A[0], ancilla=be.enc__join(bb, A[1:]))
    A = [bb.add(Hadamard(), q=r['h']), *be.enc__split(bb, r['ancilla'], full.ancilla_bitsize + 2)]
    bb.free(bb.add(be.enc_XGate(), q=r['ctrl']))
    return bb.finalize(system=o['system'], ancilla=be.enc__join(bb, A), resource=o['resource'])


@pytest.mark.parametrize('kind', wk.opened_KINDS)
def test_walk_projects_to_independent_hermitian_dilation(kind):
    full = small_full()
    dt = 0.02
    lam = 0.7
    walk = wk.opened_Walk(full, kind, dt, lam, site=1)
    pair = be.model_build_pair(Lx=2, Ly=2, N=1, n_max=1, b=1, B=1, mu=np.zeros((2, 2)), modes=('direct',))
    D = 1 << full.system_bitsize
    raw = check_rounded_tcl_columns(pair, range(D)) * full.alpha
    H = (raw + raw.conj().T) / 2
    J = reference_jump_matrix(reference_jump(2, 2, 1, 1, 1, kind, site=1), range(D))
    target = np.block([[math.sqrt(dt) * H, math.sqrt(lam) * J.conj().T], [math.sqrt(lam) * J, np.zeros_like(H)]]) / (math.sqrt(dt) * full.alpha + math.sqrt(lam))
    cols = [0, 1, 64, D + 1, D + 64]
    got, diag = check_projected_columns(projected_dilation(walk), cols)
    assert np.linalg.norm(got - target[:, cols]) < 1e-10
    assert diag['catalyst_leakage'] < 1e-10


@pytest.mark.parametrize('kind', wk.opened_KINDS)
def test_disabled_walk_and_exact_adjoint_on_nonzero_signals(kind):
    full = small_full()
    walk = wk.opened_Walk(full, kind, 0.02, 0.7)
    tape = tape_gate_tape(walk)
    args = dict(ctrl=0, selection=(1 << full.ancilla_bitsize + 2) + 339, system=65)
    state = check_run(tape, state=check_catalyst_state(tape, args))
    assert check_state_distance(tape, state, args) < 1e-10
    bb, s = BloqBuilder.from_signature(walk.signature)
    s = bb.add_d(walk, **s)
    s = bb.add_d(be.enc_coherent(walk).adjoint(), **s)
    identity = tape_gate_tape(bb.finalize(**s))
    args['ctrl'] = 1
    state = check_run(identity, state=check_catalyst_state(identity, args))
    assert check_state_distance(identity, state, args) < 1e-10


def both_catalysts(tape, args):
    base = sum((tape_value_mask(tape.inputs[k], v) for k, v in args.items()))
    a = np.asarray(PhaseGradientState(len(tape.inputs['resource']), exponent=1).tensor_contract()).reshape(-1)
    b = np.asarray(PhaseGradientState(len(tape.inputs['rotation_resource']), exponent=1).tensor_contract()).reshape(-1)
    keys = [base | tape_value_mask(tape.inputs['resource'], i) | tape_value_mask(tape.inputs['rotation_resource'], j) for i in range(len(a)) for j in range(len(b))]
    return (np.array(keys, dtype=object), np.kron(a, b))


@pytest.mark.parametrize('inverse,cv', [(False, 1), (True, 1), (False, 0)])
def test_synthesized_walk_matches_rounded_native_with_two_catalysts(inverse, cv):
    walk = wk.opened_Walk(small_full(), 'photon_loss', 0.02, 0.7)
    a, b = [tape_gate_tape(gq.opened_RoundedWalk(walk, 2, inverse, cv, native)) for native in (False, True)]
    args = dict(ctrl=1, selection=3, system=65)
    states = [check_run(t, state=both_catalysts(t, args)) for t in (a, b)]
    left, right = [logical_state(t, s, list(t.outputs)) for t, s in zip((a, b), states)]
    assert state_distance(left, right) < 1e-10


def test_fresh_environment_and_persistent_catalysts():
    walk = wk.opened_Walk(small_full(), 'atomic_loss', 0.02, 0.7)
    op = gq.opened_StepChannel(walk, 2, 1)
    leaves = wk.rotation_clifford_t_leaves(op)
    zeros = sum((n for b, n in leaves.items() if isinstance(b, gq.opened_ZeroState)))
    discards = sum((n for b, n in leaves.items() if isinstance(b, gq.opened_DiscardQ)))
    assert zeros == discards == walk.full.ancilla_bitsize + 5
    count = wk.rotation_leaf_cost({b: n for b, n in leaves.items() if not isinstance(b, (gq.opened_ZeroState, gq.opened_DiscardQ))})
    expected = wk.rotation_leaf_cost(wk.rotation_clifford_t_leaves(gq.opened_EvolutionSchedule(walk, 2, 1)))
    assert count['T'] == expected['T']
    assert set((r.name for r in op.signature)) == {'physical', 'resource', 'rotation_resource'}


def _closed_sector(q,Lx=4,Ly=4,nu=1.,hop=.2,g=1.):
    """Original TCL Hamiltonian in an exact excitation sector, cutoff >=q."""
    import itertools
    L=Lx*Ly;basis=[]
    for chosen in itertools.combinations_with_replacement(range(2*L),q):
        if any(chosen.count(j)>1 for j in chosen if j>=L):continue
        s=[0]*(2*L)
        for j in chosen:s[j]+=1
        basis.append(tuple(s))
    index={s:i for i,s in enumerate(basis)};H=np.zeros((len(basis),len(basis)))
    for col,s in enumerate(basis):
        H[col,col]=-nu*q
        for i in range(L):
            if s[L+i]:
                target=list(s);target[L+i]-=1;target[i]+=1
                row=index[tuple(target)];H[row,col]+=g*math.sqrt(s[i]+1);H[col,row]+=g*math.sqrt(s[i]+1)
            if s[i]:
                x,y=i%Lx,i//Lx
                for xx,yy in (((x+1)%Lx,y),((x-1)%Lx,y),(x,(y+1)%Ly),(x,(y-1)%Ly)):
                    j=yy*Lx+xx;target=list(s);target[i]-=1;target[j]+=1
                    H[index[tuple(target)],col]-=hop*math.sqrt(s[i]*(s[j]+1))
    return basis,H


def run_closed():
    """Independent sector matrices, vacuum dispersion, correlator sum rule and evolution.

    Original physical Hamiltonian nu=1, Delta=0, g=1, hopping=0.2, periodic
    4x4 lattice, one emitter per cavity, cutoff 2. No Qualtran cost ledger.
    """
    import itertools
    from scipy.linalg import expm
    from scipy.special import jv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    Lx=Ly=4;L=16;nu=1.;hop=.2;g=1.
    b0,H0=_closed_sector(0);b1,H1=_closed_sector(1);b2,H2=_closed_sector(2)
    E1,U1=np.linalg.eigh(H1);E2,U2=np.linalg.eigh(H2)
    analytic=[]
    for kx,ky in itertools.product(2*np.pi*np.arange(4)/4,repeat=2):
        ep=-2*hop*(np.cos(kx)+np.cos(ky))-nu;ea=-nu
        center=(ep+ea)/2;split=np.sqrt(((ep-ea)/2)**2+g*g)
        analytic.extend((center-split,center+split))
    dispersion_error=float(np.max(abs(np.sort(analytic)-E1)))
    ground=U1[:,0];Aplus=np.zeros((len(b2),len(b1)));Aminus=np.zeros((1,len(b1)))
    i2={s:i for i,s in enumerate(b2)}
    for col,s in enumerate(b1):
        target=list(s);target[0]+=1;Aplus[i2[tuple(target)],col]=math.sqrt(s[0]+1)
        if s[0]:Aminus[0,col]=math.sqrt(s[0])
    addition=abs(U2.conj().T@Aplus@ground)**2;removal=abs(Aminus@ground)**2
    sum_rule_error=float(abs(addition.sum()-removal.sum()-1))
    initial=np.zeros(len(b1));initial[next(i for i,s in enumerate(b1) if s[0]==1)]=1
    times=np.linspace(0,4,81);states=U1@((U1.conj().T@initial)[:,None]*np.exp(-1j*E1[:,None]*times))
    exact=expm(-4j*H1)@initial
    exponential_error=float(np.linalg.norm(states[:,-1]-exact))
    photon_number=np.array([sum(s[:L]) for s in b1]);emitter_number=1-photon_number
    pops=np.abs(states)**2;ph=photon_number@pops;at=emitter_number@pops
    alpha=float(np.max(abs(E1)));A=H1/alpha
    v0=initial;v1=A@initial;cheb=[v0,v1]
    for k in range(2,65):cheb.append(2*A@cheb[-1]-cheb[-2])
    convergence=[]
    for degree in (8,16,32,64):
        approx=jv(0,4*alpha)*cheb[0].astype(complex)
        for k in range(1,degree+1):approx+=2*(-1j)**k*jv(k,4*alpha)*cheb[k]
        convergence.append(dict(degree=degree,state_error=float(np.linalg.norm(approx-exact))))
    assert max(dispersion_error,sum_rule_error,exponential_error)<1e-11
    assert max(abs(ph+at-1))<1e-11
    assert convergence[-1]['state_error']<1e-11
    out=Path(__file__).resolve().parent/'results';out.mkdir(exist_ok=True)
    pd.DataFrame(dict(time=times,photon_population=ph,emitter_population=at)).to_csv(out/'closed_trajectories.csv',index=False)
    pd.DataFrame(convergence).to_csv(out/'closed_convergence.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(9,3.3),layout='constrained')
    axes[0].plot(times,ph,label='Photons');axes[0].plot(times,at,label='Emitters');axes[0].legend()
    axes[0].set(xlabel='gt',ylabel='Total excitation population',title='Closed-system evolution')
    axes[1].semilogy([r['degree'] for r in convergence],[r['state_error'] for r in convergence],'o-')
    axes[1].set(xlabel='Chebyshev degree',ylabel='Final state error',title='Polynomial vs exact evolution')
    fig.savefig(out/'closed_validation.png',dpi=160);plt.close(fig)
    result=dict(status='PASS',sector_dimensions=[len(b0),len(b1),len(b2)],
        dispersion_eigenvalue_error=dispersion_error,correlator_commutator_sum_rule_error=sum_rule_error,
        matrix_exponential_vs_eigendecomposition_error=exponential_error,
        max_excitation_conservation_error=float(max(abs(ph+at-1))),polynomial_convergence=convergence,
        scope='Small classical sector and polynomial validation; circuit tests are separate',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'closed_validation_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def reproduce_figure8():
    """Recompute Fig.8 from Eqs.17--20,119,123, without digitizing the image.

    Panel b integrates the Hann window analytically for the exact Lehmann
    poles. Panel c uses a hard-window trapezoidal rule, dt=0.1, seed=917.
    The random-phase test illustrates cancellation; it is not a shot bound.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    L=16;b0,H0=_closed_sector(0);b1,H1=_closed_sector(1);b2,H2=_closed_sector(2)
    E1,U1=np.linalg.eigh(H1);E2,U2=np.linalg.eigh(H2);E0=float(E1[0]);ground=U1[:,0]
    plus=np.zeros((len(b2),len(b1)));minus=np.zeros((1,len(b1)));index={s:i for i,s in enumerate(b2)}
    for col,s in enumerate(b1):
        target=list(s);target[0]+=1;plus[index[tuple(target)],col]=math.sqrt(s[0]+1)
        if s[0]:minus[0,col]=math.sqrt(s[0])
    weight_plus=abs(U2.conj().T@plus@ground)**2;weight_minus=float(abs(minus@ground)[0]**2)
    pole_plus=E2-E0;pole_minus=E0-float(H0[0,0])
    assert abs(weight_plus.sum()-weight_minus-1)<1e-11
    # Momentum blocks are extracted from the independently assembled matrix.
    kpath=[(0,0),(np.pi/2,0),(np.pi,0),(np.pi,np.pi/2),(np.pi,np.pi),
           (np.pi/2,np.pi/2),(0,0)]
    distances=[0,np.pi/2,np.pi,3*np.pi/2,2*np.pi,2*np.pi+np.pi/np.sqrt(2),2*np.pi+np.sqrt(2)*np.pi]
    rows=[];max_pole_error=max_weight_error=0.
    for j,(kx,ky) in enumerate(kpath):
        V=np.zeros((len(b1),2),complex)
        for col,s in enumerate(b1):
            mode=s.index(1);site=mode%L
            V[col,mode//L]=np.exp(1j*(kx*(site%4)+ky*(site//4)))/4
        block=V.conj().T@H1@V;ev,uv=np.linalg.eigh(block)
        assert np.linalg.norm(H1@V-V@block)<1e-11
        ep=-.4*(np.cos(kx)+np.cos(ky))-1.;ea=-1.
        gap=np.sqrt((ep-ea)**2+4);expected=np.array([(ep+ea-gap)/2,(ep+ea+gap)/2])
        weights=np.array([.5*(1-(ep-ea)/gap),.5*(1+(ep-ea)/gap)])
        max_pole_error=max(max_pole_error,float(max(abs(ev-expected))))
        max_weight_error=max(max_weight_error,float(max(abs(abs(uv[0])**2-weights))))
        for branch in (0,1):rows.append(dict(path_index=j,path_distance=distances[j],kx=kx,ky=ky,branch=branch,
            pole=float(ev[branch]),analytic_pole=float(expected[branch]),photon_weight=float(abs(uv[0,branch])**2)))
    omega=np.linspace(-4,1.5,1601)
    def hann_kernel(z,T):
        return T*(.5*np.sinc(z*T/np.pi)+.25*np.sinc((z+np.pi/T)*T/np.pi)+.25*np.sinc((z-np.pi/T)*T/np.pi))
    spectra={}
    for T in (20,60,200):
        spectra[T]=(hann_kernel(omega[:,None]-pole_plus[None,:],T)@weight_plus
                    -weight_minus*hann_kernel(omega-pole_minus,T))/np.pi
    # Direct time quadrature checks the analytic Hann transform, including the
    # sign/phase of the removal contribution. Selected frequencies keep memory small.
    T=60.;time=np.linspace(0,T,6001);dt=time[1]-time[0];weights=np.full(len(time),dt);weights[[0,-1]]*=.5
    cplus=weight_plus@np.exp(-1j*pole_plus[:,None]*time)
    cminus=weight_minus*np.exp(-1j*pole_minus*time)
    chosen=np.arange(0,len(omega),80)
    numerical=np.real(np.exp(1j*omega[chosen,None]*time)@(weights*(1+np.cos(np.pi*time/T))/2*(cplus-cminus)))/np.pi
    quadrature_error=float(max(abs(numerical-spectra[60][chosen])))
    assert quadrature_error<1e-7,quadrature_error
    # Both correlators receive independent complex errors of fixed magnitude.
    time_error=np.linspace(0,60,601);w=np.full(601,.1);w[[0,-1]]*=.5
    rng=np.random.default_rng(917);random_errors=np.exp(2j*np.pi*rng.random((2,len(w))))
    propagation=np.exp(1j*omega[:,None]*time_error)@((random_errors[0]-random_errors[1])*w)/np.pi
    epsilon=np.logspace(-5,-1,7);observed=epsilon*np.max(abs(propagation.real));bounds=2*epsilon*w.sum()/np.pi
    assert np.all(observed<=bounds)
    assert max_pole_error<1e-11 and max_weight_error<1e-11
    out=Path(__file__).resolve().parent/'results';out.mkdir(exist_ok=True)
    dispersion=pd.DataFrame(rows);dispersion.to_csv(out/'figure8_vacuum_dispersion.csv',index=False)
    pd.DataFrame(dict(omega=omega,**{f'spectrum_T{T}':s for T,s in spectra.items()})).to_csv(out/'figure8_spectra.csv',index=False)
    pd.DataFrame(dict(epsilon_C=epsilon,maximum_spectral_deviation=observed,deterministic_bound=bounds)).to_csv(out/'figure8_error_propagation.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(11.8,3.55),layout='constrained')
    for branch,color,marker in ((0,'#1f77b4','o'),(1,'#ff7f0e','s')):
        d=dispersion[dispersion.branch==branch]
        axes[0].plot(d.path_distance,d.analytic_pole,color=color,lw=1.2)
        axes[0].scatter(d.path_distance,d.pole,s=12+55*d.photon_weight,marker=marker,
                        facecolors='none',edgecolors=color,zorder=3)
    axes[0].set(xticks=[distances[j] for j in (0,2,4,6)],xticklabels=[r'$\Gamma$','X','M',r'$\Gamma$'],
                ylabel=r'Pole frequency $\omega/g$',title='(a) Vacuum dispersion')
    for offset,(T,color,style) in enumerate(((20,'#1f77b4','-'),(60,'#ff7f0e','--'),(200,'#9467bd',':'))):
        axes[1].plot(omega,.8*spectra[T]/max(abs(spectra[T]))+offset,color=color,ls=style,lw=1.1,label=fr'$gT_{{\max}}={T}$')
    axes[1].axvline(pole_minus,color='.4',ls='-.',lw=.9)
    axes[1].set(xlim=(-4,1.5),xlabel=r'$\omega/g$',ylabel='Scaled spectrum + offset',title='(b) Single-polariton reference')
    axes[1].legend(fontsize=7,loc='upper left',frameon=False)
    axes[2].loglog(epsilon,observed,'o-',ms=4,label='Random-phase errors')
    axes[2].loglog(epsilon,bounds,'--',color='#ff7f0e',label='Deterministic bound')
    axes[2].set(xlabel=r'$\epsilon_C$',ylabel='Max. spectral deviation',title='(c) Hard-window propagation')
    axes[2].legend(fontsize=7,loc='upper left',frameon=False)
    fig.savefig(out/'figure8_closed_system.png',dpi=190);fig.savefig(out/'figure8_closed_system.svg');plt.close(fig)
    result=dict(status='PASS',parameters=dict(Lx=4,Ly=4,N=1,n_max=2,hopping=.2,coupling=1,Delta=0,nu=1),
        reference='Lowest Q=1 eigenstate; Q=2 addition and vacuum removal',reference_energy=E0,
        addition_weight=float(weight_plus.sum()),removal_weight=weight_minus,removal_pole=pole_minus,
        nearest_addition_pole=float(pole_plus[np.argmin(abs(pole_plus-pole_minus))]),
        pole_frequency_error=max_pole_error,photon_weight_error=max_weight_error,
        commutator_sum_rule_error=float(abs(weight_plus.sum()-weight_minus-1)),hann_vs_time_quadrature_error=quadrature_error,
        windows=[20,60,200],frequency_grid_points=len(omega),error_test_seed=917,hard_window_time_points=len(w),
        deterministic_bound_over_observed=float(bounds[0]/observed[0]),
        scope='Recalculated physical Figure 8; same definitions and parameters, with explicit numerical grids. Not digitized image data or a shot-concentration certificate.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'figure8_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def test_figure8_spectral_reference():
    report=reproduce_figure8()
    assert abs(report['addition_weight']-1.0429)<5e-5
    assert abs(report['removal_weight']-.0429)<5e-5
    assert abs(report['removal_pole']+2.477033)<1e-6


def run_open():
    """Full two-cavity open benchmark retained from validate_open.py."""
    open_physics_main()
    path=open_physics_HERE/'open_validation_summary.json'
    result=json.loads(path.read_text())
    result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path.write_text(json.dumps(result,indent=2)+'\n')
    return result


@pytest.mark.parametrize('mode',['direct','paired','sparse'])
def test_closed_block_encoding_against_ladder_reference(mode):
    pair=be.model_build_pair(Lx=2,Ly=2,N=1,n_max=2,b=2,B=1,
        mu=np.zeros((2,2)),modes=('direct',mode))
    cols=[0,1,17,1024,2048,4095]
    got,diag=check_projected_columns(pair[mode]['full'],cols)
    target=check_rounded_tcl_columns(pair,cols)
    assert np.linalg.norm(got-target)<1e-10
    assert diag['catalyst_leakage']<1e-10


def run_gate_tests():
    """Run pytest and refresh the source-hashed notebook validation record."""
    import subprocess
    path=Path(__file__).resolve()
    result=subprocess.run([sys.executable,'-m','pytest','-q',str(path)],
        cwd=path.parents[1],capture_output=True,text=True)
    print(result.stdout,end='')
    if result.stderr:print(result.stderr,file=sys.stderr,end='')
    if result.returncode==0:
        record=dict(exit_code=0,pytest_output=result.stdout.strip(),
            validation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),resources_sha256=gq.source_hashes())
        (path.parent/'results/gate_validation.json').write_text(json.dumps(record,indent=2)+'\n')
    return result.returncode


if __name__=='__main__':
    import argparse,hashlib
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['closed','open','gates']);args=parser.parse_args()
    if args.action=='closed':print(json.dumps(run_closed(),indent=2))
    elif args.action=='open':run_open()
    else:raise SystemExit(run_gate_tests())
