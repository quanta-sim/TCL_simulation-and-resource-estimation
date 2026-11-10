"""Consolidated TCL block encoding implementation.

Static, readable definitions; no embedded modules or parent-bundle imports.
Internal prefixes distinguish shared definitions merged from the development code.
"""
from __future__ import annotations
from qualtran._infra.adjoint import Adjoint as enc_Adjoint
from qualtran.bloqs.mcmt.and_bloq import And as enc_And
from qualtran.bloqs.state_preparation.black_box_prepare import BlackBoxPrepare as enc_BlackBoxPrepare
from qualtran import Bloq as enc_Bloq
from qualtran import BloqBuilder as enc_BloqBuilder
from qualtran.bloqs.basic_gates import CNOT as enc_CNOT
from qualtran.bloqs.basic_gates import CSwap as enc_CSwap
from qualtran.bloqs.bookkeeping import Cast as enc_Cast
from qualtran import CompositeBloq as enc_CompositeBloq
from collections import Counter as enc_Counter
from qualtran.resource_counting import GateCounts as enc_GateCounts
from qualtran._infra.quantum_graph import LeftDangle as enc_LeftDangle
from qualtran.bloqs.reflections.prepare_identity import PrepareIdentity as enc_PrepareIdentity
from qualtran import QAny as enc_QAny
from qualtran import QBit as enc_QBit
from qualtran.resource_counting import QECGatesCost as enc_QECGatesCost
from qualtran import QUInt as enc_QUInt
from qualtran import Register as enc_Register
from qualtran._infra.quantum_graph import RightDangle as enc_RightDangle
from qualtran import Signature as enc_Signature
from qualtran.bloqs.basic_gates import Toffoli as enc_Toffoli
from qualtran.bloqs.basic_gates import XGate as enc_XGate
from qualtran.bloqs.bookkeeping._bookkeeping_bloq import _BookkeepingBloq as enc__BookkeepingBloq
from qualtran._infra.adjoint import _adjoint_final_soqs as enc__adjoint_final_soqs
from qualtran._infra.composite_bloq import _cxns_to_soq_dict as enc__cxns_to_soq_dict
from qualtran._infra.composite_bloq import _map_flat_soqs as enc__map_flat_soqs
from qualtran._infra.composite_bloq import _reg_to_soq as enc__reg_to_soq
from qualtran._infra.composite_bloq import _update_flat_soq_map as enc__update_flat_soq_map
import attrs as enc_attrs
from functools import cached_property as enc_cached_property
from attrs import frozen as enc_frozen
from qualtran.resource_counting import get_cost_value as enc_get_cost_value
from qualtran.bloqs.mcmt.specialized_ctrl import get_ctrl_system_1bit_cv as enc_get_ctrl_system_1bit_cv
from functools import lru_cache as enc_lru_cache
import numpy as enc_np
from typing import Optional as core_Optional
from typing import Sequence as core_Sequence
import math as core_math
import numpy as core_np
from qualtran.bloqs.bookkeeping import Allocate as model_Allocate
from qualtran.bloqs.block_encoding.block_encoding_base import BlockEncoding as model_BlockEncoding
from qualtran import Bloq as model_Bloq
from qualtran import BloqBuilder as model_BloqBuilder
from qualtran.bloqs.basic_gates import CZ as model_CZ
from qualtran.bloqs.bookkeeping import Free as model_Free
from qualtran.bloqs.basic_gates import Hadamard as model_Hadamard
from typing import Optional as model_Optional
from qualtran import QAny as model_QAny
from qualtran import QBit as model_QBit
from qualtran import QUInt as model_QUInt
from qualtran import Register as model_Register
from qualtran.bloqs.basic_gates import Ry as model_Ry
from qualtran.bloqs.basic_gates import SGate as model_SGate
from qualtran import Side as model_Side
from qualtran import Signature as model_Signature
from qualtran.bloqs.basic_gates import XGate as model_XGate
from qualtran.bloqs.basic_gates import ZGate as model_ZGate
import attrs as model_attrs
from functools import cached_property as model_cached_property
from attrs import frozen as model_frozen
from functools import lru_cache as model_lru_cache
import math as model_math
import numpy as model_np


# tcl_gqsp_pipeline_clean.py -- retained active definitions

def core_q_ph_from_nmax(n_max: int) -> int:
    return core_math.ceil(core_math.log2(n_max + 1))

def core_normalize_mu_2d(mu: core_np.ndarray, Lx: int, Ly: int) -> core_np.ndarray:
    arr = core_np.asarray(mu, dtype=float)
    if arr.shape == (Lx * Ly,):
        arr = arr.reshape(Ly, Lx)
    if arr.shape != (Ly, Lx):
        raise ValueError(f'mu must have shape ({Ly},{Lx}) or ({Lx * Ly},).')
    return arr

def core_normalize_g_2d(g: core_np.ndarray, Lx: int, Ly: int, N: int) -> core_np.ndarray:
    arr = core_np.asarray(g, dtype=float)
    if arr.shape == (Lx * Ly, N):
        arr = arr.reshape(Ly, Lx, N)
    if arr.shape != (Ly, Lx, N):
        raise ValueError(f'g must have shape ({Ly},{Lx},{N}) or ({Lx * Ly},{N}).')
    return arr

def core_build_tcl_local_coefficients_2d(Lx: int, Ly: int, N: int, q_ph: int, n_max: int, mu: core_np.ndarray, Delta: float, g: core_np.ndarray, beta_lobe: core_Optional[core_np.ndarray | float]=None, keep_identity_shift: bool=True):
    """Build labels and coefficients for H_loc.

    Labels:
        ('id',)
        ('ph', x, y, r)
        ('at', x, y, alpha)
        ('int', x, y, alpha)

    The interaction branch coefficient is g * beta_lobe, while the selected
    operation block-encodes B/beta_lobe.
    """
    mu = core_normalize_mu_2d(mu, Lx, Ly)
    g = core_normalize_g_2d(g, Lx, Ly, N)
    photon_dim = 2 ** q_ph
    if n_max < 1 or n_max >= photon_dim:
        raise ValueError(f'n_max must satisfy 1 <= n_max < 2**q_ph; received n_max={n_max}, q_ph={q_ph}.')
    required_beta = core_math.sqrt(n_max)
    if beta_lobe is None:
        beta_arr = required_beta * core_np.ones((Ly, Lx, N), dtype=float)
    elif core_np.isscalar(beta_lobe):
        beta_arr = float(beta_lobe) * core_np.ones((Ly, Lx, N), dtype=float)
    else:
        beta_arr = core_np.asarray(beta_lobe, dtype=float)
        if beta_arr.shape == (Lx * Ly, N):
            beta_arr = beta_arr.reshape(Ly, Lx, N)
        if beta_arr.shape != (Ly, Lx, N):
            raise ValueError(f'beta_lobe must have shape ({Ly},{Lx},{N}) or ({Lx * Ly},{N}).')
    if not core_np.allclose(beta_arr, required_beta, atol=1e-12, rtol=0.0):
        raise ValueError('The current physical LOBE circuit uses beta=sqrt(n_max). Set beta_lobe=None or supply beta_lobe=sqrt(n_max).')
    labels: list[tuple] = []
    coeffs: list[float] = []
    C_id = -0.5 * sum((mu[y, x] * (photon_dim - 1) + N * (mu[y, x] + Delta) for y in range(Ly) for x in range(Lx)))
    if keep_identity_shift:
        labels.append(('id',))
        coeffs.append(C_id)
    for y in range(Ly):
        for x in range(Lx):
            for r in range(q_ph):
                labels.append(('ph', x, y, r))
                coeffs.append(mu[y, x] * 2 ** r / 2.0)
    for y in range(Ly):
        for x in range(Lx):
            for alpha in range(N):
                labels.append(('at', x, y, alpha))
                coeffs.append((mu[y, x] + Delta) / 2.0)
    for y in range(Ly):
        for x in range(Lx):
            for alpha in range(N):
                labels.append(('int', x, y, alpha))
                coeffs.append(g[y, x, alpha] * beta_arr[y, x, alpha])
    return (labels, core_np.asarray(coeffs, dtype=float), C_id, beta_arr)

def core_integerize_probabilities(probs: core_Sequence[float], total_integer: int) -> core_np.ndarray:
    probs = core_np.asarray(probs, dtype=float)
    probs = probs / core_np.sum(probs)
    raw = probs * total_integer
    ints = core_np.floor(raw).astype(int)
    residuals = raw - ints
    remaining = int(total_integer - core_np.sum(ints))
    if remaining > 0:
        order = core_np.argsort(-residuals)
        ints[order[:remaining]] += 1
    elif remaining < 0:
        order = core_np.argsort(residuals)
        for j in order[:-remaining]:
            if ints[j] > 0:
                ints[j] -= 1
    if int(core_np.sum(ints)) != int(total_integer):
        raise RuntimeError('Integer probability rounding failed.')
    return ints

def core_build_integer_alias_table_from_weights(integer_weights: core_Sequence[int], mu_bits: int):
    work = core_np.asarray(integer_weights, dtype=int).copy()
    M = len(work)
    D = 2 ** mu_bits
    if int(core_np.sum(work)) != M * D:
        raise ValueError('The integer weights must sum to M*2**mu_bits.')
    keep = core_np.zeros(M, dtype=int)
    alt = core_np.arange(M, dtype=int)
    small = [i for i in range(M) if work[i] < D]
    large = [i for i in range(M) if work[i] > D]
    while small and large:
        s = small.pop()
        l = large.pop()
        keep[s] = work[s]
        alt[s] = l
        work[l] -= D - keep[s]
        if work[l] < D:
            small.append(l)
        elif work[l] > D:
            large.append(l)
        else:
            keep[l] = D
            alt[l] = l
    for i in small + large:
        keep[i] = D
        alt[i] = i
    for i in range(M):
        if keep[i] == 0 and integer_weights[i] == D:
            keep[i] = D
            alt[i] = i
    return (keep, alt)

def core_probabilities_from_integer_alias_table(keep: core_Sequence[int], alt: core_Sequence[int], mu_bits: int) -> core_np.ndarray:
    keep = core_np.asarray(keep, dtype=int)
    alt = core_np.asarray(alt, dtype=int)
    M = len(keep)
    D = 2 ** mu_bits
    probs = core_np.zeros(M, dtype=float)
    for x in range(M):
        probs[x] += keep[x] / (M * D)
        probs[alt[x]] += (D - keep[x]) / (M * D)
    return probs


# tcl_direct_register.py -- retained active definitions

def enc__split(bb, x, n):
    return list(bb.split(x))

def enc__join(bb, bits, dtype=None):
    return bb.join(enc_np.asarray(bits, dtype=object), dtype=dtype or enc_QAny(len(bits)))

def enc__cx(bb, a, b):
    o = bb.add_d(enc_CNOT(), ctrl=a, target=b)
    return (o['ctrl'], o['target'])

def enc__ccx(bb, a, b, c):
    o = bb.add_d(enc_Toffoli(), ctrl=enc_np.asarray([a, b], dtype=object), target=c)
    return (*list(o['ctrl']), o['target'])

def enc__inc(bb, enable, bits, decrement=False):
    """Appendix B2: MSB-first wires, modular arithmetic, clean prefix erasure."""
    x = list(reversed(bits))
    q = len(x)
    carry = [bb.allocate(dtype=enc_QBit()) for _ in range(max(0, q - 2))]
    flags = [enable, *carry]
    for k in range(1, q - 1):
        if decrement:
            x[k - 1] = bb.add(enc_XGate(), q=x[k - 1])
        flags[k - 1], x[k - 1], flags[k] = enc__ccx(bb, flags[k - 1], x[k - 1], flags[k])
        if decrement:
            x[k - 1] = bb.add(enc_XGate(), q=x[k - 1])
    if q >= 2:
        if decrement:
            x[q - 2] = bb.add(enc_XGate(), q=x[q - 2])
        flags[q - 2], x[q - 2], x[q - 1] = enc__ccx(bb, flags[q - 2], x[q - 2], x[q - 1])
        if decrement:
            x[q - 2] = bb.add(enc_XGate(), q=x[q - 2])
    for k in reversed(range(1, q - 1)):
        flags[k], x[k] = enc__cx(bb, flags[k], x[k])
        if decrement:
            x[k - 1] = bb.add(enc_XGate(), q=x[k - 1])
        flags[k - 1], x[k - 1], flags[k] = enc__ccx(bb, flags[k - 1], x[k - 1], flags[k])
        if decrement:
            x[k - 1] = bb.add(enc_XGate(), q=x[k - 1])
    flags[0], x[0] = enc__cx(bb, flags[0], x[0])
    for c in flags[1:]:
        bb.free(c)
    return (flags[0], list(reversed(x)))

def enc__unary(bb, address, leaf, enable=None):
    """Appendix B1 traversal; callback updates a one-element leaf-flag list.

    Without enable the root is a classical 1: its children are the address bit
    and its complement, so no root CCXs or root workspace are needed.
    """
    a = list(address)

    def visit(depth, flag, offset):
        if depth == len(a):
            box = [flag]
            leaf(offset, box)
            return box[0]
        f = bb.allocate(dtype=enc_QBit())
        a[depth] = bb.add(enc_XGate(), q=a[depth])
        flag, a[depth], f = enc__ccx(bb, flag, a[depth], f)
        a[depth] = bb.add(enc_XGate(), q=a[depth])
        f = visit(depth + 1, f, offset)
        flag, f = enc__cx(bb, flag, f)
        f = visit(depth + 1, f, offset + 2 ** (len(a) - depth - 1))
        flag, a[depth], f = enc__ccx(bb, flag, a[depth], f)
        bb.free(f)
        return flag
    if enable is None:
        if not a:
            raise ValueError('At least two table entries are required.')
        a[0] = bb.add(enc_XGate(), q=a[0])
        a[0] = visit(1, a[0], 0)
        a[0] = bb.add(enc_XGate(), q=a[0])
        a[0] = visit(1, a[0], 2 ** (len(a) - 1))
    else:
        enable = visit(0, enable, 0)
    return (a, enable)

@enc_lru_cache(None)
def enc_build_selected_shift(L: int, q_ph: int, decrement=False):
    """Enabled in-place shift at a coherent binary site address."""
    if L < 2 or L & L - 1 or q_ph < 1:
        raise ValueError('L must be a power of two >=2 and q_ph >=1.')
    bb = enc_BloqBuilder()
    e = bb.add_register('enable', enc_QBit())
    a = enc__split(bb, bb.add_register('address', enc_QUInt(L.bit_length() - 1)), L.bit_length() - 1)
    data = enc__split(bb, bb.add_register('data', enc_QAny(L * q_ph)), L * q_ph)

    def leaf(site, flag):
        flag[0], data[site * q_ph:(site + 1) * q_ph] = enc__inc(bb, flag[0], data[site * q_ph:(site + 1) * q_ph], decrement)
    a, e = enc__unary(bb, a, leaf, e)
    return bb.finalize(enable=e, address=enc__join(bb, a, enc_QUInt(len(a))), data=enc__join(bb, data))

@enc_lru_cache(None)
def enc_build_indexed_load(L: int, q_ph: int):
    """One coherent unary traversal reads a whole quantum occupation word."""
    if L < 2 or L & L - 1 or q_ph < 1:
        raise ValueError('L must be a power of two >=2 and q_ph >=1.')
    bb = enc_BloqBuilder()
    a = enc__split(bb, bb.add_register('address', enc_QUInt(L.bit_length() - 1)), L.bit_length() - 1)
    data = enc__split(bb, bb.add_register('data', enc_QAny(L * q_ph)), L * q_ph)
    word = enc__split(bb, bb.add_register('word', enc_QAny(q_ph)), q_ph)

    def leaf(site, flag):
        for k in range(q_ph):
            flag[0], data[site * q_ph + k], word[k] = enc__ccx(bb, flag[0], data[site * q_ph + k], word[k])
    a, _ = enc__unary(bb, a, leaf)
    return bb.finalize(address=enc__join(bb, a, enc_QUInt(len(a))), data=enc__join(bb, data), word=enc__join(bb, word))

@enc_lru_cache(None)
def enc_build_destination(Lx: int, Ly: int):
    """XOR destination i=(x+dx mod Lx,y+dy mod Ly) into a clean word.

    The specified initialized action requires destination=0; exact gate reversal
    erases it.  Coordinate bit order is [y(MSB..LSB), x(MSB..LSB)].
    """
    if min(Lx, Ly) < 2 or any((v & v - 1 for v in (Lx, Ly))):
        raise ValueError('Power-of-two Lx,Ly >=2 required.')
    nx, ny = (Lx.bit_length() - 1, Ly.bit_length() - 1)
    bb = enc_BloqBuilder()
    src = enc__split(bb, bb.add_register('source', enc_QUInt(nx + ny)), nx + ny)
    axis = bb.add_register('axis', enc_QBit())
    sign = bb.add_register('sign', enc_QBit())
    dst = enc__split(bb, bb.add_register('destination', enc_QUInt(nx + ny)), nx + ny)
    for k in range(nx + ny):
        src[k], dst[k] = enc__cx(bb, src[k], dst[k])
    for k in range(nx + ny):
        sign, dst[k] = enc__cx(bb, sign, dst[k])
    axis = bb.add(enc_XGate(), q=axis)
    axis, dst[ny:] = enc__inc(bb, axis, dst[ny:])
    axis = bb.add(enc_XGate(), q=axis)
    axis, dst[:ny] = enc__inc(bb, axis, dst[:ny])
    for k in range(nx + ny):
        sign, dst[k] = enc__cx(bb, sign, dst[k])
    return bb.finalize(source=enc__join(bb, src, enc_QUInt(nx + ny)), axis=axis, sign=sign, destination=enc__join(bb, dst, enc_QUInt(nx + ny)))

@enc_lru_cache(None)
def enc_build_classical_lookup(words: tuple, width: int):
    """Classical-word QROM with explicit coherent unary flags and CNOT writes."""
    D = len(words)
    if D < 2 or D & D - 1:
        raise ValueError('Table length must be a power of two >=2.')
    if any((int(v) < 0 or int(v) >= 2 ** width for v in words)):
        raise ValueError('Table word out of range.')
    bb = enc_BloqBuilder()
    a = enc__split(bb, bb.add_register('address', enc_QUInt(D.bit_length() - 1)), D.bit_length() - 1)
    t = enc__split(bb, bb.add_register('target', enc_QAny(width)), width)

    def leaf(j, flag):
        for k in range(width):
            if int(words[j]) >> width - k - 1 & 1:
                flag[0], t[k] = enc__cx(bb, flag[0], t[k])
    a, _ = enc__unary(bb, a, leaf)
    return bb.finalize(address=enc__join(bb, a, enc_QUInt(len(a))), target=enc__join(bb, t))

@enc_lru_cache(None)
def enc_build_row_map_update(L: int, q_ph: int):
    """Manuscript G10: separately enabled F, R, SWAP on a clean second bank.

    On y=0 maps |x,0> to |Sx,0>; e=0 is identity on arbitrary x,y.
    These are initialized-subspace unitary row-map extensions, not XOR oracles
    with a promised standard action on every nonzero y.
    """
    bb = enc_BloqBuilder()
    e = bb.add_register('enable', enc_QBit())
    j = bb.add_register('source', enc_QUInt(L.bit_length() - 1))
    i = bb.add_register('destination', enc_QUInt(L.bit_length() - 1))
    x = bb.add_register('data', enc_QAny(L * q_ph))
    y = bb.add_register('auxiliary', enc_QAny(L * q_ph))

    def bank_copy(a, b):
        nonlocal e
        aa, bbits = (enc__split(bb, a, L * q_ph), enc__split(bb, b, L * q_ph))
        for k in range(L * q_ph):
            e, aa[k], bbits[k] = enc__ccx(bb, e, aa[k], bbits[k])
        return (enc__join(bb, aa), enc__join(bb, bbits))

    def shift_pair(data):
        nonlocal e, j, i
        for key, dec in [('source', True), ('destination', False)]:
            o = bb.add_d(enc_build_selected_shift(L, q_ph, dec), enable=e, address=j if key == 'source' else i, data=data)
            e, data = (o['enable'], o['data'])
            if key == 'source':
                j = o['address']
            else:
                i = o['address']
        return data
    x, y = bank_copy(x, y)
    y = shift_pair(y)
    x = shift_pair(x)
    y, x = bank_copy(y, x)
    o = bb.add_d(enc_CSwap(L * q_ph), ctrl=e, x=x, y=y)
    return bb.finalize(enable=o['ctrl'], source=j, destination=i, data=o['x'], auxiliary=o['y'])

def enc__is_terminal(b):
    return isinstance(b, enc__BookkeepingBloq) or (type(b).__module__.startswith('qualtran.bloqs.basic_gates') and (not isinstance(b, enc_CSwap)) and (type(b).__name__ not in ('OnEach', 'Power', 'IntState', 'IntEffect', 'TwoBitCSwap')))

def enc__fast_adjoint(cb):
    """Qualtran's exact gate reversal with an O(1) wire-remapping dictionary.

    Version 0.7's list-based _map_soqs repeatedly scans all prior wires and is
    quadratic for large unary traversals.  This changes no gates or scheduling.
    """
    signature = cb.signature.adjoint()
    mapping = {}
    enc__update_flat_soq_map(zip([enc__reg_to_soq(enc_RightDangle, r) for r in cb.signature.rights()], [enc__reg_to_soq(enc_LeftDangle, r) for r in signature.lefts()]), mapping)
    bb, _ = enc_BloqBuilder.from_signature(signature)
    for inst, preds, succs in reversed(list(cb.iter_bloqnections())):
        ins = enc__cxns_to_soq_dict(inst.bloq.signature.rights(), succs, get_me=lambda x: x.left, get_assign=lambda x: x.right)
        ins = enc__map_flat_soqs(ins, mapping)
        old_out = tuple((enc__reg_to_soq(inst, r) for r in inst.bloq.signature.lefts()))
        new_out = bb.add_t(inst.bloq.adjoint(), **ins)
        enc__update_flat_soq_map(zip(old_out, new_out), mapping)
    return bb.finalize(**enc__map_flat_soqs(enc__adjoint_final_soqs(cb, signature), mapping))

@enc_frozen
class enc_Coherent(enc_Bloq):
    """An executable coherent decomposition, never a resource-count override."""
    inner: enc_Bloq

    @property
    def signature(self):
        return self.inner.signature

    @enc_cached_property
    def body(self):
        b = self.inner
        if type(b).__name__ == 'TwoBitCSwap':
            bb, s = enc_BloqBuilder.from_signature(b.signature)
            c, x, y = (s['ctrl'], s['x'], s['y'])
            y, x = enc__cx(bb, y, x)
            c, x, y = enc__ccx(bb, c, x, y)
            y, x = enc__cx(bb, y, x)
            return bb.finalize(ctrl=c, x=x, y=y)
        if isinstance(b, enc_And):
            bb, soqs = enc_BloqBuilder.from_signature(b.signature)
            c = list(soqs['ctrl'])
            t = soqs['target'] if b.uncompute else bb.allocate(dtype=enc_QBit())
            for k, cv in enumerate((b.cv1, b.cv2)):
                if cv == 0:
                    c[k] = bb.add(enc_XGate(), q=c[k])
            c[0], c[1], t = enc__ccx(bb, c[0], c[1], t)
            for k, cv in enumerate((b.cv1, b.cv2)):
                if cv == 0:
                    c[k] = bb.add(enc_XGate(), q=c[k])
            out = {'ctrl': enc_np.asarray(c, dtype=object)}
            if b.uncompute:
                bb.free(t)
            else:
                out['target'] = t
            return bb.finalize(**out)
        if isinstance(b, enc_Adjoint):
            return enc__fast_adjoint(enc_coherent(b.subbloq).decompose_bloq())
        cb = b if isinstance(b, enc_CompositeBloq) else b.decompose_bloq()
        bb, _ = enc_BloqBuilder.from_signature(cb.signature)
        mapping = {}
        for inst, ins, old_out in cb.iter_bloqsoqs():
            ins = enc__map_flat_soqs(ins, mapping)
            new_out = bb.add_t(enc_coherent(inst.bloq), **ins)
            enc__update_flat_soq_map(zip(old_out, new_out), mapping)
        return bb.finalize(**enc__map_flat_soqs(cb.final_soqs(), mapping))

    def build_composite_bloq(self, bb, **soqs):
        return bb.add_d(self.body, **soqs)

    def build_call_graph(self, ssa):
        return {self.body: 1}

    def adjoint(self):
        return enc_CoherentAdjoint(self)

@enc_frozen
class enc_CoherentAdjoint(enc_Bloq):
    forward: enc_Coherent

    @property
    def signature(self):
        return self.forward.signature.adjoint()

    @enc_cached_property
    def body(self):
        return enc__fast_adjoint(self.forward.body)

    def build_composite_bloq(self, bb, **soqs):
        return bb.add_d(self.body, **soqs)

    def adjoint(self):
        return self.forward

@enc_lru_cache(None)
def enc_coherent(b):
    return b if enc__is_terminal(b) or isinstance(b, (enc_Coherent, enc_CoherentAdjoint)) else enc_Coherent(b)

class enc__ZeroSignal:

    @property
    def signal_state(self):
        return enc_BlackBoxPrepare(enc_PrepareIdentity.from_bitsizes([self.ancilla_bitsize]))

    def get_ctrl_system(self, ctrl_spec):
        return enc_get_ctrl_system_1bit_cv(self, ctrl_spec=ctrl_spec, current_ctrl_bit=self.control_val, get_ctrl_bloq_and_ctrl_reg_name=lambda cv: (enc_attrs.evolve(self, control_val=int(cv)), 'ctrl'))

    @property
    def signature(self):
        regs = [enc_Register('system', enc_QAny(self.system_bitsize)), enc_Register('ancilla', enc_QAny(self.ancilla_bitsize))]
        if self.resource_bitsize:
            regs.append(enc_Register('resource', enc_QAny(self.resource_bitsize)))
        if self.control_val is not None:
            regs.insert(0, enc_Register('ctrl', enc_QBit()))
        return enc_Signature(regs)

@enc_lru_cache(None)
def enc_compiled_leaf_counts(bloq):
    """Visit actual decompositions, aggregating repeated compiled subcircuits."""
    b = enc_coherent(bloq)
    if enc__is_terminal(b):
        return enc_Counter({b: 1})
    cb = b.body if isinstance(b, (enc_Coherent, enc_CoherentAdjoint)) else b.decompose_bloq()
    counts = enc_Counter()
    for child, multiplicity in enc_Counter((inst.bloq for inst in cb.bloq_instances)).items():
        for leaf, n in enc_compiled_leaf_counts(child).items():
            counts[leaf] += n * multiplicity
    return counts

@enc_lru_cache(None)
def enc_compiled_peak_qubits(bloq):
    """Peak of a concrete recursive topological schedule, with free-wire reuse."""
    b = enc_coherent(bloq)
    initial = sum((int(r.total_bits()) for r in b.signature.lefts()))
    if enc__is_terminal(b):
        return max(initial, sum((int(r.total_bits()) for r in b.signature.rights())))
    cb = b.body if isinstance(b, (enc_Coherent, enc_CoherentAdjoint)) else b.decompose_bloq()
    live = peak = initial
    for inst, _, _ in cb.iter_bloqsoqs():
        child = inst.bloq
        inputs = sum((int(r.total_bits()) for r in child.signature.lefts()))
        outputs = sum((int(r.total_bits()) for r in child.signature.rights()))
        peak = max(peak, live - inputs + enc_compiled_peak_qubits(child))
        live += outputs - inputs
    if live != sum((int(r.total_bits()) for r in cb.signature.rights())):
        raise AssertionError('Unbalanced compiled allocation schedule.')
    return peak

def enc_circuit_resources(bloq, *, ancilla_bits=0, catalyst_bits=0, extra_photon_bits=0):
    leaves = enc_compiled_leaf_counts(bloq)
    total = None
    for leaf, n in leaves.items():
        cost = enc_get_cost_value(leaf, enc_QECGatesCost()) * n
        total = cost if total is None else total + cost
    if total is None:
        total = enc_GateCounts()
    if total.and_bloq or total.measurement:
        raise AssertionError('Measurement-assisted primitive survived coherent compilation.')
    peak = enc_compiled_peak_qubits(bloq)
    interface = sum((int(r.total_bits()) for r in bloq.signature.lefts()))
    return dict(coherent_ccx=int(total.toffoli), direct_T=int(total.t), T_with_7T_CCX=int(total.t + 7 * total.toffoli), arbitrary_rotations=int(total.rotation), clifford=int(total.clifford), temporary_AND=0, measurement=0, block_ancilla_bits=int(ancilla_bits), reusable_catalyst_bits=int(catalyst_bits), clean_workspace_peak=peak - interface, peak_logical_qubits=peak, additional_photon_register_bits=int(extra_photon_bits), compiled_leaf_instances=sum(leaves.values()), compiled_leaf_types=len(leaves), count_status='recursive executable decomposition; rotations unsynthesized')


# tcl_manuscript_explicit.py -- retained active definitions

def model__cz(bb, a, b):
    o = bb.add_d(model_CZ(), q1=a, q2=b)
    return (o['q1'], o['q2'])

def model__fredkin(bb, c, a, b):
    a, b = enc__cx(bb, a, b)
    c, b, a = enc__ccx(bb, c, b, a)
    a, b = enc__cx(bb, a, b)
    return (c, a, b)

@model_lru_cache(None)
def model_comparator(B):
    """Clean lexicographic >= comparator; adapted from their gate trace.

    Registers are MSB first; the arithmetic loop uses LSB first lists.
    All equality/XOR scratch is coherently erased before returning v.
    """
    if B < 1:
        raise ValueError('B must be positive.')
    bb = model_BloqBuilder()
    r = list(reversed(enc__split(bb, bb.add_register('random', model_QAny(B)), B)))
    t = list(reversed(enc__split(bb, bb.add_register('threshold', model_QAny(B)), B)))
    v = bb.add_register('flag', model_QBit())
    d = [bb.allocate(dtype=model_QBit()) for _ in range(B)]
    eq = [bb.allocate(dtype=model_QBit()) for _ in range(B)]
    g = bb.allocate(dtype=model_QBit())
    for i in range(B):
        r[i], d[i] = enc__cx(bb, r[i], d[i])
        t[i], d[i] = enc__cx(bb, t[i], d[i])
        d[i] = bb.add(model_XGate(), q=d[i])
    d[-1], eq[-1] = enc__cx(bb, d[-1], eq[-1])
    for i in range(B - 2, -1, -1):
        eq[i + 1], d[i], eq[i] = enc__ccx(bb, eq[i + 1], d[i], eq[i])
    t[-1] = bb.add(model_XGate(), q=t[-1])
    r[-1], t[-1], v = enc__ccx(bb, r[-1], t[-1], v)
    t[-1] = bb.add(model_XGate(), q=t[-1])
    for i in range(B - 2, -1, -1):
        eq[i + 1], r[i], g = enc__ccx(bb, eq[i + 1], r[i], g)
        t[i] = bb.add(model_XGate(), q=t[i])
        g, t[i], v = enc__ccx(bb, g, t[i], v)
        t[i] = bb.add(model_XGate(), q=t[i])
        eq[i + 1], r[i], g = enc__ccx(bb, eq[i + 1], r[i], g)
    eq[0], v = enc__cx(bb, eq[0], v)
    for i in range(B - 1):
        eq[i + 1], d[i], eq[i] = enc__ccx(bb, eq[i + 1], d[i], eq[i])
    d[-1], eq[-1] = enc__cx(bb, d[-1], eq[-1])
    for i in range(B):
        d[i] = bb.add(model_XGate(), q=d[i])
        t[i], d[i] = enc__cx(bb, t[i], d[i])
        r[i], d[i] = enc__cx(bb, r[i], d[i])
    for w in d + eq + [g]:
        bb.free(w)
    return bb.finalize(random=enc__join(bb, list(reversed(r))), threshold=enc__join(bb, list(reversed(t))), flag=v)

@model_frozen(cache_hash=True)
class model_AliasPreparation(model_Bloq):
    """H, classical QROM, >= comparator, Fredkins, with retained alias outputs.

    Full-threshold columns have alternative=column and stored threshold=0.
    The resulting always-swap exchanges identical values, so T=2**B needs
    no extra threshold bit. No uniform-table optimization is performed.
    """
    weights: tuple
    B: int

    def __attrs_post_init__(self):
        if len(self.weights) < 2 or len(self.weights) & len(self.weights) - 1:
            raise ValueError('Power-of-two table required.')
        if self.B < 1 or min(self.weights) < 0 or sum(self.weights) <= 0:
            raise ValueError('Invalid weights or precision.')

    @property
    def d(self):
        return len(self.weights).bit_length() - 1

    @property
    def width(self):
        return 2 * self.d + 2 * self.B + 1

    @property
    def signature(self):
        return model_Signature.build(selection=self.d, random=self.B, alternative=self.d, threshold=self.B, flag=1)

    @model_cached_property
    def data(self):
        probs = model_np.asarray(self.weights) / sum(self.weights)
        iw = core_integerize_probabilities(probs, len(probs) * 2 ** self.B)
        keep, alt = core_build_integer_alias_table_from_weights(iw, self.B)
        alt = model_np.where(keep == 2 ** self.B, model_np.arange(len(probs)), alt)
        threshold = model_np.where(keep == 2 ** self.B, 0, keep)
        approx = core_probabilities_from_integer_alias_table(keep, alt, self.B)
        return dict(keep=keep, alt=alt, threshold=threshold, approx_probs=approx, exact_probs=probs, l1_prob_error=float(model_np.sum(abs(approx - probs))))

    def build_composite_bloq(self, bb, selection, random, alternative, threshold, flag):
        col = enc__split(bb, selection, self.d)
        rnd = enc__split(bb, random, self.B)
        col = [bb.add(model_Hadamard(), q=w) for w in col]
        rnd = [bb.add(model_Hadamard(), q=w) for w in rnd]
        words = tuple((int(a) << self.B | int(t) for a, t in zip(self.data['alt'], self.data['threshold'])))
        target = enc__join(bb, [*enc__split(bb, alternative, self.d), *enc__split(bb, threshold, self.B)])
        o = bb.add_d(enc_build_classical_lookup(words, self.d + self.B), address=enc__join(bb, col, model_QUInt(self.d)), target=target)
        col = enc__split(bb, o['address'], self.d)
        out = enc__split(bb, o['target'], self.d + self.B)
        alt, thr = (out[:self.d], out[self.d:])
        o = bb.add_d(model_comparator(self.B), random=enc__join(bb, rnd), threshold=enc__join(bb, thr), flag=flag)
        rnd, thr, flag = (enc__split(bb, o['random'], self.B), enc__split(bb, o['threshold'], self.B), o['flag'])
        for i in range(self.d):
            flag, col[i], alt[i] = model__fredkin(bb, flag, col[i], alt[i])
        return dict(selection=enc__join(bb, col), random=enc__join(bb, rnd), alternative=enc__join(bb, alt), threshold=enc__join(bb, thr), flag=flag)

@model_lru_cache(None)
def model_gradient_add(b):
    """Catalyst-aware addition using coherent carries and top-bit CZs.

    The highest catalyst bit is unchanged; its |-> eigenvalue supplies the
    top carry phase. This is an explicit Clifford+CCX circuit, including b=1.
    """
    if b < 1:
        raise ValueError('b must be positive.')
    bb = model_BloqBuilder()
    a = list(reversed(enc__split(bb, bb.add_register('angle', model_QAny(b)), b)))
    z = list(reversed(enc__split(bb, bb.add_register('phase_grad', model_QAny(b + 1)), b + 1)))
    c = [None] + [bb.allocate(dtype=model_QBit()) for _ in range(b - 1)]

    def carry(i):
        if i:
            c[i], a[i] = enc__cx(bb, c[i], a[i])
            c[i], z[i] = enc__cx(bb, c[i], z[i])
        a[i], z[i], c[i + 1] = enc__ccx(bb, a[i], z[i], c[i + 1])
        if i:
            c[i], c[i + 1] = enc__cx(bb, c[i], c[i + 1])
            c[i], z[i] = enc__cx(bb, c[i], z[i])
            c[i], a[i] = enc__cx(bb, c[i], a[i])
    for i in range(b - 1):
        carry(i)
    a[b - 1], z[b - 1] = model__cz(bb, a[b - 1], z[b - 1])
    if b > 1:
        a[b - 1], c[b - 1] = model__cz(bb, a[b - 1], c[b - 1])
        z[b - 1], c[b - 1] = model__cz(bb, z[b - 1], c[b - 1])
    for i in reversed(range(b)):
        a[i], z[i] = enc__cx(bb, a[i], z[i])
        if i:
            c[i], z[i] = enc__cx(bb, c[i], z[i])
            carry(i - 1)
    for w in c[1:]:
        bb.free(w)
    return bb.finalize(angle=enc__join(bb, list(reversed(a))), phase_grad=enc__join(bb, list(reversed(z))))

@model_lru_cache(None)
def model_angle_oracle(table: tuple, b: int):
    bb = model_BloqBuilder()
    e = bb.add_register('enable', model_QBit())
    n = bb.add_register('occupation', model_QAny(len(table).bit_length() - 1))
    sig = bb.add_register('signal', model_QBit())
    G = bb.add_register('phase_grad', model_QAny(b + 1))
    angle = bb.allocate(dtype=model_QAny(b))
    lookup = enc_build_classical_lookup(table, b)
    n = bb.add(enc_Cast(model_QAny(len(table).bit_length() - 1), model_QUInt(len(table).bit_length() - 1)), reg=n)
    o = bb.add_d(lookup, address=n, target=angle)
    n, angle = (o['address'], o['target'])
    a = enc__split(bb, angle, b)
    mask = [bb.allocate(dtype=model_QBit()) for _ in range(b)]
    for i in range(b):
        e, a[i], mask[i] = enc__ccx(bb, e, a[i], mask[i])
    sig = bb.add(model_SGate().adjoint(), q=sig)
    sig = bb.add(model_Hadamard(), q=sig)
    z = enc__split(bb, G, b + 1)
    for i in range(b + 1):
        sig, z[i] = enc__cx(bb, sig, z[i])
    o = bb.add_d(model_gradient_add(b), angle=enc__join(bb, mask), phase_grad=enc__join(bb, z))
    masked, G = (o['angle'], o['phase_grad'])
    z = enc__split(bb, G, b + 1)
    for i in range(b + 1):
        sig, z[i] = enc__cx(bb, sig, z[i])
    G = enc__join(bb, z)
    sig = bb.add(model_Hadamard(), q=sig)
    sig = bb.add(model_SGate(), q=sig)
    mask = enc__split(bb, masked, b)
    for i in reversed(range(b)):
        e, a[i], mask[i] = enc__ccx(bb, e, a[i], mask[i])
    for w in mask:
        bb.free(w)
    o = bb.add_d(lookup.adjoint(), address=n, target=enc__join(bb, a))
    n, angle = (o['address'], o['target'])
    bb.free(angle)
    n = bb.add(enc_Cast(model_QUInt(len(table).bit_length() - 1), model_QAny(len(table).bit_length() - 1)), reg=n)
    return bb.finalize(enable=e, occupation=n, signal=sig, phase_grad=G)

def model_factor_tables(n_max, b):
    if b < 1:
        raise ValueError('b must be positive.')
    R = 2 ** core_q_ph_from_nmax(n_max)
    A = [model_math.sqrt(n / n_max) if n <= n_max else 0 for n in range(R)]
    B = [model_math.sqrt((n + 1) / n_max) if n < n_max else 0 for n in range(R)]
    return tuple((tuple((int(round(2 * model_math.acos(v) * 2 ** b / (2 * model_math.pi))) for v in tab)) for tab in (A, B)))

@model_lru_cache(None)
def model_emitter_read(E):
    """Route/copy/unroute quantum data with the manuscript Fredkin tree."""
    if E < 2 or E & E - 1:
        raise ValueError('Power-of-two E>=2 required.')
    d = E.bit_length() - 1
    bb = model_BloqBuilder()
    a = list(reversed(enc__split(bb, bb.add_register('address', model_QAny(d)), d)))
    data = enc__split(bb, bb.add_register('data', model_QAny(E)), E)
    sample = bb.add_register('sample', model_QBit())
    for level in range(d):
        step = 1 << level
        for block in range(0, E, 2 * step):
            a[level], data[block], data[block + step] = model__fredkin(bb, a[level], data[block], data[block + step])
    data[0], sample = enc__cx(bb, data[0], sample)
    for level in reversed(range(d)):
        step = 1 << level
        for block in range(0, E, 2 * step):
            a[level], data[block], data[block + step] = model__fredkin(bb, a[level], data[block], data[block + step])
    return bb.finalize(address=enc__join(bb, list(reversed(a))), data=enc__join(bb, data), sample=sample)

@model_lru_cache(None)
def model_emitter_flip(E):
    bb = model_BloqBuilder()
    e = bb.add_register('enable', model_QBit())
    a = enc__split(bb, bb.add_register('address', model_QAny(E.bit_length() - 1)), E.bit_length() - 1)
    data = enc__split(bb, bb.add_register('data', model_QAny(E)), E)

    def leaf(j, flag):
        flag[0], data[j] = enc__cx(bb, flag[0], data[j])
    a, e = enc__unary(bb, a, leaf, e)
    return bb.finalize(enable=e, address=enc__join(bb, a), data=enc__join(bb, data))

@model_lru_cache(None)
def model_build_paired_update(L, q):
    """Manuscript G paired variant: CX(x,y), C_e(S_x S_y), CX(x,y).

    On y=0, the initialized action is (x,0) -> (Sx,0). At e=0 the
    unconditional CNOT banks cancel on arbitrary x,y. This circuit does not
    implement separately enabled forward/reverse row oracles.
    """
    bb = model_BloqBuilder()
    e = bb.add_register('enable', model_QBit())
    j = bb.add_register('source', model_QUInt(L.bit_length() - 1))
    i = bb.add_register('destination', model_QUInt(L.bit_length() - 1))
    x = bb.add_register('data', model_QAny(L * q))
    y = bb.add_register('auxiliary', model_QAny(L * q))

    def copy_bank(x, y):
        xx, yy = (enc__split(bb, x, L * q), enc__split(bb, y, L * q))
        for k in range(L * q):
            xx[k], yy[k] = enc__cx(bb, xx[k], yy[k])
        return (enc__join(bb, xx), enc__join(bb, yy))

    def shift_bank(data):
        nonlocal e, j, i
        for source, dec in ((True, True), (False, False)):
            o = bb.add_d(enc_build_selected_shift(L, q, dec), enable=e, address=j if source else i, data=data)
            e, data = (o['enable'], o['data'])
            if source:
                j = o['address']
            else:
                i = o['address']
        return data
    x, y = copy_bank(x, y)
    x = shift_bank(x)
    y = shift_bank(y)
    x, y = copy_bank(x, y)
    return bb.finalize(enable=e, source=j, destination=i, data=x, auxiliary=y)

@model_frozen(cache_hash=True)
class model_LocalSelect(model_Bloq):
    Lx: int
    Ly: int
    N: int
    n_max: int
    b: int
    labels: tuple
    coefficients: tuple

    @property
    def L(self):
        return self.Lx * self.Ly

    @property
    def q(self):
        return core_q_ph_from_nmax(self.n_max)

    @property
    def p(self):
        return len(self.labels).bit_length() - 1

    @property
    def signature(self):
        return model_Signature.build(enable=1, selection=self.p, signal=1, photons=self.L * self.q, emitters=self.L * self.N, resource=self.b + 1)

    def build_composite_bloq(self, bb, enable, selection, signal, photons, emitters, resource):
        a = enc__split(bb, selection, self.p)
        ph = enc__split(bb, photons, self.L * self.q)
        at = enc__split(bb, emitters, self.L * self.N)

        def leaf(k, flag):
            label, coeff = (self.labels[k], self.coefficients[k])
            if label[0] in ('ph', 'at'):
                _, x, y, r = label
                site = y * self.Lx + x
                if label[0] == 'ph':
                    pos = site * self.q + self.q - 1 - r
                    flag[0], ph[pos] = model__cz(bb, flag[0], ph[pos])
                else:
                    pos = site * self.N + r
                    flag[0], at[pos] = model__cz(bb, flag[0], at[pos])
            if coeff < 0:
                flag[0] = bb.add(model_ZGate(), q=flag[0])
        a, enable = enc__unary(bb, a, leaf, enable)
        E = self.L * self.N
        d = E.bit_length() - 1
        t = self.p - d
        flags = [enable] + [bb.allocate(dtype=model_QBit()) for _ in range(t)]
        for i in range(t):
            a[i] = bb.add(model_XGate(), q=a[i])
            flags[i], a[i], flags[i + 1] = enc__ccx(bb, flags[i], a[i], flags[i + 1])
            a[i] = bb.add(model_XGate(), q=a[i])
        ex = flags[-1]
        addr = enc__join(bb, a[t:])
        sample = bb.allocate(dtype=model_QBit())
        o = bb.add_d(model_emitter_read(E), address=addr, data=enc__join(bb, at), sample=sample)
        addr, emitters, sample = (o['address'], o['data'], o['sample'])
        addrb = enc__split(bb, addr, d)
        site = enc__join(bb, addrb[:self.L.bit_length() - 1], model_QUInt(self.L.bit_length() - 1))
        rest = addrb[self.L.bit_length() - 1:]

        def shift(decrement):
            nonlocal ex, sample, site, photons
            flag = bb.allocate(dtype=model_QBit())
            if decrement:
                sample = bb.add(model_XGate(), q=sample)
            ex, sample, flag = enc__ccx(bb, ex, sample, flag)
            if decrement:
                sample = bb.add(model_XGate(), q=sample)
            o = bb.add_d(enc_build_selected_shift(self.L, self.q, decrement), enable=flag, address=site, data=photons)
            flag, site, photons = (o['enable'], o['address'], o['data'])
            if decrement:
                sample = bb.add(model_XGate(), q=sample)
            ex, sample, flag = enc__ccx(bb, ex, sample, flag)
            if decrement:
                sample = bb.add(model_XGate(), q=sample)
            bb.free(flag)
        photons = enc__join(bb, ph)
        shift(False)
        word = bb.allocate(dtype=model_QAny(self.q))
        load = enc_build_indexed_load(self.L, self.q)
        o = bb.add_d(load, address=site, data=photons, word=word)
        site, photons, word = (o['address'], o['data'], o['word'])
        o = bb.add_d(model_angle_oracle(model_factor_tables(self.n_max, self.b)[0], self.b), enable=ex, occupation=word, signal=signal, phase_grad=resource)
        ex, word, signal, resource = (o['enable'], o['occupation'], o['signal'], o['phase_grad'])
        o = bb.add_d(load.adjoint(), address=site, data=photons, word=word)
        site, photons, word = (o['address'], o['data'], o['word'])
        bb.free(word)
        shift(True)
        addr = enc__join(bb, [*enc__split(bb, site, self.L.bit_length() - 1), *rest])
        o = bb.add_d(model_emitter_flip(E), enable=ex, address=addr, data=emitters)
        ex, addr, emitters = (o['enable'], o['address'], o['data'])
        ex, sample = enc__cx(bb, ex, sample)
        o = bb.add_d(model_emitter_read(E).adjoint(), address=addr, data=emitters, sample=sample)
        addr, emitters, sample = (o['address'], o['data'], o['sample'])
        bb.free(sample)
        a[t:] = enc__split(bb, addr, d)
        flags[-1] = ex
        for i in reversed(range(t)):
            a[i] = bb.add(model_XGate(), q=a[i])
            flags[i], a[i], flags[i + 1] = enc__ccx(bb, flags[i], a[i], flags[i + 1])
            a[i] = bb.add(model_XGate(), q=a[i])
        for w in flags[1:]:
            bb.free(w)
        return dict(enable=flags[0], selection=enc__join(bb, a), signal=signal, photons=photons, emitters=emitters, resource=resource)

@model_frozen(cache_hash=True)
class model_HoppingSelect(model_Bloq):
    Lx: int
    Ly: int
    n_max: int
    b: int
    t_table: tuple
    mode: str = 'direct'

    @property
    def L(self):
        return self.Lx * self.Ly

    @property
    def q(self):
        return core_q_ph_from_nmax(self.n_max)

    @property
    def ell(self):
        return self.L.bit_length() - 1

    @property
    def signature(self):
        return model_Signature.build(enable=1, selection=self.ell + 2, signals=2, photons=self.L * self.q, resource=self.b + 1)

    def __attrs_post_init__(self):
        if self.mode not in ('direct', 'sparse', 'paired'):
            raise ValueError('Unknown photon update.')
        if len(self.t_table) != 4 * self.L:
            raise ValueError('Expected 4L directed couplings.')

    def build_composite_bloq(self, bb, enable, selection, signals, photons, resource):
        label = enc__split(bb, selection, self.ell + 2)
        j = enc__join(bb, label[:-2], model_QUInt(self.ell))
        axis, sign = label[-2:]
        sig = enc__split(bb, signals, 2)
        i = bb.allocate(dtype=model_QUInt(self.ell))
        dest = enc_build_destination(self.Lx, self.Ly)
        o = bb.add_d(dest, source=j, axis=axis, sign=sign, destination=i)
        j, axis, sign, i = (o[k] for k in ('source', 'axis', 'sign', 'destination'))
        load = enc_build_indexed_load(self.L, self.q)
        for idx, table in enumerate(model_factor_tables(self.n_max, self.b)):
            word = bb.allocate(dtype=model_QAny(self.q))
            o = bb.add_d(load, address=j if idx == 0 else i, data=photons, word=word)
            if idx == 0:
                j = o['address']
            else:
                i = o['address']
            photons, word = (o['data'], o['word'])
            o = bb.add_d(model_angle_oracle(table, self.b), enable=enable, occupation=word, signal=sig[idx], phase_grad=resource)
            enable, word, sig[idx], resource = (o['enable'], o['occupation'], o['signal'], o['phase_grad'])
            o = bb.add_d(load.adjoint(), address=j if idx == 0 else i, data=photons, word=word)
            if idx == 0:
                j = o['address']
            else:
                i = o['address']
            photons, word = (o['data'], o['word'])
            bb.free(word)
        label = [*enc__split(bb, j, self.ell), axis, sign]
        negative = tuple((t > 0 for t in self.t_table))
        if all(negative):
            enable = bb.add(model_ZGate(), q=enable)
        elif any(negative):

            def leaf(k, flag):
                if negative[k]:
                    flag[0] = bb.add(model_ZGate(), q=flag[0])
            label, enable = enc__unary(bb, label, leaf, enable)
        j = enc__join(bb, label[:-2], model_QUInt(self.ell))
        axis, sign = label[-2:]
        if self.mode == 'direct':
            for src, dec in ((True, True), (False, False)):
                o = bb.add_d(enc_build_selected_shift(self.L, self.q, dec), enable=enable, address=j if src else i, data=photons)
                enable, photons = (o['enable'], o['data'])
                if src:
                    j = o['address']
                else:
                    i = o['address']
        else:
            aux = bb.allocate(dtype=model_QAny(self.L * self.q))
            update = enc_build_row_map_update(self.L, self.q) if self.mode == 'sparse' else model_build_paired_update(self.L, self.q)
            o = bb.add_d(enc_coherent(update), enable=enable, source=j, destination=i, data=photons, auxiliary=aux)
            enable, j, i, photons, aux = (o[k] for k in ('enable', 'source', 'destination', 'data', 'auxiliary'))
            bb.free(aux)
        o = bb.add_d(dest.adjoint(), source=j, axis=axis, sign=sign, destination=i)
        j, axis, sign, i = (o[k] for k in ('source', 'axis', 'sign', 'destination'))
        bb.free(i)
        return dict(enable=enable, selection=enc__join(bb, [*enc__split(bb, j, self.ell), axis, sign]), signals=enc__join(bb, sig), photons=photons, resource=resource)

def model__take_prep(bb, prep, bits):
    soqs = {}
    offset = 0
    for reg in prep.signature:
        n = int(reg.total_bits())
        soqs[reg.name] = enc__join(bb, bits[offset:offset + n], reg.dtype)
        offset += n
    return soqs

def model__prep_bits(bb, prep, soqs):
    return [w for reg in prep.signature for w in enc__split(bb, soqs[reg.name], int(reg.total_bits()))]

@model_frozen(cache_hash=True)
class model_ManuscriptTCL(enc__ZeroSignal, model_BlockEncoding):
    local: model_LocalSelect
    hop: model_HoppingSelect
    local_weights: tuple
    B: int
    rotation_eps: float = 1e-10
    control_val: model_Optional[int] = None

    @model_cached_property
    def prep_local(self):
        return model_AliasPreparation(self.local_weights, self.B)

    @model_cached_property
    def prep_hop(self):
        return model_AliasPreparation(tuple((abs(t) for t in self.hop.t_table)), self.B)

    @property
    def system_bitsize(self):
        return self.local.L * (self.local.q + self.local.N)

    @property
    def ancilla_bitsize(self):
        return 4 + self.prep_local.width + self.prep_hop.width

    @property
    def resource_bitsize(self):
        return self.local.b + 1

    @property
    def alpha_local(self):
        return float(sum(self.local_weights))

    @property
    def alpha_hop(self):
        return self.hop.n_max * sum((abs(t) for t in self.hop.t_table))

    @property
    def alpha(self):
        return self.alpha_local + self.alpha_hop

    @property
    def epsilon(self):
        ex = sum((abs(c) for l, c in zip(self.local.labels, self.local.coefficients) if l[0] == 'int'))
        return (ex + 2 * self.alpha_hop) * model_math.pi / 2 ** (self.local.b + 1) + self.alpha_local * self.prep_local.data['l1_prob_error'] + self.alpha_hop * self.prep_hop.data['l1_prob_error']

    def build_composite_bloq(self, bb, system, ancilla, resource, ctrl=None):
        bits = enc__split(bb, ancilla, self.ancilla_bitsize)
        s, eta, sig1, sig2 = bits[:4]
        nl = self.prep_local.width
        loc = model__take_prep(bb, self.prep_local, bits[4:4 + nl])
        hop = model__take_prep(bb, self.prep_hop, bits[4 + nl:])
        theta = 2 * model_math.acos(model_math.sqrt(self.alpha_local / self.alpha))
        s = bb.add(model_Ry(theta, eps=self.rotation_eps), q=s)
        loc = bb.add_d(self.prep_local, **loc)
        hop = bb.add_d(self.prep_hop, **hop)
        data = enc__split(bb, system, self.system_bitsize)
        ph = []
        at = []
        for j in range(self.local.L):
            off = j * (self.local.q + self.local.N)
            ph.extend(data[off:off + self.local.q])
            at.extend(data[off + self.local.q:off + self.local.q + self.local.N])
        photons, emitters = (enc__join(bb, ph), enc__join(bb, at))
        e = ctrl
        if e is None:
            e = bb.add(model_XGate(), q=bb.allocate(dtype=model_QBit()))
        elif self.control_val == 0:
            e = bb.add(model_XGate(), q=e)
        flag = bb.allocate(dtype=model_QBit())
        s = bb.add(model_XGate(), q=s)
        e, s, flag = enc__ccx(bb, e, s, flag)
        o = bb.add_d(self.local, enable=flag, selection=loc['selection'], signal=eta, photons=photons, emitters=emitters, resource=resource)
        flag, loc['selection'], eta, photons, emitters, resource = (o[k] for k in ('enable', 'selection', 'signal', 'photons', 'emitters', 'resource'))
        e, s, flag = enc__ccx(bb, e, s, flag)
        s = bb.add(model_XGate(), q=s)
        e, s, flag = enc__ccx(bb, e, s, flag)
        o = bb.add_d(self.hop, enable=flag, selection=hop['selection'], signals=enc__join(bb, [sig1, sig2]), photons=photons, resource=resource)
        flag, hop['selection'], signals, photons, resource = (o[k] for k in ('enable', 'selection', 'signals', 'photons', 'resource'))
        sig1, sig2 = enc__split(bb, signals, 2)
        e, s, flag = enc__ccx(bb, e, s, flag)
        bb.free(flag)
        if ctrl is None:
            e = bb.add(model_XGate(), q=e)
            bb.free(e)
        else:
            ctrl = bb.add(model_XGate(), q=e) if self.control_val == 0 else e
        ph, at = (enc__split(bb, photons, self.local.L * self.local.q), enc__split(bb, emitters, self.local.L * self.local.N))
        data = []
        for j in range(self.local.L):
            data.extend(ph[j * self.local.q:(j + 1) * self.local.q] + at[j * self.local.N:(j + 1) * self.local.N])
        hop = bb.add_d(enc_coherent(self.prep_hop).adjoint(), **hop)
        loc = bb.add_d(enc_coherent(self.prep_local).adjoint(), **loc)
        s = bb.add(model_Ry(-theta, eps=self.rotation_eps), q=s)
        out = dict(system=enc__join(bb, data), ancilla=enc__join(bb, [s, eta, sig1, sig2, *model__prep_bits(bb, self.prep_local, loc), *model__prep_bits(bb, self.prep_hop, hop)]), resource=resource)
        if self.control_val is not None:
            out['ctrl'] = ctrl
        return out

@model_frozen(cache_hash=True)
class model_DoublyEnabledEncoding(model_Bloq):
    full: model_ManuscriptTCL
    h_value: int = 0

    @property
    def signature(self):
        return model_Signature.build(ctrl=1, h=1, system=self.full.system_bitsize, ancilla=self.full.ancilla_bitsize, resource=self.full.resource_bitsize)

    def build_composite_bloq(self, bb, ctrl, h, **soqs):
        e = bb.allocate(dtype=model_QBit())
        if self.h_value == 0:
            h = bb.add(model_XGate(), q=h)
        ctrl, h, e = enc__ccx(bb, ctrl, h, e)
        if self.h_value == 0:
            h = bb.add(model_XGate(), q=h)
        o = bb.add_d(model_attrs.evolve(self.full, control_val=1), ctrl=e, **soqs)
        e = o.pop('ctrl')
        if self.h_value == 0:
            h = bb.add(model_XGate(), q=h)
        ctrl, h, e = enc__ccx(bb, ctrl, h, e)
        if self.h_value == 0:
            h = bb.add(model_XGate(), q=h)
        bb.free(e)
        return dict(ctrl=ctrl, h=h, **o)

@model_lru_cache(None)
def model_reflection(a):
    bb = model_BloqBuilder()
    q = bb.add_register('ctrl', model_QBit())
    h = bb.add_register('h', model_QBit())
    A = bb.add_register('ancilla', model_QAny(a))
    h = bb.add(model_Hadamard(), q=h)
    bits = [h, *enc__split(bb, A, a)]
    bits = [bb.add(model_XGate(), q=w) for w in bits]
    r = [bb.allocate(dtype=model_QBit()) for _ in range(a)]
    bits[0], bits[1], r[0] = enc__ccx(bb, bits[0], bits[1], r[0])
    for i in range(1, a):
        r[i - 1], bits[i + 1], r[i] = enc__ccx(bb, r[i - 1], bits[i + 1], r[i])
    q, r[-1] = model__cz(bb, q, r[-1])
    for i in reversed(range(1, a)):
        r[i - 1], bits[i + 1], r[i] = enc__ccx(bb, r[i - 1], bits[i + 1], r[i])
    bits[0], bits[1], r[0] = enc__ccx(bb, bits[0], bits[1], r[0])
    for w in r:
        bb.free(w)
    bits = [bb.add(model_XGate(), q=w) for w in bits]
    h = bb.add(model_Hadamard(), q=bits[0])
    q = bb.add(model_ZGate(), q=q)
    return bb.finalize(ctrl=q, h=h, ancilla=enc__join(bb, bits[1:]))

@model_frozen(cache_hash=True)
class model_ControlledWalk(model_Bloq):
    full: model_ManuscriptTCL

    @property
    def signature(self):
        return model_Signature.build(ctrl=1, system=self.full.system_bitsize, selection=self.full.ancilla_bitsize + 1, resource=self.full.resource_bitsize)

    def build_composite_bloq(self, bb, ctrl, system, selection, resource):
        bits = enc__split(bb, selection, self.full.ancilla_bitsize + 1)
        h = bits[0]
        A = enc__join(bb, bits[1:])
        ctrl, h = enc__cx(bb, ctrl, h)
        o = bb.add_d(model_DoublyEnabledEncoding(self.full, 0), ctrl=ctrl, h=h, system=system, ancilla=A, resource=resource)
        o = bb.add_d(enc_coherent(model_DoublyEnabledEncoding(self.full, 1)).adjoint(), **o)
        ctrl, h, system, A, resource = (o[k] for k in ('ctrl', 'h', 'system', 'ancilla', 'resource'))
        o = bb.add_d(model_reflection(self.full.ancilla_bitsize), ctrl=ctrl, h=h, ancilla=A)
        return dict(ctrl=o['ctrl'], system=system, selection=enc__join(bb, [o['h'], *enc__split(bb, o['ancilla'], self.full.ancilla_bitsize)]), resource=resource)

def model_build_pair(*, Lx=2, Ly=2, N=1, n_max=2, b=5, B=3, mu=None, g=None, t_table=None, Delta=0.0, keep_identity_shift=False, modes=('direct', 'sparse')):
    if min(Lx, Ly) < 2 or any((v & v - 1 for v in (Lx, Ly, N))) or N < 1:
        raise ValueError('Power-of-two geometry and N required.')
    q = core_q_ph_from_nmax(n_max)
    L = Lx * Ly
    mu = model_np.ones((Ly, Lx)) if mu is None else core_normalize_mu_2d(mu, Lx, Ly)
    g = model_np.ones((Ly, Lx, N)) if g is None else core_normalize_g_2d(g, Lx, Ly, N)
    t_table = model_np.full((Ly, Lx, 2, 2), 0.05) if t_table is None else model_np.asarray(t_table).reshape(Ly, Lx, 2, 2)
    oldlabels, coeffs, _, _ = core_build_tcl_local_coefficients_2d(Lx, Ly, N, q, n_max, mu, Delta, g, keep_identity_shift=keep_identity_shift)
    lookup = dict(zip(oldlabels, coeffs))
    labels = []
    for kind in ('int', 'at'):
        labels.extend(((kind, x, y, n) for y in range(Ly) for x in range(Lx) for n in range(N)))
    labels.extend((('ph', x, y, r) for y in range(Ly) for x in range(Lx) for r in range(q)))
    labels.append(('id',))
    D = 1 << (1 + L * (q + 2 * N) - 1).bit_length()
    labels.extend([('pad',)] * (D - len(labels)))
    coeffs = tuple((float(lookup.get(label, 0)) for label in labels))
    local = model_LocalSelect(Lx, Ly, N, n_max, b, tuple(labels), coeffs)
    result = dict(Lx=Lx, Ly=Ly, N=N, n_max=n_max, b=b, B=B, angle_bits=b + 1, alias_bits=B, mu=mu, g=g, t_table=t_table, Delta=Delta, keep_identity_shift=keep_identity_shift, local=local)
    for mode in modes:
        hop = model_HoppingSelect(Lx, Ly, n_max, b, tuple(t_table.ravel()), mode)
        full = model_ManuscriptTCL(local, hop, tuple((abs(c) for c in coeffs)), B)
        result[mode] = dict(hopping_select=hop, full=full, encoding_call=model_DoublyEnabledEncoding(full), walk=model_ControlledWalk(full))
    return result

@model_frozen(cache_hash=True)
class model_BankedCircuit(model_Bloq):
    """Route actual allocate/free instructions through an explicit clean bank.

    This is a wire-allocation transformation. It changes no physical gates.
    Each recursive call receives the currently unused wires and returns them;
    its required high-water mark is checked rather than assumed. Left/right
    bank sizes differ for primitives that create or consume output registers.
    """
    inner: model_Bloq
    free_bits: int

    @property
    def final_free_bits(self):
        return self.free_bits + sum((int(r.total_bits()) for r in self.inner.signature.lefts())) - sum((int(r.total_bits()) for r in self.inner.signature.rights()))

    @property
    def signature(self):
        regs = list(self.inner.signature)
        if self.free_bits:
            regs.append(model_Register('scratch_in', model_QAny(self.free_bits), side=model_Side.LEFT))
        if self.final_free_bits:
            regs.append(model_Register('scratch_out', model_QAny(self.final_free_bits), side=model_Side.RIGHT))
        return model_Signature(regs)

    def build_composite_bloq(self, bb, **soqs):
        free = enc__split(bb, soqs.pop('scratch_in'), self.free_bits) if self.free_bits else []
        b = enc_coherent(self.inner)
        cb = b.body if isinstance(b, (enc_Coherent, enc_CoherentAdjoint)) else b.decompose_bloq()
        mapping = {}
        for reg in cb.signature.lefts():
            old = enc__reg_to_soq(enc_LeftDangle, reg)
            enc__update_flat_soq_map([(old, soqs[reg.name])], mapping)
        for inst, ins, oldout in cb.iter_bloqsoqs():
            ins = enc__map_flat_soqs(ins, mapping)
            child = inst.bloq
            if isinstance(child, model_Allocate):
                n = int(child.dtype.num_qubits)
                if len(free) < n:
                    raise ValueError(f'Workspace bank exhausted in {self.inner}: need {n}, have {len(free)}.')
                allocated = free[:n]
                free = free[n:]
                out = {'reg': enc__join(bb, allocated, child.dtype)}
            elif isinstance(child, model_Free):
                free.extend(enc__split(bb, ins['reg'], int(child.dtype.num_qubits)))
                out = {}
            elif enc__is_terminal(child):
                out = bb.add_d(child, **ins)
            else:
                n_inputs = sum((int(r.total_bits()) for r in child.signature.lefts()))
                required = enc_compiled_peak_qubits(child) - n_inputs
                if required > len(free):
                    raise ValueError('Insufficient workspace for child.')
                used, held = (free[:required], free[required:])
                call = model_BankedCircuit(child, required)
                if call.final_free_bits < 0:
                    raise ValueError('Insufficient workspace for child outputs.')
                if used:
                    ins['scratch_in'] = enc__join(bb, used)
                out = bb.add_d(call, **ins)
                returned = enc__split(bb, out.pop('scratch_out'), call.final_free_bits) if call.final_free_bits else []
                free = returned + held
            newout = tuple((out[r.name] for r in child.signature.rights()))
            enc__update_flat_soq_map(zip(oldout, newout), mapping)
        result = enc__map_flat_soqs(cb.final_soqs(), mapping)
        if len(free) != self.final_free_bits:
            raise AssertionError('Bank balance mismatch.')
        if free:
            result['scratch_out'] = enc__join(bb, free)
        return result

def model_declared_workspace(full):
    """Manuscript's deliberately conservative fixed bank allocation.

    Used to ALLOCATE wires, never to supply a measured resource count. The
    BankedCircuit transformation verifies that every actual subcircuit fits.
    The sparse second photon bank is reserved separately for the full query.
    """
    loc = full.local
    lx = loc.Lx.bit_length() - 1
    ly = loc.Ly.bit_length() - 1
    ell = lx + ly
    d_ex = loc.p - ell - (loc.N.bit_length() - 1)
    w = d_ex + 3 + ell + max(lx - 2, ly - 2, 0) + loc.q + max(loc.p, loc.q) + max(loc.q - 2, 0) + (3 * loc.b - 1) + 1 + (2 * full.B + 1)
    return max(full.ancilla_bitsize, w) + (loc.L * loc.q if full.hop.mode in ('sparse', 'paired') else 0)
