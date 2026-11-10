"""Consolidated TCL walk implementation.

Static, readable definitions; no embedded modules or parent-bundle imports.
Internal prefixes distinguish shared definitions merged from the development code.
"""
from __future__ import annotations
import block_encoding as be
from qualtran import Bloq as opened_Bloq
from qualtran import QBit as opened_QBit
from qualtran.bloqs.basic_gates import Ry as opened_Ry
from qualtran import Signature as opened_Signature
from qualtran.bloqs.basic_gates import XGate as opened_XGate
from attrs import frozen as opened_frozen
import math as opened_math
from qualtran._infra.adjoint import Adjoint as enc_Adjoint
from qualtran._infra.composite_bloq import _map_flat_soqs as enc__map_flat_soqs
from qualtran._infra.composite_bloq import _update_flat_soq_map as enc__update_flat_soq_map
from qualtran import Signature as model_Signature
from qualtran import Bloq as rotation_Bloq
from qualtran import BloqBuilder as rotation_BloqBuilder
from collections import Counter as rotation_Counter
from qualtran.resource_counting import GateCounts as rotation_GateCounts
from qualtran.bloqs.basic_gates import Hadamard as rotation_Hadamard
from qualtran import QAny as rotation_QAny
from qualtran.resource_counting import QECGatesCost as rotation_QECGatesCost
from qualtran.bloqs.basic_gates import SGate as rotation_SGate
from qualtran import Signature as rotation_Signature
from qualtran.bloqs.basic_gates import TGate as rotation_TGate
from qualtran.bloqs.basic_gates import Toffoli as rotation_Toffoli
from qualtran.bloqs.basic_gates import ZGate as rotation_ZGate
from attrs import frozen as rotation_frozen
from qualtran.resource_counting import get_cost_value as rotation_get_cost_value
from functools import lru_cache as rotation_lru_cache
import mpmath as rotation_mp
import numpy as rotation_np
from qualtran import Bloq as closed_Bloq
from qualtran import BloqBuilder as closed_BloqBuilder
from collections import Counter as closed_Counter
from qualtran import QAny as closed_QAny
from qualtran import QBit as closed_QBit
from qualtran.bloqs.basic_gates import Ry as closed_Ry
from qualtran import Signature as closed_Signature
from qualtran.bloqs.basic_gates import XGate as closed_XGate
from functools import cached_property as closed_cached_property
from attrs import frozen as closed_frozen
import math as closed_math


# tcl_manuscript_table_VI.py -- retained active definitions

rotation__split, rotation__join, rotation__cx, rotation__ccx = (be.enc__split, be.enc__join, be.enc__cx, be.enc__ccx)

def rotation_precision_plan(lx, ly, N, n_max, tau='1', epsilon='0.001', dps=80):
    """D9-D10, G16 and the smallest degree certified by G19; no lookup table."""
    with rotation_mp.workdps(dps):
        L = lx * ly
        q = n_max.bit_length()
        tau = abs(rotation_mp.mpf(str(tau)))
        eps = rotation_mp.mpf(str(epsilon))
        if tau <= 0 or not 0 < eps < 1:
            raise ValueError('Positive time and 0<epsilon<1 required.')
        alpha_int = L * N * rotation_mp.sqrt(n_max)
        alpha_hop = 4 * L * rotation_mp.mpf('0.05') * n_max
        alpha_local = rotation_mp.mpf(L) * (2 ** q - 1 + N) / 2 + alpha_int
        alpha = alpha_local + alpha_hop
        ceil_log2 = lambda x: max(1, int(rotation_mp.ceil(rotation_mp.log(x, 2))))
        b = ceil_log2(4 * rotation_mp.pi * tau * (2 * alpha_hop + alpha_int) / eps)
        B = ceil_log2(8 * tau * alpha / eps)
        x = alpha * tau
        target = eps ** 2 / 32

        def tail(K):
            n = K + 1
            if n <= x:
                return rotation_mp.inf
            y = rotation_mp.acosh(n / x)
            return 4 * rotation_mp.exp(-n * y + rotation_mp.sqrt(n * n - x * x)) / -rotation_mp.expm1(-y)
        lower = max(0, int(rotation_mp.floor(x)) - 1)
        upper = max(1, lower + 1)
        while tail(upper) > target:
            upper = max(upper + 1, 2 * upper)
        while upper - lower > 1:
            middle = (upper + lower) // 2
            if tail(middle) <= target:
                upper = middle
            else:
                lower = middle
        K = upper
        n_signal = 2 * K + 1
        m = max(ceil_log2(4 * rotation_mp.pi * tau * alpha / eps), ceil_log2(6 * rotation_mp.pi * n_signal / eps))
        bound = tail(K)
        assert bound <= target and (K == 0 or tail(K - 1) > target)
        assert 2 * tau * alpha * rotation_mp.pi / 2 ** (m + 1) <= eps / 4
        assert 3 * n_signal * rotation_mp.pi / 2 ** (m + 1) <= eps / 4
        return dict(Lx=lx, Ly=ly, L=L, N=N, n_max=n_max, tau=str(tau), epsilon=str(eps), b=b, B=B, K=K, m=m, n_signal=n_signal, walk_queries=2 * K, alpha=rotation_mp.nstr(alpha, dps), alpha_local=rotation_mp.nstr(alpha_local, dps), alpha_hop=rotation_mp.nstr(alpha_hop, dps), alpha_int=rotation_mp.nstr(alpha_int, dps), two_eta_bound=rotation_mp.nstr(bound, dps), previous_degree_bound=rotation_mp.nstr(tail(K - 1), dps), polynomial_budget=rotation_mp.nstr(target, dps), outer_rotation_error_bound=rotation_mp.nstr(2 * tau * alpha * rotation_mp.pi / 2 ** (m + 1), dps), signal_rotation_error_bound=rotation_mp.nstr(3 * n_signal * rotation_mp.pi / 2 ** (m + 1), dps), planning_decimal_digits=dps, energy_origin='identity omitted')

def rotation_rounded_word(theta, bits):
    with rotation_mp.workdps(80):
        return int(rotation_mp.floor(rotation_mp.mpf(str(theta)) * 2 ** bits / (2 * rotation_mp.pi) + rotation_mp.mpf('0.5')))

@rotation_frozen(cache_hash=True)
class rotation_ControlledGridRotation(rotation_Bloq):
    """Exact controlled Ry/Rz at theta=2*pi*k/2**bits on a positive catalyst.

    Only Clifford word-loading/erasure and the winding Z depend on k. The
    coherent gradient adder is identical for every k, including zero. Retain
    this manuscript generic schedule rather than constant-folding its adder.
    """
    bits: int
    k: int
    axis: str = 'z'

    def __attrs_post_init__(self):
        if self.bits < 1 or self.axis not in ('y', 'z'):
            raise ValueError('Invalid rotation.')

    @property
    def signature(self):
        return rotation_Signature.build(enable=1, target=1, rotation_resource=self.bits + 1)

    def build_composite_bloq(self, bb, enable, target, rotation_resource):
        modulus = 1 << self.bits
        winding, word = divmod(self.k, modulus)
        angle = rotation__split(bb, bb.allocate(dtype=rotation_QAny(self.bits)), self.bits)
        for i in range(self.bits):
            if word >> self.bits - i - 1 & 1:
                enable, angle[i] = rotation__cx(bb, enable, angle[i])
        if self.axis == 'y':
            target = bb.add(rotation_SGate().adjoint(), q=target)
            target = bb.add(rotation_Hadamard(), q=target)
        z = rotation__split(bb, rotation_resource, self.bits + 1)
        for i in range(self.bits + 1):
            target, z[i] = rotation__cx(bb, target, z[i])
        out = bb.add_d(be.model_gradient_add(self.bits), angle=rotation__join(bb, angle), phase_grad=rotation__join(bb, z))
        angle = rotation__split(bb, out['angle'], self.bits)
        z = rotation__split(bb, out['phase_grad'], self.bits + 1)
        for i in range(self.bits + 1):
            target, z[i] = rotation__cx(bb, target, z[i])
        rotation_resource = rotation__join(bb, z)
        if self.axis == 'y':
            target = bb.add(rotation_Hadamard(), q=target)
            target = bb.add(rotation_SGate(), q=target)
        for i in range(self.bits):
            if word >> self.bits - i - 1 & 1:
                enable, angle[i] = rotation__cx(bb, enable, angle[i])
        bb.free(rotation__join(bb, angle))
        if winding % 2:
            enable = bb.add(rotation_ZGate(), q=enable)
        return dict(enable=enable, target=target, rotation_resource=rotation_resource)

@rotation_lru_cache(None)
def rotation_exact_seven_t_toffoli():
    """Exact ancilla-free CCX decomposition, including its absolute phase."""
    bb, soqs = rotation_BloqBuilder.from_signature(rotation_Toffoli().signature)
    a, b = soqs['ctrl']
    t = soqs['target']
    t = bb.add(rotation_Hadamard(), q=t)
    b, t = rotation__cx(bb, b, t)
    t = bb.add(rotation_TGate().adjoint(), q=t)
    a, t = rotation__cx(bb, a, t)
    t = bb.add(rotation_TGate(), q=t)
    b, t = rotation__cx(bb, b, t)
    t = bb.add(rotation_TGate().adjoint(), q=t)
    a, t = rotation__cx(bb, a, t)
    b = bb.add(rotation_TGate(), q=b)
    t = bb.add(rotation_TGate(), q=t)
    t = bb.add(rotation_Hadamard(), q=t)
    a, b = rotation__cx(bb, a, b)
    a = bb.add(rotation_TGate(), q=a)
    b = bb.add(rotation_TGate().adjoint(), q=b)
    a, b = rotation__cx(bb, a, b)
    return bb.finalize(ctrl=rotation_np.array([a, b], dtype=object), target=t)

def rotation_clifford_t_leaves(bloq):
    out = rotation_Counter()
    toffoli_leaves = be.enc_compiled_leaf_counts(rotation_exact_seven_t_toffoli())
    for leaf, n in be.enc_compiled_leaf_counts(bloq).items():
        if isinstance(leaf, rotation_Toffoli):
            for physical, mult in toffoli_leaves.items():
                out[physical] += n * mult
        else:
            out[leaf] += n
    return out

def rotation_leaf_cost(leaves):
    total = rotation_GateCounts()
    for leaf, n in leaves.items():
        total += rotation_get_cost_value(leaf, rotation_QECGatesCost()) * n
    if total.and_bloq or total.measurement or total.rotation or total.toffoli:
        raise AssertionError(f'Non-Clifford+T leaf remains: {total}')
    return dict(T=int(total.t), Clifford=int(total.clifford), physical_leaf_instances=sum(leaves.values()))

def rotation_printed_tail_bound(value):
    """Five significant digits rounded upward, preserving a stated bound."""
    with rotation_mp.workdps(80):
        x = rotation_mp.mpf(str(value))
        exponent = int(rotation_mp.floor(rotation_mp.log10(x)))
        factor = rotation_mp.power(10, 4 - exponent)
        rounded = rotation_mp.ceil(x * factor) / factor
        return f'{float(rounded):.4e}'


# tcl_revised_manuscript.py -- retained active definitions

def closed_base(b):
    while isinstance(b, be.enc_Coherent):
        b = b.inner
    return b

@closed_frozen(cache_hash=True)
class closed_F(closed_Bloq):
    full: be.model_ManuscriptTCL

    @property
    def signature(self):
        return model_Signature.build(ancilla=self.full.ancilla_bitsize)

    def build_composite_bloq(self, bb, ancilla):
        full = self.full
        bits = be.enc__split(bb, ancilla, full.ancilla_bitsize)
        s, eta, s1, s2 = bits[:4]
        nl = full.prep_local.width
        loc = be.model__take_prep(bb, full.prep_local, bits[4:4 + nl])
        hop = be.model__take_prep(bb, full.prep_hop, bits[4 + nl:])
        s = bb.add(closed_Ry(2 * closed_math.acos(closed_math.sqrt(full.alpha_local / full.alpha))), q=s)
        loc = bb.add_d(full.prep_local, **loc)
        hop = bb.add_d(full.prep_hop, **hop)
        return dict(ancilla=be.enc__join(bb, [s, eta, s1, s2, *be.model__prep_bits(bb, full.prep_local, loc), *be.model__prep_bits(bb, full.prep_hop, hop)]))

@closed_frozen(cache_hash=True)
class closed_StripPreparations(closed_Bloq):
    inner: closed_Bloq

    @property
    def signature(self):
        return self.inner.signature

    @closed_cached_property
    def body(self):
        b = closed_base(self.inner)
        if isinstance(b, be.enc_CoherentAdjoint):
            return be.enc__fast_adjoint(closed_StripPreparations(b.forward).body)
        if isinstance(b, enc_Adjoint):
            return be.enc__fast_adjoint(closed_StripPreparations(b.subbloq).body)
        cb = be.enc_coherent(b).body
        bb, _ = closed_BloqBuilder.from_signature(cb.signature)
        mapping = {}
        removed = closed_Counter()
        for inst, ins, oldout in cb.iter_bloqsoqs():
            ins = enc__map_flat_soqs(ins, mapping)
            child = inst.bloq
            under = closed_base(child)
            strip = isinstance(b, be.model_ManuscriptTCL) and (isinstance(under, (be.model_AliasPreparation, closed_Ry)) or (isinstance(under, be.enc_CoherentAdjoint) and isinstance(closed_base(under.forward), be.model_AliasPreparation)))
            if strip:
                removed['rotation' if isinstance(under, closed_Ry) else 'alias'] += 1
                out = ins
            elif isinstance(b, be.model_ManuscriptTCL) or be.enc__is_terminal(child):
                out = bb.add_d(child, **ins)
            else:
                out = bb.add_d(closed_StripPreparations(child), **ins)
            enc__update_flat_soq_map(zip(oldout, tuple((out[r.name] for r in child.signature.rights()))), mapping)
        if isinstance(b, be.model_ManuscriptTCL):
            assert removed == dict(alias=4, rotation=2), removed
        return bb.finalize(**enc__map_flat_soqs(cb.final_soqs(), mapping))

    def build_composite_bloq(self, bb, **soqs):
        return bb.add_d(self.body, **soqs)

@closed_frozen(cache_hash=True)
class closed_SharedWalk(closed_Bloq):
    full: be.model_ManuscriptTCL

    @property
    def signature(self):
        return be.model_ControlledWalk(self.full).signature

    def build_composite_bloq(self, bb, ctrl, system, selection, resource):
        h, *A = be.enc__split(bb, selection, self.full.ancilla_bitsize + 1)
        A = bb.add(closed_F(self.full), ancilla=be.enc__join(bb, A))
        ctrl, h = be.enc__cx(bb, ctrl, h)
        out = bb.add_d(closed_StripPreparations(be.model_DoublyEnabledEncoding(self.full, 0)), ctrl=ctrl, h=h, system=system, ancilla=A, resource=resource)
        out = bb.add_d(be.enc_coherent(closed_StripPreparations(be.model_DoublyEnabledEncoding(self.full, 1))).adjoint(), **out)
        ctrl, h, system, A, resource = (out[k] for k in ('ctrl', 'h', 'system', 'ancilla', 'resource'))
        A = bb.add(be.enc_coherent(closed_F(self.full)).adjoint(), ancilla=A)
        out = bb.add_d(be.model_reflection(self.full.ancilla_bitsize), ctrl=ctrl, h=h, ancilla=A)
        return dict(ctrl=out['ctrl'], system=system, selection=be.enc__join(bb, [out['h'], *be.enc__split(bb, out['ancilla'], self.full.ancilla_bitsize)]), resource=resource)

@closed_frozen(cache_hash=True)
class closed_FixedSharedWalk(closed_Bloq):
    full: be.model_ManuscriptTCL

    @property
    def signature(self):
        return closed_SharedWalk(self.full).signature

    def build_composite_bloq(self, bb, **soqs):
        n = be.model_declared_workspace(self.full)
        bank = bb.allocate(dtype=closed_QAny(n))
        out = bb.add_d(be.model_BankedCircuit(closed_SharedWalk(self.full), n), **soqs, scratch_in=bank)
        bb.free(out.pop('scratch_out'))
        return out

@closed_frozen(cache_hash=True)
class closed_GridRotation(closed_Bloq):
    """Unconditionally executed generic grid rotation, retaining absolute phase.

    The constant enable is allocated inside the arithmetic workspace, not as
    a persistent Hadamard-test wire. Its exact CCX/T body is the existing
    validated signed-adder implementation, including winding corrections.
    """
    bits: int
    k: int
    axis: str = 'z'

    @property
    def signature(self):
        return closed_Signature.build(target=1, rotation_resource=self.bits + 1)

    def build_composite_bloq(self, bb, target, rotation_resource):
        e = bb.add(closed_XGate(), q=bb.allocate(dtype=closed_QBit()))
        out = bb.add_d(rotation_ControlledGridRotation(self.bits, self.k, self.axis), enable=e, target=target, rotation_resource=rotation_resource)
        e = bb.add(closed_XGate(), q=out.pop('enable'))
        bb.free(e)
        return out


# tcl_appendix_h.py -- retained active definitions

opened_KINDS = ('photon_loss', 'atomic_loss', 'dephasing')

@opened_frozen(cache_hash=True)
class opened_Jump(opened_Bloq):
    """H1--H4: fixed classical address, supplied enable, shared eta signal."""
    full: be.model_ManuscriptTCL
    kind: str
    site: int = 0
    atom: int = 0

    def __attrs_post_init__(self):
        if self.kind not in opened_KINDS:
            raise ValueError('Unknown jump.')
        if not 0 <= self.site < self.full.local.L or not 0 <= self.atom < self.full.local.N:
            raise ValueError('Invalid classical address.')

    @property
    def signature(self):
        return opened_Signature.build(enable=1, system=self.full.system_bitsize, eta=1, resource=self.full.resource_bitsize)

    def build_composite_bloq(self, bb, enable, system, eta, resource):
        loc = self.full.local
        data = be.enc__split(bb, system, self.full.system_bitsize)
        off = self.site * (loc.q + loc.N)
        if self.kind == 'photon_loss':
            word = be.enc__join(bb, data[off:off + loc.q])
            out = bb.add_d(be.model_angle_oracle(be.model_factor_tables(loc.n_max, loc.b)[0], loc.b), enable=enable, occupation=word, signal=eta, phase_grad=resource)
            enable, word, eta, resource = (out[k] for k in ('enable', 'occupation', 'signal', 'phase_grad'))
            bits = be.enc__split(bb, word, loc.q)
            enable, bits = be.enc__inc(bb, enable, bits, decrement=True)
            data[off:off + loc.q] = bits
        elif self.kind == 'atomic_loss':
            j = off + loc.q + self.atom
            enable, eta = be.enc__cx(bb, enable, eta)
            enable, data[j], eta = be.enc__ccx(bb, enable, data[j], eta)
            enable, data[j] = be.enc__cx(bb, enable, data[j])
        else:
            enable, data[off + loc.q + self.atom] = be.model__cz(bb, enable, data[off + loc.q + self.atom])
        return dict(enable=enable, system=be.enc__join(bb, data), eta=eta, resource=resource)

@opened_frozen(cache_hash=True)
class opened_Walk(opened_Bloq):
    """Fig.14: shared FD, supplied SELECT enables and reflection on h,beta,f,A.

    Selection register layout h,beta,f,A; system layout d,physical. Three
    query flags are reused by H and J and released before the reflection.
    """
    full: be.model_ManuscriptTCL
    kind: str
    dt: float
    lambda_D: float
    site: int = 0
    atom: int = 0

    @property
    def signature(self):
        return opened_Signature.build(ctrl=1, selection=self.full.ancilla_bitsize + 3, system=self.full.system_bitsize + 1, resource=self.full.resource_bitsize)

    @property
    def beta_angle(self):
        hw = opened_math.sqrt(self.dt) * self.full.alpha
        return 2 * opened_math.acos(opened_math.sqrt(hw / (hw + opened_math.sqrt(self.lambda_D))))

    def build_composite_bloq(self, bb, ctrl, selection, system, resource):
        h, beta, f, *A = be.enc__split(bb, selection, self.full.ancilla_bitsize + 3)
        d, *data = be.enc__split(bb, system, self.full.system_bitsize + 1)
        physical = be.enc__join(bb, data)
        A = bb.add(closed_F(self.full), ancilla=be.enc__join(bb, A))
        beta = bb.add(opened_Ry(self.beta_angle), q=beta)
        ctrl, h = be.enc__cx(bb, ctrl, h)
        u, v, e = [bb.allocate(dtype=opened_QBit()) for _ in range(3)]
        beta = bb.add(opened_XGate(), q=beta)
        ctrl, beta, u = be.enc__ccx(bb, ctrl, beta, u)
        d = bb.add(opened_XGate(), q=d)
        u, d, v = be.enc__ccx(bb, u, d, v)
        d = bb.add(opened_XGate(), q=d)
        select = closed_StripPreparations(self.full.controlled())
        for hv in (0, 1):
            if hv == 0:
                h = bb.add(opened_XGate(), q=h)
            v, h, e = be.enc__ccx(bb, v, h, e)
            if hv == 0:
                h = bb.add(opened_XGate(), q=h)
            op = select if hv == 0 else be.enc_coherent(select).adjoint()
            out = bb.add_d(op, ctrl=e, system=physical, ancilla=A, resource=resource)
            e, physical, A, resource = (out[k] for k in ('ctrl', 'system', 'ancilla', 'resource'))
            if hv == 0:
                h = bb.add(opened_XGate(), q=h)
            v, h, e = be.enc__ccx(bb, v, h, e)
            if hv == 0:
                h = bb.add(opened_XGate(), q=h)
        u, d, f = be.enc__ccx(bb, u, d, f)
        d = bb.add(opened_XGate(), q=d)
        u, d, v = be.enc__ccx(bb, u, d, v)
        d = bb.add(opened_XGate(), q=d)
        ctrl, beta, u = be.enc__ccx(bb, ctrl, beta, u)
        beta = bb.add(opened_XGate(), q=beta)
        ctrl, beta, u = be.enc__ccx(bb, ctrl, beta, u)
        Abits = be.enc__split(bb, A, self.full.ancilla_bitsize)
        eta = Abits[1]
        if self.kind == 'dephasing':
            u, d = be.enc__cx(bb, u, d)
            out = bb.add_d(opened_Jump(self.full, self.kind, self.site, self.atom), enable=u, system=physical, eta=eta, resource=resource)
            u, physical, eta, resource = (out[k] for k in ('enable', 'system', 'eta', 'resource'))
        else:
            d = bb.add(opened_XGate(), q=d)
            u, d, v = be.enc__ccx(bb, u, d, v)
            d = bb.add(opened_XGate(), q=d)
            u, d, e = be.enc__ccx(bb, u, d, e)
            jump = opened_Jump(self.full, self.kind, self.site, self.atom)
            for flag, inverse in ((v, False), (e, True)):
                op = be.enc_coherent(jump).adjoint() if inverse else jump
                out = bb.add_d(op, enable=flag, system=physical, eta=eta, resource=resource)
                physical, eta, resource = (out[k] for k in ('system', 'eta', 'resource'))
                if inverse:
                    e = out['enable']
                else:
                    v = out['enable']
            u, d, e = be.enc__ccx(bb, u, d, e)
            d = bb.add(opened_XGate(), q=d)
            u, d, v = be.enc__ccx(bb, u, d, v)
            d = bb.add(opened_XGate(), q=d)
            u, d = be.enc__cx(bb, u, d)
        Abits[1] = eta
        A = be.enc__join(bb, Abits)
        ctrl, beta, u = be.enc__ccx(bb, ctrl, beta, u)
        for flag in (u, v, e):
            bb.free(flag)
        beta = bb.add(opened_Ry(-self.beta_angle), q=beta)
        A = bb.add(be.enc_coherent(closed_F(self.full)).adjoint(), ancilla=A)
        out = bb.add_d(be.model_reflection(self.full.ancilla_bitsize + 2), ctrl=ctrl, h=h, ancilla=be.enc__join(bb, [beta, f, *be.enc__split(bb, A, self.full.ancilla_bitsize)]))
        return dict(ctrl=out['ctrl'], selection=be.enc__join(bb, [out['h'], *be.enc__split(bb, out['ancilla'], self.full.ancilla_bitsize + 2)]), system=be.enc__join(bb, [d, *be.enc__split(bb, physical, self.full.system_bitsize)]), resource=resource)
