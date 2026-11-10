"""Consolidated TCL gqsp implementation.

Static, readable definitions; no embedded modules or parent-bundle imports.
Internal prefixes distinguish shared definitions merged from the development code.
"""
from __future__ import annotations
import block_encoding as be
import walk as wk
from qualtran import Bloq as opened_Bloq
from qualtran import BloqBuilder as opened_BloqBuilder
from qualtran.bloqs.basic_gates.discard import DiscardQ as opened_DiscardQ
from qualtran.bloqs.basic_gates import Hadamard as opened_Hadamard
from qualtran import QAny as opened_QAny
from qualtran import Register as opened_Register
from qualtran.bloqs.basic_gates import Ry as opened_Ry
from qualtran import Signature as opened_Signature
from qualtran.bloqs.basic_gates import XGate as opened_XGate
from qualtran.bloqs.basic_gates import ZeroState as opened_ZeroState
from functools import cached_property as opened_cached_property
from attrs import frozen as opened_frozen
import math as opened_math
import mpmath as opened_mp
from qualtran._infra.composite_bloq import _map_flat_soqs as enc__map_flat_soqs
from qualtran._infra.composite_bloq import _update_flat_soq_map as enc__update_flat_soq_map
import mpmath as phase_mp
import numpy as phase_np
import pandas as phase_pd
from qualtran import Bloq as closed_Bloq
from qualtran import BloqBuilder as closed_BloqBuilder
from qualtran import QAny as closed_QAny
from qualtran import Register as closed_Register
from qualtran.bloqs.basic_gates import Ry as closed_Ry
from qualtran import Signature as closed_Signature
from qualtran.bloqs.basic_gates import XGate as closed_XGate
from functools import cached_property as closed_cached_property
from attrs import frozen as closed_frozen
import math as closed_math
import mpmath as closed_mp
import numpy as closed_np
from qualtran.bloqs.mcmt.and_bloq import And as audit_And
from qualtran import BloqBuilder as audit_BloqBuilder
from collections import Counter as audit_Counter
from qualtran.resource_counting import GateCounts as audit_GateCounts
from qualtran.bloqs.basic_gates import Hadamard as audit_Hadamard
from pathlib import Path as audit_Path
from qualtran import QAny as audit_QAny
from qualtran import QBit as audit_QBit
from qualtran.resource_counting import QECGatesCost as audit_QECGatesCost
from qualtran import QUInt as audit_QUInt
from qualtran.bloqs.basic_gates import SGate as audit_SGate
from qualtran.bloqs.basic_gates import XGate as audit_XGate
from qualtran.bloqs.bookkeeping._bookkeeping_bloq import _BookkeepingBloq as audit__BookkeepingBloq
from qualtran.resource_counting import get_cost_value as audit_get_cost_value
import hashlib as audit_hashlib
import json as audit_json
from functools import lru_cache as audit_lru_cache
import numpy as audit_np
import pandas as audit_pd


# tcl_revised_manuscript.py -- retained active definitions

closed_POINTS = ((4, 4, 1, 3), (8, 8, 2, 7), (16, 16, 4, 15))

def closed_precision_plan(lx, ly, N, n_max, tau='1', epsilon='.001', dps=80):
    """D9/D10/G16/G19 in the uniform conserved-charge frame; no count formula."""
    with closed_mp.workdps(dps):
        t = abs(closed_mp.mpf(str(tau)))
        eps = closed_mp.mpf(str(epsilon))
        L = lx * ly
        if t <= 0 or not 0 < eps < 1:
            raise ValueError('Positive time and 0<epsilon<1 required.')
        ai = L * N * closed_mp.sqrt(n_max)
        ah = closed_mp.mpf('.2') * L * n_max
        alpha = ai + ah
        clog = lambda x: max(1, int(closed_mp.ceil(closed_mp.log(x, 2))))
        b = clog(4 * closed_mp.pi * t * (2 * ah + ai) / eps)
        B = clog(8 * t * alpha / eps)
        x = alpha * t

        def tail(k):
            if k + 1 <= x:
                return closed_mp.inf
            y = closed_mp.acosh((k + 1) / x)
            return 4 * closed_mp.exp(-(k + 1) * y + closed_mp.sqrt((k + 1) ** 2 - x * x)) / -closed_mp.expm1(-y)
        lo = max(0, int(closed_mp.floor(x)) - 1)
        hi = max(1, lo + 1)
        while tail(hi) > eps ** 2 / 32:
            hi = max(hi + 1, 2 * hi)
        while hi - lo > 1:
            middle = (hi + lo) // 2
            if tail(middle) <= eps ** 2 / 32:
                hi = middle
            else:
                lo = middle
        K = hi
        ns = 2 * K + 1
        m = max(clog(4 * closed_mp.pi * t * alpha / eps), clog(6 * closed_mp.pi * ns / eps))
        return dict(Lx=lx, Ly=ly, L=L, N=N, n_max=n_max, tau=str(t), epsilon=str(eps), b=b, B=B, K=K, m=m, n_signal=ns, walk_queries=2 * K, alpha=closed_mp.nstr(alpha, dps), alpha_local=closed_mp.nstr(ai, dps), alpha_int=closed_mp.nstr(ai, dps), alpha_hop=closed_mp.nstr(ah, dps), two_eta_bound=closed_mp.nstr(tail(K), dps), previous_degree_bound=closed_mp.nstr(tail(K - 1), dps), polynomial_budget=closed_mp.nstr(eps ** 2 / 32, dps), outer_rotation_error_bound=closed_mp.nstr(2 * t * alpha * closed_mp.pi / 2 ** (m + 1), dps), signal_rotation_error_bound=closed_mp.nstr(3 * ns * closed_mp.pi / 2 ** (m + 1), dps), energy_origin='uniform conserved-charge term removed; nu=1, Delta=0', planning_decimal_digits=dps)

def closed_build_pair(plan, modes=('direct', 'sparse', 'paired')):
    p = plan
    pair = be.model_build_pair(Lx=p['Lx'], Ly=p['Ly'], N=p['N'], n_max=p['n_max'], b=p['b'], B=p['B'], mu=closed_np.zeros((p['Ly'], p['Lx'])), modes=modes)
    for mode in modes:
        assert abs(pair[mode]['full'].alpha - float(p['alpha'])) < 1e-11 * float(p['alpha'])
    return pair

@closed_frozen(cache_hash=True)
class closed_ReplaceSharedOuterRotations(closed_Bloq):
    inner: closed_Bloq
    bits: int
    native_reference: bool = False

    @property
    def signature(self):
        return closed_Signature([*self.inner.signature, closed_Register('rotation_resource', closed_QAny(self.bits + 1))])

    @closed_cached_property
    def body(self):
        bb, soqs = closed_BloqBuilder.from_signature(self.signature)
        grad = soqs.pop('rotation_resource')
        b = wk.closed_base(self.inner)
        unchanged = b
        while isinstance(unchanged, be.enc_CoherentAdjoint):
            unchanged = wk.closed_base(unchanged.forward)
        if isinstance(b, closed_Ry):
            k = wk.rotation_rounded_word(b.angle, self.bits)
            if self.native_reference:
                result = dict(q=bb.add(closed_Ry(2 * closed_math.pi * k / 2 ** self.bits), q=soqs['q']))
            else:
                out = bb.add_d(wk.closed_GridRotation(self.bits, k, 'y'), target=soqs['q'], rotation_resource=grad)
                result = dict(q=out['target'])
                grad = out['rotation_resource']
        elif isinstance(unchanged, (be.model_AliasPreparation, be.model_LocalSelect, be.model_HoppingSelect, wk.closed_StripPreparations)):
            assert not any((isinstance(leaf, closed_Ry) for leaf in be.enc_compiled_leaf_counts(self.inner)))
            result = bb.add_d(self.inner, **soqs)
        elif be.enc__is_terminal(self.inner):
            result = bb.add_d(self.inner, **soqs)
        else:
            old = be.enc_coherent(self.inner)
            cb = old.body if isinstance(old, (be.enc_Coherent, be.enc_CoherentAdjoint)) else old.decompose_bloq()
            mapping = {}
            for inst, ins, oldout in cb.iter_bloqsoqs():
                ins = enc__map_flat_soqs(ins, mapping)
                out = bb.add_d(closed_ReplaceSharedOuterRotations(inst.bloq, self.bits, self.native_reference), **ins, rotation_resource=grad)
                grad = out.pop('rotation_resource')
                enc__update_flat_soq_map(zip(oldout, tuple((out[r.name] for r in inst.bloq.signature.rights()))), mapping)
            result = enc__map_flat_soqs(cb.final_soqs(), mapping)
        return bb.finalize(**result, rotation_resource=grad)

    def build_composite_bloq(self, bb, **soqs):
        return bb.add_d(self.body, **soqs)

@closed_frozen(cache_hash=True)
class closed_RoundedSharedWalk(closed_Bloq):
    full: be.model_ManuscriptTCL
    bits: int
    inverse: bool = False
    control_value: int = 1
    native_reference: bool = False

    @property
    def signature(self):
        return closed_ReplaceSharedOuterRotations(wk.closed_SharedWalk(self.full), self.bits).signature

    def build_composite_bloq(self, bb, ctrl, **soqs):
        if self.control_value == 0:
            ctrl = bb.add(closed_XGate(), q=ctrl)
        inner = be.enc_coherent(wk.closed_SharedWalk(self.full)).adjoint() if self.inverse else wk.closed_SharedWalk(self.full)
        replaced = closed_ReplaceSharedOuterRotations(inner, self.bits, self.native_reference)
        n = be.model_declared_workspace(self.full)
        bank = bb.allocate(dtype=closed_QAny(n))
        out = bb.add_d(be.model_BankedCircuit(replaced, n), ctrl=ctrl, **soqs, scratch_in=bank)
        bb.free(out.pop('scratch_out'))
        if self.control_value == 0:
            out['ctrl'] = bb.add(closed_XGate(), q=out['ctrl'])
        return out

@closed_frozen(cache_hash=True)
class closed_EvolutionSchedule(closed_Bloq):
    """Ordered executable generic schedule; words may be a numerical phase list.

    Empty words select a constant cost probe for every Euler factor. That
    case is explicitly a phase-independent resource bound, not a synthesized
    evolution. Word-independent adder T cost makes this bound valid for the
    numerical GQSP list at the derived precision.
    """
    full: be.model_ManuscriptTCL
    bits: int
    degree: int
    words: tuple = ()

    @property
    def signature(self):
        return closed_RoundedSharedWalk(self.full, self.bits).signature

    def build_composite_bloq(self, bb, **soqs):
        if self.words and len(self.words) != 2 * self.degree + 1:
            raise ValueError('Need 2K+1 signal gates.')
        for j in range(2 * self.degree + 1):
            if j:
                inverse = j <= self.degree
                soqs = bb.add_d(closed_RoundedSharedWalk(self.full, self.bits, inverse=inverse, control_value=1 if inverse else 0), **soqs)
            words = self.words[j] if self.words else ((1 << self.bits) - 1,) * 3
            for axis, k in zip(('z', 'y', 'z'), words):
                out = bb.add_d(wk.closed_GridRotation(self.bits, int(k), axis), target=soqs['ctrl'], rotation_resource=soqs['rotation_resource'])
                soqs['ctrl'], soqs['rotation_resource'] = (out['target'], out['rotation_resource'])
        return soqs


# tcl_appendix_h.py -- retained active definitions

def opened_plan(lx=4, ly=4, N=1, n_max=3, *, T='10', mu_min='.5', epsilon_g='.1', failure_probability='.01', reported_times=1, kappa='.1', gamma1='.1', gamma_phi='0', dps=90):
    """IXE 126--129, sampling-only F3 allowance, and H14--H15.

    The example uses Eq.103 at channel tolerance e/3, and e/3 for implementation, with exact
    preparation. The dilation remainder is NOT certified by this allocation.
    N_step is NOT claimed sufficient for the complete channel error.
    """
    if min(lx, ly) < 2 or any((v & v - 1 for v in (lx, ly, N))):
        raise ValueError('Power-of-two sides >=2 and positive power-of-two N required.')
    if n_max < 2 or reported_times < 1:
        raise ValueError('This photon-statistics plan requires cutoff >=2 and at least one time.')
    with opened_mp.workdps(dps):
        t, mu, eg, delta = map(lambda s: opened_mp.mpf(str(s)), (T, mu_min, epsilon_g, failure_probability))
        kap, gam, phi = map(lambda s: opened_mp.mpf(str(s)), (kappa, gamma1, gamma_phi))
        if min(t, mu, eg) <= 0 or not 0 < delta < 1 or min(kap, gam, phi) < 0:
            raise ValueError('Invalid time, lower occupation bound, target, failure probability or rates.')
        if eg * mu > opened_mp.mpf(n_max - 1) / 8:
            raise ValueError('Eq. 126 requires epsilon_g*mu_min <= (nmax-1)/8.')
        L = lx * ly
        en = mu * opened_mp.sqrt(eg * mu / (16 * (n_max - 1)))
        ep = opened_mp.mpf(3) / 4 * eg * mu ** 2
        e = min(en / (2 * n_max), ep / (2 * n_max * (n_max - 1)))
        shots = int(opened_mp.ceil(2 * max(n_max ** 2 / en ** 2, (n_max * (n_max - 1)) ** 2 / ep ** 2) * opened_mp.log(4 * L * reported_times / delta)))
        weights = (L * n_max * kap, L * N * gam, L * N * phi / 2)
        lam = sum(weights)
        if lam <= 0:
            raise ValueError('At least one dissipative channel required.')
        steps = max(1, int(opened_mp.ceil(32 * t * t * lam ** 2 / (e / 3))))
        dt = t / steps
        es = e / (6 * steps)
        ai = L * N * opened_mp.sqrt(n_max)
        ah = opened_mp.mpf('.2') * L * n_max
        alpha = ai + ah
        x = alpha * dt + opened_mp.sqrt(lam * dt)
        clog = lambda v: max(1, int(opened_mp.ceil(opened_mp.log(v, 2))))
        Phi = dt * (2 * ah + ai) + (opened_mp.sqrt(lam * dt) if kap else 0)
        b = clog(4 * opened_mp.pi * Phi / es)
        B = clog(8 * alpha * dt / es)

        def tail(k):
            if k + 1 <= x:
                return opened_mp.inf
            y = opened_mp.acosh((k + 1) / x)
            return 4 * opened_mp.exp(-(k + 1) * y + opened_mp.sqrt((k + 1) ** 2 - x * x)) / -opened_mp.expm1(-y)
        lo = max(-1, int(opened_mp.floor(x)) - 1)
        hi = max(0, lo + 1)
        while tail(hi) > es ** 2 / 32:
            hi = max(hi + 1, 2 * hi)
        while hi - lo > 1:
            mid = (hi + lo) // 2
            if tail(mid) <= es ** 2 / 32:
                hi = mid
            else:
                lo = mid
        K = hi
        nk = 2 * K + 1
        m = max(clog(8 * opened_mp.pi * x / es), clog(6 * opened_mp.pi * nk / es))
        f = lambda v: opened_mp.nstr(v, dps)
        return dict(Lx=lx, Ly=ly, L=L, N=N, n_max=n_max, T=f(t), mu_min=f(mu), epsilon_g=f(eg), failure_probability=f(delta), reported_times=reported_times, kappa=f(kap), gamma1=f(gam), gamma_phi=f(phi), epsilon_n=f(en), epsilon_phi=f(ep), state_tolerance=f(e), N_shot=shots, N_step=steps, dt=f(dt), epsilon_step=f(es), alpha=f(alpha), alpha_int=f(ai), alpha_hop=f(ah), lambda_D=f(lam), x_step=f(x), probabilities={k: float(v / lam) for k, v in zip(wk.opened_KINDS, weights)}, b=b, B=B, K=K, m=m, n_signal=nk, walk_queries=2 * K, two_eta_bound=f(tail(K)), previous_degree_bound=f(tail(K - 1)), polynomial_budget=f(es ** 2 / 32), sampling_error_bound=f(8 * t * lam ** 2 * dt), sampling_budget=f(e / 12), implementation_error_bound=f(2 * steps * es), implementation_budget=f(e / 3), energy_frame='uniform conserved-charge term removed; nu=1, Delta=0, g=1, t=0.05', full_channel_error_certified=False, dilation_remainder_budgeted=False, planning_decimal_digits=dps)

def opened_build_full(p):
    return closed_build_pair(p, modes=('direct',))['direct']['full']

@opened_frozen(cache_hash=True)
class opened_ReplaceRotations(opened_Bloq):
    inner: opened_Bloq
    bits: int
    angle_words: tuple
    native_reference: bool = False

    @property
    def signature(self):
        return opened_Signature([*self.inner.signature, opened_Register('rotation_resource', opened_QAny(self.bits + 1))])

    @opened_cached_property
    def body(self):
        bb, soqs = opened_BloqBuilder.from_signature(self.signature)
        grad = soqs.pop('rotation_resource')
        b = wk.closed_base(self.inner)
        unchanged = b
        while isinstance(unchanged, be.enc_CoherentAdjoint):
            unchanged = wk.closed_base(unchanged.forward)
        if isinstance(b, opened_Ry):
            word = dict(self.angle_words)[b.angle]
            if self.native_reference:
                result = dict(q=bb.add(opened_Ry(2 * opened_math.pi * word / 2 ** self.bits), q=soqs['q']))
            else:
                out = bb.add_d(wk.closed_GridRotation(self.bits, word, 'y'), target=soqs['q'], rotation_resource=grad)
                result = dict(q=out['target'])
                grad = out['rotation_resource']
        elif isinstance(unchanged, (be.model_AliasPreparation, wk.closed_StripPreparations, wk.opened_Jump)):
            assert not any((isinstance(x, opened_Ry) for x in be.enc_compiled_leaf_counts(self.inner)))
            result = bb.add_d(self.inner, **soqs)
        elif be.enc__is_terminal(self.inner):
            result = bb.add_d(self.inner, **soqs)
        else:
            co = be.enc_coherent(self.inner)
            cb = co.body
            mapping = {}
            for inst, ins, oldout in cb.iter_bloqsoqs():
                ins = enc__map_flat_soqs(ins, mapping)
                out = bb.add_d(opened_ReplaceRotations(inst.bloq, self.bits, self.angle_words, self.native_reference), **ins, rotation_resource=grad)
                grad = out.pop('rotation_resource')
                enc__update_flat_soq_map(zip(oldout, tuple((out[r.name] for r in inst.bloq.signature.rights()))), mapping)
            result = enc__map_flat_soqs(cb.final_soqs(), mapping)
        return bb.finalize(**result, rotation_resource=grad)

    def build_composite_bloq(self, bb, **soqs):
        return bb.add_d(self.body, **soqs)

def opened_outer_words(walk, bits):
    with opened_mp.workdps(100):
        loc = walk.full.local
        ai = opened_mp.mpf(loc.L) * loc.N * opened_mp.sqrt(loc.n_max)
        ah = opened_mp.mpf('.2') * loc.L * loc.n_max
        angles = (2 * opened_mp.acos(opened_mp.sqrt(ai / (ai + ah))), 2 * opened_mp.acos(opened_mp.sqrt(opened_mp.sqrt(opened_mp.mpf(str(walk.dt))) * (ai + ah) / (opened_mp.sqrt(opened_mp.mpf(str(walk.dt))) * (ai + ah) + opened_mp.sqrt(opened_mp.mpf(str(walk.lambda_D)))))))
        native = (2 * opened_math.acos(opened_math.sqrt(walk.full.alpha_local / walk.full.alpha)), walk.beta_angle)
        return tuple(((sign * float(a), sign * int(opened_mp.floor(theta * 2 ** bits / (2 * opened_mp.pi) + opened_mp.mpf('.5')))) for a, theta in zip(native, angles) for sign in (1, -1)))

def opened_scratch_width(full, m):
    w = be.model_declared_workspace(full)
    return max(full.ancilla_bitsize + 2, w + 3, 2 * m - 1)

@opened_frozen(cache_hash=True)
class opened_RoundedWalk(opened_Bloq):
    walk: wk.opened_Walk
    bits: int
    inverse: bool = False
    control_value: int = 1
    native_reference: bool = False

    @property
    def signature(self):
        return opened_ReplaceRotations(self.walk, self.bits, opened_outer_words(self.walk, self.bits)).signature

    def build_composite_bloq(self, bb, ctrl, **soqs):
        if self.control_value == 0:
            ctrl = bb.add(opened_XGate(), q=ctrl)
        op = be.enc_coherent(self.walk).adjoint() if self.inverse else self.walk
        replaced = opened_ReplaceRotations(op, self.bits, opened_outer_words(self.walk, self.bits), self.native_reference)
        n = opened_scratch_width(self.walk.full, self.bits)
        bank = bb.allocate(dtype=opened_QAny(n))
        out = bb.add_d(be.model_BankedCircuit(replaced, n), ctrl=ctrl, **soqs, scratch_in=bank)
        bb.free(out.pop('scratch_out'))
        if self.control_value == 0:
            out['ctrl'] = bb.add(opened_XGate(), q=out['ctrl'])
        return out

@opened_frozen(cache_hash=True)
class opened_EvolutionSchedule(opened_Bloq):
    """Ordered 2K queries and 3(2K+1) generic fixed-grid axial rotations."""
    walk: wk.opened_Walk
    bits: int
    degree: int

    @property
    def signature(self):
        return opened_RoundedWalk(self.walk, self.bits).signature

    def build_composite_bloq(self, bb, **soqs):
        for j in range(2 * self.degree + 1):
            if j:
                inverse = j <= self.degree
                soqs = bb.add_d(opened_RoundedWalk(self.walk, self.bits, inverse, 1 if inverse else 0), **soqs)
            for axis in ('z', 'y', 'z'):
                out = bb.add_d(wk.closed_GridRotation(self.bits, (1 << self.bits) - 1, axis), target=soqs['ctrl'], rotation_resource=soqs['rotation_resource'])
                soqs['ctrl'], soqs['rotation_resource'] = (out['target'], out['rotation_resource'])
        return soqs

@opened_frozen(cache_hash=True)
class opened_StepChannel(opened_Bloq):
    """One sampled timestep with fresh signals; catalysts persist, no postselection.

    Zero-T initialization/discard leaves are retained for separate auditing.
    Numerical GQSP phases remain unspecified in this generic cost schedule.
    """
    walk: wk.opened_Walk
    bits: int
    degree: int

    @property
    def signature(self):
        return opened_Signature.build(physical=self.walk.full.system_bitsize, resource=self.walk.full.resource_bitsize, rotation_resource=self.bits + 1)

    def build_composite_bloq(self, bb, physical, resource, rotation_resource):
        full = self.walk.full
        signal = bb.add(opened_ZeroState())
        d = bb.add(opened_ZeroState())
        A = [bb.add(opened_ZeroState()) for _ in range(full.ancilla_bitsize + 3)]
        A[0] = bb.add(opened_Hadamard(), q=A[0])
        out = bb.add_d(opened_EvolutionSchedule(self.walk, self.bits, self.degree), ctrl=signal, selection=be.enc__join(bb, A), system=be.enc__join(bb, [d, *be.enc__split(bb, physical, full.system_bitsize)]), resource=resource, rotation_resource=rotation_resource)
        bb.add(opened_DiscardQ(), q=out['ctrl'])
        for bit in be.enc__split(bb, out['selection'], full.ancilla_bitsize + 3):
            bb.add(opened_DiscardQ(), q=bit)
        d, *data = be.enc__split(bb, out['system'], full.system_bitsize + 1)
        bb.add(opened_DiscardQ(), q=d)
        return dict(physical=be.enc__join(bb, data), resource=out['resource'], rotation_resource=out['rotation_resource'])


# tcl_gqsp_numerical_first_case.py -- retained active definitions

def phase_rz(theta):
    return phase_np.diag([phase_np.exp(-0.5j * theta), phase_np.exp(0.5j * theta)])

def phase_ry(theta):
    c, s = (phase_np.cos(theta / 2), phase_np.sin(theta / 2))
    return phase_np.array([[c, -s], [s, c]], dtype=complex)

def phase_polynomial_data(plan):
    """80-digit Bessel coefficients; bound the omitted numerical tail as in G19.

    The denominator uses the summed eta_K plus a certified residual allowance,
    rather than substituting G19's much larger displayed upper bound for eta_K.
    Float conversion and subsequent phase synthesis are checked separately.
    """
    with phase_mp.workdps(80):
        K = plan['K']
        x = phase_mp.mpf(plan['alpha']) * phase_mp.mpf(plan['tau'])
        stop = K + 1

        def residual(n):
            y = phase_mp.acosh(n / x)
            return 2 * phase_mp.exp(-n * y + phase_mp.sqrt(n * n - x * x)) / -phase_mp.expm1(-y)
        while residual(stop) > phase_mp.mpf('1e-65'):
            stop += 1
        tail_remainder = residual(stop)
        eta_sum = 2 * phase_mp.fsum((abs(phase_mp.besselj(j, x)) for j in range(K + 1, stop)))
        eta = eta_sum + tail_remainder
        P = phase_np.array([complex((-1j) ** j * phase_mp.besselj(j, x) / (1 + eta)) for j in range(-K, K + 1)])
        metadata = dict(eta_sum=phase_mp.nstr(eta_sum, 75), eta_used=phase_mp.nstr(eta, 75), remaining_tail_allowance=phase_mp.nstr(tail_remainder, 75), summed_tail_orders=[K + 1, stop - 1], decimal_digits=80, two_eta_used=phase_mp.nstr(2 * eta, 75))
    return (P, metadata)

def phase_complementary_polynomial(P, fft_size=1 << 17):
    """Numerical Fejer--Riesz factorization through the analytic log on a grid.

    Reverse-conjugate the outer factor to choose the inside-root factor. This
    orientation makes downward phase extraction stable for this instance.
    The finite FFT is NOT itself an error certificate: polynomial coefficients
    reconstructed from the final rotations supply the independent residual.
    """
    if fft_size < 4 * len(P) or fft_size & fft_size - 1:
        raise ValueError('Use a power-of-two FFT grid larger than the polynomial.')
    values = phase_np.fft.ifft(P, n=fft_size) * fft_size
    gap = 1 - abs(values) ** 2
    if gap.min() <= 0:
        raise ArithmeticError('Polynomial is not strictly contractive at this precision.')
    fourier = phase_np.fft.fft(phase_np.log(gap)) / fft_size
    analytic = phase_np.zeros(fft_size, complex)
    analytic[0] = fourier[0] / 2
    analytic[1:fft_size // 2] = fourier[1:fft_size // 2]
    q_values = phase_np.exp(phase_np.fft.ifft(analytic) * fft_size)
    outer = (phase_np.fft.fft(q_values) / fft_size)[:len(P)]
    Q = outer[::-1].conj()
    actual = phase_np.fft.ifft(Q, n=fft_size) * fft_size
    return (Q, dict(fft_size=fft_size, minimum_contractive_gap=float(gap.min()), sampled_complement_residual=float(phase_np.max(abs(abs(values) ** 2 + abs(actual) ** 2 - 1)))))

def phase_phase_factors(P, Q):
    """Qualtran's downward recurrence without its fixed 1e-10 phase cutoff.

    Tiny nonzero coefficients must retain their arguments. atan2(abs(b),abs(a))
    avoids divisions; arg(a)-arg(b) avoids underflow in a*conj(b).
    """
    if len(P) != len(Q):
        raise ValueError('Complementary polynomials must have equal length.')
    S = phase_np.array([P, Q], dtype=complex)
    theta = phase_np.zeros(len(P))
    phi = phase_np.zeros(len(P))
    lambd = 0.0
    for d in reversed(range(len(P))):
        a, b = S[:, d]
        theta[d] = phase_np.arctan2(abs(b), abs(a))
        phi[d] = phase_np.angle(a) - phase_np.angle(b)
        if d == 0:
            lambd = phase_np.angle(b)
        else:
            R = phase_raw_rotation(theta[d], phi[d], 0)
            S = R.conj().T @ S
            S = phase_np.array([S[0, 1:d + 1], S[1, :d]])
    return (theta, phi, float(lambd))

def phase_raw_rotation(theta, phi, lambd):
    """The U(2) matrix used by installed Qualtran's SU2RotationGate."""
    c, s = (phase_np.cos(theta), phase_np.sin(theta))
    return phase_np.array([[phase_np.exp(1j * (lambd + phi)) * c, phase_np.exp(1j * phi) * s], [phase_np.exp(1j * lambd) * s, -c]])

def phase_su2_euler_sequence(theta, phi, lambd, bits):
    raw = phase_np.array([phase_raw_rotation(t, p, lambd if j == 0 else 0) for j, (t, p) in enumerate(zip(theta, phi))])
    delta = phase_np.angle(phase_np.linalg.det(raw)) / 2
    V = raw * phase_np.exp(-1j * delta[:, None, None])
    gamma = float(delta.sum())
    V[-1] = phase_rz(-2 * gamma) @ V[-1]
    rows = []
    rounded = []
    for j, v in enumerate(V):
        a, c = (phase_np.angle(v[0, 0]), phase_np.angle(v[1, 0]))
        angles = [-c - a, 2 * phase_np.arctan2(abs(v[1, 0]), abs(v[0, 0])), c - a]
        words = [wk.rotation_rounded_word(float(angle), bits) for angle in angles]
        grid = 2 * phase_np.pi * phase_np.array(words) / 2 ** bits
        restored = phase_rz(angles[2]) @ phase_ry(angles[1]) @ phase_rz(angles[0])
        if phase_np.max(abs(restored - v)) > 1e-12:
            raise ArithmeticError('Euler conversion lost a global phase.')
        rounded.append(phase_rz(grid[2]) @ phase_ry(grid[1]) @ phase_rz(grid[0]))
        rows.append(dict(j=j, theta=float(theta[j]), phi=float(phi[j]), lambd=float(lambd if j == 0 else 0), z_first=angles[0], y_middle=angles[1], z_last=angles[2], k_z_first=words[0], k_y_middle=words[1], k_z_last=words[2], rounding_max_radians=float(phase_np.max(abs(phase_np.array(angles) - grid)))))
    return (raw, V, phase_np.array(rounded), phase_pd.DataFrame(rows), gamma)

def phase_spectral_response(rotations, z, K):
    """Evaluate the actual ordered 2x2 signal sequence at walk eigenvalue z."""
    z = phase_np.atleast_1d(z).astype(complex)
    S = phase_np.broadcast_to(rotations[0], (len(z), 2, 2)).copy()
    for j, rotation in enumerate(rotations[1:], 1):
        if j <= K:
            S[:, 1, :] *= z.conj()[:, None]
        else:
            S[:, 0, :] *= z[:, None]
        S = phase_np.einsum('ij,njk->nik', rotation, S)
    return S

def phase_reconstruct_polynomials(rotations):
    """Recover P,Q by coefficient arithmetic, independently of sampling z."""
    S = phase_np.asarray(rotations[0][:, :1], dtype=complex)
    for rotation in rotations[1:]:
        lifted = phase_np.zeros((2, S.shape[1] + 1), complex)
        lifted[0, 1:] = S[0]
        lifted[1, :-1] = S[1]
        S = rotation @ lifted
    return S

def phase_build_phases(plan=None, fft_size=1 << 17):
    plan = wk.rotation_precision_plan(4, 4, 1, 3) if plan is None else plan
    P, tail = phase_polynomial_data(plan)
    Q, factor = phase_complementary_polynomial(P, fft_size)
    theta, phi, lambd = phase_phase_factors(P, Q)
    raw, rotations, rounded, phases, gamma = phase_su2_euler_sequence(theta, phi, lambd, plan['m'])
    reconstructed = phase_reconstruct_polynomials(rotations)
    residual = float(phase_np.sum(abs(reconstructed[0] - P)))
    if residual > 1e-10:
        raise ArithmeticError(f'Phase reconstruction failed: coefficient residual={residual}.')
    return dict(plan=plan, tail=tail, factor=factor, P=P, Q=Q, raw_rotations=raw, rotations=rotations, rounded_rotations=rounded, phases=phases, aggregate_removed_U2_phase=gamma, reconstruction_coefficient_l1=residual)

def phase_validate_spectral(data, grid_size=16385):
    p = data['plan']
    K = p['K']
    x = float(p['alpha']) * float(p['tau'])
    angle = phase_np.linspace(0, 2 * phase_np.pi, grid_size)
    z = phase_np.exp(1j * angle)
    target = phase_np.exp(-1j * x * phase_np.cos(angle))
    ideal = phase_spectral_response(data['rotations'], z, K)
    rounded = phase_spectral_response(data['rounded_rotations'], z, K)
    polynomial = phase_np.polynomial.polynomial.polyval(z, data['P']) / z ** K
    err_ideal = abs(ideal[:, 0, 0] - target)
    err_rounded = abs(rounded[:, 0, 0] - target)
    state_ideal = phase_np.sqrt(err_ideal ** 2 + abs(ideal[:, 1, 0]) ** 2)
    state_rounded = phase_np.sqrt(err_rounded ** 2 + abs(rounded[:, 1, 0]) ** 2)
    difference = float(phase_np.linalg.norm(ideal - rounded, ord=2, axis=(1, 2)).max())
    rounding_bound = 3 * p['n_signal'] * phase_np.pi / 2 ** (p['m'] + 1)
    coefficient_residual = data['reconstruction_coefficient_l1']
    poly_bound = float(data['tail']['two_eta_used']) + coefficient_residual
    initialized_bound = phase_np.sqrt(2 * poly_bound) + rounding_bound
    assert difference <= rounding_bound + 1e-12
    assert err_ideal.max() <= poly_bound + 1e-12
    assert state_rounded.max() <= initialized_bound + 1e-12
    certificate = dict(grid_points=grid_size, unrounded_polynomial_reconstruction_max=float(abs(ideal[:, 0, 0] - polynomial).max()), reconstruction_coefficient_l1=coefficient_residual, unrounded_evolution_block_max=float(err_ideal.max()), rounded_evolution_block_max=float(err_rounded.max()), unrounded_initialized_state_max=float(state_ideal.max()), rounded_initialized_state_max=float(state_rounded.max()), sampled_full_signal_operator_rounding_error=difference, analytic_signal_rounding_bound=rounding_bound, numerical_uniform_polynomial_error_bound=poly_bound, numerical_uniform_initialized_signal_error_bound=float(initialized_bound), numerical_coefficient_bounds_are_interval_certified=False, complete_lattice_statevector_simulated=False, Hamiltonian_encoding_error_included_in_signal_errors=False, outer_selection_rotation_error_included_in_signal_errors=False)
    frame = phase_pd.DataFrame(dict(walk_phase=angle, normalized_energy=phase_np.cos(angle), target_real=target.real, target_imag=target.imag, unrounded_block_error=err_ideal, rounded_block_error=err_rounded, unrounded_initialized_error=state_ideal, rounded_initialized_error=state_rounded))
    return (certificate, frame)


# tcl_table_II_qualtran_audit.py -- retained active definitions

audit_ROOT = audit_Path(__file__).resolve().parent

audit_OUT = audit_ROOT / 'output/tcl_manuscript_explicit'

audit_CASES = ((4, 4, 1, 3), (8, 8, 2, 7), (16, 16, 4, 15))

audit__split, audit__join, audit__cx, audit__ccx = (be.enc__split, be.enc__join, be.enc__cx, be.enc__ccx)

def audit__and(bb, a, b, target=None, cvs=(1, 1)):
    gate = audit_And(*cvs, uncompute=target is not None)
    args = dict(ctrl=audit_np.array([a, b], dtype=object))
    if target is not None:
        args['target'] = target
    out = bb.add_d(gate, **args)
    return (*out['ctrl'], out.get('target'))

def audit__unary(bb, address, leaf, enable=None):
    """Measurement-erased address flags; callbacks preserve their flag/control."""
    address = list(address)

    def visit(depth, flag, offset):
        if depth == len(address):
            box = [flag]
            leaf(offset, box)
            return box[0]
        flag, address[depth], child = audit__and(bb, flag, address[depth], cvs=(1, 0))
        child = visit(depth + 1, child, offset)
        flag, child = audit__cx(bb, flag, child)
        child = visit(depth + 1, child, offset + (1 << len(address) - depth - 1))
        flag, address[depth], _ = audit__and(bb, flag, address[depth], child)
        return flag
    if enable is not None:
        enable = visit(0, enable, 0)
    else:
        address[0] = bb.add(audit_XGate(), q=address[0])
        address[0] = visit(1, address[0], 0)
        address[0] = bb.add(audit_XGate(), q=address[0])
        address[0] = visit(1, address[0], 1 << len(address) - 1)
    return (address, enable)

@audit_lru_cache(None)
def audit_lookup(words, width):
    """General enabled unary QROM; retain the ledger's unoptimized root."""
    D = len(words)
    d = (D - 1).bit_length()
    assert D == 1 << d and D >= 2
    bb = audit_BloqBuilder()
    a = audit__split(bb, bb.add_register('address', audit_QUInt(d)), d)
    t = audit__split(bb, bb.add_register('target', audit_QAny(width)), width)
    root = bb.add(audit_XGate(), q=bb.allocate(dtype=audit_QBit()))

    def leaf(k, flag):
        for bit in range(width):
            if words[k] >> width - bit - 1 & 1:
                flag[0], t[bit] = audit__cx(bb, flag[0], t[bit])
    a, root = audit__unary(bb, a, leaf, root)
    root = bb.add(audit_XGate(), q=root)
    bb.free(root)
    return bb.finalize(address=audit__join(bb, a, audit_QUInt(d)), target=audit__join(bb, t))

@audit_lru_cache(None)
def audit_indexed_load(L, q):
    """Hadamard-conjugated unary phase read, as specified in the manuscript.

    For each word bit H(target) turns an addressed CZ into an XOR read.
    Use the target bit itself as the unary root: leaf gates are Clifford CZs,
    not CCXs with two quantum controls. Reuse O(log L) flag workspace across
    word bits. This construction tests the ledger's actual LOAD convention.
    """
    d = (L - 1).bit_length()
    assert L == 1 << d
    bb = audit_BloqBuilder()
    a = audit__split(bb, bb.add_register('address', audit_QUInt(d)), d)
    data = audit__split(bb, bb.add_register('data', audit_QAny(L * q)), L * q)
    word = audit__split(bb, bb.add_register('word', audit_QAny(q)), q)
    for bit in range(q):
        word[bit] = bb.add(audit_Hadamard(), q=word[bit])

        def leaf(site, flag):
            flag[0], data[site * q + bit] = be.model__cz(bb, flag[0], data[site * q + bit])
        a, word[bit] = audit__unary(bb, a, leaf, word[bit])
        word[bit] = bb.add(audit_Hadamard(), q=word[bit])
    return bb.finalize(address=audit__join(bb, a, audit_QUInt(d)), data=audit__join(bb, data), word=audit__join(bb, word))

def audit__inc(bb, enable, bits, decrement=False):
    x = list(reversed(bits))
    flags = [enable]
    for k in range(1, len(x) - 1):
        flags[k - 1], x[k - 1], carry = audit__and(bb, flags[k - 1], x[k - 1], cvs=(1, 0 if decrement else 1))
        flags.append(carry)
    if len(x) >= 2:
        if decrement:
            x[-2] = bb.add(audit_XGate(), q=x[-2])
        flags[-1], x[-2], x[-1] = audit__ccx(bb, flags[-1], x[-2], x[-1])
        if decrement:
            x[-2] = bb.add(audit_XGate(), q=x[-2])
    for k in reversed(range(1, len(x) - 1)):
        flags[k], x[k] = audit__cx(bb, flags[k], x[k])
        flags[k - 1], x[k - 1], _ = audit__and(bb, flags[k - 1], x[k - 1], flags[k], cvs=(1, 0 if decrement else 1))
    enable, x[0] = audit__cx(bb, flags[0], x[0])
    return (enable, list(reversed(x)))

@audit_lru_cache(None)
def audit_selected_shift(L, q, decrement=False):
    d = (L - 1).bit_length()
    bb = audit_BloqBuilder()
    enable = bb.add_register('enable', audit_QBit())
    a = audit__split(bb, bb.add_register('address', audit_QUInt(d)), d)
    data = audit__split(bb, bb.add_register('data', audit_QAny(L * q)), L * q)

    def leaf(site, flag):
        flag[0], data[site * q:(site + 1) * q] = audit__inc(bb, flag[0], data[site * q:(site + 1) * q], decrement)
    a, enable = audit__unary(bb, a, leaf, enable)
    return bb.finalize(enable=enable, address=audit__join(bb, a, audit_QUInt(d)), data=audit__join(bb, data))

@audit_lru_cache(None)
def audit_destination(lx, ly):
    nx, ny = (lx.bit_length() - 1, ly.bit_length() - 1)
    bb = audit_BloqBuilder()
    src = audit__split(bb, bb.add_register('source', audit_QUInt(nx + ny)), nx + ny)
    axis = bb.add_register('axis', audit_QBit())
    sign = bb.add_register('sign', audit_QBit())
    dst = audit__split(bb, bb.add_register('destination', audit_QUInt(nx + ny)), nx + ny)
    for k in range(nx + ny):
        src[k], dst[k] = audit__cx(bb, src[k], dst[k])
    for k in range(nx + ny):
        sign, dst[k] = audit__cx(bb, sign, dst[k])
    axis = bb.add(audit_XGate(), q=axis)
    axis, dst[ny:] = audit__inc(bb, axis, dst[ny:])
    axis = bb.add(audit_XGate(), q=axis)
    axis, dst[:ny] = audit__inc(bb, axis, dst[:ny])
    for k in range(nx + ny):
        sign, dst[k] = audit__cx(bb, sign, dst[k])
    return bb.finalize(source=audit__join(bb, src, audit_QUInt(nx + ny)), axis=axis, sign=sign, destination=audit__join(bb, dst, audit_QUInt(nx + ny)))

@audit_lru_cache(None)
def audit_sampled_delta_write(L, q, decrement=False):
    """XOR old XOR (old +/- 1) into an addressed word, using a retained sample.

    Prefixes depend on the sampled original occupation, not the target bank.
    Thus this is a true XOR write on arbitrary target data. It is the loaded-
    occupation update used inside the Table II row oracles.
    """
    d = L.bit_length() - 1
    bb = audit_BloqBuilder()
    e = bb.add_register('enable', audit_QBit())
    address = audit__split(bb, bb.add_register('address', audit_QUInt(d)), d)
    sample = list(reversed(audit__split(bb, bb.add_register('sample', audit_QAny(q)), q)))
    data = audit__split(bb, bb.add_register('data', audit_QAny(L * q)), L * q)
    prefixes = [e]
    for bit in range(1, q):
        prefixes[bit - 1], sample[bit - 1], flag = audit__and(bb, prefixes[bit - 1], sample[bit - 1], cvs=(1, 0 if decrement else 1))
        prefixes.append(flag)
    for bit in range(q):

        def leaf(site, flag):
            pos = site * q + q - 1 - bit
            flag[0], data[pos] = audit__cx(bb, flag[0], data[pos])
        address, prefixes[bit] = audit__unary(bb, address, leaf, prefixes[bit])
    for bit in reversed(range(1, q)):
        prefixes[bit - 1], sample[bit - 1], _ = audit__and(bb, prefixes[bit - 1], sample[bit - 1], prefixes[bit], cvs=(1, 0 if decrement else 1))
    return bb.finalize(enable=prefixes[0], address=audit__join(bb, address, audit_QUInt(d)), sample=audit__join(bb, list(reversed(sample))), data=audit__join(bb, data))

@audit_lru_cache(None)
def audit_row_oracle(lx, ly, q, inverse_permutation=False):
    """Full XOR oracle |x,y> -> |x,y XOR S^(+/-1)(x)>.

    Compute destination and two input occupations, copy the input bank with
    CNOTs, then use those samples to write the decrement/increment bit deltas.
    Input occupations remain unchanged, so their LOAD adjoints erase samples.
    The inverse-permutation oracle increments the source and decrements the
    destination. No external branch control is included (Table II scope).
    """
    L = lx * ly
    d = L.bit_length() - 1
    bb = audit_BloqBuilder()
    source = bb.add_register('source', audit_QUInt(d))
    axis = bb.add_register('axis', audit_QBit())
    sign = bb.add_register('sign', audit_QBit())
    data = bb.add_register('data', audit_QAny(L * q))
    auxiliary = bb.add_register('auxiliary', audit_QAny(L * q))
    dest = bb.allocate(dtype=audit_QUInt(d))
    o = bb.add_d(audit_destination(lx, ly), source=source, axis=axis, sign=sign, destination=dest)
    source, axis, sign, dest = (o[k] for k in ('source', 'axis', 'sign', 'destination'))
    samples = [bb.allocate(dtype=audit_QAny(q)) for _ in range(2)]
    addresses = [source, dest]
    for k in range(2):
        o = bb.add_d(audit_indexed_load(L, q), address=addresses[k], data=data, word=samples[k])
        addresses[k], data, samples[k] = (o[x] for x in ('address', 'data', 'word'))
    x, y = (audit__split(bb, data, L * q), audit__split(bb, auxiliary, L * q))
    for k in range(L * q):
        x[k], y[k] = audit__cx(bb, x[k], y[k])
    data, auxiliary = (audit__join(bb, x), audit__join(bb, y))
    e = bb.add(audit_XGate(), q=bb.allocate(dtype=audit_QBit()))
    for k in range(2):
        decrement = (k == 0) != inverse_permutation
        o = bb.add_d(audit_sampled_delta_write(L, q, decrement), enable=e, address=addresses[k], sample=samples[k], data=auxiliary)
        e, addresses[k], samples[k], auxiliary = (o[x] for x in ('enable', 'address', 'sample', 'data'))
    e = bb.add(audit_XGate(), q=e)
    bb.free(e)
    for k in reversed(range(2)):
        o = bb.add_d(audit_indexed_load(L, q), address=addresses[k], data=data, word=samples[k])
        addresses[k], data, samples[k] = (o[x] for x in ('address', 'data', 'word'))
        bb.free(samples[k])
    o = bb.add_d(be.enc__fast_adjoint(audit_destination(lx, ly)), source=addresses[0], axis=axis, sign=sign, destination=addresses[1])
    bb.free(o['destination'])
    return bb.finalize(source=o['source'], axis=o['axis'], sign=o['sign'], data=data, auxiliary=auxiliary)

def audit__majority(bb, a, b, c, target=None):
    """Carry majority = c XOR ((a XOR c) AND (b XOR c))."""
    if target is not None:
        c, target = audit__cx(bb, c, target)
    c, a = audit__cx(bb, c, a)
    c, b = audit__cx(bb, c, b)
    a, b, out = audit__and(bb, a, b, target)
    c, b = audit__cx(bb, c, b)
    c, a = audit__cx(bb, c, a)
    if target is None:
        c, out = audit__cx(bb, c, out)
    return (a, b, c, out)

@audit_lru_cache(None)
def audit_compare_ge(B):
    """Ripple r>=threshold XOR flag; one temporary AND per carry bit."""
    bb = audit_BloqBuilder()
    r = list(reversed(audit__split(bb, bb.add_register('random', audit_QAny(B)), B)))
    t = list(reversed(audit__split(bb, bb.add_register('threshold', audit_QAny(B)), B)))
    flag = bb.add_register('flag', audit_QBit())
    carries = []
    r[0], t[0], c = audit__and(bb, r[0], t[0], cvs=(0, 1))
    c = bb.add(audit_XGate(), q=c)
    carries.append(c)
    for i in range(1, B):
        t[i] = bb.add(audit_XGate(), q=t[i])
        r[i], t[i], carries[i - 1], c = audit__majority(bb, r[i], t[i], carries[i - 1])
        t[i] = bb.add(audit_XGate(), q=t[i])
        carries.append(c)
    carries[-1], flag = audit__cx(bb, carries[-1], flag)
    for i in reversed(range(1, B)):
        t[i] = bb.add(audit_XGate(), q=t[i])
        r[i], t[i], carries[i - 1], _ = audit__majority(bb, r[i], t[i], carries[i - 1], carries[i])
        t[i] = bb.add(audit_XGate(), q=t[i])
    carries[0] = bb.add(audit_XGate(), q=carries[0])
    r[0], t[0], _ = audit__and(bb, r[0], t[0], carries[0], cvs=(0, 1))
    return bb.finalize(random=audit__join(bb, list(reversed(r))), threshold=audit__join(bb, list(reversed(t))), flag=flag)

@audit_lru_cache(None)
def audit_alias_preparation(weights, B):
    original = be.model_AliasPreparation(weights, B)
    d = original.d
    bb, soqs = audit_BloqBuilder.from_signature(original.signature)
    a = audit__split(bb, soqs['selection'], d)
    rnd = audit__split(bb, soqs['random'], B)
    a = [bb.add(audit_Hadamard(), q=w) for w in a]
    rnd = [bb.add(audit_Hadamard(), q=w) for w in rnd]
    words = tuple((int(alt) << B | int(thr) for alt, thr in zip(original.data['alt'], original.data['threshold'])))
    o = bb.add_d(audit_lookup(words, d + B), address=audit__join(bb, a, audit_QUInt(d)), target=audit__join(bb, [*audit__split(bb, soqs['alternative'], d), *audit__split(bb, soqs['threshold'], B)]))
    a = audit__split(bb, o['address'], d)
    out = audit__split(bb, o['target'], d + B)
    alt, thr = (out[:d], out[d:])
    o = bb.add_d(audit_compare_ge(B), random=audit__join(bb, rnd), threshold=audit__join(bb, thr), flag=soqs['flag'])
    flag = o['flag']
    for k in range(d):
        flag, a[k], alt[k] = be.model__fredkin(bb, flag, a[k], alt[k])
    return bb.finalize(selection=audit__join(bb, a), random=o['random'], alternative=audit__join(bb, alt), threshold=o['threshold'], flag=flag)

@audit_lru_cache(None)
def audit_gradient_add(b):
    bb = audit_BloqBuilder()
    a = list(reversed(audit__split(bb, bb.add_register('angle', audit_QAny(b)), b)))
    z = list(reversed(audit__split(bb, bb.add_register('phase_grad', audit_QAny(b + 1)), b + 1)))
    c = [None]
    for i in range(b - 1):
        if i == 0:
            a[i], z[i], carry = audit__and(bb, a[i], z[i])
        else:
            a[i], z[i], c[i], carry = audit__majority(bb, a[i], z[i], c[i])
        c.append(carry)
    a[-1], z[-2] = be.model__cz(bb, a[-1], z[-2])
    if b > 1:
        a[-1], c[-1] = be.model__cz(bb, a[-1], c[-1])
        z[-2], c[-1] = be.model__cz(bb, z[-2], c[-1])
    for i in reversed(range(b)):
        a[i], z[i] = audit__cx(bb, a[i], z[i])
        if i:
            c[i], z[i] = audit__cx(bb, c[i], z[i])
            if i == 1:
                a[0], z[0], _ = audit__and(bb, a[0], z[0], c[1])
            else:
                a[i - 1], z[i - 1], c[i - 1], _ = audit__majority(bb, a[i - 1], z[i - 1], c[i - 1], c[i])
    return bb.finalize(angle=audit__join(bb, list(reversed(a))), phase_grad=audit__join(bb, list(reversed(z))))

@audit_lru_cache(None)
def audit_rotation(b):
    """Stored-angle Ry, without the enable mask excluded by Table II."""
    bb = audit_BloqBuilder()
    angle = bb.add_register('angle', audit_QAny(b))
    signal = bb.add_register('signal', audit_QBit())
    grad = bb.add_register('phase_grad', audit_QAny(b + 1))
    signal = bb.add(audit_SGate().adjoint(), q=signal)
    signal = bb.add(audit_Hadamard(), q=signal)
    z = audit__split(bb, grad, b + 1)
    for k in range(b + 1):
        signal, z[k] = audit__cx(bb, signal, z[k])
    out = bb.add_d(audit_gradient_add(b), angle=angle, phase_grad=audit__join(bb, z))
    z = audit__split(bb, out['phase_grad'], b + 1)
    for k in range(b + 1):
        signal, z[k] = audit__cx(bb, signal, z[k])
    signal = bb.add(audit_Hadamard(), q=signal)
    signal = bb.add(audit_SGate(), q=signal)
    return bb.finalize(angle=out['angle'], signal=signal, phase_grad=audit__join(bb, z))

@audit_lru_cache(None)
def audit_routing(D):
    """Generic enabled unary phase SELECT routing; one reusable signal wire."""
    bb = audit_BloqBuilder()
    d = D.bit_length() - 1
    a = audit__split(bb, bb.add_register('address', audit_QUInt(d)), d)
    enable = bb.add_register('enable', audit_QBit())
    signal = bb.add_register('signal', audit_QBit())

    def leaf(k, flag):
        nonlocal signal
        if k % 3:
            flag[0], signal = be.model__cz(bb, flag[0], signal)
    a, enable = audit__unary(bb, a, leaf, enable)
    return bb.finalize(address=audit__join(bb, a, audit_QUInt(d)), enable=enable, signal=signal)

@audit_lru_cache(None)
def audit_physical_leaves(bloq):
    if isinstance(bloq, (audit_And, audit__BookkeepingBloq)) or be.enc__is_terminal(bloq):
        return audit_Counter({bloq: 1})
    result = audit_Counter()
    body = bloq if hasattr(bloq, 'bloq_instances') else bloq.decompose_bloq()
    for child, mult in audit_Counter((inst.bloq for inst in body.bloq_instances)).items():
        for leaf, n in audit_physical_leaves(child).items():
            result[leaf] += mult * n
    return result

def audit_measured(bloq, *, adjoint=False):
    leaves = audit_physical_leaves(bloq)
    if adjoint:
        reversed_leaves = audit_Counter()
        for leaf, n in leaves.items():
            reversed_leaves[leaf.adjoint()] += n
        leaves = reversed_leaves
    cost = audit_GateCounts()
    for leaf, n in leaves.items():
        cost += audit_get_cost_value(leaf, audit_QECGatesCost()) * n
    return dict(temporary_AND=int(cost.and_bloq), ordinary_CCX=int(cost.toffoli), measurement=int(cost.measurement), Toffoli_family=int(cost.and_bloq + cost.toffoli), direct_T=int(cost.t), arbitrary_rotations=int(cost.rotation), actual_leaf_instances=sum(leaves.values()))

def audit_compile_audit():
    """Audit declared arithmetic primitives and aggregate their ledger multiplicities.

    O_F/O_R are counted as complete executable XOR row oracles. No aggregate
    here is advertised as an integrated local/hopping TCL encoding or cW count.
    """
    audit_OUT.mkdir(parents=True, exist_ok=True)
    primitives, lines, summary, leaves_export = ([], [], [], [])
    for lx, ly, N, n in audit_CASES:
        L = lx * ly
        q = n.bit_length()
        R = 1 << q
        D = 1 << (1 + L * (q + 2 * N) - 1).bit_length()
        pair = be.model_build_pair(Lx=lx, Ly=ly, N=N, n_max=n, b=20, B=20, keep_identity_shift=True)
        full = pair['direct']['full']
        blocks = dict(PREP_hop=audit_alias_preparation(full.prep_hop.weights, 20), PREP_local=audit_alias_preparation(full.prep_local.weights, 20), local_routing=audit_routing(D), LOAD=audit_indexed_load(L, q), INC=audit_selected_shift(L, q), DEC=audit_selected_shift(L, q, True), DEST=audit_destination(lx, ly), rotation=audit_rotation(20), O_F=audit_row_oracle(lx, ly, q), O_R=audit_row_oracle(lx, ly, q, True))
        for tag, table in zip(('table_A', 'table_B'), be.model_factor_tables(n, 20)):
            blocks[tag] = audit_lookup(table, 20)
        counts = {}
        for tag, block in blocks.items():
            counts[tag] = audit_measured(block)
            inverse_counts = audit_measured(block, adjoint=True)
            for field in ('Toffoli_family', 'temporary_AND', 'ordinary_CCX', 'measurement'):
                assert inverse_counts[field] == counts[tag][field], (tag, field)
            primitives.append(dict(Lx=lx, Ly=ly, L=L, N=N, n_max=n, primitive=tag, **counts[tag], adjoint_Toffoli_family=inverse_counts['Toffoli_family'], adjoint_measurement=inverse_counts['measurement']))
            for leaf, mult in audit_physical_leaves(block).items():
                leaves_export.append(dict(L=L, N=N, n_max=n, primitive=tag, leaf_type=type(leaf).__name__, leaf=str(leaf), leaf_repr=repr(leaf), multiplicity=mult, uncompute=bool(getattr(leaf, 'uncompute', False))))
        I = q * (L - 1)
        E = 2 * (lx.bit_length() - 1 + (ly.bit_length() - 1)) + 4

        def A(d):
            return d - 1 + 20 + (d - 1).bit_length()

        def add(branch, name, reference, parts):
            row = dict(Lx=lx, Ly=ly, L=L, N=N, n_max=n, branch=branch, line_item=name, analytical=reference, components=audit_json.dumps(parts, sort_keys=True))
            for field in ('Toffoli_family', 'temporary_AND', 'ordinary_CCX', 'measurement'):
                row[field] = sum((counts[key][field] * mult for key, mult in parts.items()))
            row['difference'] = row['Toffoli_family'] - reference
            lines.append(row)
        add('common', 'Hopping PREP and inverse', 2 * A(4 * L), dict(PREP_hop=2))
        add('common', 'Local PREP and inverse', 2 * A(D), dict(PREP_local=2))
        add('common', 'Local SELECT unary routing', D - 1, dict(local_routing=1))
        add('common', 'Exchange increment and decrement', 2 * I + 2 * (q - 1), dict(INC=1, DEC=1))
        add('common', 'Shared angle load, table, rotation', 2 * I + 2 * (R - 1) + 19, dict(LOAD=2, table_A=2, rotation=1))
        add('direct', 'Destination and inverse', 2 * E, dict(DEST=2))
        add('direct', 'Two occupation loads and inverses', 4 * I, dict(LOAD=4))
        add('direct', 'Tables A, B and inverses', 4 * (R - 1), dict(table_A=2, table_B=2))
        add('direct', 'Two phase-gradient rotations', 38, dict(rotation=2))
        add('direct', 'Direct shift plus carry/borrow chains', 2 * I + 2 * (q - 1), dict(INC=1, DEC=1))
        add('sparse', 'Forward XOR row oracle', 2 * E + 6 * I + 2 * (q - 1), dict(O_F=1))
        add('sparse', 'Reverse XOR row oracle', 2 * E + 6 * I + 2 * (q - 1), dict(O_R=1))
        add('sparse', 'Tables A, B and inverses', 4 * (R - 1), dict(table_A=2, table_B=2))
        add('sparse', 'Two phase-gradient rotations', 38, dict(rotation=2))
        case_lines = [r for r in lines if r['L'] == L]
        for item in case_lines:
            expected_difference = -2 * (E - counts['DEST']['Toffoli_family']) if item['line_item'] in ('Destination and inverse', 'Forward XOR row oracle', 'Reverse XOR row oracle') else 0
            assert item['difference'] == expected_difference, item
        row = dict(lattice=f'{lx} x {ly}', L=L, N=N, n_max=n, b=20, b_alias=20)
        for mode in ('direct', 'sparse'):
            selected = [r for r in case_lines if r['branch'] in ('common', mode)]
            row[mode + '_analytical'] = sum((r['analytical'] for r in selected))
            row[mode + '_qualtran_primitive_subtotal'] = sum((r['Toffoli_family'] for r in selected))
            row[mode + '_difference'] = row[mode + '_qualtran_primitive_subtotal'] - row[mode + '_analytical']
        row['exact_match'] = row['direct_difference'] == row['sparse_difference'] == 0
        row['direct_destination_bound_slack'] = 2 * (E - counts['DEST']['Toffoli_family'])
        row['sparse_destination_bound_slack'] = 4 * (E - counts['DEST']['Toffoli_family'])
        row['direct_upper_bound_with_DEST_slack'] = row['direct_qualtran_primitive_subtotal'] + row['direct_destination_bound_slack']
        row['sparse_upper_bound_with_DEST_slack'] = row['sparse_qualtran_primitive_subtotal'] + row['sparse_destination_bound_slack']
        row['primitive_ledger_bound_supported'] = all((r['difference'] <= 0 for r in case_lines))
        assert row['direct_upper_bound_with_DEST_slack'] == row['direct_analytical']
        assert row['sparse_upper_bound_with_DEST_slack'] == row['sparse_analytical']
        row['alpha_TCL'] = round(full.alpha)
        row['alpha_TCL_0'] = round(full.alpha - abs(next((c for lab, c in zip(full.local.labels, full.local.coefficients) if lab[0] == 'id'))))
        row['full_block_encoding_verified'] = False
        summary.append(row)
    outputs = dict(primitives=audit_pd.DataFrame(primitives), line_items=audit_pd.DataFrame(lines), summary=audit_pd.DataFrame(summary), physical_leaves=audit_pd.DataFrame(leaves_export))
    outputs['physical_leaves'] = outputs['physical_leaves'].sort_values(['L', 'N', 'n_max', 'primitive', 'leaf_type', 'leaf_repr', 'uncompute']).reset_index(drop=True)
    for name, frame in outputs.items():
        frame.to_csv(audit_OUT / f'table_II_qualtran_{name}.csv', index=False)
    report = dict(scope='Table III: executable primitive audit; not an integrated TCL block encoding', count_source='Actual executable decompositions; QECGatesCost on physical leaves only', measurement_assisted=True, ledger_formula_used_as_measured_count=False, complete_block_encoding_compiled=False, full_row_oracles_compiled=True, all_non_DEST_line_items_match=all((r['difference'] == 0 for r in lines if r['line_item'] not in ('Destination and inverse', 'Forward XOR row oracle', 'Reverse XOR row oracle'))), primitive_ledger_bounds_supported=all((r['primitive_ledger_bound_supported'] for r in summary)), exact_match=all((r['exact_match'] for r in summary)), sources_sha256=source_hashes(), limitations=['Complete forward/reverse XOR row oracles, but the aggregate combines separately counted primitives.', 'Destination slack is reported, not padded into measured circuits.', 'Not a coherent-query or full evolution count.'])
    (audit_OUT / 'table_II_qualtran_provenance.json').write_text(audit_json.dumps(report, indent=2) + '\n')
    return (outputs, report)

# Public notebook API: shared planning, compilation, tables and plots.
from pathlib import Path
import hashlib
import json
import sys
import subprocess
from collections import Counter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
CLOSED_CASES = ((4,4,1,3),(8,8,2,7),(16,16,4,15))
OPEN_CASES = tuple((x,y,N,3) for x,y in ((4,4),(4,8),(8,8)) for N in (1,2,4))
closed_plan = closed_precision_plan
open_plan = opened_plan
build_closed = closed_build_pair
build_open = opened_build_full
count_native = be.enc_circuit_resources
count_T_leaves = wk.rotation_clifford_t_leaves
leaf_cost = wk.rotation_leaf_cost


def source_hashes():
    return {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
            for n in ('block_encoding.py','walk.py','gqsp.py')}


def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def _point_path(family,job):
    return RESULTS/family/('point_'+'_'.join(map(str,job))+'.json')


def _current(path):
    return path.exists() and json.loads(path.read_text()).get('sources_sha256')==source_hashes()


def _count_evolution(full,p):
    """Count actual forward/inverse queries and Euler modules, then repeat.

    No huge flattened circuit is materialized; the ordered schedule has K of
    each query direction and 2K+1 Z/Y/Z factors. This equals recursive counting
    of the complete hierarchy and retains generic word-independent adders.
    """
    leaves=Counter();peak=0
    for inverse in (False,True):
        op=closed_RoundedSharedWalk(full,p['m'],inverse=inverse,control_value=1 if inverse else 0)
        for b,n in count_T_leaves(op).items():leaves[b]+=p['K']*n
        peak=max(peak,be.enc_compiled_peak_qubits(op))
    for axis,multiplicity in (('z',2*p['n_signal']),('y',p['n_signal'])):
        op=wk.closed_GridRotation(p['m'],(1<<p['m'])-1,axis)
        for b,n in count_T_leaves(op).items():leaves[b]+=multiplicity*n
    return dict(**leaf_cost(leaves),peak_logical_qubits=peak)


def compile_closed(job):
    p=closed_plan(*job);pair=build_closed(p);modes={}
    for mode in ('direct','paired','sparse'):
        full=pair[mode]['full']
        print('Closed',job,mode,flush=True)
        encoding=count_native(be.model_DoublyEnabledEncoding(full))
        walk=count_native(wk.closed_FixedSharedWalk(full))
        assert encoding['arbitrary_rotations']==walk['arbitrary_rotations']==2
        evolution=_count_evolution(full,p)
        expected=7*(2*p['K']*walk['coherent_ccx']+(8*p['K']+6*p['n_signal'])*(p['m']-1))
        assert evolution['T']==expected
        modes[mode]=dict(a=full.ancilla_bitsize,encoding=encoding,walk=walk,evolution=evolution)
        print(' ',evolution['T'],'T;',evolution['peak_logical_qubits'],'qubits',flush=True)
    r=dict(plan=p,modes=modes,sources_sha256=source_hashes(),numerical_phases_generated=False,
           scope='Generic GQSP schedule, supplied catalysts, seven-T coherent CCXs; no external Hadamard control')
    write_json(_point_path('closed',job),r)
    return r


def compile_open(job):
    p=open_plan(*job);full=build_open(p);channels=[]
    cdir=count_native(be.model_DoublyEnabledEncoding(full))['coherent_ccx']
    cf=2*count_native(wk.closed_F(full))['coherent_ccx']
    components=dict(C_dir=cdir,C_F=cf,a=full.ancilla_bitsize,workspace_w=be.model_declared_workspace(full))
    for kind in wk.opened_KINDS:
        walk=wk.opened_Walk(full,kind,float(p['dt']),float(p['lambda_D']))
        native=count_native(walk)
        cj=0 if kind=='dephasing' else 2*count_native(wk.opened_Jump(full,kind))['coherent_ccx']+4
        assert native['coherent_ccx']==2*cdir-cf+2*full.ancilla_bitsize+11+cj
        op=opened_EvolutionSchedule(walk,p['m'],p['K'])
        count=leaf_cost(count_T_leaves(op));peak=be.enc_compiled_peak_qubits(op)
        formula=7*(2*p['K']*native['coherent_ccx']+(16*p['K']+6*p['n_signal'])*(p['m']-1))
        assert count['T']==formula
        assert peak==full.system_bitsize+3+full.ancilla_bitsize+2+full.resource_bitsize+p['m']+1+opened_scratch_width(full,p['m'])
        channels.append(dict(channel=kind,probability=p['probabilities'][kind],C_J=cj,
            walk_Toffolis=native['coherent_ccx'],T_step=count['T'],peak_logical_qubits=peak,
            arithmetic_T_step=14*p['K']*native['coherent_ccx'],synthesis_T_step=count['T']-14*p['K']*native['coherent_ccx']))
        print('Open',job,kind,count['T'],'T/step;',peak,'qubits',flush=True)
    mean=sum(r['probability']*r['T_step'] for r in channels)
    worst=max(r['T_step'] for r in channels if r['probability']>0)
    totals=dict(mean_T_step=mean,worst_T_step=worst,mean_T_run=mean*p['N_step'],worst_T_run=worst*p['N_step'],
        N_shot=p['N_shot'],N_step=p['N_step'],total_mean_T=mean*p['N_step']*p['N_shot'],
        peak_logical_qubits=channels[0]['peak_logical_qubits'])
    r=dict(plan=p,components=components,channels=channels,totals=totals,sources_sha256=source_hashes(),
        full_channel_error_certified=False,dilation_remainder_budgeted=False,numerical_phases_generated=False,
        exclusions=['Input-state preparation','Catalyst preparation','Physical error correction'])
    write_json(_point_path('open',job),r);return r


def ensure_resources(family='all',force=False):
    families=('closed','open') if family=='all' else (family,)
    for name in families:
        for job in CLOSED_CASES if name=='closed' else OPEN_CASES:
            if force or not _current(_point_path(name,job)):
                # Isolated compiler processes bound memory/cache growth.
                subprocess.run([sys.executable,str(Path(__file__).resolve()),name+'-point',*map(str,job)],check=True,cwd=ROOT)
    return collect_tables()


def collect_tables():
    tables={}
    if all(_current(_point_path('closed',j)) for j in CLOSED_CASES):
        ii=[];vi=[]
        for job in CLOSED_CASES:
            r=json.loads(_point_path('closed',job).read_text());p=r['plan'];d,s,a=[r['modes'][m] for m in ('direct','sparse','paired')]
            ii.append(dict(L=p['L'],N=p['N'],n_max=p['n_max'],b=p['b'],B=p['B'],a=d['a'],
                cU_direct=d['encoding']['coherent_ccx'],cU_row=s['encoding']['coherent_ccx'],
                cW_direct=d['walk']['coherent_ccx'],cW_row=s['walk']['coherent_ccx'],cW_pair=a['walk']['coherent_ccx'],
                saving_percent=100*(1-d['walk']['coherent_ccx']/a['walk']['coherent_ccx']),
                P_ph=p['L']*p['n_max'].bit_length(),Q_direct=d['walk']['peak_logical_qubits'],Q_pair=a['walk']['peak_logical_qubits']))
            vi.append(dict(L=p['L'],N=p['N'],n_max=p['n_max'],K=p['K'],m=p['m'],two_eta_bound=float(p['two_eta_bound']),
                bound_rounded_up=wk.rotation_printed_tail_bound(p['two_eta_bound']),
                T_U_direct=d['evolution']['T'],T_U_pair=a['evolution']['T'],
                Q_U_direct=d['evolution']['peak_logical_qubits'],Q_U_pair=a['evolution']['peak_logical_qubits']))
        tables['II']=pd.DataFrame(ii);tables['VI']=pd.DataFrame(vi)
    if all(_current(_point_path('open',j)) for j in OPEN_CASES):
        rows=[]
        for job in OPEN_CASES:
            r=json.loads(_point_path('open',job).read_text());p=r['plan']
            rows.append(dict(lattice=f'{p["Lx"]}x{p["Ly"]}',**{k:p[k] for k in ('Lx','Ly','L','N','n_max','b','B','K','m')},**r['totals']))
        tables['VII_style']=pd.DataFrame(rows)
    RESULTS.mkdir(parents=True,exist_ok=True)
    for name,table in tables.items():table.to_csv(RESULTS/f'table_{name}.csv',index=False)
    return tables


def manuscript_tables(force=False):
    tables=ensure_resources(force=force)
    # Independent manuscript transcriptions, consulted only after compilation.
    refII=pd.DataFrame([(16,1,3,20,19,108,2308,2496,3690,4066,3874,4.75,32,297,329),
        (64,2,7,23,22,128,8834,10046,14410,16834,15682,8.11,192,613,805),
        (256,4,15,27,26,154,52212,58864,83720,97024,90880,7.88,1024,2399,3423)],columns=tables['II'].columns)
    refVI=pd.DataFrame([(16,1,3,62,22,3386082,3545794,320,352),
        (64,2,7,485,25,99474508,108111388,639,831),
        (256,4,15,4863,28,5718208314,6205675434,2428,3452)],
        columns=['L','N','n_max','K','m','T_U_direct','T_U_pair','Q_U_direct','Q_U_pair'])
    diffII=tables['II'].copy()
    for col in diffII:
        diffII[col]=(diffII[col].round(2) if col=='saving_percent' else diffII[col])-refII[col]
    diffVI=tables['VI'][refVI.columns]-refVI
    assert not diffII.to_numpy().any() and not diffVI.to_numpy().any()
    diffII.to_csv(RESULTS/'table_II_differences.csv',index=False);diffVI.to_csv(RESULTS/'table_VI_differences.csv',index=False)
    # Primitive ledger: explicitly distinct from full coherent query accounting.
    global audit_OUT
    audit_OUT=RESULTS/'arithmetic'
    provenance=audit_OUT/'table_II_qualtran_provenance.json'
    if force or not _current(provenance):audit_compile_audit()
    tables['III']=pd.read_csv(audit_OUT/'table_II_qualtran_summary.csv')
    tables['IV']=pd.read_csv(audit_OUT/'table_II_qualtran_line_items.csv')
    np.testing.assert_array_equal(tables['III'].direct_analytical,[1016,4186,24854])
    np.testing.assert_array_equal(tables['III'].sparse_analytical,[1222,5356,31020])
    tables['III'].to_csv(RESULTS/'table_III_arithmetic.csv',index=False)
    r=json.loads(_point_path('open',OPEN_CASES[0]).read_text())
    printed={'photon_loss':(282,4435,429786),'atomic_loss':(6,4159,406602),'dephasing':(0,4153,406098)}
    comparison=[]
    for row in r['channels']:
        cj,cw,t=printed[row['channel']]
        comparison.append(dict(channel=row['channel'],reported_T_step=t,qualtran_T_step=row['T_step'],
            difference_T=row['T_step']-t,difference_CJ=row['C_J']-cj,difference_walk=row['walk_Toffolis']-cw,
            reported_qubits=434,qualtran_qubits=row['peak_logical_qubits'],difference_qubits=row['peak_logical_qubits']-434))
    tables['H']=pd.DataFrame(comparison)
    assert not tables['H'].filter(like='difference').to_numpy().any()
    tables['H'].to_csv(RESULTS/'appendix_H_comparison.csv',index=False)
    write_json(RESULTS/'manifest.json',dict(sources_sha256=source_hashes(),tables=['II','III','IV','VI','Appendix H','VII-style extension'],
        full_open_channel_error_certified=False,table_III_scope='Primitive arithmetic bounds; DEST slack retained',
        table_VI_certificate_note='Two printed decimal tail bounds round down; use upward-rounded certificates in table_VI.csv.'))
    return tables


def resource_heatmaps():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    tables=collect_tables()
    for family,fields in [('open',('mean_T_run','total_mean_T','peak_logical_qubits')),
                          ('closed',('T_U_direct','T_U_pair','Q_U_direct'))]:
        table=tables['VII_style'] if family=='open' else closed_heatmap_table()
        fig,axes=plt.subplots(1,3,figsize=(13,4.6),layout='constrained')
        for ax,field in zip(axes,fields):
            mat=np.array([[float(table[(table.N==N)&(table.lattice==xy)].iloc[0][field]) for xy in ('4x4','4x8','8x8')] for N in (1,2,4)])
            xt=('4x4','4x8','8x8');yt=('1','2','4');xlabel='Lattice';ylabel='Emitters per cavity'
            is_T='T' in field.split('_')
            norm=LogNorm(vmin=mat.min(),vmax=mat.max()) if is_T else None
            im=ax.imshow(mat,origin='lower',aspect='auto',cmap='viridis',norm=norm)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    label=f'{mat[i,j]:.2e}' if is_T else f'{mat[i,j]:.0f}'
                    if i==j==0:label+=' *'
                    ax.text(j,i,label,ha='center',va='center',color='white',fontsize=9,bbox=dict(facecolor='black',alpha=.3,edgecolor='none'))
            titles={'mean_T_run':'Mean T gates per evolution','total_mean_T':'Mean T gates across measurement runs',
                'peak_logical_qubits':'Logical qubits (two catalysts)', 'T_U_direct':'Direct evolution T gates',
                'T_U_pair':'Paired evolution T gates','Q_U_direct':'Direct logical qubits'}
            ax.set(xticks=range(len(xt)),xticklabels=xt,yticks=range(len(yt)),yticklabels=yt,xlabel=xlabel,ylabel=ylabel,title=titles[field])
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((-.49,-.49),.98,.98,fill=False,edgecolor='#ffb000',lw=2.5))
            fig.colorbar(im,ax=ax,shrink=.7)
        fig.suptitle('Appendix H uniform family: cutoff 3, T=10' if family=='open' else 'Closed uniform family: cutoff 3, T=1, target error 0.001')
        fig.supxlabel('Star: manuscript example. Sampling allowance only; dilation remainder not certified.' if family=='open' else 'Star: first Table VI case. Generic GQSP schedule; state and catalyst preparations excluded.',fontsize=10)
        fig.savefig(RESULTS/f'{family}_resource_heatmaps.png',dpi=170);plt.close(fig)


def figure7_point(lx,ly,n_max):
    """Fresh complete controlled-walk counts at N=2; no evolution synthesis."""
    plan=closed_plan(lx,ly,2,n_max);pair=build_closed(plan);rows=[]
    for mode in ('direct','paired','sparse'):
        full=pair[mode]['full']
        cost=count_native(wk.closed_FixedSharedWalk(full))
        assert cost['arbitrary_rotations']==2
        assert cost['temporary_AND']==cost['measurement']==0
        rows.append(dict(Lx=lx,Ly=ly,L=lx*ly,N=2,n_max=n_max,mode=mode,
            b=plan['b'],B=plan['B'],alpha=plan['alpha'],
            coherent_ccx=cost['coherent_ccx'],peak_logical_qubits=cost['peak_logical_qubits'],
            arbitrary_rotations=cost['arbitrary_rotations']))
        print(f'Figure 7 {lx}x{ly} cutoff={n_max} {mode}: {cost["coherent_ccx"]} CCX, {cost["peak_logical_qubits"]} qubits',flush=True)
    record=dict(plan=plan,resources=rows,sources_sha256=source_hashes(),
        method='Actual FixedSharedWalk recursive decomposition and verified clean-bank allocation',
        outer_GQSP_rotation_synthesis_included=False,catalyst_preparation_included=False)
    write_json(RESULTS/'figure7'/f'point_{lx}_{ly}_{n_max}.json',record)
    return record


def figure7_table(force=False,workers=2):
    """Compile 20 parameter instances, then export the 60 measured rows."""
    path=RESULTS/'figure7_resource_table.csv';manifest=RESULTS/'figure7/manifest.json'
    if not force and path.exists() and _current(manifest):
        record=json.loads(manifest.read_text())
        if record['table_sha256']==hashlib.sha256(path.read_bytes()).hexdigest():
            table=pd.read_csv(path)
            plot_figure7(table)
            return table
    from concurrent.futures import ThreadPoolExecutor
    geometries=((4,4),(4,8),(8,8),(8,16),(16,16));cutoffs=(3,7,15,31)
    jobs=[(x,y,n) for x,y in geometries for n in cutoffs]
    def run(job):
        path=RESULTS/'figure7'/('point_'+'_'.join(map(str,job))+'.json')
        if force or not _current(path):
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'figure7-point',*map(str,job)],check=True,cwd=ROOT)
    with ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(run,jobs))
    rows=[]
    for job in jobs:
        path=RESULTS/'figure7'/('point_'+'_'.join(map(str,job))+'.json')
        assert _current(path)
        rows.extend(json.loads(path.read_text())['resources'])
    table=pd.DataFrame(rows)
    assert len(table)==60 and not table.duplicated(['Lx','Ly','n_max','mode']).any()
    path=RESULTS/'figure7_resource_table.csv';table.to_csv(path,index=False)
    write_json(RESULTS/'figure7/manifest.json',dict(sources_sha256=source_hashes(),
        table_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=60,parameter_instances=20,
        geometries=geometries,cutoffs=cutoffs,N=2,tau=1,epsilon=.001,
        counting='Complete coherent controlled walks; supplied occupation catalyst; reusable clean workspace',
        exclusions=['Outer and GQSP rotation synthesis','Catalyst preparation'],hard_coded_resource_points=False))
    # Render by reading the exported table, not the in-memory compilation rows.
    plot_figure7(pd.read_csv(path))
    return pd.read_csv(path)


def plot_figure7(table):
    """Every plotted resource coordinate is read from the generated table."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    geometries=((4,4),(4,8),(8,8),(8,16),(16,16));cutoffs=(3,7,15,31)
    if len(table)!=60 or table.duplicated(['Lx','Ly','n_max','mode']).any():
        raise ValueError('Figure 7 requires the complete 60-row measured resource table.')
    styles={'direct':('o','#0072B2','Direct register'),
            'paired':('^','#D55E00','Paired two registers'),
            'sparse':('s','#9C4DCC','Independent row maps')}
    fig,axes=plt.subplots(2,3,figsize=(12.8,7.3),layout='constrained')
    for ax,(lx,ly) in zip(axes.flat,geometries):
        sub=table[(table.Lx==lx)&(table.Ly==ly)]
        assert len(sub)==12 and sub.N.eq(2).all()
        for n in cutoffs:
            block=sub[sub.n_max==n].set_index('mode').loc[list(styles)]
            a=block.loc['direct']
            for mode in ('paired','sparse'):
                b=block.loc[mode]
                ax.plot([a.peak_logical_qubits,b.peak_logical_qubits],[a.coherent_ccx,b.coherent_ccx],color='.65',lw=.8,zorder=1)
            for mode,(marker,color,_) in styles.items():
                b=block.loc[mode]
                ax.plot(b.peak_logical_qubits,b.coherent_ccx,marker=marker,color=color,
                        mfc=color if mode=='direct' else 'none',ms=4.5,ls='none',zorder=3)
            ax.annotate(str(n),(a.peak_logical_qubits,a.coherent_ccx),xytext=(4,-9),textcoords='offset points',fontsize=8)
        ax.set(title=f'{lx} x {ly}, N=2',xlabel='Peak logical qubits per walk',ylabel='Toffoli gates per shared walk')
        ax.grid(alpha=.25);ax.margins(.09,.15)
    ax=axes.flat[-1];ax.axis('off')
    handles=[Line2D([],[],marker=marker,color=color,mfc=color if mode=='direct' else 'none',ls='none',label=label)
             for mode,(marker,color,label) in styles.items()]
    ax.legend(handles=handles,loc='upper left',frameon=False)
    ax.text(.03,.62,'Shared preparation\nUniform potential removed\n'+r'$|\tau|=1,\ \epsilon_{\rm sim}=10^{-3}$'+'\n'+
        r'$t=0.05,\ g=\nu=1,\ \Delta=0$'+'\nPoint labels give photon cutoffs\n\nCoordinates from the generated\n60-row Qualtran resource table.',
        va='top',transform=ax.transAxes,fontsize=10,linespacing=1.6)
    fig.savefig(RESULTS/'figure7_shared_walk_resources.png',dpi=180)
    fig.savefig(RESULTS/'figure7_shared_walk_resources.svg');plt.close(fig)


def closed_heatmap_point(lx,ly,N):
    """Fixed-cutoff closed evolution from counted shared walks and rotations.

    Reuse the source-hashed Table II/Figure 7 walk counts where they cover
    the requested instance. Compile missing walks only. T totals aggregate
    actual seven-T walk leaves and measured fixed-angle rotation modules.
    """
    job=(lx,ly,N,3);p=closed_plan(*job);pair=build_closed(p,modes=('direct','paired'))
    old_path=_point_path('closed',job);old=json.loads(old_path.read_text()) if _current(old_path) else None
    fig7_manifest=RESULTS/'figure7/manifest.json';fig7_csv=RESULTS/'figure7_resource_table.csv'
    fig7=None
    if N==2 and fig7_csv.exists() and _current(fig7_manifest):
        m=json.loads(fig7_manifest.read_text())
        assert m['table_sha256']==hashlib.sha256(fig7_csv.read_bytes()).hexdigest()
        fig7=pd.read_csv(fig7_csv)
    rotation_cost={axis:leaf_cost(count_T_leaves(wk.closed_GridRotation(p['m'],(1<<p['m'])-1,axis)))['T'] for axis in ('y','z')}
    modes={}
    for mode in ('direct','paired'):
        full=pair[mode]['full'];origin='Fresh recursive shared-walk decomposition'
        if old:
            native=old['modes'][mode]['walk'];origin='Existing source-hashed manuscript case'
        elif fig7 is not None:
            row=fig7[(fig7.Lx==lx)&(fig7.Ly==ly)&(fig7.n_max==3)&(fig7['mode']==mode)].iloc[0]
            assert int(row.b)==p['b'] and int(row.B)==p['B']
            native=dict(coherent_ccx=int(row.coherent_ccx),peak_logical_qubits=int(row.peak_logical_qubits))
            origin='Existing verified Figure 7 walk table'
        else:native=count_native(wk.closed_FixedSharedWalk(full))
        # Two outer rotations per shared walk, then Z/Y/Z per signal factor.
        arithmetic=7*(2*p['K'])*native['coherent_ccx']
        synthesis=2*p['K']*2*rotation_cost['y']+p['n_signal']*(2*rotation_cost['z']+rotation_cost['y'])
        # The fixed-angle adder plus its constant enable fits in the already
        # allocated clean bank; only its persistent catalyst adds live wires.
        assert be.model_declared_workspace(full)>=2*p['m']
        peak=native['peak_logical_qubits']+p['m']+1
        total=arithmetic+synthesis
        if old:
            assert total==old['modes'][mode]['evolution']['T']
            assert peak==old['modes'][mode]['evolution']['peak_logical_qubits']
        modes[mode]=dict(T=total,arithmetic_T=arithmetic,rotation_T=synthesis,
            peak_logical_qubits=peak,walk_Toffolis=native['coherent_ccx'],native_count_origin=origin)
        print(f'Closed grid {lx}x{ly} N={N} {mode}: {total} T, {peak} qubits; {origin}',flush=True)
    record=dict(plan=p,modes=modes,measured_rotation_T=rotation_cost,sources_sha256=source_hashes(),
        method='Measured complete shared-walk counts and measured fixed-angle modules, repeated in the specified GQSP schedule',
        numerical_phases_generated=False,measurement_repetitions_included=False,
        input_and_catalyst_preparations_included=False)
    write_json(RESULTS/'closed_heatmaps'/f'point_{lx}_{ly}_{N}.json',record)
    return record


def closed_heatmap_table(force=False,workers=2):
    """Nine independent lattice/emitter cases, cutoff 3, T=1, epsilon=.001."""
    from concurrent.futures import ThreadPoolExecutor
    jobs=[(x,y,N) for x,y in ((4,4),(4,8),(8,8)) for N in (1,2,4)]
    path=RESULTS/'closed_heatmap_resource_table.csv';manifest=RESULTS/'closed_heatmaps/manifest.json'
    if not force and path.exists() and _current(manifest):
        m=json.loads(manifest.read_text())
        if m['table_sha256']==hashlib.sha256(path.read_bytes()).hexdigest():return pd.read_csv(path)
    def run(job):
        point=RESULTS/'closed_heatmaps'/('point_'+'_'.join(map(str,job))+'.json')
        if force or not _current(point):
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'closed-grid-point',*map(str,job)],cwd=ROOT,check=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(run,jobs))
    rows=[]
    for x,y,N in jobs:
        point=RESULTS/'closed_heatmaps'/f'point_{x}_{y}_{N}.json';assert _current(point)
        r=json.loads(point.read_text());p=r['plan'];a,b=r['modes']['direct'],r['modes']['paired']
        rows.append(dict(lattice=f'{x}x{y}',Lx=x,Ly=y,L=x*y,N=N,n_max=3,tau=1,epsilon=.001,
            b=p['b'],B=p['B'],K=p['K'],m=p['m'],T_U_direct=a['T'],T_U_pair=b['T'],
            Q_U_direct=a['peak_logical_qubits'],Q_U_pair=b['peak_logical_qubits']))
    table=pd.DataFrame(rows);assert len(table)==9 and not table.duplicated(['Lx','Ly','N']).any()
    table.to_csv(path,index=False)
    write_json(manifest,dict(sources_sha256=source_hashes(),rows=9,cutoff=3,tau=1,epsilon=.001,
        table_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        method='Counted coherent shared walks and synthesized rotation modules; generic GQSP schedule',
        original_Table_VI_first_case_matches=True))
    return table


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['closed-point','open-point','compile','tables','plots','figure7','figure7-point','closed-grid','closed-grid-point'])
    parser.add_argument('job',nargs='*',type=int);args=parser.parse_args()
    if args.action=='closed-grid-point':closed_heatmap_point(*args.job)
    elif args.action=='closed-grid':closed_heatmap_table();resource_heatmaps()
    elif args.action=='figure7-point':figure7_point(*args.job)
    elif args.action=='figure7':figure7_table()
    elif args.action=='closed-point':compile_closed(tuple(args.job))
    elif args.action=='open-point':compile_open(tuple(args.job))
    elif args.action=='compile':ensure_resources()
    elif args.action=='tables':manuscript_tables()
    else:resource_heatmaps()
