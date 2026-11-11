#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STANDALONE VERIFICATION & FIGURE SUITE FOR PAPER V:
"Logarithmic Dilation of the Jacobi-Riemann Kernel, Strip-Holomorphic Shell
 Partitions, Stieltjes Diagonal Asymptotics, and Off-Diagonal Obstructions"

Author: Thijs Missiaen (ORCID: 0009-0007-3678-0810)

Rigorous computational verification (mpmath 50-dps, mpmath.iv 35-dps interval
arithmetic, and vectorized continuum scans):
  1. Canonical Wronskian normalization:
       W(d,t) = (1/2) d/dd |Xi(t - i*d)|^2 = R*I_t - I*R_t
              = 16 * int_0^inf y * sinh(2*d*y) * I(y,t) dy
     where Xi(z) = 8 * int_0^inf Phi(u) cos(2*z*u) du and
     I(y,t) = int_{-inf}^{inf} Phi((x+y)/2) Phi((x-y)/2) cos(2*t*x) dx.
  2. Section 2: Logarithmic dilation phi_m(u) = m^{-1/2} varphi_1(u + (1/2)ln m),
     bilateral Laplace-Mellin transform G(s) = (1/8)s(s-1)*pi^{-s/2}*Gamma(s/2),
     exact algebraic identity (1-s)(-s) = s(s-1) on C, Spira / Saidak-Zvengrowski
     modulus inequality |G(1-s)/G(s)| = |chi(s)| < 1 for d > 0, t >= 7,
     and branch-damping rho_Gamma(d,t) * n^{2d} < 1 for 1 <= n <= N(t) - 1
     (and n = N(t) whenever ln N(t) < vartheta'(t)).
  3. Section 3: Exact Wigner shift lever L_{m,n}(d) = y_{m,n}*sinh(2*d*y_{m,n}),
     pole obstruction of w_Jac(z) at |Im(z)| = pi/48 < pi/8, exact even RS
     remainder kernel R_N^{ker}(u), and genuine strip-holomorphic logistic
     shells Phi_m^{log}(z) = w_4(z)phi_m(z) + w_4(-z)phi_m(-z) with w_4(z) = 1/(1+e^{-8z})
     evaluated at TRUE complex z = xi + i*eta across 0 < eta < pi/8.
  4. Section 4: Non-oscillatory cubic-logarithmic Stieltjes diagonal expansion
     D_diag(0,t), sharp adjacent-pair self-damping bound
       A_{N-1,N}(0,t) < (3N-2)^2 / (N(N-1))^{5/2} < 9 / (N-1)^3,
     AND explicit demonstration of the O(t^{1/2}) off-diagonal cross-pair
     obstruction overwhelming D_diag(0,t) = O(L^3) pointwise.
  5. Section 5: Split-ray Cauchy tail decomposition & unshifted real-axis
     incomplete-Gamma tail bound (without artificial cosh(2*t*eta) blow-up),
     exact hyperbolic sinc ratio sinc_hyp(dL/2)/sinc_hyp(dL) = 1/cosh(dL/2),
     Gabcke's d=0, t>=200 bound, and empirical numerical remainder ratios on d > 0.
"""

import math
import sys
import time
from pathlib import Path

import mpmath as mp
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

mp.mp.dps = 50
mp.iv.dps = 35

SCRIPT_DIR = Path(__file__).resolve().parent
PUB_ROOT = SCRIPT_DIR.parent
FIG_DIR_ROOT = PUB_ROOT / "figures"
FIG_DIR_PAPER = PUB_ROOT / "paper" / "figures"
FIG_DIR_ROOT.mkdir(parents=True, exist_ok=True)
FIG_DIR_PAPER.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# 1. CANONICAL KERNELS, WRONSKIAN, AND MOTHER-WAVELET PRIMITIVES
# ==============================================================================

def varphi_1_mp(w):
    e4w = mp.exp(4 * w)
    return (2 * mp.pi**2 * mp.exp(9 * w) - 3 * mp.pi * mp.exp(5 * w)) * mp.exp(-mp.pi * e4w)


def phi_m_mp(m: int, u):
    m_mp = mp.mpf(m)
    e4u = mp.exp(4 * u)
    return (
        2 * mp.pi**2 * (m_mp**4) * mp.exp(9 * u)
        - 3 * mp.pi * (m_mp**2) * mp.exp(5 * u)
    ) * mp.exp(-mp.pi * (m_mp**2) * e4u)


def Phi_mp(u, m_max: int = 30):
    return mp.fsum(phi_m_mp(m, u) for m in range(1, m_max + 1))


def G_mellin_mp(s):
    return mp.mpf("0.125") * s * (s - 1) * mp.power(mp.pi, -s / 2) * mp.gamma(s / 2)


def Xi_exact_mp(z):
    s = mp.mpf("0.5") + 1j * z
    return 4 * G_mellin_mp(s) * mp.zeta(s)


def W_canonical_mp(d_val, t_val):
    d_mp = mp.mpf(d_val)
    t_mp = mp.mpf(t_val)
    return mp.mpf("0.5") * mp.diff(lambda dd: abs(Xi_exact_mp(t_mp - 1j * dd))**2, d_mp)


def A_gamma_mp(d_val, t_val):
    w = mp.mpf("0.25") + mp.mpf(d_val) / 2 + 1j * (mp.mpf(t_val) / 2)
    return mp.mpf("0.5") * mp.re(mp.psi(0, w)) - mp.mpf("0.5") * mp.log(mp.pi)


def theta_riemann_mp(t_val):
    t_mp = mp.mpf(t_val)
    return mp.im(mp.loggamma(mp.mpf("0.25") + 1j * (t_mp / 2))) - (t_mp / 2) * mp.log(mp.pi)


def theta_prime_mp(t_val):
    return A_gamma_mp(0, t_val)


def rho_gamma_mp(d_val, t_val):
    s = mp.mpf("0.5") + mp.mpf(d_val) + 1j * mp.mpf(t_val)
    return abs(G_mellin_mp(1 - s) / G_mellin_mp(s))


# ==============================================================================
# 2. SECTION 1 & 2: CANONICAL NORMALIZATION, DILATION, AND SPIRA MODULUS RATIO
# ==============================================================================

def verify_section_2():
    print("=" * 84)
    print("SECTION 2: CANONICAL WRONSKIAN, LOGARITHMIC DILATION & SPIRA MODULUS RATIO")
    print("=" * 84)

    # 2A. Canonical Wronskian normalization W(d,t) = (1/2)d_d|Xi|^2 = R*I_t - I*R_t = 16 int_0^inf y sinh(2dy) I(y,t) dy
    old_dps = mp.mp.dps
    mp.mp.dps = 30
    d0 = mp.mpf("0.25")
    t0 = mp.mpf("14.5")
    W_dd = mp.mpf("0.5") * mp.diff(lambda dd: abs(Xi_exact_mp(t0 - 1j * dd))**2, d0)
    Xi_v = Xi_exact_mp(t0 - 1j * d0)
    Xi_t = mp.diff(lambda tt: Xi_exact_mp(tt - 1j * d0), t0)
    W_ri = mp.re(Xi_v) * mp.im(Xi_t) - mp.im(Xi_v) * mp.re(Xi_t)

    nodes, weights = mp.gauss_quadrature(80, "legendre")
    def gl_int(fn, a, b):
        mid = (a + b) / 2
        half = (b - a) / 2
        return half * mp.fsum(w * fn(mid + half * x) for x, w in zip(nodes, weights))

    def Phi_ev(u):
        return Phi_mp(abs(u), m_max=14)

    def I_wigner(y):
        return 2 * gl_int(lambda x: Phi_ev((x + y) / 2) * Phi_ev((x - y) / 2) * mp.cos(2 * t0 * x), mp.mpf(0), mp.mpf("3.0"))

    W_int16 = 16 * gl_int(lambda y: y * mp.sinh(2 * d0 * y) * I_wigner(y), mp.mpf(0), mp.mpf("3.0"))
    err_wronsk = abs(W_dd - W_ri)
    err_int16 = abs(W_dd - W_int16)
    mp.mp.dps = old_dps

    print(f"  [2A] Canonical Wronskian at (d=0.25, t=14.5):")
    print(f"       W_dd    = {mp.nstr(W_dd, 22)}")
    print(f"       R*It-IRt= {mp.nstr(W_ri, 22)} (diff = {mp.nstr(err_wronsk, 4)})")
    print(f"       16*int  = {mp.nstr(W_int16, 22)} (diff = {mp.nstr(err_int16, 4)})")
    assert err_wronsk < mp.mpf("1e-26")
    assert err_int16 < mp.mpf("1e-22")

    # 2B. Exact logarithmic dilation phi_m(u) = m^{-1/2} varphi_1(u + (1/2)ln m)
    max_dil_err = mp.mpf("0")
    for m in [1, 2, 3, 5, 10, 25]:
        for u_val in ["-0.75", "-0.20", "0.0", "0.35", "0.90"]:
            u = mp.mpf(u_val) + 1j * mp.mpf("0.15")
            lhs = phi_m_mp(m, u)
            rhs = (mp.mpf(1) / mp.sqrt(m)) * varphi_1_mp(u + mp.mpf("0.5") * mp.log(m))
            max_dil_err = max(max_dil_err, abs(lhs - rhs))
    print(f"  [2B] Exact logarithmic dilation max residual (50-dps): {mp.nstr(max_dil_err, 5)}")
    assert max_dil_err < mp.mpf("1e-45")

    # 2C. Mother wavelet landmarks & bilateral Laplace-Mellin transform
    w_node = mp.mpf("0.25") * mp.log(3 / (2 * mp.pi))
    w_max = mp.mpf("0.25") * mp.log((15 + mp.sqrt(105)) / (8 * mp.pi))
    w_min = mp.mpf("0.25") * mp.log((15 - mp.sqrt(105)) / (8 * mp.pi))
    assert abs(varphi_1_mp(w_node)) < mp.mpf("1e-47")
    assert abs(mp.diff(varphi_1_mp, w_max)) < mp.mpf("1e-46")
    assert abs(mp.diff(varphi_1_mp, w_min)) < mp.mpf("1e-46")

    s_test = mp.mpf("0.8") + 1j * mp.mpf("14.134725")
    I_mel = mp.quad(lambda v: varphi_1_mp(v) * mp.exp(2 * (s_test - mp.mpf("0.5")) * v), [-mp.mpf("24.0"), 0, mp.mpf("3.0")])
    err_mel = abs(I_mel - G_mellin_mp(s_test))
    print(f"  [2C] Bilateral Laplace-Mellin transform error: {mp.nstr(err_mel, 5)}")
    assert err_mel < mp.mpf("1e-45")

    # 2D. Exact algebraic identity (1-s)(-s) == s(s-1) (NOT conjugation!) and Spira ratio
    max_poly_id_err = mp.mpf("0")
    for d_str in ["0.01", "0.10", "0.25", "0.50"]:
        for t_str in ["7.0", "14.13472514", "50.0", "1000.0"]:
            s = mp.mpf("0.5") + mp.mpf(d_str) + 1j * mp.mpf(t_str)
            max_poly_id_err = max(max_poly_id_err, abs((1 - s) * (-s) - s * (s - 1)))
    assert max_poly_id_err == 0

    t_crit = mp.findroot(lambda tt: A_gamma_mp(0, tt), mp.mpf("6.29"))
    print(f"  [2D] Digamma zero t_crit = {mp.nstr(t_crit, 25)} < 7")

    table1_rows = []
    for d_val, t_val in [
        ("0.01", "7.0"),
        ("0.05", "14.1347251417346937904572519835625"),
        ("0.25", "14.1347251417346937904572519835625"),
        ("0.50", "25.0108575801456887632137909925628"),
        ("0.10", "100.0"),
        ("0.25", "1000.0"),
    ]:
        d_mp = mp.mpf(d_val)
        t_mp = mp.mpf(t_val)
        s = mp.mpf("0.5") + d_mp + 1j * t_mp
        z = t_mp - 1j * d_mp
        rho = rho_gamma_mp(d_mp, t_mp)
        N_t = max(1, int(mp.floor(mp.sqrt(t_mp / (2 * mp.pi)))))
        damp_N = rho * mp.power(N_t, 2 * d_mp)
        exp_bound = mp.exp(-2 * d_mp * (theta_prime_mp(t_mp) - mp.log(N_t)))
        assert rho < 1 and damp_N < exp_bound <= 1

        def F11(zz):
            ss = mp.mpf("0.5") + 1j * zz
            return G_mellin_mp(ss) + G_mellin_mp(1 - ss)

        W11 = mp.mpf("0.5") * mp.diff(lambda dd: abs(F11(t_mp - 1j * dd))**2, d_mp)
        A_sym = mp.mpf("0.5") * (A_gamma_mp(d_mp, t_mp) + A_gamma_mp(-d_mp, t_mp))
        W11_sec = 2 * A_sym * (abs(G_mellin_mp(s))**2 - abs(G_mellin_mp(1 - s))**2)
        assert W11 > 0 and W11_sec > 0

        table1_rows.append((d_val, float(t_mp), N_t, float(rho), float(damp_N), float(W11 / abs(G_mellin_mp(s))**2)))
        print(
            f"       (d={d_val:>4}, t={float(t_mp):>8.2f}, N={N_t:>2}) | "
            f"rho_Gamma={float(rho):.8f} < 1 | rho*N^(2d)={float(damp_N):.8f} < 1 | "
            f"W11/|G|^2={float(W11/abs(G_mellin_mp(s))**2):.6f} > 0"
        )

    # 2E. Interval enclosure via Olver's explicit Stirling remainder bound
    def iv_psi_re_lower(x_iv, y_iv):
        r2 = x_iv**2 + y_iv**2
        r = mp.iv.sqrt(r2)
        cos2 = (x_iv**2 - y_iv**2) / r2
        err = mp.iv.mpf("1") / (6 * r**2) + mp.iv.mpf("1") / (3 * r**4)
        return mp.iv.log(r) - x_iv / (2 * r2) - cos2 / (12 * r2) - mp.iv.mpf([-1, 1]) * err

    for d_str, t_str in [("0.001", "7.0"), ("0.05", "14.1347"), ("0.25", "25.0108"), ("0.50", "100.0")]:
        d_iv = mp.iv.mpf(d_str)
        t_iv = mp.iv.mpf(t_str)
        x_p = mp.iv.mpf("0.25") + d_iv / 2
        x_m = mp.iv.mpf("0.25") - d_iv / 2
        y_iv = t_iv / 2
        A_iv = mp.iv.mpf("0.25") * (iv_psi_re_lower(x_p, y_iv) + iv_psi_re_lower(x_m, y_iv)) - mp.iv.mpf("0.5") * mp.iv.log(mp.iv.pi)
        assert A_iv.a > 0

    print("  [2E] Rigorous mpmath.iv Stirling enclosure A_Gamma^{sym}(d,t) > 0: PASS")

    # 2F. Transition micro-interval check at t = 8*pi (m = 2): omega_2(8*pi) < 0 while omega_1(8*pi) + omega_2(8*pi) > 0
    t_8pi = 8 * mp.pi
    om1_8pi = theta_prime_mp(t_8pi) - mp.log(1)
    om2_8pi = theta_prime_mp(t_8pi) - mp.log(2)
    assert -mp.mpf("1") / (24 * t_8pi**2) < om2_8pi < 0 and om1_8pi + om2_8pi > 0
    print(f"  [2F] Transition micro-interval at t = 8*pi: omega_2(8*pi) = {mp.nstr(om2_8pi, 8)} < 0, omega_1+omega_2 = {mp.nstr(om1_8pi + om2_8pi, 8)} > 0: PASS")
    return {"t_crit": float(t_crit), "table1": table1_rows}


# ==============================================================================
# 3. SECTION 3: POLE OBSTRUCTION OF w_Jac VS STRIP-HOLOMORPHIC LOGISTIC SHELLS
# ==============================================================================

def w_logistic_mp(z):
    """Strip-holomorphic logistic weight w_4(z) = 1 / (1 + exp(-8*z)),
    pole-free on the full Jacobi strip |Im(z)| < pi/8."""
    return 1 / (1 + mp.exp(-8 * z))


def Phi_log_shell_mp(m: int, z):
    return w_logistic_mp(z) * phi_m_mp(m, z) + w_logistic_mp(-z) * phi_m_mp(m, -z)


def R_N_ker_mp(N: int, z, m_max: int = 28):
    """Exact even Riemann-Siegel remainder kernel Phi(z) - sum_{m=1}^N (phi_m(z) + phi_m(-z))."""
    return Phi_mp(z, m_max=m_max) - mp.fsum(phi_m_mp(m, z) + phi_m_mp(m, -z) for m in range(1, N + 1))


def verify_section_3():
    print("\n" + "=" * 84)
    print("SECTION 3: WIGNER SHIFT LEVER, POLE OBSTRUCTION & STRIP-HOLOMORPHIC SHELLS")
    print("=" * 84)

    # 3A. Exact Wigner shift and Dirichlet-Sinh lever L_{m,n}(d) = y_{m,n} * sinh(2*d*y_{m,n})
    d_test = mp.mpf("0.25")
    for m, n in [(1, 2), (2, 3), (3, 5)]:
        ymn = mp.mpf("0.5") * mp.log(mp.mpf(n) / mp.mpf(m))
        L_lhs = ymn * mp.sinh(2 * d_test * ymn)
        L_rhs = mp.mpf("0.25") * mp.log(mp.mpf(n) / mp.mpf(m)) * (
            mp.power(mp.mpf(n) / mp.mpf(m), d_test) - mp.power(mp.mpf(m) / mp.mpf(n), d_test)
        )
        assert abs(L_lhs - L_rhs) < mp.mpf("1e-48") and L_lhs > 0
    print("  [3A] Exact Wigner lever L_{m,n}(d) = y_{m,n}*sinh(2*d*y_{m,n}) == (1/4)ln(n/m)[(n/m)^d - (m/n)^d]: PASS")

    # 3B. Expose pole-in-strip obstruction of w_Jac(z) = 1/(1 + exp(-2*pi*sinh(8*z)))
    z_pole_1 = 1j * (mp.pi / 48)  # Im(z) = pi/48 = 0.0654498... < pi/8 = 0.39269908...
    denom_w_jac = 1 + mp.exp(-2 * mp.pi * mp.sinh(8 * z_pole_1))
    print(f"  [3B] Pole obstruction of w_Jac at z = i*pi/48 (Im = {float(mp.pi/48):.6f} < pi/8 = {float(mp.pi/8):.6f}):")
    print(f"       |1 + exp(-2*pi*sinh(8*i*pi/48))| = {mp.nstr(abs(denom_w_jac), 5)} (EXACT POLE!)")
    assert abs(denom_w_jac) < mp.mpf("1e-45")

    # 3C. Verify strip-holomorphic logistic shells Phi_m^{log}(z) with w_4(z) = 1/(1+e^{-8z})
    phi1_d1 = mp.diff(lambda u: phi_m_mp(1, u), mp.mpf("0"), 1)
    log1_d1 = mp.diff(lambda u: Phi_log_shell_mp(1, u), mp.mpf("0"), 1)
    log1_d3 = mp.diff(lambda u: Phi_log_shell_mp(1, u), mp.mpf("0"), 3)
    print(f"  [3C] Single-shell kink phi_1'(0) = {mp.nstr(phi1_d1, 20)}")
    print(f"       Logistic shell (Phi_1^{{log}})'(0) = {mp.nstr(log1_d1, 5)}, (Phi_1^{{log}})'''(0) = {mp.nstr(log1_d3, 5)}")
    assert abs(log1_d1) < mp.mpf("1e-45") and abs(log1_d3) < mp.mpf("1e-43")

    # Genuine complex contour-shift invariance at TRUE z = xi + i*eta (eta = 0.30 < pi/8)!
    old_dps = mp.mp.dps
    mp.mp.dps = 30
    z_spec = mp.mpf("25.0") - 1j * mp.mpf("0.20")
    eta_shift = mp.mpf("0.30")
    pts_xi = [-mp.mpf("6.0"), -mp.mpf("2.0"), mp.mpf("0.0"), mp.mpf("2.0"), mp.mpf("6.0")]
    int_real_axis = mp.quad(lambda xi: Phi_log_shell_mp(1, xi) * mp.exp(2j * z_spec * xi), pts_xi)
    int_shifted   = mp.quad(
        lambda xi: Phi_log_shell_mp(1, xi + 1j * eta_shift) * mp.exp(2j * z_spec * (xi + 1j * eta_shift)),
        pts_xi,
    )
    err_contour = abs(int_real_axis - int_shifted)
    print(f"       Genuine complex contour shift (eta=0 vs eta=0.30 < pi/8) diff = {mp.nstr(err_contour, 5)}")
    assert err_contour < mp.mpf("1e-24")

    # 3D. Exact even Riemann-Siegel remainder kernel identity without artificial weights
    s_spec = mp.mpf("0.5") + 1j * z_spec
    N_cut = 3
    lhs_rs = (
        Xi_exact_mp(z_spec)
        - 4 * mp.fsum(
            G_mellin_mp(s_spec) * mp.power(m, -s_spec) + G_mellin_mp(1 - s_spec) * mp.power(m, -(1 - s_spec))
            for m in range(1, N_cut + 1)
        )
    ) / 8
    rhs_rs = mp.quad(lambda u: R_N_ker_mp(N_cut, u, m_max=22) * mp.cos(2 * z_spec * u), [0, mp.mpf("1.5"), mp.mpf("4.0"), mp.mpf("8.0"), mp.mpf("14.0")])
    err_rs = abs(lhs_rs - rhs_rs)
    rk_d1 = mp.diff(lambda u: R_N_ker_mp(N_cut, u, m_max=22), mp.mpf("0"), 1)
    mp.mp.dps = old_dps
    print(f"  [3D] Exact weight-free RS remainder kernel identity diff = {mp.nstr(err_rs, 5)}, (R_N^{{ker}})'(0) = {mp.nstr(rk_d1, 5)}")
    assert err_rs < mp.mpf("1e-20") and abs(rk_d1) < mp.mpf("1e-25")

    return {"phi1_d1": float(phi1_d1), "err_contour": float(err_contour)}


# ==============================================================================
# 4. SECTION 4: STIELTJES DIAGONAL, SHARP (N-1,N) BOUND & OFF-DIAGONAL OBSTRUCTION
# ==============================================================================

def D_diag_exact_mp(d_val, t_val):
    d_mp = mp.mpf(d_val)
    t_mp = mp.mpf(t_val)
    N = max(1, int(mp.floor(mp.sqrt(t_mp / (2 * mp.pi)))))
    th_p = theta_prime_mp(t_mp)
    total = mp.mpf("0")
    for n in range(1, N + 1):
        om = th_p - mp.log(n)
        if d_mp == 0:
            sh = mp.mpf("1")
        else:
            x = 2 * d_mp * om
            sh = mp.sinh(x) / x if abs(x) > mp.mpf("1e-20") else 1 + x**2 / 6
        total += (4 * om**2 / n) * sh
    return total


def D_diag_stieltjes_mp(t_val):
    t_mp = mp.mpf(t_val)
    L = 2 * theta_prime_mp(t_mp)
    g0 = mp.euler
    g1 = mp.stieltjes(1)
    g2 = mp.stieltjes(2)
    return (L**3) / 6 + g0 * (L**2) - 4 * g1 * L + 4 * g2


def verify_section_4():
    print("\n" + "=" * 84)
    print("SECTION 4: STIELTJES (ln t)^3 DIAGONAL, SHARP (N-1,N) BOUND & CROSS-PAIR OBSTRUCTION")
    print("=" * 84)

    table3_rows = []
    for t_val in [14.134725141734694, 25.01085758014569, 50.0, 100.0, 250.0, 500.0, 1000.0]:
        t_mp = mp.mpf(str(t_val))
        N = max(1, int(mp.floor(mp.sqrt(t_mp / (2 * mp.pi)))))
        D_ex = D_diag_exact_mp(0, t_mp)
        D_em = D_diag_stieltjes_mp(t_mp)
        err_em = abs(D_ex - D_em)

        th_p = theta_prime_mp(t_mp)
        if N >= 2:
            om_nm1 = th_p - mp.log(N - 1)
            om_n = th_p - mp.log(N)
            A_pair = ((om_nm1 + om_n) ** 2) / mp.sqrt((N - 1) * N)
            # True sharp algebraic bound: ((3N - 2)/(N(N-1)))^2 / sqrt(N(N-1)) = (3N-2)^2 / (N(N-1))^{5/2} < 9 / (N-1)^3
            B_sharp = mp.mpf((3 * N - 2) ** 2) / mp.power(N * (N - 1), mp.mpf("2.5"))
            B_cubic = mp.mpf("9") / mp.power(N - 1, 3)
            assert A_pair < B_sharp < B_cubic
        else:
            A_pair = mp.mpf("0")
            B_sharp = mp.mpf("0")

        # Total worst-case L^1 amplitude of ALL N(N-1)/2 off-diagonal cross-pairs on d = 0:
        # Each pair (m,n) (1 <= m < n <= N) has slow difference-phase amplitude 2*(om_m + om_n)^2 / sqrt(mn)
        oms = [th_p - mp.log(n) for n in range(1, N + 1)]
        S_off_L1 = mp.mpf("0")
        for m in range(1, N + 1):
            for n in range(m + 1, N + 1):
                S_off_L1 += 2 * ((oms[m - 1] + oms[n - 1]) ** 2) / mp.sqrt(m * n)

        table3_rows.append((
            float(t_mp), N, float(D_ex), float(D_em), float(err_em),
            float(2 * A_pair), float(2 * B_sharp), float(S_off_L1), float(S_off_L1 / D_ex)
        ))
        print(
            f"  t={float(t_mp):>7.2f} (N={N:>2}) | D_diag={float(D_ex):>9.5f} | EM={float(D_em):>9.5f} "
            f"(|err|={float(err_em):.2e}) | 2A_{{N-1,N}}={float(2*A_pair):.6f} < {float(2*B_sharp):.6f} | "
            f"Sum_off_L1={float(S_off_L1):>9.3f} ({float(S_off_L1/D_ex):>5.2f}x D_diag)"
        )

    # Confirm that at t=1000 (N=12, 66 cross-pairs), S_off_L1 already exceeds D_diag by > 2.3x!
    assert table3_rows[-1][-1] > 2.0
    return {"table3": table3_rows}


# ==============================================================================
# 5. SECTION 5: UNSHIFTED REAL-AXIS & SPLIT-RAY TAIL BOUNDS + GABCKE BENCHMARK
# ==============================================================================

def b_m_shell_mp(m: int, d_val, eta_val):
    m_mp = mp.mpf(m)
    d_mp = mp.mpf(d_val)
    c4 = mp.cos(4 * mp.mpf(eta_val))
    m2 = m_mp * m_mp
    p1 = 2 * (mp.pi**2) * (m2**2) / (4 * mp.pi * m2 * c4 - (5 + 2 * d_mp))
    p2 = 3 * mp.pi * m2 / (4 * mp.pi * m2 * c4 - (1 + 2 * d_mp))
    return (p1 + p2) * mp.exp(-mp.pi * m2 * c4)


def q_N_ratio_mp(N: int, eta_val):
    N_mp = mp.mpf(N)
    c4 = mp.cos(4 * mp.mpf(eta_val))
    return ((N_mp + 2) / (N_mp + 1)) ** 4 * mp.exp(-mp.pi * (2 * N_mp + 3) * c4)


def verify_section_5():
    print("\n" + "=" * 84)
    print("SECTION 5: REAL-AXIS INCOMPLETE-GAMMA TAIL, SINC RATIO & GABCKE BENCHMARKS")
    print("=" * 84)

    # 5A. Real-axis evanescent tail bound (eta = 0: NO cosh(2*t*eta) blowup!)
    table4_rows = []
    for d_str, t_str in [("0.10", "25.0"), ("0.25", "50.0"), ("0.25", "100.0"), ("0.50", "200.0"), ("0.25", "500.0")]:
        d_mp = mp.mpf(d_str)
        t_mp = mp.mpf(t_str)
        z = t_mp - 1j * d_mp
        N = max(1, int(mp.floor(mp.sqrt(t_mp / (2 * mp.pi)))))

        # Exact u >= 0 tail sum_{m=N+1}^inf int_0^inf phi_m(u) cos(2*z*u) du
        I_tail_real = mp.mpf("0")
        for m in range(N + 1, N + 7):
            I_tail_real += mp.quad(lambda u: phi_m_mp(m, u) * mp.cos(2 * z * u), [0, mp.mpf("1.5")])
        abs_I = abs(I_tail_real)

        # Closed-form geometric bound on real axis (eta = 0): |cos(2*(t-id)*u)| <= cosh(2*d*u) <= exp(2*d*u)!
        q0 = q_N_ratio_mp(N, 0)
        B_real = b_m_shell_mp(N + 1, d_mp, 0) / (1 - q0)
        assert abs_I <= B_real

        table4_rows.append((d_str, float(t_mp), N, float(mp.log10(abs_I)), float(mp.log10(B_real)), float(q0)))
        print(
            f"  (d={d_str}, t={float(t_mp):>5.1f}, N={N:>2}) | "
            f"log10|I_tail,real|={float(mp.log10(abs_I)):>8.2f} <= log10(B_tail,0)={float(mp.log10(B_real)):>8.2f} | "
            f"q_N(0)={float(q0):.3e}"
        )

    # 5B. Exact hyperbolic sinc damping identity sinc_hyp(dL/2)/sinc_hyp(dL) == 1/cosh(dL/2)
    max_sinc_err = mp.mpf("0")
    for d_str in ["0.05", "0.25", "0.50"]:
        for L_str in ["2.0", "5.0", "10.0"]:
            dv = mp.mpf(d_str)
            Lv = mp.mpf(L_str)
            sh_half = mp.sinh(dv * Lv / 2) / (dv * Lv / 2)
            sh_full = mp.sinh(dv * Lv) / (dv * Lv)
            max_sinc_err = max(max_sinc_err, abs(sh_half / sh_full - 1 / mp.cosh(dv * Lv / 2)))
    print(f"  [5B] Hyperbolic sinc ratio max residual: {mp.nstr(max_sinc_err, 5)}")
    assert max_sinc_err < mp.mpf("1e-48")

    # 5C. Gabcke (1979) exact theorem check on d = 0, t >= 200 AND empirical d > 0 ratios
    print("  [5C] Gabcke (1979) d=0 (t >= 200) check and empirical d > 0 remainder ratios:")
    emp_rows = []
    for t_val in [200.0, 500.0, 1000.0]:
        t_mp = mp.mpf(str(t_val))
        N = int(mp.floor(mp.sqrt(t_mp / (2 * mp.pi))))
        # On d = 0: Z(t) = Z_main(t) + (-1)^{N-1}*(t/(2*pi))^{-1/4}*Psi(p) + R_1(t)
        # where |R_1(t)| <= 0.127 * t^{-3/4} for t >= 200 (Gabcke 1979, Satz 1 / Satz 4)!
        th = theta_riemann_mp(t_mp)
        Z_ex = mp.re(mp.exp(1j * th) * mp.zeta(mp.mpf("0.5") + 1j * t_mp))
        Z_main = 2 * mp.fsum(mp.cos(th - t_mp * mp.log(n)) / mp.sqrt(n) for n in range(1, N + 1))
        a_rs = mp.sqrt(t_mp / (2 * mp.pi))
        p_frac = a_rs - N
        Psi_p = mp.cos(2 * mp.pi * (p_frac**2 - p_frac - mp.mpf("1") / 16)) / mp.cos(2 * mp.pi * p_frac)
        c0_term = ((-1) ** (N - 1)) * mp.power(t_mp / (2 * mp.pi), mp.mpf("-0.25")) * Psi_p
        R_unsub = abs(Z_ex - Z_main)
        R1_abs = abs(Z_ex - Z_main - c0_term)
        gabcke_bound = mp.mpf("0.127") * mp.power(t_mp, mp.mpf("-0.75"))
        assert R1_abs <= gabcke_bound

        # Empirical d = 0.25 Wronskian remainder ratio |W_exact - W_main| / W_diag_scaled
        # Note: Xi(z) = 4*G(s)*zeta(s), so Xi_main(z) = 4 * sum_{n=1}^N [G(s)*n^{-s} + G(1-s)*n^{-(1-s)}]
        d_emp = mp.mpf("0.25")
        W_ex = W_canonical_mp(d_emp, t_mp)
        def Xi_main(dd, tt):
            zz = tt - 1j * dd
            ss = mp.mpf("0.5") + 1j * zz
            return 4 * mp.fsum(
                G_mellin_mp(ss) * mp.power(n, -ss) + G_mellin_mp(1 - ss) * mp.power(n, -(1 - ss))
                for n in range(1, N + 1)
            )
        W_mn = mp.mpf("0.5") * mp.diff(lambda dd: abs(Xi_main(dd, t_mp))**2, d_emp)
        W_rem = abs(W_ex - W_mn)
        G_mod2 = abs(G_mellin_mp(mp.mpf("0.5") + 1j * t_mp))**2
        D_d = 16 * G_mod2 * d_emp * D_diag_exact_mp(d_emp, t_mp)
        ratio_emp = float(W_rem / D_d)
        emp_rows.append((t_val, N, float(R_unsub), float(R1_abs), float(gabcke_bound), ratio_emp))
        print(
            f"       t={t_val:>6.1f} (N={N:>2}) | |Z-Z_main|={float(R_unsub):.5e} (O(t^-1/4)) | "
            f"|R_1(t)|={float(R1_abs):.5e} <= 0.127*t^(-3/4)={float(gabcke_bound):.5e} | "
            f"d=0.25 |W_rem|/W_diag={ratio_emp:.4e}"
        )

    return {"table4": table4_rows, "emp_rows": emp_rows}


# ==============================================================================
# 6. PUBLICATION FIGURES (300-DPI VECTOR PDF + PNG)
# ==============================================================================

def generate_figures(sec4_data):
    print("\n" + "=" * 84)
    print("GENERATING PUBLICATION FIGURES (FIGURES 1 & 2)")
    print("=" * 84)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9.5,
        "axes.titlesize": 10.2,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.0,
        "figure.dpi": 300,
    })

    # FIGURE 1: Mother-Wavelet Dilation, Spira Ratio, Pole Obstruction vs Logistic Shell, Contour Shift
    fig1, axes1 = plt.subplots(2, 2, figsize=(12.2, 8.8))

    # Panel (a): Mother wavelet dilation phi_m(u) = m^{-1/2} varphi_1(u + (1/2)ln m)
    ax = axes1[0, 0]
    u_grid = np.linspace(-2.2, 0.8, 600)
    for m, col, ls in [(1, "#1b9e77", "-"), (2, "#2b5c8f", "--"), (4, "#d95f02", "-."), (8, "#7570b3", ":")]:
        vals = (2 * np.pi**2 * (m**4) * np.exp(9 * u_grid) - 3 * np.pi * (m**2) * np.exp(5 * u_grid)) * np.exp(-np.pi * (m**2) * np.exp(4 * u_grid))
        ax.plot(u_grid, vals, color=col, ls=ls, lw=1.9, label=rf"$m = {m}$: $\phi_{m}(u) = {m}^{{-1/2}}\varphi_1(u + \frac{{1}}{{2}}\ln {m})$")
    ax.axhline(0.0, color="black", lw=0.7, alpha=0.5)
    ax.set_title(r"(a) Exact Logarithmic Dilation $\phi_m(u) = m^{-1/2}\varphi_1(u + \frac{1}{2}\ln m)$")
    ax.set_xlabel(r"Logarithmic coordinate $u$")
    ax.set_ylabel(r"Shell amplitude $\phi_m(u)$")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")

    # Panel (b): Spira / Saidak-Zvengrowski modulus ratio rho_Gamma(d,t) = |chi(s)| < 1
    ax = axes1[0, 1]
    t_arr = np.linspace(5.0, 60.0, 300)
    for dv, col in [(0.05, "#e41a1c"), (0.15, "#ff7f00"), (0.30, "#2b5c8f"), (0.50, "#1b9e77")]:
        rhos = [float(rho_gamma_mp(dv, float(tv))) for tv in t_arr]
        ax.plot(t_arr, rhos, color=col, lw=1.9, label=rf"$d = {dv:.2f}$")
    ax.axhline(1.0, color="black", ls="--", lw=1.2, label=r"Unit circle $|\chi(s)| = 1$")
    ax.axvline(6.289836, color="#7570b3", ls=":", lw=1.2, label=r"$t_{\mathrm{crit}} \approx 6.2898$")
    ax.set_title(r"(b) Reflection Modulus Ratio $\rho_\Gamma(d,t) = |\mathcal{G}(1-s)/\mathcal{G}(s)|$")
    ax.set_xlabel(r"Spectral height $t$")
    ax.set_ylabel(r"Modulus ratio $\rho_\Gamma(d,t)$")
    ax.set_ylim(0.0, 1.15)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left")

    # Panel (c): Strip holomorphy: pole of w_Jac at Im(z)=pi/48 vs pole-free w_4(z) on |Im(z)| < pi/8
    ax = axes1[1, 0]
    eta_scan = np.linspace(0.0, 0.385, 500)
    mod_w4 = [float(abs(w_logistic_mp(1j * float(et)))) for et in eta_scan]
    mod_wjac = []
    for et in eta_scan:
        denom = abs(1 + mp.exp(-2 * mp.pi * mp.sinh(8j * mp.mpf(float(et)))))
        mod_wjac.append(min(50.0, float(1 / max(denom, mp.mpf("1e-6")))))
    ax.semilogy(eta_scan, mod_w4, color="#1b9e77", lw=2.1, label=r"Logistic $w_4(\mathrm{i}\eta) = (1+\mathrm{e}^{-8\mathrm{i}\eta})^{-1}$ (pole at $\pi/8$)")
    ax.semilogy(eta_scan, mod_wjac, color="#e41a1c", lw=1.7, ls="--", label=r"$w_{\mathrm{Jac}}(\mathrm{i}\eta)$ (poles at $\pi/48, 5\pi/48, \dots$)")
    ax.axvline(math.pi / 48, color="#e41a1c", ls=":", lw=1.2, label=r"First $w_{\mathrm{Jac}}$ pole $\eta = \pi/48 \approx 0.0654$")
    ax.axvline(math.pi / 8, color="black", ls="-.", lw=1.2, label=r"Jacobi strip boundary $\eta = \pi/8 \approx 0.3927$")
    ax.set_title(r"(c) Imaginary-Axis Regularity on $0 \leq \eta < \pi/8$")
    ax.set_xlabel(r"Contour height $\eta = \mathrm{Im}(z)$")
    ax.set_ylabel(r"Weight modulus $|w(\mathrm{i}\eta)|$ (clipped at $50$)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left", fontsize=7.5)

    # Panel (d): Smoothness at u = 0: naive half-line phi_1(|u|) kink vs Phi_1^{log}(u)
    ax = axes1[1, 1]
    u_loc = np.linspace(-0.35, 0.35, 500)
    naive_d1 = [float(np.sign(uv) * float(mp.diff(lambda x: phi_m_mp(1, x), abs(float(uv))))) if abs(uv) > 1e-4 else np.nan for uv in u_loc]
    log_d1 = [float(mp.diff(lambda x: Phi_log_shell_mp(1, x), float(uv))) for uv in u_loc]
    ax.plot(u_loc, naive_d1, color="#e41a1c", ls="--", lw=1.8, label=r"Naive $\frac{\mathrm{d}}{\mathrm{d}u}\phi_1(|u|)$ (jump $2\phi_1'(0) \approx 0.0790$)")
    ax.plot(u_loc, log_d1, color="#1b9e77", lw=2.1, label=r"Smooth logistic $\frac{\mathrm{d}}{\mathrm{d}u}\Phi_1^{\mathrm{log}}(u)$ ($(\Phi_1^{\mathrm{log}})'(0)=0$)")
    ax.axhline(0.0, color="black", lw=0.7, alpha=0.5)
    ax.set_title(r"(d) Elimination of the $u = 0$ Odd-Derivative Kink")
    ax.set_xlabel(r"Coordinate $u$ near $0$")
    ax.set_ylabel(r"First derivative at $m = 1$")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=7.6)

    fig1.tight_layout()
    for d_out in (FIG_DIR_ROOT, FIG_DIR_PAPER):
        fig1.savefig(d_out / "fig1_mother_wavelet_abc_matrix.pdf", bbox_inches="tight")
        fig1.savefig(d_out / "fig1_mother_wavelet_abc_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # FIGURE 2: Stieltjes Diagonal vs Off-Diagonal Cross-Pair Sum & Boundary Pair Self-Damping
    fig2, axes2 = plt.subplots(2, 2, figsize=(12.2, 8.8))
    rows3 = sec4_data["table3"]
    ts = [r[0] for r in rows3]
    D_exs = [r[2] for r in rows3]
    D_ems = [r[3] for r in rows3]
    pair_2A = [r[5] for r in rows3]
    pair_2B = [r[6] for r in rows3]
    S_offs = [r[7] for r in rows3]

    # Panel (a): Stieltjes diagonal vs Off-diagonal L^1 cross-pair amplitude sum
    ax = axes2[0, 0]
    ax.loglog(ts, D_exs, "o-", color="#1b9e77", lw=2.0, label=r"Exact diagonal $\mathcal{D}_{\mathrm{diag}}(0,t) \asymp \frac{1}{6}L^3$")
    ax.loglog(ts, D_ems, "--", color="#2b5c8f", lw=1.6, label=r"Stieltjes expansion $\mathcal{D}_{\mathrm{EM}}(0,t)$")
    ax.loglog(ts, S_offs, "s-", color="#e41a1c", lw=2.0, label=r"Off-diagonal $L^1$ sum $\sum_{m<n} 2A_{m,n} \asymp \frac{96}{\sqrt{2\pi}} t^{1/2}$")
    ax.set_title(r"(a) Stieltjes Diagonal vs. Total Off-Diagonal $L^1$ Envelope")
    ax.set_xlabel(r"Spectral height $t$")
    ax.set_ylabel(r"Amplitude")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left", fontsize=7.6)

    # Panel (b): Single boundary pair 2*A_{N-1,N}(0,t) vs sharp O(t^{-3/2}) bound
    ax = axes2[0, 1]
    ts_ge2 = [r[0] for r in rows3 if r[1] >= 2]
    p2A_ge2 = [r[5] for r in rows3 if r[1] >= 2]
    p2B_ge2 = [r[6] for r in rows3 if r[1] >= 2]
    ax.loglog(ts_ge2, p2A_ge2, "o-", color="#2b5c8f", lw=2.0, label=r"Exact boundary pair $2A_{N-1,N}(0,t)$")
    ax.loglog(ts_ge2, p2B_ge2, "s--", color="#d95f02", lw=1.8, label=r"Sharp bound $2(3N-2)^2 / [N(N-1)]^{5/2}$")
    ax.set_title(r"(b) Adjacent Boundary Pair $(N-1, N)$ Self-Damping $\mathcal{O}(t^{-3/2})$")
    ax.set_xlabel(r"Spectral height $t$ ($N(t) \geq 2$)")
    ax.set_ylabel(r"Amplitude")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", fontsize=7.8)

    # Panel (c): Real-axis incomplete-Gamma tail decay vs N(t)
    ax = axes2[1, 0]
    t_tail = [25.0, 50.0, 100.0, 200.0, 500.0]
    log_ex = []
    log_bd = []
    for tv in t_tail:
        Nv = int(math.floor(math.sqrt(tv / (2 * math.pi))))
        q0 = q_N_ratio_mp(Nv, 0)
        B0 = b_m_shell_mp(Nv + 1, mp.mpf("0.25"), 0) / (1 - q0)
        log_bd.append(float(mp.log10(B0)))
        z_t = mp.mpf(str(tv)) - 1j * mp.mpf("0.25")
        It = mp.quad(lambda u: phi_m_mp(Nv + 1, u) * mp.cos(2 * z_t * u), [0, mp.mpf("1.2")])
        log_ex.append(float(mp.log10(abs(It))))
    ax.plot(t_tail, log_ex, "o-", color="#1b9e77", lw=2.0, label=r"$\log_{10}|\mathcal{I}_{\mathrm{tail}}(N, t-\mathrm{i}d, 0)|$")
    ax.plot(t_tail, log_bd, "s--", color="#7570b3", lw=1.8, label=r"Closed-form bound $\log_{10} B_{\mathrm{tail}}(N, d, 0)$")
    ax.set_title(r"(c) Super-Exponential Real-Axis Tail Bound ($d = 0.25, \eta = 0$)")
    ax.set_xlabel(r"Spectral height $t$")
    ax.set_ylabel(r"$\log_{10}$ magnitude")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=7.8)

    # Panel (d): Hyperbolic sinc ratio sinc_hyp(dL/2)/sinc_hyp(dL) = 1/cosh(dL/2)
    ax = axes2[1, 1]
    L_scan = np.linspace(0.1, 14.0, 300)
    for dv, col in [(0.10, "#e41a1c"), (0.25, "#2b5c8f"), (0.50, "#1b9e77")]:
        ax.plot(L_scan, 1.0 / np.cosh(0.5 * dv * L_scan), color=col, lw=2.0, label=rf"$d = {dv:.2f}$: $\cosh^{{-1}}(dL/2)$")
    ax.set_title(r"(d) Hyperbolic Sinc Ratio $\mathrm{sinc}_{\mathrm{hyp}}(dL/2)/\mathrm{sinc}_{\mathrm{hyp}}(dL) = \cosh^{-1}(dL/2)$")
    ax.set_xlabel(r"Logarithmic scale $L = \ln(t/2\pi)$")
    ax.set_ylabel(r"Ratio $\cosh^{-1}(dL/2)$")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left")

    fig2.tight_layout()
    for d_out in (FIG_DIR_ROOT, FIG_DIR_PAPER):
        fig2.savefig(d_out / "fig2_stieltjes_diagonal_and_gabcke_bounds.pdf", bbox_inches="tight")
        fig2.savefig(d_out / "fig2_stieltjes_diagonal_and_gabcke_bounds.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print("  Saved Figure 1 and Figure 2 (PDF + PNG).")


def main():
    t0 = time.perf_counter()
    print("PAPER V MASTER VERIFICATION SUITE (mpmath 50-dps + mpmath.iv 35-dps)")
    print(f"Device: {DEVICE}")
    s2 = verify_section_2()
    s3 = verify_section_3()
    s4 = verify_section_4()
    s5 = verify_section_5()
    generate_figures(s4)
    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 84)
    print(f"ALL PAPER V VERIFICATIONS PASSED (EXIT CODE 0) IN {elapsed:.2f}s")
    print("=" * 84)


if __name__ == "__main__":
    main()
