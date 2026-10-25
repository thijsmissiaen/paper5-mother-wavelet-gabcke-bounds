---
name: mother-wavelet-gabcke-bounds
description: Scientific knowledge skill for Paper V of the Transverse Wronskian series on Riemann's Xi-function. Contains the proven analytical identities for the universal mother-wavelet log-shift dilation of the Jacobi-Riemann kernel, exact polynomial cancellation |(1-s)(-s)| = |s(s-1)| in the bilateral Laplace-Mellin transform, the 3-line reflection-ellipse proof of W_{1,1}(d,t) > 0, the Symmetric Branch-Damping Inequality, the 2D Wigner phase-space shift and smooth Jacobi-Sinh shell elimination of the u = 0 odd-derivative kink, the non-oscillatory cubic-logarithmic Stieltjes diagonal expansion with O(t^{-3/2}) boundary mode self-damping, and explicit incomplete-Gamma and two-dimensional Gabcke-Olver saddle-point remainder envelopes.
---

# Universal Mother-Wavelet Decomposition, Smooth Jacobi–Sinh Shells, and Explicit Saddle-Point Remainder Bounds (Paper V)

## 1. Scope and Mathematical Architecture

This skill encapsulates the analytical theorems, exact integral identities, rigorous `mpmath.iv` interval-arithmetic certifications, and GPU continuum verification procedures of **Paper V** in the series on the Transverse Wronskian of Riemann's $\Xi$-Function. Paper V bridges the single-integral Jacobi–Riemann representation of $\Xi(z)$ with the finite Riemann–Siegel Dirichlet sum of length $N(t) = \lfloor\sqrt{t/2\pi}\rfloor$ across the half-strip $\mathcal{S} _ {1/2}^+ = \lbrace (d,t) \in (0, 1/2) \times [14, \infty)\rbrace$ through four coupled analytical pillars:

1. **Universal Mother-Wavelet Dilation and Exact Polynomial Cancellation (Section 2)**: Every shell $\phi_m(u)$ of the Jacobi–Riemann kernel $\Phi(u) = \sum _ {m=1}^\infty \phi_m(u)$ is an exact logarithmic translation and $m^{-1/2}$ amplitude dilation of a single universal mother wavelet $\varphi_1(w)$. Its bilateral Laplace–Mellin transform $\mathcal{G}(s)$ equals $\frac{1}{8}s(s-1)\pi^{-s/2}\Gamma(s/2)$. Because $\lvert(1-s)(-s)\rvert \equiv \lvert s(s-1)\rvert$ holds identically for $s = \frac{1}{2} + d + \mathrm{i}t$, the polynomial prefactor cancels completely in the reflection modulus ratio $\rho _ \Gamma(d,t) = \lvert\mathcal{G}(1-s)/\mathcal{G}(s)\rvert < 1$, yielding a 3-line reflection-ellipse proof of $W _ {1,1}(d,t) > 0$ and the **Symmetric Branch-Damping Inequality** across all active modes $1 \le n \le N(t)$.
2. **Saddle-Point Truncation $N(t)$, 2D Wigner Shift, and Smooth Jacobi–Sinh Shells (Section 3)**: In the 2D Wigner coordinates $(x,y) = (u+v, u-v)$, the cross-shell product $\phi_m(u)\phi_n(v)$ localizes around $(x _ {m,n}, y _ {m,n}) = (-\frac{1}{2}\ln(mn), \frac{1}{2}\ln(n/m))$ with strictly positive Dirichlet–Sinh lever $\mathcal{L} _ {m,n}(d) = y _ {m,n}\sinh(d y _ {m,n}) > 0$. Naive half-line restriction $\phi_m(\lvert u\rvert)$ introduces an odd-derivative jump at $u = 0$ ($\phi_1'(0) = -\sum _ {m=2}^\infty \phi_m'(0) \approx +0.03949877$), creating a spurious $\mathcal{O}(t^{-2})\mathrm{e}^{\pi t/2}$ integration-by-parts boundary artifact. Replacing $\phi_m(\lvert u\rvert)$ by the smooth self-mirrored Jacobi–Sinh shells $\Phi_m^{\mathrm{ABC}}(z)$ eliminates all odd-order jumps at $u = 0$ while preserving exact partition summation.
3. **Non-Oscillatory $(\ln t)^3$ Stieltjes Diagonal and $\mathcal{O}(t^{-3/2})$ Boundary Self-Damping (Section 4)**: On the diagonal $m = n$, the oscillating phases satisfy $\sin^2\phi_n + \cos^2\phi_n \equiv 1$ and the hyperbolic difference vanishes identically, producing a strictly positive non-oscillatory Stieltjes cubic-logarithmic expansion $\mathcal{D} _ {\mathrm{diag}}(0,t) = \frac{1}{6}L^3 + \gamma _ {\mathrm{E}}L^2 - 4\gamma_1^{\mathrm{St}}L + 4\gamma_2^{\mathrm{St}} + \mathcal{O}(t^{-3/2}\ln^2 t)$ with $L = 2\vartheta'(t)$. Simultaneously, the slowest adjacent boundary pair $(N-1,N)$ has local saddle velocity $\omega_n(t) = \vartheta'(t) - \ln n \to 0$ as $n \to N(t)$, causing its cross-amplitude $A _ {N-1,N}(0,t) := \frac{(\omega _ {N-1}(t)+\omega_N(t))^2}{\sqrt{(N-1)N}} \le \frac{9}{4 (N(t)-1)^3} = \mathcal{O}(t^{-3/2}) \to 0$ to self-damp quadratically.
4. **Closed-Form Incomplete-Gamma Tail Bounds and 2D Gabcke–Olver Remainder Envelopes (Section 5)**: On the shifted horizontal contour $u = \xi + \mathrm{i}\eta$ ($0 < \eta < \pi/8$), exact integration by parts in $w = \mathrm{e}^{4\xi} \ge 1$ yields a closed-form geometric-series bound $B _ {\mathrm{tail}}(N,d,t,\eta)$ and a uniform Gaussian barrier $\mathcal{B} _ {\mathrm{Gauss}}(m,N(t),\eta)$. Combining the $u = 0$ mirror-wall damping identity $\mathrm{sinc} _ {\mathrm{hyp}}(dL/2)/\mathrm{sinc} _ {\mathrm{hyp}}(dL) \equiv \cosh(dL/2)^{-1} \le 2(t/2\pi)^{-d/2}$ with Gabcke's $d = 0$ remainder theorem yields the two-dimensional Wronskian remainder envelope $B _ {\widetilde{W}}^{(C_0)}(d,t)$.

## 2. Canonical Identities and Theorems

### 2.1 Universal Mother Wavelet and Bilateral Laplace–Mellin Transform

Define the **universal mother wavelet** on $\mathbb{R}$ (and holomorphically on $\mathbb{C}$) by:

$$\varphi_1(w) := \left(2\pi^2\mathrm{e}^{9w} - 3\pi\mathrm{e}^{5w}\right)\exp\left(-\pi\mathrm{e}^{4w}\right).$$

Every mode $\phi_m(u) = (2\pi^2 m^4\mathrm{e}^{9u} - 3\pi m^2\mathrm{e}^{5u})\exp(-\pi m^2\mathrm{e}^{4u})$ of the Jacobi–Riemann kernel $\Phi(u) = \sum _ {m=1}^\infty \phi_m(u)$ satisfies the exact log-shift dilation identity:

$$\phi_m(u) = m^{-1/2} \varphi_1\left(u + \frac{1}{2}\ln m\right) \qquad (m \ge 1,   u \in \mathbb{C}).$$

The mother wavelet possesses a unique real zero node, a global positive peak, and a negative secondary trough at:

$$w _ {\mathrm{node}} = \frac{1}{4}\ln\left(\frac{3}{2\pi}\right) \approx -0.1848161944, \qquad w _ {\max} = \frac{1}{4}\ln\left(\frac{15+\sqrt{105}}{8\pi}\right) \approx +0.0011334898, \qquad w _ {\min} = \frac{1}{4}\ln\left(\frac{15-\sqrt{105}}{8\pi}\right) \approx -0.4163462679.$$

Substituting $x = \pi\mathrm{e}^{4v}$ into the bilateral Laplace–Mellin integral (which converges absolutely for $\mathrm{Re}(s) > -2$ since $\varphi_1(v) \sim -3\pi\mathrm{e}^{5v}$ as $v \to -\infty$, with meromorphic continuation to $\mathbb{C}$) yields the exact completed Gamma factor:

$$\mathcal{G}(s) := \int _ {-\infty}^{\infty} \varphi_1(v) \mathrm{e}^{2(s-1/2)v} \mathrm{d}v = \frac{1}{8}s(s-1)\pi^{-s/2}\Gamma\left(\frac{s}{2}\right), \qquad \xi(s) = \frac{1}{2}s(s-1)\pi^{-s/2}\Gamma\left(\frac{s}{2}\right)\zeta(s) = 4 \mathcal{G}(s)\zeta(s).$$

Consequently, the bilateral integral of the $m$-th shell decomposes into the $m$-th Dirichlet pair:

$$F _ {m,\pm}(z) := 2\int _ {-\infty}^{\infty} \phi_m(u)\cos(2zu) \mathrm{d}u = \mathcal{G}(s) m^{-s} + \mathcal{G}(1-s) m^{-(1-s)} \qquad \left(s = \frac{1}{2} + d + \mathrm{i}t,   z = t - \mathrm{i}d\right).$$

### 2.2 Exact Polynomial Cancellation, Reflection Ellipse, and Symmetric Branch Damping (Theorem 2.2)

For $s = \frac{1}{2} + d + \mathrm{i}t$, we have $s(s-1) = (d + \mathrm{i}t)^2 - \frac{1}{4}$ and $(1-s)(-s) = (-d + \mathrm{i}t)^2 - \frac{1}{4} = \overline{s(s-1)}$, so $\lvert(1-s)(-s)\rvert \equiv \lvert s(s-1)\rvert$ identically across the entire complex plane. The reflection modulus ratio therefore simplifies to the pure Archimedean Gamma ratio:

$$\rho _ \Gamma(d,t) := \left\lvert\frac{\mathcal{G}(1-s)}{\mathcal{G}(s)}\right\rvert = \pi^d \left\lvert\frac{\Gamma\left(\frac{1}{4} - \frac{d}{2} + \mathrm{i}\frac{t}{2}\right)}{\Gamma\left(\frac{1}{4} + \frac{d}{2} + \mathrm{i}\frac{t}{2}\right)}\right\rvert = \exp\left(-2\int_0^d \mathcal{A} _ \Gamma^{\mathrm{sym}}(u,t) \mathrm{d}u\right),$$

where $\mathcal{A} _ \Gamma(d,t) = \frac{1}{2}\mathrm{Re} \psi(\frac{1}{4} + \frac{d}{2} + \mathrm{i}\frac{t}{2}) - \frac{1}{2}\ln\pi > 0$ for all $t \ge 7 > t _ {\mathrm{crit}} = 6.2898359888\dots$. This gives the 3-line minor-axis bound and secular self-Wronskian positivity (up to $\mathcal{O}(d/t^2)$ oscillatory cross-derivative terms):

$$\lvert F _ {1,1}(t - \mathrm{i}d)\rvert = \lvert\mathcal{G}(s) + \mathcal{G}(1-s)\rvert \ge \lvert\mathcal{G}(s)\rvert\left(1 - \rho _ \Gamma(d,t)\right) > 0, \qquad W _ {1,1}^{\mathrm{sec}}(d,t) = 2\mathcal{A} _ \Gamma^{\mathrm{sym}}(d,t)\left(\lvert\mathcal{G}(s)\rvert^2 - \lvert\mathcal{G}(1-s)\rvert^2\right) > 0.$$

Furthermore, because $\mathcal{A} _ \Gamma^{\mathrm{sym}}(u,t) > \mathcal{A} _ \Gamma(0,t) = \vartheta'(t)$ by positive digamma curvature, $\rho _ \Gamma(d,t) < \mathrm{e}^{-2d\vartheta'(t)}$, which yields the **Symmetric Branch-Damping Inequality** for every active mode $1 \le n \le N(t)$:

$$\rho _ \Gamma(d,t) n^{-1/2+d} < \mathrm{e}^{-2d\omega_n(t)} n^{-1/2-d} \le n^{-1/2-d}, \qquad \omega_n(t) := \vartheta'(t) - \ln n \ge 0.$$

### 2.3 Two-Dimensional Wigner Shift and Smooth Jacobi–Sinh Shells (Theorem 3.1)

In the 2D Wigner coordinates $x = u+v$ and $y = u-v$, the cross-shell product $\phi_m(u)\phi_n(v)$ translates the mother-wavelet tensor product to $(x _ {m,n}, y _ {m,n}) = (-\frac{1}{2}\ln(mn), \frac{1}{2}\ln(n/m))$:

$$\phi_m\left(\frac{x+y}{2}\right)\phi_n\left(\frac{x-y}{2}\right) = \frac{1}{\sqrt{mn}} \varphi_1\left(\frac{(x-x _ {m,n}) + (y-y _ {m,n})}{2}\right)\varphi_1\left(\frac{(x-x _ {m,n}) - (y-y _ {m,n})}{2}\right).$$

Writing $y = y _ {m,n} + \tilde{y}$, the transverse lever decomposes into an even part in $\tilde{y}$ and an odd part in $\tilde{y}$:

$$y\sinh(dy) = \left[\mathcal{L} _ {m,n}(d)\cosh(d\tilde{y}) + y _ {m,n}\cosh(dy _ {m,n})\tilde{y}\sinh(d\tilde{y})\right] + \left[y _ {m,n}\cosh(dy _ {m,n})\sinh(d\tilde{y}) + \sinh(dy _ {m,n})\tilde{y}\cosh(d\tilde{y})\right],$$

where the **Dirichlet–Sinh lever** at the Wigner center $y _ {m,n}$ is strictly positive for all $m \neq n$ and $d > 0$:

$$\mathcal{L} _ {m,n}(d) := y _ {m,n}\sinh(dy _ {m,n}) = \frac{1}{4}\ln\left(\frac{n}{m}\right)\left[\left(\frac{n}{m}\right)^{d/2} - \left(\frac{m}{n}\right)^{d/2}\right] > 0.$$

Because individual shells satisfy $\phi_1'(0) = 18\pi^2\mathrm{e}^{-\pi} - 15\pi\mathrm{e}^{-\pi} - 4\pi(2\pi^2-3\pi)\mathrm{e}^{-\pi} \approx +0.0394987708125 \neq 0$, naive half-line shells $\phi_m(\lvert u\rvert)$ suffer a $C^1$ jump at $u = 0$. The smooth **Jacobi–Sinh shells**:

$$\Phi_m^{\mathrm{ABC}}(z) := w _ {\mathrm{Jac}}(z) \phi_m(z) + w _ {\mathrm{Jac}}(-z) \phi_m(-z), \qquad w _ {\mathrm{Jac}}(z) := \frac{1}{1 + \exp(-2\pi\sinh(8z))},$$

are even, holistic on $\lvert\mathrm{Im}(z)\rvert < \pi/8$, satisfy $(\Phi_m^{\mathrm{ABC}})^{(2k+1)}(0) = 0$ for all $k \ge 0$, and sum identically to $\Phi(z)$ whenever $\Phi(-z) = \Phi(z)$.

### 2.4 Non-Oscillatory $(\ln t)^3$ Stieltjes Diagonal and $\mathcal{O}(t^{-3/2})$ Boundary Self-Damping (Theorems 4.1 & 4.2)

In the leading-branch Wronskian sum $\widehat{W} _ {\mathrm{lead}}(d,t) = \mathcal{D} _ {\mathrm{diag}}(d,t) + \mathcal{O} _ {\mathrm{off}}(d,t)$, the diagonal terms $m = n$ eliminate all oscillating phases via $\sin^2\phi_n(t) + \cos^2\phi_n(t) \equiv 1$ and $(\omega_n-\omega_n)\sinh(d(\omega_n-\omega_n)) \equiv 0$:

$$\mathcal{D} _ {\mathrm{diag}}(d,t) := \sum _ {n=1}^{N(t)} \frac{4 \omega_n(t)^2}{n} \mathrm{sinc} _ {\mathrm{hyp}}\left(2d \omega_n(t)\right) > 0, \qquad \mathrm{sinc} _ {\mathrm{hyp}}(x) := \frac{\sinh x}{x}.$$

Applying Euler–Maclaurin summation in terms of the Stieltjes constants $\gamma _ {\mathrm{E}} = \gamma_0^{\mathrm{St}}, \gamma_1^{\mathrm{St}}, \gamma_2^{\mathrm{St}}$ and $L = 2\vartheta'(t) = \ln(t/2\pi) + \mathcal{O}(t^{-2})$ yields:

$$\mathcal{D} _ {\mathrm{diag}}(0,t) = \frac{1}{6}L^3 + \gamma _ {\mathrm{E}}L^2 - 4\gamma_1^{\mathrm{St}}L + 4\gamma_2^{\mathrm{St}} + \mathcal{O}\left(t^{-3/2}\ln^2 t\right), \qquad \mathcal{D} _ {\mathrm{diag}}(d,t) = \frac{dL\cosh(dL) - \sinh(dL)}{2d^3} + \mathcal{O}\left(\mathrm{e}^{dL}L\right).$$

Because the integrand $f(x) = (\frac{1}{2}L - \ln x)^2/x$ vanishes together with its first derivative at the upper cutoff $x = \mathrm{e}^{L/2} = \sqrt{t/2\pi}$, the sawtooth jump terms of order $\mathcal{O}(\delta_t)$ and $\mathcal{O}(\delta_t^2)$ at $N(t) = \lfloor\sqrt{t/2\pi}\rfloor$ vanish identically. Simultaneously, for the slowest adjacent boundary mode $(m,n) = (N-1,N)$ with minimum frequency gap $\Omega _ {\min}(t) = \ln(N/(N-1)) \sim \sqrt{2\pi/t}$, the slow difference-frequency cross-amplitude $A _ {N-1,N}(0,t)$ satisfies (for $N(t) \ge 3$, and $\le 4(N(t)-1)^{-3}$ for all $N(t) \ge 2$):

$$A _ {N-1,N}(0,t) := \frac{(\omega _ {N-1}(t) + \omega_N(t))^2}{\sqrt{(N-1)N}} \le \frac{9}{4 (N(t)-1)^3} = \mathcal{O}\left(t^{-3/2}\right) \longrightarrow 0, \qquad \frac{2A _ {N-1,N}(0,t)}{\mathcal{D} _ {\mathrm{diag}}(0,t)} = \mathcal{O}\left(t^{-3/2}(\ln t)^{-3}\right) \longrightarrow 0.$$

### 2.5 Incomplete-Gamma Tail Bound and Two-Dimensional Gabcke–Olver Envelopes (Theorems 5.1 & 5.2)

On the shifted horizontal ray $u = \xi + \mathrm{i}\eta$ ($\xi \ge 0$, $0 < \eta < \pi/8$, $\cos 4\eta > 0$), substituting $w = \mathrm{e}^{4\xi} \ge 1$ and integrating by parts yields the closed-form geometric-series tail bound:

$$\left\lvert\mathcal{I} _ {\mathrm{tail}}(N, z, \eta)\right\rvert \le B _ {\mathrm{tail}}(N,d,t,\eta) := \cosh(2t\eta) \frac{b _ {N+1}(d,\eta)}{1 - q_N(\eta)},$$

where for every $m \ge N+1$ the single-shell bound $b_m(d,\eta)$ and geometric ratio $q_N(\eta) < 1$ are:

$$b_m(d,\eta) := \left[\frac{2\pi^2 m^4}{4\pi m^2\cos 4\eta - (5+2d)} + \frac{3\pi m^2}{4\pi m^2\cos 4\eta - (1+2d)}\right]\mathrm{e}^{-\pi m^2\cos 4\eta}, \qquad q_N(\eta) := \left(\frac{N+2}{N+1}\right)^4\mathrm{e}^{-\pi(2N+3)\cos 4\eta}.$$

Relative to the saddle peak at $m = N(t)$, every tail shell $m \ge N(t)+1$ is suppressed by the uniform Gaussian barrier $\mathcal{B} _ {\mathrm{Gauss}}(m, N(t), \eta) = \exp(-\pi(m^2-N(t)^2)\cos 4\eta) \le \exp(-4\pi(N(t)+1)\cos 4\eta)$ for $m \ge N(t)+2$.

Furthermore, because the $u = 0$ remainder originates at half the logarithmic lever of the main sum ($\bar{u} _ {\mathrm{rem}} \approx \frac{1}{4}\ln(t/2\pi) = L/4$), the transverse amplification ratio obeys the exact **Mirror-Wall Damping Identity**:

$$\frac{\mathrm{sinc} _ {\mathrm{hyp}}(dL/2)}{\mathrm{sinc} _ {\mathrm{hyp}}(dL)} = \frac{\sinh(dL/2)/(dL/2)}{\sinh(dL)/(dL)} \equiv \frac{1}{\cosh(dL/2)} \le 2\left(\frac{t}{2\pi}\right)^{-d/2} \qquad (d \ge 0,   L > 0).$$

Combining this identity with Gabcke's $d = 0$ remainder theorem yields the two-dimensional Wronskian remainder envelope:

$$B _ {\widetilde{W}}^{(C_0)}(d,t) := 0.650 L \mathrm{sinc} _ {\mathrm{hyp}}\left(\frac{dL}{2}\right)t^{-3/4}\mathrm{e}^{-\pi t/2}, \qquad \frac{B _ {\widetilde{W}}^{(C_0)}(d,t)}{\mathcal{D} _ {\mathrm{diag}}(d,t) \mathrm{e}^{-\pi t/2}} = \mathcal{O}\left(\frac{t^{-3/4}(\ln t)^{-2}}{\cosh(dL/2)}\right) \longrightarrow 0.$$

## 3. Reproducible Verification Workflow

Run the standalone verification and figure-generation suite from the repository root:

```bash
python scripts/verify_paper5_mother_wavelet_gabcke_bounds.py
```

The script executes in under 15 seconds on CPU/GPU, performs 50-decimal `mpmath` evaluations, rigorous `mpmath.iv` (`iv.dps = 35`) interval enclosures using Olver's explicit Stirling remainder bound, and batched PyTorch `float64`/`complex128` continuum scans across $1{,}250{,}000 + 440{,}000 + 502{,}000$ grid cells, and regenerates Figures 1 and 2 (`figures/fig1_mother_wavelet_abc_matrix.pdf` and `figures/fig2_stieltjes_diagonal_and_gabcke_bounds.pdf`).
