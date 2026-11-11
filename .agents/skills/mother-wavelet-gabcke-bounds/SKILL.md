---
name: mother-wavelet-gabcke-bounds
description: Scientific knowledge skill for Paper V of the Transverse Wronskian series on Riemann's Xi-function. Contains the canonical Wronskian normalization W(d,t) = 16 int_0^infty y sinh(2dy) I(y,t) dy, logarithmic shell dilation of the Jacobi-Riemann kernel, exact algebraic identity (1-s)(-s) = s(s-1) and Spira-Saidak-Zvengrowski modulus inequality |chi(s)| < 1, strip-holomorphic remainder kernel R_N^{ker}(z) and logistic partition w_4(z) vs the Im(z) = pi/48 pole obstruction of sinh(8z) cutoffs, cubic-logarithmic Stieltjes diagonal expansion, sharp (N-1,N) boundary bound vs O(t^{1/2}) off-diagonal cross-pair dominance, and t-uniform real-axis incomplete-Gamma tail bounds.
---

# Logarithmic Dilation of the Jacobi–Riemann Kernel, Strip-Holomorphic Shell Partitions, and Stieltjes Diagonal Asymptotics (Paper V)

## 1. Canonical Normalization and Scope

Paper V analyzes the individual theta shells $\phi _ m(u) = (2\pi^2 m^4\mathrm{e}^{9u} - 3\pi m^2\mathrm{e}^{5u})\exp(-\pi m^2\mathrm{e}^{4u})$ of Riemann's kernel $\Phi(u) = \sum _ {m=1}^{\infty} \phi _ m(u)$ under the canonical normalization

$$\Xi(z) = 8\int _ 0^{\infty} \Phi(u)\cos(2zu)\mathrm{d}u = 4\int _ {-\infty}^{\infty} \Phi(u)\cos(2zu)\mathrm{d}u,$$

$$W(d,t) := \frac{1}{2}\frac{\partial}{\partial d}\left\lvert\Xi(t - \mathrm{i} d)\right\rvert^2 = R I _ t - I R _ t = 16\int _ 0^{\infty} y\sinh(2dy) I(y,t)\mathrm{d}y,$$

where $I(y,t) := \int _ {-\infty}^{\infty} \Phi(\frac{x+y}{2})\Phi(\frac{x-y}{2})\cos(2tx)\mathrm{d}x$ and $\widetilde{W}(d,t) := W(d,t)/d$.

## 2. Rigorous Identities and Analytical Obstructions

### 2.1 Logarithmic Shell Dilation and Spira–Saidak–Zvengrowski Branch Damping

With $\varphi _ 1(w) := (2\pi^2\mathrm{e}^{9w} - 3\pi\mathrm{e}^{5w})\exp(-\pi\mathrm{e}^{4w})$, every shell satisfies $\phi _ m(u) = m^{-1/2}\varphi _ 1(u + \frac{1}{2}\ln m)$ and bilateral Laplace–Mellin transform ($\mathrm{Re}(s) > -2$):

$$\mathcal{G}(s) := \int _ {-\infty}^{\infty} \varphi _ 1(v)\mathrm{e}^{2(s-1/2)v}\mathrm{d}v = \frac{1}{8}s(s-1)\pi^{-s/2}\Gamma\left(\frac{s}{2}\right), \qquad \xi(s) = 4\mathcal{G}(s)\zeta(s).$$

Because $(1-s)(-s) = s^2 - s \equiv s(s-1)$ holds algebraically for all $s \in \mathbb{C}$ (do not confuse $1-s$ with $1-\bar{s}$), $\mathcal{G}(1-s)/\mathcal{G}(s) = \chi(s)$. By tetragamma convexity ($\mathrm{Re} \psi'' > 0$) for $t \ge 7 > t _ {\mathrm{crit}} \approx 6.289836$ and $d \in (0, 1/2]$ (Spira 1965; Saidak–Zvengrowski 2003):

$$\rho _ \Gamma(d,t) := \left\lvert\frac{\mathcal{G}(1-s)}{\mathcal{G}(s)}\right\rvert = \lvert\chi(s)\rvert < \mathrm{e}^{-2d\vartheta'(t)} < 1, \qquad \rho _ \Gamma(d,t) n^{2d} < \mathrm{e}^{-2d\omega _ n(t)} < 1 \quad (1 \le n \le N(t) - 1, \text{ and } n = N(t) \text{ when } \ln N(t) < \vartheta'(t)).$$

**Critical Scope Boundary:** Because $\rho _ \Gamma(d,t) < 1$ depends only on $\Gamma(s/2)$, it holds identically for Davenport–Heilbronn Dirichlet series that lack an Euler product and have off-critical zeros.

### 2.2 Canonical Wigner Lever, Pole Obstruction of $w _ {\mathrm{Jac}}$, and Weight-Free Remainder Kernel

In Wigner coordinates $(x,y) = (u+v, u-v)$ centered at $(x _ {m,n}, y _ {m,n}) = (-\frac{1}{2}\ln(mn), \frac{1}{2}\ln(n/m))$, the canonical lever is

$$\mathcal{L} _ {m,n}(d) := y _ {m,n}\sinh(2dy _ {m,n}) = \frac{1}{4}\ln\left(\frac{n}{m}\right)\left[\left(\frac{n}{m}\right)^d - \left(\frac{m}{n}\right)^d\right] > 0 \qquad (m \ne n, d > 0).$$

- **Pole Obstruction:** The double-exponential cutoff $w _ {\mathrm{Jac}}(z) = (1 + \exp(-2\pi\sinh(8z)))^{-1}$ has poles at $2\pi\sinh(8z) = \pm\mathrm{i}\pi$, giving $\lvert\mathrm{Im}(z)\rvert = \frac{\pi}{48} \approx 0.06545 < \frac{\pi}{8}$ inside the strip of holomorphy of $\Phi(z)$.
- **Weight-Free Remainder Kernel:** Separating the bilateral main sum requires no cutoff weight at all:

$$\Xi(z) = 4\sum _ {m=1}^N \left[\mathcal{G}(s)m^{-s} + \mathcal{G}(1-s)m^{-(1-s)}\right] + 8\int _ 0^{\infty} R _ N^{\mathrm{ker}}(u)\cos(2zu)\mathrm{d}u, \qquad R _ N^{\mathrm{ker}}(z) := \Phi(z) - \sum _ {m=1}^N \left[\phi _ m(z) + \phi _ m(-z)\right],$$

where $R _ N^{\mathrm{ker}}(z)$ is even and holomorphic throughout $\lvert\mathrm{Im}(z)\rvert < \frac{\pi}{8}$. When an infinite shell partition is desired, the logistic weight $w _ 4(z) := (1 + \mathrm{e}^{-8z})^{-1}$ places its nearest poles at $\lvert\mathrm{Im}(z)\rvert = \frac{\pi}{8}$.

### 2.3 Stieltjes Diagonal Expansion vs Off-Diagonal Cross-Pair Multiplicity

With $L := 2\vartheta'(t) = \ln(t/2\pi) + \mathcal{O}(t^{-2})$ and $N = N(t) = \lfloor\sqrt{t/2\pi}\rfloor \ge 2$:

$$\mathcal{D} _ {\mathrm{diag}}(0,t) := \sum _ {n=1}^{N(t)} \frac{4\omega _ n(t)^2}{n} = \frac{1}{6}L^3 + \gamma _ {\mathrm{E}}L^2 - 4\gamma _ 1^{\mathrm{St}}L + 4\gamma _ 2^{\mathrm{St}} + \mathcal{O}\left(t^{-3/2}\ln^2 t\right),$$

$$0 < A _ {N-1,N}(0,t) := \frac{(\omega _ {N-1}(t)+\omega _ N(t))^2}{\sqrt{(N-1)N}} < \frac{(3N-2)^2}{[N(N-1)]^{5/2}} < \frac{9}{(N-1)^3} = \mathcal{O}\left(t^{-3/2}\right).$$

**Critical Obstruction:** Across all $\binom{N(t)}{2}$ pairs, the $L^1$ off-diagonal amplitude sums to $\mathcal{S} _ {\mathrm{off}}^{L^1}(0,t) = \sum _ {1 \le m < n \le N(t)} 2A _ {m,n}(0,t) = \frac{96}{\sqrt{2\pi}} t^{1/2} + \mathcal{O}((\ln t)^2 t^{1/4}) \gg \mathcal{D} _ {\mathrm{diag}}(0,t)$, overwhelming the diagonal whenever phases align.

### 2.4 Real-Axis Incomplete-Gamma Tail Bound and Gabcke Domain

On the real ray $\eta = 0$ (avoiding the vertical-segment omission and $\cosh(2t\eta)$ growth of unsplit half-ray shifts):

$$\left\lvert\mathcal{I} _ {\mathrm{tail}}(N,z,0)\right\rvert \le \frac{b _ {N+1}(d,0)}{1 - q _ N(0)}, \qquad b _ m(d,0) := \left[\frac{2\pi^2 m^4}{4\pi m^2 - (5+2d)} + \frac{3\pi m^2}{4\pi m^2 - (1+2d)}\right]\mathrm{e}^{-\pi m^2}, \qquad q _ N(0) := \left(\frac{N+2}{N+1}\right)^4 \mathrm{e}^{-\pi(2N+3)} < 1,$$

and $\mathrm{sinc} _ {\mathrm{hyp}}(dL/2)/\mathrm{sinc} _ {\mathrm{hyp}}(dL) \equiv \cosh(dL/2)^{-1} \le 2(t/2\pi)^{-d/2}$. Gabcke's explicit theorem $\lvert R _ 1(t)\rvert \le 0.127 t^{-3/4}$ holds strictly on $d = 0$ for $t \ge 200$ after subtracting $(-1)^{N-1}(t/2\pi)^{-1/4}\Psi(p)$.

## 3. Reproducible Verification

```bash
python scripts/verify_paper5_mother_wavelet_gabcke_bounds.py
```
