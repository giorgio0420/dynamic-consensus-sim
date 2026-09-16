# dynamic-consensus-sim

Simulator + interactive GUI for the non-smooth median-consensus protocol from:

> Z. A. Z. Sanai Dashti, C. Seatzu, M. Franceschelli, *"Dynamic Consensus on the Median
> Value in Open Multi-Agent Systems"*, IEEE Transactions on Automatic Control.

This is an **independent, from-scratch implementation** of the mathematical protocol
described in the paper (equations, not text or figures). Algorithms and formulas are not
copyrightable — only the paper's specific text/figures are, and neither is reproduced
here. The PDF itself is not included in this repository.

## The protocol

Each agent `i` holds a state `x_i` and a time-varying reference `u_i(t)`. Over a graph
`G(t)`:

```
ẋ_i = -λ Σ_{j∈N_i} sign(x_i - x_j) - α sign(x_i - u_i)
```

`λ` pulls neighbors to agreement, `α` pulls each agent toward its own reference. Because
`sign` is discontinuous, trajectories are simulated as Filippov solutions via small-step
explicit Euler (standard practice for sliding-mode / non-smooth consensus dynamics).

Two settings are implemented:

- **Closed network** (Theorem 4.1/4.2): fixed graph, agents reach consensus in finite time
  on the median of their (moving) references, provided `n·Π < α < 2λ/n`.
- **Open network** (Theorem 4.3): agents join/leave with a minimum dwell time `Δτ`; a
  joining agent starts at the mean of its neighbours (eq. 6). Consensus error stays
  bounded by `B` instead of vanishing, provided `B ≤ (α/n_max − Π)·Δτ`.

## Run

```bash
pip install -r requirements.txt
python -m tests.test_dynamics   # sanity checks
streamlit run app.py            # GUI
```

The GUI lets you pick closed/open scenario, tune `λ`, `α`, graph size, dwell time and
reference band `B`, run the simulation, and watch every agent's state converge (or track)
the median live, together with the theoretical bounds (`T1`, `B`) overlaid on the error
curve.

## Math reference

Notation: `x_i(t)` state, `u_i(t)` reference, `N_i(t)` neighbours, `n(t) ≤ n_max` agents,
`λ` coupling gain, `α` fidelity gain, `Π` bound on `|u̇_i|`, `B` reference band width,
`Δτ` minimum dwell time between network events.

**Median (Def. 3.1).** Not "sorted middle element" — variational:

$$\chi(t) = \arg\min_{\bar u} \sum_j |u_j(t) - \bar u|$$

For `n` even the minimizer is a pair; `m(u)` is their mean and `M(u) = [min, max]` is the
median interval, length `L_M`. Absolute value gives the median (vs. squared error giving
the mean) — that's why the protocol below uses `sign` where linear consensus uses the
Laplacian.

**Protocol (eq. 5).**

$$\dot x_i = -\lambda \sum_{j\in N_i(t)} \text{sign}(x_i - x_j) \;-\; \alpha\, \text{sign}(x_i - u_i)$$

with `sign(0)=0` for the ODE, and the set-valued `SIGN(0)=[-1,1]` for the Filippov analysis
below. In matrix form (`E` incidence matrix, `L=EEᵀ`): linear consensus is `ẋ=-Lx`, this
protocol is `ẋ = -λ E·sign(Eᵀx) - α·sign(x-u)` — same graph structure, `sign` replaces the
raw disagreement. Consequence: the algebraic connectivity `λ2` does not appear in the
convergence rate; only `n_max`, `λ`, `α` do.

**Join rule (eq. 6).** A joining agent starts at the mean of its neighbours:

$$x_i(t) = \frac{1}{|N_i(t)|}\sum_{j\in N_i(t)} x_j(t)$$

so it lands in the convex hull of existing states — it can never become the new max/min,
which is what keeps the Lyapunov argument valid across join events (§ below).

**Non-smooth machinery (Defs. 2.1–2.3, needed because `sign` is discontinuous).**

*Filippov solution* — replace the ODE with a differential inclusion:

$$\dot x \in K(x) \triangleq \bigcap_{\delta>0}\ \bigcap_{\mu(N)=0} \overline{\text{co}}\{f(B(x,\delta)\setminus N,\ t)\}$$

At a discontinuity the derivative becomes the convex hull of the one-sided limits
(`K(0)=[-1,1]` for `sign`). The field of eq. 5 is measurable and uniformly bounded, so a
Filippov solution exists for every initial condition.

*Clarke gradient* — same idea applied to a locally Lipschitz Lyapunov function `V`:

$$\partial V(x) \triangleq \text{co}\Big\{\lim_{i\to\infty}\nabla V(x_i)\ \Big|\ x_i\to x,\ x_i\notin\Omega_V\cup N\Big\}$$

For `V(x)=|x|`: `∂V(0)=[-1,1]`, the same object as `SIGN(0)`.

*Set-valued Lie derivative* —

$$\tilde{\mathcal L}V(x) \triangleq \{a\in\mathbb R\ |\ \exists v\in K(x):\ \zeta\cdot v = a\ \ \forall \zeta\in\partial V(x)\}$$

*Theorem 2.5 (finite-time convergence).* If `V=0` on the consensus subspace, `V>0`
elsewhere, and `dV/dt ≤ -ε < 0` a.e. off the subspace, then `V` (and `x`) reach it by
`t = V(x(0))/ε`. A constant-slope bound integrates to a straight line hitting zero at a
finite time — unlike smooth consensus (`V̇ ≤ -cV` integrates to an exponential that never
truly reaches zero). This is why `sign`-based consensus is finite-time.

**Theorem 4.1 — consensus (closed or open network).** With `0 < α < 2λ/n_max`, all agents
agree within `T1 = (max xᵢ(0) − min xᵢ(0)) / μ2`, `μ2 = 2(2λ/n_max − α)`, using the
Lyapunov function

$$V_1(x) = \text{mean}_{i\in I_{max}} x_i - \text{mean}_{i\in I_{min}} x_i$$

(gap between the top group and bottom group). A joining agent leaves `V1` unchanged
(eq. 6 keeps it inside the convex hull); a leaving agent can only shrink `V1` — so the
closed-network bound survives network churn unchanged.

**Theorem 4.2 — the consensus value is the median (closed network only).** With
`n·Π < α < 2λ/n`, the common value `c(t)` enters the median interval `M(u(t))` in finite
time. Summing eq. 5 over all agents cancels the coupling term (undirected graph), leaving
`ċ = -(α/n) Σ sign(c - u_i)`: a majority vote among agents whose reference is above/below
`c`, strength at least `α/n`, against a median that drifts at most `Π` — negative net drift
iff `α > nΠ`.

**Theorem 4.3 — bounded error (open network).** With `n_max·Π < α < 2λ/n_max` and

$$B \le \left(\frac{\alpha}{n_{max}} - \Pi\right)\Delta\tau$$

the tracking error `|c(t) - m(u(t))| ≤ B` for all `t ≥ T3` (bounded, not zero — the price
of allowing agents to join/leave). Each network event can move the median by at most `B`;
between events the continuous dynamics shrinks the error by `(α/n_max - Π)·Δτ`. The
condition above is exactly "net decrement per dwell period `D > 0`".

## Layout

```
dynamic_consensus/
  dynamics.py   protocol RHS, median/interval, Lyapunov V1
  graphs.py     random connected graph generator
  simulate.py   closed-network and open-network integrators
app.py          Streamlit GUI
tests/          assert-based self-checks
```
