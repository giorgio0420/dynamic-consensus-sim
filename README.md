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

## Layout

```
dynamic_consensus/
  dynamics.py   protocol RHS, median/interval, Lyapunov V1
  graphs.py     random connected graph generator
  simulate.py   closed-network and open-network integrators
app.py          Streamlit GUI
tests/          assert-based self-checks
```
