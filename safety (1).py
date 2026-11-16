"""Requirement-level attribution for safety shields (water tank and grid world).

Prints every number reported in the paper: coalition values, global and
local attributions, isolation/removal/Banzhaf baselines, restrictiveness shares over
all winning configurations and over reachable configurations, sensitivity table and
runtime figures.
Usage:  python safety.py
"""
import itertools, math, time, platform, resource
import numpy as np

VI_TOL = 1e-11          # value iteration stops when the sup-norm update is below this
TIE_TOL = 1e-9          # actions within this tolerance of the best Q-value are optimal

# ----------------------------------------------------------------- water tank
T_CAP, T_SCALE = 100, 2
T_LMAX = T_CAP * T_SCALE
T_INIT = 50
CLOSE, OPEN = 0, 1
T_ACT = (CLOSE, OPEN)
T_REQS = (1, 2, 3, 4)
T_LONG = {1: "min level", 2: "capacity", 3: "close-hold", 4: "open-hold"}
N1 = N2 = 2
OK, ERR12 = 0, 1

# ----------------------------------------------------------------- grid world
LAYOUT = ["G........", ".........", ".........", "..e......",
          "##e###.##", "..e......", ".........", ".........", "S........"]
ROWS, COLS = len(LAYOUT), len(LAYOUT[0])
NCELL = ROWS * COLS
VEH_PATH = [(4, 2), (4, 2), (4, 2), (3, 2), (2, 2), (3, 2)]   # phase 0..5, initial phase 0
NPHASE = len(VEH_PATH)
GACT = [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]            # up, down, right, left, stay
GACT_NAME = ["up", "down", "right", "left", "stay"]
NACT = len(GACT)
G_REQS = (1, 2, 3)
G_LONG = {1: "obstacles", 2: "vehicle", 3: "exposure"}
M1 = M2 = 2
cell = lambda r, c: r * COLS + c
rc = lambda i: (int(i // COLS), int(i % COLS))


def subsets(reqs):
    return [frozenset(C) for k in range(len(reqs) + 1)
            for C in itertools.combinations(reqs, k)]


T_SUB = subsets(T_REQS)
G_SUB = subsets(G_REQS)


def fmt(C):
    return "{" + ",".join(str(i) for i in sorted(C)) + "}" if C else "{}"


# ----------------------------------------------------------------- game theory
def shapley(v, players):
    n = len(players)
    f = [math.factorial(i) for i in range(n + 1)]
    phi = {i: 0.0 for i in players}
    for i in players:
        for k in range(n):
            for C in itertools.combinations([p for p in players if p != i], k):
                S = frozenset(C)
                phi[i] += f[k] * f[n - k - 1] / f[n] * (v(S | {i}) - v(S))
    return phi


def banzhaf(v, players):
    n = len(players)
    b = {i: 0.0 for i in players}
    for i in players:
        for k in range(n):
            for C in itertools.combinations([p for p in players if p != i], k):
                S = frozenset(C)
                b[i] += (v(S | {i}) - v(S)) / 2 ** (n - 1)
    return b


def minimal_cores(tab, players):
    out = []
    for k in range(len(players) + 1):
        for C in itertools.combinations(players, k):
            if tab[frozenset(C)] == 1 and not any(set(c) <= set(C) for c in out):
                out.append(C)
    return out


def monotone(v, reqs):
    for C in subsets(reqs):
        for j in reqs:
            if j not in C and v(C | {j}) < v(C) - 1e-9:
                return False
    return True


def pct(phi):
    t = sum(phi.values())
    return {i: (100.0 * phi[i] / t if abs(t) > 1e-12 else 0.0) for i in phi}


# ----------------------------------------------------------------- shared tools
def reachable(NEXT, sh, init, n):
    """Configurations reachable from init when the agent may take any action
    admitted by sh and the environment any successor admitted by the model."""
    seen = np.zeros(n, bool); seen[init] = True
    frontier = np.array([init])
    while frontier.size:
        new = []
        for a in NEXT:
            src = frontier[sh[a][frontier]]
            if src.size:
                new.append(NEXT[a][src].ravel())
        if not new:
            break
        cand = np.unique(np.concatenate(new))
        cand = cand[~seen[cand]]
        seen[cand] = True
        frontier = cand
    return seen


def q_values(NEXT, sh, R, V, gamma, stochastic):
    acts = sorted(NEXT)
    Q = np.full((len(acts), len(R)), -np.inf)
    for a in acts:
        cont = V[NEXT[a]].mean(axis=1) if stochastic else V[NEXT[a]]
        Q[a] = np.where(sh[a], R + gamma * cont, -np.inf)
    return Q


def optimal_mixture(Q):
    """Policy that randomizes uniformly over all optimal actions (no tie-breaking)."""
    m = Q.max(axis=0)
    live = np.isfinite(m)
    opt = np.isfinite(Q) & (np.abs(Q - np.where(live, m, 0.0)) < TIE_TOL)
    P = opt / np.maximum(opt.sum(axis=0), 1)
    return P, live, opt


def occupancy(NEXT, P, live, init, n, gamma, iters=20000, tol=1e-14):
    """Normalized discounted occupancy measure, Eq. (24)."""
    mu0 = np.zeros(n); mu0[init] = 1.0 - gamma
    d = mu0.copy()
    for _ in range(iters):
        nd = mu0.copy()
        for a in NEXT:
            nx = NEXT[a]
            w = gamma * d * P[a] * live
            if nx.ndim == 2:
                np.add.at(nd, nx.ravel(), np.repeat(w / nx.shape[1], nx.shape[1]))
            else:
                np.add.at(nd, nx, w)
        if np.abs(nd - d).max() < tol:
            break
        d = nd
    d = np.where(d > 1e-15, d, 0.0)
    return d / d.sum()


# ----------------------------------------------------------------- water tank
def hold_automata(k):
    """Hold of k steps including the switching step.
    phi3: 0 = last action open, 1 = last action close, 2..k = pending closes.
    phi4: 0 = last action close, 1 = last action open, 2..k = pending opens."""
    n, err = k + 2, k + 1
    nxt = lambda p: p + 1 if p < k else 1
    d3 = np.zeros((n, 2), int); d4 = np.zeros((n, 2), int)
    d3[0] = [2 if k >= 2 else 1, 0]; d3[1] = [1, 0]
    d4[0] = [0, 2 if k >= 2 else 1]; d4[1] = [0, 1]
    for p in range(2, k + 1):
        d3[p] = [nxt(p), err]
        d4[p] = [err, nxt(p)]
    d3[err] = [err, err]; d4[err] = [err, err]
    F = np.ones(n, bool); F[err] = False
    return n, d3, d4, F


class TankGame:
    """Common game G = Q_R x Q_E. The abstraction state is the water level, which
    is also the MDP state, so a configuration (s, g) is identified with g.
    The MDP draws the net level change uniformly from the abstraction successors;
    clipping to [0, capacity] is part of the dynamics for every coalition."""

    def __init__(self, i_min=1.0, i_max=2.0, o_min=0.0, o_max=1.0,
                 hold=3, gamma=0.95, lstar=99.0, sigma=4.0):
        s = T_SCALE
        self.gamma, self.lstar, self.sigma = gamma, lstar, sigma
        self.N3, self.D3, self.D4, self.F = hold_automata(hold)
        self.N4 = self.N3
        self.deltas = {
            OPEN:  np.arange(round((i_min - o_max) * s),
                             round((i_max - o_min) * s) + 1),
            CLOSE: np.arange(round(-o_max * s), round(-o_min * s) + 1)}
        self.n = (T_LMAX + 1) * N1 * N2 * self.N3 * self.N4
        idx = np.arange(self.n)
        self.q4 = idx % self.N4
        self.q3 = (idx // self.N4) % self.N3
        self.q2 = (idx // (self.N4 * self.N3)) % N2
        self.q1 = (idx // (self.N4 * self.N3 * N2)) % N1
        self.lev = idx // (self.N4 * self.N3 * N2 * N1)
        self.init = self.index(T_INIT * T_SCALE, OK, OK, 1, 0)   # initial valve closed
        self._transitions()

    def index(self, lev, q1, q2, q3, q4):
        return (((lev * N1 + q1) * N2 + q2) * self.N3 + q3) * self.N4 + q4

    def _transitions(self):
        self.NEXT = {}
        for a in T_ACT:
            d = self.deltas[a]
            nlev = np.clip(self.lev[:, None] + d[None, :], 0, T_LMAX)
            nq1 = np.where((self.q1[:, None] == ERR12) | (nlev == 0), ERR12, OK)
            nq2 = np.where((self.q2[:, None] == ERR12) | (nlev == T_LMAX), ERR12, OK)
            nq3 = self.D3[self.q3, a][:, None] * np.ones_like(nlev)
            nq4 = self.D4[self.q4, a][:, None] * np.ones_like(nlev)
            self.NEXT[a] = self.index(nlev, nq1, nq2, nq3, nq4)

    def safe(self, C):
        m = np.ones(self.n, bool)
        if 1 in C: m &= self.q1 == OK
        if 2 in C: m &= self.q2 == OK
        if 3 in C: m &= self.F[self.q3]
        if 4 in C: m &= self.F[self.q4]
        return m

    def solve(self, C):
        W = self.safe(C)
        while True:
            ok = {a: W[self.NEXT[a]].all(axis=1) for a in T_ACT}
            Wn = W & (ok[CLOSE] | ok[OPEN])
            if np.array_equal(Wn, W):
                return W, ok
            W = Wn

    def rewards(self):
        l = self.lev / T_SCALE
        return -(1.0 - np.exp(-((l - self.lstar) ** 2) / (2 * self.sigma ** 2)))


def t_vi(g, sh, R, iters=20000):
    V = np.zeros(g.n)
    for _ in range(iters):
        Q = q_values(g.NEXT, sh, R, V, g.gamma, True)
        m = Q.max(axis=0)
        Vn = np.where(np.isfinite(m), m, 0.0)
        if np.abs(Vn - V).max() < VI_TOL:
            return Vn
        V = Vn
    return V


def t_threshold(g, data, C, action, low=False):
    sh = data[C]["sh"][action]
    q3, q4 = (1, 0) if action == OPEN else (0, 1)
    v = [l for l in range(T_LMAX + 1) if sh[g.index(l, OK, OK, q3, q4)]]
    if not v:
        return None
    return (min(v) if low else max(v)) / T_SCALE


def t_solve_all(game):
    R = game.rewards()
    data = {}; tsyn = tvi = 0.0
    for C in T_SUB:
        a = time.perf_counter(); W, sh = game.solve(C)
        b = time.perf_counter(); V = t_vi(game, sh, R)
        c = time.perf_counter()
        tsyn += b - a; tvi += c - b
        data[C] = dict(W=W, sh=sh, V=V, J=float(V[game.init]))
    return data, R, tsyn, tvi


# ----------------------------------------------------------------- grid world
class GridGame:
    """Configuration = (agent cell, vehicle phase, monitors). The vehicle phase is
    shared by the abstraction and the collision monitor. A collision occurs when
    the agent's next cell equals the vehicle's next cell (swaps are not counted).
    Moves off the grid leave the agent in place. The goal is absorbing."""

    def __init__(self, gamma=0.98, max_expo=2):
        self.gamma, self.max_expo = gamma, max_expo
        self.M3 = max_expo + 2; self.E_ERR = self.M3 - 1
        self.n = NCELL * NPHASE * M1 * M2 * self.M3
        idx = np.arange(self.n)
        self.q3 = idx % self.M3
        self.q2 = (idx // self.M3) % M2
        self.q1 = (idx // (self.M3 * M2)) % M1
        self.ph = (idx // (self.M3 * M2 * M1)) % NPHASE
        self.cl = idx // (self.M3 * M2 * M1 * NPHASE)
        w, e, s, gl = set(), set(), None, None
        for r, row in enumerate(LAYOUT):
            for c, ch in enumerate(row):
                if ch == "#": w.add((r, c))
                elif ch == "e": e.add((r, c))
                elif ch == "S": s = (r, c)
                elif ch == "G": gl = (r, c)
        self.WALLS, self.EXPO, self.START, self.GOAL = w, e, s, gl
        self.is_wall = np.array([rc(c) in w for c in range(NCELL)])
        self.is_expo = np.array([rc(c) in e for c in range(NCELL)])
        self.veh = np.array([cell(*VEH_PATH[p]) for p in range(NPHASE)])
        self.goal = cell(*gl)
        self.init = self.index(cell(*s), 0, OK, OK, 0)
        self._transitions()

    def index(self, c, ph, q1, q2, q3):
        return (((c * NPHASE + ph) * M1 + q1) * M2 + q2) * self.M3 + q3

    def _transitions(self):
        self.NEXT = {}
        r, c = self.cl // COLS, self.cl % COLS
        for a, (dr, dc) in enumerate(GACT):
            nr = np.clip(r + dr, 0, ROWS - 1)
            nc = np.clip(c + dc, 0, COLS - 1)
            ncell = np.where(self.cl == self.goal, self.goal, nr * COLS + nc)
            nph = (self.ph + 1) % NPHASE
            nq1 = np.where((self.q1 == ERR12) | self.is_wall[ncell], ERR12, OK)
            nq2 = np.where((self.q2 == ERR12) | (ncell == self.veh[nph]), ERR12, OK)
            inc = np.minimum(self.q3 + 1, self.E_ERR)
            nq3 = np.where(self.q3 == self.E_ERR, self.E_ERR,
                           np.where(self.is_expo[ncell], inc, 0))
            self.NEXT[a] = self.index(ncell, nph, nq1, nq2, nq3)

    def safe(self, C):
        m = np.ones(self.n, bool)
        if 1 in C: m &= self.q1 == OK
        if 2 in C: m &= self.q2 == OK
        if 3 in C: m &= self.q3 != self.E_ERR
        return m

    def solve(self, C):
        W = self.safe(C)
        while True:
            ok = {a: W[self.NEXT[a]] for a in range(NACT)}
            Wn = W & np.logical_or.reduce([ok[a] for a in range(NACT)])
            if np.array_equal(Wn, W):
                return W, ok
            W = Wn

    def rewards(self):
        # -1 for every step taken from a non-goal cell (including the step that
        # enters the goal), 0 afterwards
        return np.where(self.cl == self.goal, 0.0, -1.0)


def g_vi(g, sh, R, iters=50000):
    V = np.zeros(g.n); DEAD = -1.0 / (1.0 - g.gamma)
    for _ in range(iters):
        Q = q_values(g.NEXT, sh, R, V, g.gamma, False)
        m = Q.max(axis=0)
        Vn = np.where(np.isfinite(m), m, DEAD)
        Vn = np.where(g.cl == g.goal, 0.0, Vn)
        if np.abs(Vn - V).max() < VI_TOL:
            return Vn
        V = Vn
    return V


def g_solve_all(game):
    R = game.rewards()
    data = {}; tsyn = tvi = 0.0
    for C in G_SUB:
        a = time.perf_counter(); W, sh = game.solve(C)
        b = time.perf_counter(); V = g_vi(game, sh, R)
        c = time.perf_counter()
        tsyn += b - a; tvi += c - b
        data[C] = dict(W=W, sh=sh, V=V, J=float(V[game.init]))
    return data, R, tsyn, tvi


# ----------------------------------------------------------------- reporting
def line(t):
    print("\n" + "=" * 78); print(t); print("=" * 78)


def first_order_policy(opt, live, order):
    """Deterministic policy selecting the first optimal action in `order`."""
    n = opt.shape[1]
    chosen = np.full(n, -1)
    for a in order:
        chosen = np.where((chosen < 0) & opt[a], a, chosen)
    P = np.zeros(opt.shape)
    P[np.maximum(chosen, 0), np.arange(n)] = live
    return P


def coalition_table(data, reqs):
    empty = frozenset()
    print("\nTable of coalition values")
    print(f"  {'C':<11}{'J*(C)':>14}{'v_glo(C)':>12}")
    for C in subsets(reqs):
        print(f"  {fmt(C):<11}{data[C]['J']:14.4f}{data[empty]['J'] - data[C]['J']:12.4f}")


def analyse(game, data, R, reqs, names, stochastic, order):
    """Global attribution, baselines and restrictiveness shares, as in the paper."""
    full, empty = frozenset(reqs), frozenset()
    acts = sorted(game.NEXT)
    sh, W = data[full]["sh"], data[full]["W"]
    X = reachable(game.NEXT, sh, game.init, game.n) & W
    Q = q_values(game.NEXT, sh, R, data[full]["V"], game.gamma, stochastic)
    _, live, opt = optimal_mixture(Q)
    rho = lambda C: sum((~data[C]["sh"][a]).astype(float) for a in acts)
    gJ = lambda C: data[empty]["J"] - data[C]["J"]
    res = lambda mu: (lambda C: float((mu * rho(C)).sum()))

    mu_all = W.astype(float) / W.sum()           # all winning configurations
    mu_rch = X.astype(float) / X.sum()           # reachable configurations
    mu_occ = occupancy(game.NEXT, first_order_policy(opt, live, order), live,
                       game.init, game.n, game.gamma)
    mu_rev = occupancy(game.NEXT, first_order_policy(opt, live, order[::-1]), live,
                       game.init, game.n, game.gamma)

    phiJ = shapley(gJ, reqs)
    shares = {k: pct(shapley(res(mu), reqs)) for k, mu in
              [("uniform", mu_all), ("reachable", mu_rch), ("occupancy", mu_occ),
               ("occ. reversed ties", mu_rev)]}
    bz = banzhaf(gJ, reqs); tot = gJ(full); sb = sum(bz.values())
    bzn = {i: bz[i] * tot / sb for i in reqs}
    iso = {i: gJ(frozenset({i})) for i in reqs}
    rem = {i: gJ(full) - gJ(full - {i}) for i in reqs}

    print(f"\nconfigurations: all winning={int(W.sum())}  reachable={int(X.sum())}  "
          f"reachable with ties={int((X & (opt.sum(axis=0) > 1)).sum())}")
    print(f"monotone: v_glo={monotone(gJ, reqs)}  "
          f"v_res={all(monotone(res(m), reqs) for m in (mu_all, mu_rch, mu_occ))}")
    print(f"efficiency: sum phi={sum(phiJ.values()):.6f}  v(N)={tot:.6f}")
    print(f"\nshares (%)        {'return':>8}" + "".join(f"{k:>20}" for k in shares))
    pJ = pct(phiJ)
    for i in reqs:
        print(f"phi{i} {names[i]:<13}{abs(pJ[i]):8.1f}" +
              "".join(f"{abs(shares[k][i]):20.1f}" for k in shares))
    print(f"\n{'requirement':<18}{'isolation':>11}{'removal':>10}{'Shapley':>10}"
          f"{'Banzhaf norm.':>15}")
    for i in reqs:
        print(f"phi{i} {names[i]:<13}{iso[i]:11.4f}{rem[i]:10.4f}{phiJ[i]:10.4f}{bzn[i]:15.4f}")
    print(f"{'sum':<18}{sum(iso.values()):11.4f}{sum(rem.values()):10.4f}"
          f"{sum(phiJ.values()):10.4f}{sum(bzn.values()):15.4f}")
    print(f"{'total cost':<18}{tot:11.4f}")


def run_tank():
    line("WATER TANK")
    game = TankGame()
    data, R, tsyn, tvi = t_solve_all(game)
    full, empty = frozenset(T_REQS), frozenset()
    print(f"|G|={game.n} game states  coalitions={len(T_SUB)}  "
          f"synthesis={tsyn:.2f}s  value iteration={tvi:.2f}s")
    print(f"net level change (litres, uniform): open {game.deltas[OPEN]/T_SCALE}, "
          f"close {game.deltas[CLOSE]/T_SCALE}")
    print(f"thresholds: open up to {t_threshold(game, data, full, OPEN)} l, "
          f"close from {t_threshold(game, data, full, CLOSE, True)} l")
    coalition_table(data, T_REQS)
    analyse(game, data, R, T_REQS, T_LONG, True, [CLOSE, OPEN])

    print("\nlocal attribution: bands and minimal unsafe cores (no hold pending)")
    W = data[full]["W"]
    for action, q3, q4, name in [(OPEN, 1, 0, "open"), (CLOSE, 0, 1, "close")]:
        prev = None
        for l in range(T_LMAX + 1):
            gi = game.index(l, OK, OK, q3, q4)
            if not W[gi]:
                continue
            tab = {C: (0.0 if data[C]["sh"][action][gi] else 1.0) for C in T_SUB}
            key = None
            if tab[full] == 1:
                ph = shapley(lambda C: tab[C], T_REQS)
                key = (tuple(round(ph[i], 3) for i in T_REQS),
                       tuple(map(tuple, minimal_cores(tab, T_REQS))))
            if key != prev and key is not None:
                print(f"  blocking '{name}' from {l/T_SCALE:6.1f} l: "
                      f"shapley={key[0]}  cores={key[1]}")
            prev = key
    return tsyn, tvi, game.n


def run_grid():
    line("GRID WORLD")
    game = GridGame()
    data, R, tsyn, tvi = g_solve_all(game)
    full = frozenset(G_REQS)
    print(f"|G|={game.n} game states  coalitions={len(G_SUB)}  "
          f"synthesis={tsyn:.3f}s  value iteration={tvi:.3f}s")
    coalition_table(data, G_REQS)
    analyse(game, data, R, G_REQS, G_LONG, False, list(range(NACT)))

    print("\nlocal attribution and minimal unsafe cores (all winning configurations)")
    sh, W = data[full]["sh"], data[full]["W"]
    patt, memo = {}, {}
    acc = {i: np.zeros(NCELL) for i in G_REQS}; cnt = np.zeros(NCELL)
    for gi in np.where(W)[0]:
        for a in range(NACT):
            if sh[a][gi]:
                continue
            key = tuple(0 if data[C]["sh"][a][gi] else 1 for C in G_SUB)
            if key not in memo:
                tab = {C: float(k) for C, k in zip(G_SUB, key)}
                memo[key] = (shapley(lambda C: tab[C], G_REQS),
                             tuple(map(tuple, minimal_cores(tab, G_REQS))))
            ph, cs = memo[key]
            patt[cs] = patt.get(cs, 0) + 1
            for i in G_REQS:
                acc[i][game.cl[gi]] += ph[i]
            cnt[game.cl[gi]] += 1
    print(f"  blocked state-action pairs: {sum(patt.values())}")
    print(f"  with more than one minimal core: {sum(v for k, v in patt.items() if len(k) > 1)}")
    for k, v in sorted(patt.items(), key=lambda x: -x[1]):
        print(f"    cores={k}  occurrences={v}")
    print("  cells with joint responsibility (mean attribution per cell, max < 0.9):")
    for c in range(NCELL):
        if cnt[c]:
            v = {i: acc[i][c] / cnt[c] for i in G_REQS}
            if max(v.values()) < 0.9:
                print(f"    cell {rc(c)}: " + "  ".join(f"phi{i}={v[i]:.2f}" for i in G_REQS))
    return tsyn, tvi, game.n


def run_sensitivity():
    line("SENSITIVITY: cost of safety and Shapley shares (%), one parameter at a time")
    print(f"{'water tank':<20}{'cost':>10}" + "".join(f"{'phi'+str(i):>8}" for i in T_REQS))
    for name, kw in [("baseline", {}),
                     ("gamma = 0.90", dict(gamma=0.90)), ("gamma = 0.99", dict(gamma=0.99)),
                     ("l* = 95", dict(lstar=95.0)), ("l* = 97", dict(lstar=97.0))]:
        d, _, _, _ = t_solve_all(TankGame(**kw))
        f = lambda C: d[frozenset()]["J"] - d[C]["J"]
        p = pct(shapley(f, T_REQS))
        print(f"{name:<20}{f(frozenset(T_REQS)):10.4f}" +
              "".join(f"{abs(p[i]):8.1f}" for i in T_REQS))
    print(f"\n{'grid world':<20}{'cost':>10}" + "".join(f"{'phi'+str(i):>8}" for i in G_REQS))
    for name, kw in [("baseline", {}),
                     ("gamma = 0.95", dict(gamma=0.95)), ("gamma = 0.99", dict(gamma=0.99)),
                     ("max exposure = 1", dict(max_expo=1)), ("max exposure = 3", dict(max_expo=3))]:
        d, _, _, _ = g_solve_all(GridGame(**kw))
        f = lambda C: d[frozenset()]["J"] - d[C]["J"]
        p = pct(shapley(f, G_REQS))
        print(f"{name:<20}{f(frozenset(G_REQS)):10.4f}" +
              "".join(f"{abs(p[i]):8.1f}" for i in G_REQS))


if __name__ == "__main__":
    ts_t, tv_t, n_t = run_tank()
    ts_g, tv_g, n_g = run_grid()
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024   # before sensitivity runs
    line("RUNTIME (baseline benchmarks; wall-clock time of shield synthesis and value "
         "iteration over all coalitions)")
    print(f"water tank: {n_t} game states, 16 coalitions, synthesis {ts_t:.2f} s, "
          f"value iteration {tv_t:.2f} s, total {ts_t + tv_t:.2f} s")
    print(f"grid world: {n_g} game states, 8 coalitions, synthesis {ts_g:.3f} s, "
          f"value iteration {tv_g:.3f} s, total {ts_g + tv_g:.3f} s")
    print(f"peak resident memory of the process: {mem:.0f} MiB")
    print(f"Python {platform.python_version()}, NumPy {np.__version__}, "
          f"{platform.processor() or platform.machine()}")
    run_sensitivity()
