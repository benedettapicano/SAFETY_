import itertools, math, time
import numpy as np

T_CAP, T_SCALE = 100, 2
T_LMAX = T_CAP * T_SCALE
T_GAMMA = 0.95
T_LSTAR, T_SIGMA, T_INIT = 99.0, 4.0, 50
CLOSE, OPEN = 0, 1
T_ACT = (CLOSE, OPEN)
T_REQS = (1, 2, 3, 4)
T_LONG = {1: "min level", 2: "capacity", 3: "close-hold", 4: "open-hold"}
N1 = N2 = 2
N3 = N4 = 5
OK, ERR12 = 0, 1
DELTA3 = np.array([[2, 0], [1, 0], [3, 4], [1, 4], [4, 4]])
DELTA4 = np.array([[0, 2], [0, 1], [4, 3], [4, 1], [4, 4]])
F3 = np.array([True, True, True, True, False])
F4 = np.array([True, True, True, True, False])

LAYOUT = ["G........", ".........", ".........", "..e......",
          "##e###.##", "..e......", ".........", ".........", "S........"]
ROWS, COLS = len(LAYOUT), len(LAYOUT[0])
NCELL = ROWS * COLS
VEH_PATH = [(4, 2), (4, 2), (4, 2), (3, 2), (2, 2), (3, 2)]
NPHASE = len(VEH_PATH)
MAX_EXPO, G_GAMMA = 2, 0.98
GACT = [(-1, 0), (1, 0), (0, 1), (0, -1), (0, 0)]
NACT = len(GACT)
G_REQS = (1, 2, 3)
G_LONG = {1: "obstacles", 2: "vehicle", 3: "exposure"}
M1 = M2 = 2
M3 = MAX_EXPO + 2
E_ERR = M3 - 1
cell = lambda r, c: r * COLS + c
rc = lambda i: (i // COLS, i % COLS)


def subsets(reqs):
    return [frozenset(C) for k in range(len(reqs) + 1)
            for C in itertools.combinations(reqs, k)]


T_SUB = subsets(T_REQS)
G_SUB = subsets(G_REQS)


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


class TankGame:
    def __init__(self, i_min=1.0, i_max=2.0, o_min=0.0, o_max=1.0):
        s = T_SCALE
        self.deltas = {
            OPEN:  np.arange(round((i_min - o_max) * s),
                             round((i_max - o_min) * s) + 1),
            CLOSE: np.arange(round(-o_max * s), round(-o_min * s) + 1)}
        self.n = (T_LMAX + 1) * N1 * N2 * N3 * N4
        idx = np.arange(self.n)
        self.q4 = idx % N4
        self.q3 = (idx // N4) % N3
        self.q2 = (idx // (N4 * N3)) % N2
        self.q1 = (idx // (N4 * N3 * N2)) % N1
        self.lev = idx // (N4 * N3 * N2 * N1)
        self.init = self.index(T_INIT * T_SCALE, OK, OK, 1, 0)
        self._transitions()

    def index(self, lev, q1, q2, q3, q4):
        return (((lev * N1 + q1) * N2 + q2) * N3 + q3) * N4 + q4

    def _transitions(self):
        self.NEXT = {}
        for a in T_ACT:
            d = self.deltas[a]
            nlev = np.clip(self.lev[:, None] + d[None, :], 0, T_LMAX)
            nq1 = np.where((self.q1[:, None] == ERR12) | (nlev == 0), ERR12, OK)
            nq2 = np.where((self.q2[:, None] == ERR12) | (nlev == T_LMAX),
                           ERR12, OK)
            nq3 = DELTA3[self.q3, a][:, None] * np.ones_like(nlev)
            nq4 = DELTA4[self.q4, a][:, None] * np.ones_like(nlev)
            self.NEXT[a] = (((nlev * N1 + nq1) * N2 + nq2) * N3 + nq3) * N4 + nq4

    def safe(self, C):
        m = np.ones(self.n, bool)
        if 1 in C: m &= self.q1 == OK
        if 2 in C: m &= self.q2 == OK
        if 3 in C: m &= F3[self.q3]
        if 4 in C: m &= F4[self.q4]
        return m

    def solve(self, C):
        W = self.safe(C)
        while True:
            ok = {a: W[self.NEXT[a]].all(axis=1) for a in T_ACT}
            Wn = W & (ok[CLOSE] | ok[OPEN])
            if np.array_equal(Wn, W):
                return W, ok
            W = Wn


def t_energy(l):
    return 1.0 - math.exp(-((l - T_LSTAR) ** 2) / (2 * T_SIGMA ** 2))


def t_rewards(g):
    return np.array([-t_energy(l / T_SCALE) for l in g.lev])


def t_vi(g, sh, R, iters=5000, tol=1e-11):
    V = np.zeros(g.n)
    for _ in range(iters):
        Q = np.full((2, g.n), -np.inf)
        for a in T_ACT:
            Q[a] = np.where(sh[a], R + T_GAMMA * V[g.NEXT[a]].mean(axis=1),
                            -np.inf)
        m = Q.max(axis=0)
        Vn = np.where(np.isfinite(m), m, 0.0)
        if np.abs(Vn - V).max() < tol:
            return Vn
        V = Vn
    return V


def t_greedy(g, sh, R, V):
    Q = np.full((2, g.n), -np.inf)
    for a in T_ACT:
        Q[a] = np.where(sh[a], R + T_GAMMA * V[g.NEXT[a]].mean(axis=1), -np.inf)
    return Q.argmax(axis=0), np.isfinite(Q.max(axis=0))


def t_occupancy(g, pol, live, iters=5000, tol=1e-13):
    mu0 = np.zeros(g.n); mu0[g.init] = 1.0 - T_GAMMA
    d = mu0.copy()
    for _ in range(iters):
        nd = mu0.copy()
        for a in T_ACT:
            src = np.where(live & (pol == a))[0]
            if src.size == 0:
                continue
            nxt = g.NEXT[a][src]
            w = T_GAMMA * d[src] / nxt.shape[1]
            np.add.at(nd, nxt.ravel(), np.repeat(w, nxt.shape[1]))
        if np.abs(nd - d).max() < tol:
            return nd
        d = nd
    return d


def t_threshold(g, data, C, action, low=False):
    sh = data[C]["sh"][action]
    q3, q4 = (1, 0) if action == OPEN else (0, 1)
    v = [l for l in range(T_LMAX + 1) if sh[g.index(l, OK, OK, q3, q4)]]
    if not v:
        return None
    return (min(v) if low else max(v)) / T_SCALE


def t_solve_all(game):
    R = t_rewards(game)
    data = {}
    for C in T_SUB:
        W, sh = game.solve(C)
        V = t_vi(game, sh, R)
        data[C] = dict(W=W, sh=sh, V=V, J=float(V[game.init]))
    return data, R


class GridGame:
    def __init__(self):
        self.n = NCELL * NPHASE * M1 * M2 * M3
        idx = np.arange(self.n)
        self.q3 = idx % M3
        self.q2 = (idx // M3) % M2
        self.q1 = (idx // (M3 * M2)) % M1
        self.ph = (idx // (M3 * M2 * M1)) % NPHASE
        self.cl = idx // (M3 * M2 * M1 * NPHASE)
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
        return (((c * NPHASE + ph) * M1 + q1) * M2 + q2) * M3 + q3

    def _transitions(self):
        self.NEXT = {}
        r, c = rc(self.cl)
        for a, (dr, dc) in enumerate(GACT):
            nr = np.clip(r + dr, 0, ROWS - 1)
            nc = np.clip(c + dc, 0, COLS - 1)
            ncell = np.where(self.cl == self.goal, self.goal, nr * COLS + nc)
            nph = (self.ph + 1) % NPHASE
            nq1 = np.where((self.q1 == ERR12) | self.is_wall[ncell], ERR12, OK)
            nq2 = np.where((self.q2 == ERR12) | (ncell == self.veh[nph]),
                           ERR12, OK)
            inc = np.minimum(self.q3 + 1, E_ERR)
            nq3 = np.where(self.q3 == E_ERR, E_ERR,
                           np.where(self.is_expo[ncell], inc, 0))
            self.NEXT[a] = ((((ncell * NPHASE + nph) * M1 + nq1) * M2 + nq2)
                            * M3 + nq3)

    def safe(self, C):
        m = np.ones(self.n, bool)
        if 1 in C: m &= self.q1 == OK
        if 2 in C: m &= self.q2 == OK
        if 3 in C: m &= self.q3 != E_ERR
        return m

    def solve(self, C):
        W = self.safe(C)
        while True:
            ok = {a: W[self.NEXT[a]] for a in range(NACT)}
            Wn = W & np.logical_or.reduce([ok[a] for a in range(NACT)])
            if np.array_equal(Wn, W):
                return W, ok
            W = Wn


def g_vi(g, sh, R, iters=20000, tol=1e-11):
    V = np.zeros(g.n); DEAD = -1.0 / (1.0 - G_GAMMA)
    for _ in range(iters):
        Q = np.full((NACT, g.n), -np.inf)
        for a in range(NACT):
            Q[a] = np.where(sh[a], R + G_GAMMA * V[g.NEXT[a]], -np.inf)
        m = Q.max(axis=0)
        Vn = np.where(np.isfinite(m), m, DEAD)
        Vn = np.where(g.cl == g.goal, 0.0, Vn)
        if np.abs(Vn - V).max() < tol:
            return Vn
        V = Vn
    return V


def line(t):
    print("\n" + "=" * 74); print(t); print("=" * 74)


def run_tank():
    line("WATER TANK")
    game = TankGame(); t0 = time.time()
    data, R = t_solve_all(game)
    full, empty = frozenset(T_REQS), frozenset()
    elapsed = time.time() - t0

    pol, live = t_greedy(game, data[full]["sh"], R, data[full]["V"])
    mu = t_occupancy(game, pol, live)
    mu_occ = np.where(mu > 1e-14, mu, 0.0); mu_occ /= mu_occ.sum()
    mu_uni = data[full]["W"].astype(float); mu_uni /= mu_uni.sum()
    rho = lambda C: sum((~data[C]["sh"][a]).astype(float) for a in T_ACT)

    gJ = lambda C: data[empty]["J"] - data[C]["J"]
    gU = lambda C: float((mu_uni * rho(C)).sum())
    gO = lambda C: float((mu_occ * rho(C)).sum())
    phiJ, phiU, phiO = (shapley(gJ, T_REQS), shapley(gU, T_REQS),
                        shapley(gO, T_REQS))
    bzJ = banzhaf(gJ, T_REQS)
    standalone = {i: gJ(frozenset({i})) for i in T_REQS}
    leaveout = {i: gJ(full) - gJ(full - {i}) for i in T_REQS}

    print(f"|Q_R|={N1*N2*N3*N4}  |Q_M|={T_LMAX+1}  |G|={game.n}"
          f"  coalitions={len(T_SUB)}  time={elapsed:.1f}s")
    print(f"thresholds: open up to {t_threshold(game,data,full,OPEN)} l, "
          f"close from {t_threshold(game,data,full,CLOSE,True)} l  (paper 93, 4)")
    print(f"monotone: dJ={monotone(gJ,T_REQS)} unif={monotone(gU,T_REQS)} "
          f"occ={monotone(gO,T_REQS)}")
    print(f"efficiency: sum={sum(phiJ.values()):.6f}  v(N)={gJ(full):.6f}")
    print(f"J*(0)={data[empty]['J']:.4f}  J*(N)={data[full]['J']:.4f}")

    print(f"\n{'requirement':<18}{'dJ':>10}{'%':>7}{'unif':>9}{'%':>7}"
          f"{'occ':>9}{'%':>7}")
    pJ, pU, pO = pct(phiJ), pct(phiU), pct(phiO)
    for i in T_REQS:
        print(f"phi{i} {T_LONG[i]:<13}{phiJ[i]:10.4f}{pJ[i]:6.1f}%"
              f"{phiU[i]:9.4f}{pU[i]:6.1f}%{phiO[i]:9.4f}{pO[i]:6.1f}%")

    print(f"\n{'requirement':<18}{'isolation':>12}{'removal':>12}"
          f"{'Shapley':>11}{'Banzhaf':>11}")
    for i in T_REQS:
        print(f"phi{i} {T_LONG[i]:<13}{standalone[i]:12.4f}{leaveout[i]:12.4f}"
              f"{phiJ[i]:11.4f}{bzJ[i]:11.4f}")
    print(f"{'sum':<18}{sum(standalone.values()):12.4f}"
          f"{sum(leaveout.values()):12.4f}{sum(phiJ.values()):11.4f}"
          f"{sum(bzJ.values()):11.4f}")
    print(f"{'total cost':<18}{gJ(full):12.4f}{gJ(full):12.4f}"
          f"{gJ(full):11.4f}{gJ(full):11.4f}")
    print(f"{'|error|':<18}{abs(sum(standalone.values())-gJ(full)):12.4f}"
          f"{abs(sum(leaveout.values())-gJ(full)):12.4f}"
          f"{abs(sum(phiJ.values())-gJ(full)):11.4f}"
          f"{abs(sum(bzJ.values())-gJ(full)):11.4f}")

    print("\nlocal attribution: bands and minimal unsafe cores")
    for action, q3, q4, name in [(OPEN, 1, 0, "open"), (CLOSE, 0, 1, "close")]:
        prev = None
        for l in range(T_LMAX + 1):
            gi = game.index(l, OK, OK, q3, q4)
            if not data[full]["W"][gi]:
                continue
            tab = {C: (0.0 if data[C]["sh"][action][gi] else 1.0)
                   for C in T_SUB}
            if tab[full] == 0:
                key = None
            else:
                ph = shapley(lambda C: tab[C], T_REQS)
                cs = minimal_cores(tab, T_REQS)
                key = (tuple(round(ph[i], 3) for i in T_REQS),
                       tuple(map(tuple, cs)))
            if key != prev:
                if key is not None:
                    print(f"  blocking '{name}' from {l/T_SCALE:6.1f} l: "
                          f"shapley={key[0]}  cores={key[1]}")
                prev = key

    print("\nsafety margin")
    def margin_up(C):
        t = t_threshold(game, data, frozenset(C), OPEN)
        return T_CAP - t if t is not None else 0.0
    def margin_lo(C):
        t = t_threshold(game, data, frozenset(C), CLOSE, low=True)
        return t if t is not None else 0.0
    pu, pl = shapley(margin_up, T_REQS), shapley(margin_lo, T_REQS)
    print(f"{'requirement':<18}{'upper':>10}{'lower':>10}{'total':>10}")
    for i in T_REQS:
        print(f"phi{i} {T_LONG[i]:<13}{pu[i]:10.2f}{pl[i]:10.2f}"
              f"{pu[i]+pl[i]:10.2f}")
    print(f"{'sum':<18}{sum(pu.values()):10.2f}{sum(pl.values()):10.2f}"
          f"{sum(pu.values())+sum(pl.values()):10.2f}")

    print("\nsensitivity to the environment abstraction")
    rows = []
    for i_min in (1.0, 1.5):
        for i_max in (2.0, 2.5, 3.0):
            gg = TankGame(i_min=i_min, i_max=i_max)
            dd, _ = t_solve_all(gg)
            f = lambda C: dd[frozenset()]["J"] - dd[C]["J"]
            rows.append((f"[{i_min},{i_max}]", pct(shapley(f, T_REQS)), f(full)))
    print(f"{'inflow':<12}" + "".join(f"{'phi'+str(i):>9}" for i in T_REQS)
          + f"{'total':>10}")
    for name, p, tot in rows:
        print(f"{name:<12}" + "".join(f"{p[i]:8.1f}%" for i in T_REQS)
              + f"{tot:10.4f}")
    for i in T_REQS:
        v = [p[i] for _, p, _ in rows]
        print(f"  phi{i} {T_LONG[i]:<12} [{min(v):5.1f}%, {max(v):5.1f}%]"
              f"  width {max(v)-min(v):4.1f}")


def run_grid():
    line("GRID WORLD")
    game = GridGame(); t0 = time.time()
    R = np.where(game.cl == game.goal, 0.0, -1.0)
    data = {}
    for C in G_SUB:
        W, sh = game.solve(C)
        V = g_vi(game, sh, R)
        data[C] = dict(W=W, sh=sh, V=V, J=float(V[game.init]))
    full, empty = frozenset(G_REQS), frozenset()
    elapsed = time.time() - t0

    Q = np.full((NACT, game.n), -np.inf)
    for a in range(NACT):
        Q[a] = np.where(data[full]["sh"][a],
                        R + G_GAMMA * data[full]["V"][game.NEXT[a]], -np.inf)
    pol, live = Q.argmax(axis=0), np.isfinite(Q.max(axis=0))
    mu0 = np.zeros(game.n); mu0[game.init] = 1.0 - G_GAMMA
    d = mu0.copy()
    for _ in range(20000):
        nd = mu0.copy()
        for a in range(NACT):
            src = np.where(live & (pol == a))[0]
            if src.size:
                np.add.at(nd, game.NEXT[a][src], G_GAMMA * d[src])
        if np.abs(nd - d).max() < 1e-14:
            break
        d = nd
    mu_occ = np.where(d > 1e-15, d, 0.0); mu_occ /= mu_occ.sum()
    mu_uni = data[full]["W"].astype(float); mu_uni /= mu_uni.sum()
    rho = lambda C: sum((~data[C]["sh"][a]).astype(float) for a in range(NACT))

    gJ = lambda C: data[empty]["J"] - data[C]["J"]
    gU = lambda C: float((mu_uni * rho(C)).sum())
    gO = lambda C: float((mu_occ * rho(C)).sum())
    phiJ, phiU, phiO = (shapley(gJ, G_REQS), shapley(gU, G_REQS),
                        shapley(gO, G_REQS))
    bzJ = banzhaf(gJ, G_REQS)
    standalone = {i: gJ(frozenset({i})) for i in G_REQS}
    leaveout = {i: gJ(full) - gJ(full - {i}) for i in G_REQS}

    print(f"|Q_R|={M1*M2*M3}  |Q_M|={NCELL*NPHASE}  |G|={game.n}"
          f"  coalitions={len(G_SUB)}  time={elapsed:.2f}s")
    print(f"monotone: dJ={monotone(gJ,G_REQS)} unif={monotone(gU,G_REQS)} "
          f"occ={monotone(gO,G_REQS)}")
    print(f"efficiency: sum={sum(phiJ.values()):.6f}  v(N)={gJ(full):.6f}")
    print(f"J*(0)={data[empty]['J']:.4f}  J*(N)={data[full]['J']:.4f}")

    print(f"\n{'requirement':<17}{'dJ':>10}{'%':>7}{'unif':>9}{'%':>7}"
          f"{'occ':>9}{'%':>7}")
    pJ, pU, pO = pct(phiJ), pct(phiU), pct(phiO)
    for i in G_REQS:
        print(f"phi{i} {G_LONG[i]:<12}{phiJ[i]:10.4f}{pJ[i]:6.1f}%"
              f"{phiU[i]:9.4f}{pU[i]:6.1f}%{phiO[i]:9.4f}{pO[i]:6.1f}%")

    print(f"\n{'requirement':<17}{'isolation':>12}{'removal':>12}"
          f"{'Shapley':>11}{'Banzhaf':>11}")
    for i in G_REQS:
        print(f"phi{i} {G_LONG[i]:<12}{standalone[i]:12.4f}{leaveout[i]:12.4f}"
              f"{phiJ[i]:11.4f}{bzJ[i]:11.4f}")
    print(f"{'sum':<17}{sum(standalone.values()):12.4f}"
          f"{sum(leaveout.values()):12.4f}{sum(phiJ.values()):11.4f}"
          f"{sum(bzJ.values()):11.4f}")
    print(f"{'total cost':<17}{gJ(full):12.4f}{gJ(full):12.4f}"
          f"{gJ(full):11.4f}{gJ(full):11.4f}")
    print(f"{'|error|':<17}{abs(sum(standalone.values())-gJ(full)):12.4f}"
          f"{abs(sum(leaveout.values())-gJ(full)):12.4f}"
          f"{abs(sum(phiJ.values())-gJ(full)):11.4f}"
          f"{abs(sum(bzJ.values())-gJ(full)):11.4f}")

    print("\ncoalition values")
    for C in G_SUB:
        tag = "{" + ",".join(str(i) for i in sorted(C)) + "}" if C else "{}"
        print(f"  {tag:<11} J*={data[C]['J']:9.4f}  cost={gJ(C):7.4f}")

    print("\nlocal attribution and minimal unsafe cores")
    patt = {}
    example = None
    for gi in np.where(data[full]["W"])[0]:
        for a in range(NACT):
            if data[full]["sh"][a][gi]:
                continue
            tab = {C: (0.0 if data[C]["sh"][a][gi] else 1.0) for C in G_SUB}
            cs = tuple(map(tuple, minimal_cores(tab, G_REQS)))
            patt[cs] = patt.get(cs, 0) + 1
            if len(cs) > 1 and example is None:
                ph = shapley(lambda C: tab[C], G_REQS)
                example = (rc(game.cl[gi]), a, cs,
                           tuple(round(ph[i], 3) for i in G_REQS))
    tot = sum(patt.values())
    multi = sum(v for k, v in patt.items() if len(k) > 1)
    print(f"  blocked state-action pairs: {tot}")
    print(f"  with more than one minimal core: {multi}")
    for k, v in sorted(patt.items(), key=lambda x: -x[1]):
        print(f"    cores={k}  occurrences={v}")
    if example:
        print(f"  example: cell {example[0]}, action {example[1]}, "
              f"cores={example[2]}, shapley={example[3]}")

    print("\njoint responsibility over cells")
    acc = {i: np.zeros(NCELL) for i in G_REQS}; cnt = np.zeros(NCELL)
    memo = {}
    for gi in np.where(data[full]["W"])[0]:
        for a in range(NACT):
            if data[full]["sh"][a][gi]:
                continue
            key = tuple(0 if data[C]["sh"][a][gi] else 1 for C in G_SUB)
            if key not in memo:
                tab = {C: float(k) for C, k in zip(G_SUB, key)}
                memo[key] = shapley(lambda C: tab[C], G_REQS)
            for i in G_REQS:
                acc[i][game.cl[gi]] += memo[key][i]
            cnt[game.cl[gi]] += 1
    for c in range(NCELL):
        if cnt[c] == 0:
            continue
        v = {i: acc[i][c] / cnt[c] for i in G_REQS}
        if max(v.values()) < 0.9:
            print(f"  cell {rc(c)}: " + "  ".join(f"phi{i}={v[i]:.2f}"
                                                  for i in G_REQS))


if __name__ == "__main__":
    run_tank()
    run_grid()
