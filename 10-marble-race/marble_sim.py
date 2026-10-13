"""Marble-race physics used by marble_race.py (kept separate so `probe` stays fast).
6 marbles fall through pegs, ramps, spinning bars, a funnel and bumpers.
Plain Python + NumPy: circle-circle, circle-peg, circle-segment (moving bars) and bumper collisions.
"""
import math
import numpy as np

NC = 6
G = 1000.0
R = 34.0                          # marble radius
XL, XR = 70.0, 1010.0             # side walls
Y_FIN = 3730.0                    # finish line (marble centre crosses it)
Y_FLOOR = 3960.0
SUBS = 240                        # physics steps per second
DT = 1.0 / SUBS
E_PEG, E_SEG, E_MM, E_BUMP = 0.42, 0.35, 0.82, 1.12

# ---- static pegs ----
def _pegs():
    pts = []
    for r_, y in enumerate(np.arange(520, 1250, 135)):
        off = 80 if r_ % 2 else 0
        for x in np.arange(150 + off, 1000, 160):
            if XL + 70 < x < XR - 70:
                pts.append((x, y))
    # lower pegs between the funnel and the bumpers
    for r_, y in enumerate([3120, 3250]):
        for x in ([300, 540, 780] if r_ == 0 else [420, 660]):
            pts.append((x, y))
    return np.array(pts, float)
PEGS = _pegs()
PEG_R = 14.0

# ---- static ramps / funnel walls: (x0, y0, x1, y1), thickness radius 8 ----
SEGS = np.array([
    (XL, 1360, 640, 1490), (XR, 1680, 440, 1810), (XL, 1990, 640, 2120),     # zig-zag ramps
    (XL, 2840, 485, 3040), (XR, 2840, 595, 3040),                           # funnel (gap 110)
], float)
SEG_R = 9.0

# ---- spinning bars: (cx, cy, half-length, omega rad/s, phase) ----
SPIN = np.array([
    (330, 2330, 150, 1.9, 0.0), (750, 2330, 150, -1.9, 1.0), (540, 2560, 150, 1.5, 2.0),
    (540, 3330, 120, -2.4, 0.5),
], float)
SPIN_R = 8.0

# ---- bumpers ----
BUMP = np.array([(250, 3440), (540, 3440), (830, 3440), (395, 3590), (685, 3590)], float)
BUMP_R = 40.0


def spin_ends(t, phase_shift=0.0):
    """end points of all spinner bars at time t -> (n,4)"""
    a = SPIN[:, 4] + phase_shift + SPIN[:, 3] * t
    dx, dy = np.cos(a) * SPIN[:, 2], np.sin(a) * SPIN[:, 2]
    return np.stack([SPIN[:, 0] - dx, SPIN[:, 1] - dy, SPIN[:, 0] + dx, SPIN[:, 1] + dy], 1)


def _seg_hit(px, py, x0, y0, x1, y1):
    vx, vy = x1 - x0, y1 - y0
    L2 = vx * vx + vy * vy
    u = ((px - x0) * vx + (py - y0) * vy) / L2
    u = 0.0 if u < 0 else 1.0 if u > 1 else u
    return x0 + u * vx, y0 + u * vy, u


def simulate(seed, t_end_extra=5.0, max_t=32.0):
    rnd = np.random.default_rng(seed)
    shift = rnd.uniform(0, 2 * math.pi)                    # random spinner phase per seed
    x = np.linspace(190, 890, NC) + rnd.uniform(-14, 14, NC)
    y = np.full(NC, 130.0) + rnd.uniform(-6, 6, NC)
    vx = rnd.uniform(-80, 80, NC); vy = rnd.uniform(0, 120, NC)
    ang = np.zeros(NC); w = np.zeros(NC)                   # marble spin (for the rolling look)
    idle = np.zeros(NC)
    fin_t = [None] * NC
    hist = []                                              # per step: (x, y, ang)
    ev = []                                                # (t, kind, i, strength)
    n = int(max_t * SUBS)
    t_stop = None
    for s in range(n):
        t = s * DT
        sp = spin_ends(t, shift)
        omega = SPIN[:, 3]
        vy += G * DT
        x += vx * DT; y += vy * DT; ang += w * DT
        for i in range(NC):
            if fin_t[i] is None and y[i] >= Y_FIN:
                a_ = (Y_FIN - (y[i] - vy[i] * DT)) / max(vy[i] * DT, 1e-9)
                fin_t[i] = t - DT * (1 - a_)
                if t_stop is None: t_stop = t + t_end_extra
        for i in range(NC):
            xi, yi = x[i], y[i]
            # side walls
            if xi < XL + R: x[i] = XL + R; vx[i] = abs(vx[i]) * 0.5
            elif xi > XR - R: x[i] = XR - R; vx[i] = -abs(vx[i]) * 0.5
            if y[i] > Y_FLOOR - R:
                y[i] = Y_FLOOR - R; vy[i] = -abs(vy[i]) * 0.3; vx[i] *= 0.98
            xi, yi = x[i], y[i]
            # pegs
            m = np.abs(PEGS[:, 1] - yi) < 60
            if m.any():
                P = PEGS[m]
                dx = xi - P[:, 0]; dy = yi - P[:, 1]
                d2 = dx * dx + dy * dy; rr = R + PEG_R
                k = np.where(d2 < rr * rr)[0]
                for j in k:
                    d = math.sqrt(d2[j]) + 1e-9; nx, ny = dx[j] / d, dy[j] / d
                    x[i] = P[j, 0] + nx * rr; y[i] = P[j, 1] + ny * rr
                    vn = vx[i] * nx + vy[i] * ny
                    if vn < 0:
                        vx[i] -= (1 + E_PEG) * vn * nx; vy[i] -= (1 + E_PEG) * vn * ny
                        tvx, tvy = vx[i] - (vx[i] * nx + vy[i] * ny) * nx, vy[i] - (vx[i] * nx + vy[i] * ny) * ny
                        w[i] = 0.7 * w[i] + 0.3 * (tvx * ny - tvy * nx) / R
                        if -vn > 200: ev.append((t, "peg", i, min(1.0, -vn / 1500)))
                    xi, yi = x[i], y[i]
            # ramps and funnel
            for (x0, y0, x1, y1) in SEGS:
                if yi < min(y0, y1) - 60 or yi > max(y0, y1) + 60: continue
                cx_, cy_, _ = _seg_hit(xi, yi, x0, y0, x1, y1)
                dx, dy = xi - cx_, yi - cy_; d = math.hypot(dx, dy)
                if d < R + SEG_R:
                    d = d + 1e-9; nx, ny = dx / d, dy / d
                    x[i] = cx_ + nx * (R + SEG_R); y[i] = cy_ + ny * (R + SEG_R)
                    vn = vx[i] * nx + vy[i] * ny
                    if vn < 0:
                        vx[i] -= (1 + E_SEG) * vn * nx; vy[i] -= (1 + E_SEG) * vn * ny
                        tvx, tvy = vx[i] - (vx[i] * nx + vy[i] * ny) * nx, vy[i] - (vx[i] * nx + vy[i] * ny) * ny
                        w[i] = 0.9 * w[i] + 0.1 * (tvx * ny - tvy * nx) / R
                        if -vn > 350: ev.append((t, "ramp", i, min(1.0, -vn / 1500)))
                    xi, yi = x[i], y[i]
            # spinning bars
            for b in range(len(SPIN)):
                bx0, by0, bx1, by1 = sp[b]
                if abs(yi - SPIN[b, 1]) > SPIN[b, 2] + 70: continue
                cx_, cy_, _ = _seg_hit(xi, yi, bx0, by0, bx1, by1)
                dx, dy = xi - cx_, yi - cy_; d = math.hypot(dx, dy)
                if d < R + SPIN_R:
                    d = d + 1e-9; nx, ny = dx / d, dy / d
                    x[i] = cx_ + nx * (R + SPIN_R); y[i] = cy_ + ny * (R + SPIN_R)
                    sx = -omega[b] * (cy_ - SPIN[b, 1]); sy = omega[b] * (cx_ - SPIN[b, 0])   # bar surface velocity
                    rvn = (vx[i] - sx) * nx + (vy[i] - sy) * ny
                    if rvn < 0:
                        vx[i] -= (1 + 0.55) * rvn * nx; vy[i] -= (1 + 0.55) * rvn * ny
                        ev.append((t, "bar", i, min(1.0, -rvn / 1500)))
                    xi, yi = x[i], y[i]
            # bumpers
            for (bx, by) in BUMP:
                if abs(yi - by) > BUMP_R + R + 5: continue
                dx, dy = xi - bx, yi - by; d = math.hypot(dx, dy); rr = BUMP_R + R
                if d < rr:
                    d += 1e-9; nx, ny = dx / d, dy / d
                    x[i] = bx + nx * rr; y[i] = by + ny * rr
                    vn = vx[i] * nx + vy[i] * ny
                    if vn < 0:
                        vx[i] -= (1 + E_BUMP) * vn * nx; vy[i] -= (1 + E_BUMP) * vn * ny
                        sp_ = math.hypot(vx[i], vy[i])
                        if sp_ < 520: vx[i] *= 520 / sp_; vy[i] *= 520 / sp_
                        ev.append((t, "bump", i, min(1.0, -vn / 1200)))
                    xi, yi = x[i], y[i]
        # marble-marble
        for i in range(NC):
            for j in range(i + 1, NC):
                dx, dy = x[j] - x[i], y[j] - y[i]
                if abs(dy) > 2 * R or abs(dx) > 2 * R: continue
                d = math.hypot(dx, dy)
                if d < 2 * R:
                    d += 1e-9; nx, ny = dx / d, dy / d; ov = 2 * R - d
                    x[i] -= nx * ov / 2; y[i] -= ny * ov / 2; x[j] += nx * ov / 2; y[j] += ny * ov / 2
                    rv = (vx[j] - vx[i]) * nx + (vy[j] - vy[i]) * ny
                    if rv < 0:
                        jv = -(1 + E_MM) * rv / 2
                        vx[i] -= jv * nx; vy[i] -= jv * ny; vx[j] += jv * nx; vy[j] += jv * ny
                        if -rv > 250: ev.append((t, "clack", i, min(1.0, -rv / 1200)))
        # air drag + anti-stuck
        vx *= 1 - 0.25 * DT; vy *= 1 - 0.25 * DT
        for i in range(NC):
            if fin_t[i] is None and math.hypot(vx[i], vy[i]) < 25:
                idle[i] += DT
                if idle[i] > 0.5:
                    vx[i] += rnd.choice([-1, 1]) * rnd.uniform(160, 260); vy[i] -= 120; idle[i] = 0
            else:
                idle[i] = 0
        hist.append((x.copy(), y.copy(), ang.copy()))
        if t_stop is not None and t >= t_stop: break
        if all(f is not None for f in fin_t): break
    H_ = np.array([np.concatenate([h[0], h[1], h[2]]) for h in hist], np.float32)   # (steps, 18)
    return dict(hist=H_, fin=fin_t, ev=ev, shift=shift, steps=len(hist))


def stats(r):
    """winner, margin, lead changes, how deep the winner dropped"""
    fin = np.array([f if f is not None else 99.0 for f in r["fin"]])
    order = np.argsort(fin)
    win, margin = int(order[0]), float(fin[order[1]] - fin[order[0]])
    Y = r["hist"][:, NC:2 * NC]
    step = SUBS // 10
    lead = None; lc = 0; worst_win = 0
    for k in range(0, len(Y), step):
        yy = Y[k]
        if fin[win] < 99 and k * DT > fin[win]: break
        rk = int(np.sum(yy > yy[win]))                       # 0 = leading
        if k * DT > 2.5: worst_win = max(worst_win, rk)
        ld = int(np.argmax(yy))
        if lead is None: lead = ld
        elif ld != lead and yy[ld] > yy[lead] + 30:
            lc += 1; lead = ld
    return dict(win=win, margin=margin, lc=lc, worst=worst_win, t_win=float(fin[win]), fin=fin)


if __name__ == "__main__":
    import sys, time
    a = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    b = int(sys.argv[2]) if len(sys.argv) > 2 else a
    for sd in range(a, b + 1):
        t0 = time.time(); r = simulate(sd); st = stats(r)
        print(f"seed {sd:3d} win {st['win'] + 1} t={st['t_win']:5.2f} margin={st['margin']:5.2f} lc={st['lc']:2d} "
              f"worst={st['worst'] + 1} fin={np.round(st['fin'], 1)}  [{time.time() - t0:.1f}s]", flush=True)
