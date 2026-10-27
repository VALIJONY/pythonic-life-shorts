"""Pythonic Life — "PICK A COLOR: which one ends with the most balls?" (YouTube Shorts 1080x1920, ~14 s, loops, light theme)
Four lanes, the same rows of gates in each (x2, x3, +5, half, x0). Balls drop in, bounce off the walls and pick a gate
by chance. Good gates clone them, bad gates delete them. Most balls in the bucket wins.
Usage: python3 multiplier_gates.py  |  python3 multiplier_gates.py still 0 5
"""
import math, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
from race_kit import *

TH = LIGHT
LANES = 4
AX0, AX1, AY0, AY1 = 30, 1050, 400, 1560
WALL = 8
LW = (AX1 - AX0 - WALL * (LANES - 1)) / LANES
LCOL = [(240, 70, 80), (255, 150, 30), (40, 175, 90), (40, 130, 240)]
NAMES = ["RED", "ORANGE", "GREEN", "BLUE"]
GATES_Y = [560, 730, 900, 1070, 1240]
ROWS = [("x2", "x0"), ("x3", "half"), ("+5", "x2"), ("half", "x3"), ("x2", "+5")]      # (left, right), same in every lane
BUCKET_Y = 1400
RB = 6.5
G, VMAX = 900.0, 820.0
SPAWN_DT, SPAWN_N = 0.33, 25
CAP = 700
SUB = 4

def lane_x(l): return AX0 + l * (LW + WALL)

def simulate(seed):
    rng = np.random.default_rng(seed)
    flips = [rng.random() < 0.5 for _ in range(LANES * len(ROWS))]               # each lane mirrors some rows: same gates, other side
    lanes = []
    for l in range(LANES):
        lanes.append(dict(x=np.zeros(0), y=np.zeros(0), vx=np.zeros(0), vy=np.zeros(0), row=np.zeros(0, int), landed=0, rng=np.random.default_rng(seed * 10 + l)))
    frames, events = [], []; dt = 1 / (FPS * SUB); t_done = None
    t = 0.0
    for f in range(int(16 * FPS)):
        for s in range(SUB):
            t = (f * SUB + s) * dt
            for l, L in enumerate(lanes):
                k = int(t / SPAWN_DT)
                if k < SPAWN_N and abs(t - k * SPAWN_DT) < dt / 2:                 # drop a new ball
                    r_ = L["rng"]
                    L["x"] = np.append(L["x"], lane_x(l) + LW / 2 + r_.uniform(-20, 20)); L["y"] = np.append(L["y"], AY0 + 20.0)
                    L["vx"] = np.append(L["vx"], r_.uniform(-260, 260)); L["vy"] = np.append(L["vy"], 0.0); L["row"] = np.append(L["row"], 0)
                if not len(L["x"]): continue
                L["vy"] = np.minimum(L["vy"] + G * dt, VMAX)
                y0 = L["y"].copy()
                L["x"] += L["vx"] * dt; L["y"] += L["vy"] * dt
                lo, hi = lane_x(l) + RB, lane_x(l) + LW - RB
                m = L["x"] < lo; L["x"][m] = lo; L["vx"][m] = np.abs(L["vx"][m])
                m = L["x"] > hi; L["x"][m] = hi; L["vx"][m] = -np.abs(L["vx"][m])
                keep = np.ones(len(L["x"]), bool); add = []
                for r, gy in enumerate(GATES_Y):
                    cross = (L["row"] == r) & (y0 < gy) & (L["y"] >= gy)
                    if not cross.any(): continue
                    idx = np.where(cross)[0]
                    left = L["x"][idx] < lane_x(l) + LW / 2
                    a, b = ROWS[r]
                    if flips[l * len(ROWS) + r]: a, b = b, a
                    for j, is_left in zip(idx, left):
                        op = a if is_left else b
                        L["row"][j] = r + 1
                        if op == "x0": keep[j] = False
                        elif op == "half": keep[j] = L["rng"].random() < 0.5
                        else:
                            n = {"x2": 1, "x3": 2, "+5": 5}[op]
                            for _ in range(n): add.append((L["x"][j], L["y"][j], L["rng"].uniform(-300, 300), L["vy"][j] * 0.6, r + 1))
                        events.append((t, l, r, op, bool(is_left)))
                landed = keep & (L["y"] >= BUCKET_Y)
                L["landed"] += int(landed.sum()); keep &= ~landed
                for key in ("x", "y", "vx", "vy", "row"): L[key] = L[key][keep]
                room = CAP - len(L["x"])
                if add and room > 0:
                    add = add[:room]
                    L["x"] = np.append(L["x"], [a[0] for a in add]); L["y"] = np.append(L["y"], [a[1] for a in add])
                    L["vx"] = np.append(L["vx"], [a[2] for a in add]); L["vy"] = np.append(L["vy"], [a[3] for a in add]); L["row"] = np.append(L["row"], [a[4] for a in add])
        frames.append(([(L["x"].copy(), L["y"].copy()) for L in lanes], [L["landed"] for L in lanes]))
        if t_done is None and t > SPAWN_N * SPAWN_DT and all(len(L["x"]) == 0 for L in lanes): t_done = t
        if t_done is not None and t > t_done + 5: break
    return frames, events, flips, t_done

def counts_at(fr, t): return fr[min(int(t * FPS), len(fr) - 1)][1]

def pick_seed():
    for seed in range(1, 2000):
        fr, ev, fl, td = simulate(seed)
        if td is None or not 9.0 <= td <= 10.3: continue
        c = counts_at(fr, td); w = int(np.argmax(c)); srt = sorted(c)
        if srt[-1] == srt[-2] or srt[-1] - srt[-2] > 0.2 * srt[-1]: continue
        c3 = counts_at(fr, td - 2.0)
        if int(np.argmax(c3)) != w: return seed
    raise RuntimeError("no seed")

SEED = pick_seed()
FR, EV, FLIPS, T_WIN = simulate(SEED)
FINAL = counts_at(FR, T_WIN); WIN = int(np.argmax(FINAL))
T_END = T_WIN + 1.1
TOTAL = T_END + 3.3
print(f"seed {SEED}: {NAMES[WIN]} wins with {FINAL[WIN]} balls at {T_WIN:.2f} s, counts {FINAL}, total {TOTAL:.1f} s", flush=True)
BGI = background(TH)
GOOD = (40, 170, 90); BAD = (225, 60, 60)

def render(fi):
    t = fi / FPS; k_ = min(fi, len(FR) - 1)
    img = BGI.copy(); d = ImageDraw.Draw(img)
    d.rounded_rectangle([AX0 - 4, AY0 - 4, AX1 + 4, AY1 + 4], radius=16, fill=TH["arena"], outline=TH["border"], width=3)
    for l in range(LANES):
        lx = lane_x(l)
        d.rectangle([lx, AY0, lx + LW, AY1], fill=mix((255, 255, 255), LCOL[l], 0.06))
        if l: d.rectangle([lx - WALL, AY0, lx, AY1], fill=(200, 206, 216))
        d.polygon([(lx + LW / 2 - 26, AY0), (lx + LW / 2 + 26, AY0), (lx + LW / 2 + 12, AY0 + 26), (lx + LW / 2 - 12, AY0 + 26)], fill=LCOL[l])
        for r, gy in enumerate(GATES_Y):
            a, b = ROWS[r]
            if FLIPS[l * len(ROWS) + r]: a, b = b, a
            for side, op in ((0, a), (1, b)):
                gx0 = lx + 6 + side * LW / 2; gx1 = lx + LW / 2 - 6 + side * LW / 2
                good = op in ("x2", "x3", "+5"); col = GOOD if good else BAD
                flash = any(e[1] == l and e[2] == r and e[4] == (side == 0) and 0 <= t - e[0] < 0.08 for e in EV[-1:]) if False else False
                d.rounded_rectangle([gx0, gy - 24, gx1, gy + 24], radius=12, fill=mix((255, 255, 255), col, 0.22), outline=col, width=3)
                d.text(((gx0 + gx1) / 2, gy), {"half": "-50%"}.get(op, op), font=BLACK(30), fill=col, anchor="mm")
        # bucket
        n = FR[k_][1][l]; mx = max(1, max(FINAL))
        hgt = min(1.0, n / mx) * (AY1 - BUCKET_Y - 10)
        d.rectangle([lx + 8, BUCKET_Y, lx + LW - 8, AY1 - 4], outline=mix((255, 255, 255), LCOL[l], 0.6), width=3)
        d.rectangle([lx + 11, AY1 - 7 - hgt, lx + LW - 11, AY1 - 7], fill=mix((255, 255, 255), LCOL[l], 0.55))
        d.text((lx + LW / 2, BUCKET_Y + 70), str(n), font=BLACK(56), fill=LCOL[l], anchor="mm", stroke_width=3, stroke_fill=(255, 255, 255))
        xs, ys = FR[k_][0][l]
        for x, y in zip(xs, ys):
            d.ellipse([x - RB, y - RB, x + RB, y + RB], fill=LCOL[l], outline=mix(LCOL[l], (0, 0, 0), 0.35))
    title(d, TH, "COLOR", "")
    c = counts_at(FR, min(t, T_WIN))
    items = [(NAMES[l][0], LCOL[l], str(c[l]), True) for l in range(LANES)]
    chips(d, TH, items, t, T_WIN, WIN, y=300, per_row=4, w=232, gap=14)
    if t < T_WIN: d.text((W / 2, 1600), "Most balls in the bucket wins", font=SEMI(34), fill=TH["muted"], anchor="mm")
    banner(d, TH, t, T_WIN, 980, [(NAMES[WIN] + " ", LCOL[WIN]), ("WINS", TH["text"])], f"{FINAL[WIN]} BALLS!", LCOL[WIN], T_END + 0.3)
    if T_WIN <= t: burst(d, t, T_WIN, lane_x(WIN) + LW / 2, BUCKET_Y + 60, 9, [LCOL[WIN], (255, 196, 0), (60, 140, 255), (40, 175, 90)], 50)
    if t >= T_END:
        end_card(img, TH, t, t - T_END, [(NAMES[WIN] + "?", LCOL[WIN])], "Comment your color", "a gate, in Python:",
                 ["if ball.crosses(gate):", "    if gate.op == 'x3':", "        balls += [ball.copy() for _ in range(2)]", "    elif gate.op == 'x0':", "        balls.remove(ball)"])
    return np.asarray(img)

def audio():
    m = Mixer(TOTAL); m.beat(124, hot_from=T_WIN - 4, roots=(130.81, 98.0, 110.0, 87.31))
    scale = {"x2": 659.3, "x3": 784.0, "+5": 1046.5}; last = -1
    for (te, l, r, op, left) in EV:
        if te > T_WIN: break
        if te - last < 0.04: continue
        if op in scale: m.ping(te, scale[op] * (1 + 0.12 * r), 0.05)
        else: m.thud(te, 0.12)
        last = te
    m.win(T_WIN, T_END)
    return m.final()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "still": stills(render, sys.argv[2:]); sys.exit()
    encode(render, TOTAL, audio(), "multiplier_gates.mp4"); print("done", round(TOTAL, 1))
