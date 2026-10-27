"""Pythonic Life — "PICK A BALL: who finds the exit first?" (YouTube Shorts 1080x1920, ~14 s, loops, light theme)
Four balls start in the middle of a rotating maze (a random perfect maze with a few extra openings).
Gravity pulls them down while the maze turns; they bounce off the walls until one finds the exit.
Usage: python3 maze_escape.py  |  python3 maze_escape.py still 0 5
"""
import math, os, random, sys
from collections import deque
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
from race_kit import *

TH = LIGHT
N = 10; CS = 76.0; SIDE = N * CS              # maze cells, cell size
CX, CY = 540, 1010
RB = 18
G = 1700.0
SUB = 8
LCOL = [(240, 70, 80), (255, 150, 30), (40, 175, 90), (40, 130, 240)]
NAMES = ["RED", "ORANGE", "GREEN", "BLUE"]

def make_maze(rnd):
    walls = {(i, j): {"N", "S", "E", "W"} for i in range(N) for j in range(N)}      # i = column, j = row
    D = {"N": (0, -1, "S"), "S": (0, 1, "N"), "E": (1, 0, "W"), "W": (-1, 0, "E")}
    def open_(c, d):
        di, dj, o = D[d]; n = (c[0] + di, c[1] + dj); walls[c].discard(d)
        if n in walls: walls[n].discard(o)
    seen, stack = {(0, 0)}, [(0, 0)]
    while stack:
        c = stack[-1]; opts = [d for d in D if (c[0] + D[d][0], c[1] + D[d][1]) in walls and (c[0] + D[d][0], c[1] + D[d][1]) not in seen]
        if not opts: stack.pop(); continue
        d = rnd.choice(opts); open_(c, d); n = (c[0] + D[d][0], c[1] + D[d][1]); seen.add(n); stack.append(n)
    for _ in range(14):                                                    # a few loops so there is more than one way out
        c = (rnd.randrange(1, N - 1), rnd.randrange(1, N - 1)); open_(c, rnd.choice(list(D)))
    for c in [(4, 4), (5, 4), (4, 5), (5, 5)]:                             # open room in the middle
        for d in D:
            n = (c[0] + D[d][0], c[1] + D[d][1])
            if n in [(4, 4), (5, 4), (4, 5), (5, 5)]: open_(c, d)
    side = rnd.choice("NSEW"); k = rnd.randrange(N)
    exit_cell = {"N": (k, 0), "S": (k, N - 1), "W": (0, k), "E": (N - 1, k)}[side]
    walls[exit_cell].discard(side)
    return walls, exit_cell, side

def segments(walls):
    segs = set()
    for (i, j), ws in walls.items():
        x0, y0 = i * CS - SIDE / 2, j * CS - SIDE / 2
        if "N" in ws: segs.add((x0, y0, x0 + CS, y0))
        if "S" in ws: segs.add((x0, y0 + CS, x0 + CS, y0 + CS))
        if "W" in ws: segs.add((x0, y0, x0, y0 + CS))
        if "E" in ws: segs.add((x0 + CS, y0, x0 + CS, y0 + CS))
    return list(segs)

def dist_to_exit(walls, exit_cell):
    D = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}
    dist = {exit_cell: 0}; q = deque([exit_cell])
    while q:
        c = q.popleft()
        for d, (di, dj) in D.items():
            n = (c[0] + di, c[1] + dj)
            if d not in walls[c] and n in walls and n not in dist: dist[n] = dist[c] + 1; q.append(n)
    return dist

def angle(t, w0): return w0 * t + 0.25 * math.sin(t * 0.9)

def simulate(seed, T=16.0):
    rnd = random.Random(seed)
    walls, exit_cell, side = make_maze(rnd); segs = segments(walls)
    w0 = rnd.choice([-1, 1]) * rnd.uniform(0.45, 0.75)
    balls = [dict(x=(-0.5 if k % 2 == 0 else 0.5) * CS * 0.6, y=(-0.5 if k < 2 else 0.5) * CS * 0.6, vx=rnd.uniform(-150, 150), vy=0.0, out=None) for k in range(4)]
    frames, hits = [], []; dt = 1 / (FPS * SUB); winner = t_win = None
    for f in range(int(T * FPS)):
        for s in range(SUB):
            t = (f * SUB + s) * dt; a = angle(t, w0)
            gx, gy = G * math.sin(a), G * math.cos(a)                     # gravity seen from the maze frame
            for k, b in enumerate(balls):
                b["vx"] += gx * dt; b["vy"] += gy * dt
                sp = math.hypot(b["vx"], b["vy"])
                if sp > 900: b["vx"] *= 900 / sp; b["vy"] *= 900 / sp
                b["x"] += b["vx"] * dt; b["y"] += b["vy"] * dt
                if b["out"] is not None: continue
                for (x0, y0, x1, y1) in segs:
                    if abs(b["x"] - (x0 + x1) / 2) > CS or abs(b["y"] - (y0 + y1) / 2) > CS: continue
                    px = min(max(b["x"], x0), x1); py = min(max(b["y"], y0), y1)
                    dx, dy = b["x"] - px, b["y"] - py; d = math.hypot(dx, dy)
                    if 0 < d < RB + 4:
                        nx, ny = dx / d, dy / d; dot = b["vx"] * nx + b["vy"] * ny
                        if dot < 0:
                            b["vx"] -= 1.8 * dot * nx; b["vy"] -= 1.8 * dot * ny
                            if -dot > 160: hits.append((t, k))
                        b["x"] += nx * (RB + 4 - d); b["y"] += ny * (RB + 4 - d)
                if max(abs(b["x"]), abs(b["y"])) > SIDE / 2 + RB + 6:
                    b["out"] = t
                    if winner is None: winner, t_win = k, t
            for i in range(4):
                for j in range(i + 1, 4):
                    p, q = balls[i], balls[j]
                    dx, dy = q["x"] - p["x"], q["y"] - p["y"]; d = math.hypot(dx, dy)
                    if 0 < d < 2 * RB:
                        nx, ny = dx / d, dy / d; rel = (p["vx"] - q["vx"]) * nx + (p["vy"] - q["vy"]) * ny
                        if rel > 0: p["vx"] -= rel * nx; p["vy"] -= rel * ny; q["vx"] += rel * nx; q["vy"] += rel * ny
                        o = (2 * RB - d) / 2; p["x"] -= nx * o; p["y"] -= ny * o; q["x"] += nx * o; q["y"] += ny * o
        frames.append(([(b["x"], b["y"]) for b in balls], angle((f + 1) / FPS, w0)))
        if t_win is not None and t > t_win + 5: break
    return dict(walls=walls, segs=segs, exit=exit_cell, side=side, w0=w0), frames, hits, winner, t_win

def cell_of(x, y): return (min(N - 1, max(0, int((x + SIDE / 2) // CS))), min(N - 1, max(0, int((y + SIDE / 2) // CS))))

def pick_seed():
    for seed in range(1, 4000):
        mz, fr, hits, w, tw = simulate(seed, 11.0)
        if w is None or not 8.6 <= tw <= 10.2: continue
        dist = dist_to_exit(mz["walls"], mz["exit"])
        pos = fr[int((tw - 3) * FPS)][0]
        dd = [dist.get(cell_of(x, y), 99) for x, y in pos]
        if min(range(4), key=lambda k: dd[k]) != w: return seed
    raise RuntimeError("no seed")

SEED = pick_seed()
MZ, FR, HITS, WIN, T_WIN = simulate(SEED, 16.0)
T_END = T_WIN + 1.1
TOTAL = T_END + 3.3
DIST = dist_to_exit(MZ["walls"], MZ["exit"]); DMAX = max(DIST.values())
print(f"seed {SEED}: {NAMES[WIN]} escapes at {T_WIN:.2f} s, total {TOTAL:.1f} s", flush=True)
BGI = background(TH)

def render(fi):
    t = fi / FPS; k_ = min(fi, len(FR) - 1)
    pos, a = FR[k_]; ca, sa = math.cos(a), math.sin(a)
    S = 2; box = (0, 440, W, 1580)
    im = BGI.crop(box).resize(((box[2] - box[0]) * S, (box[3] - box[1]) * S), Image.NEAREST); d = ImageDraw.Draw(im)
    def P(x, y):                                                           # maze frame -> screen (rotated by -a)
        X = CX + x * ca + y * sa; Y = CY - x * sa + y * ca
        return ((X - box[0]) * S, (Y - box[1]) * S)
    corners = [P(-SIDE / 2, -SIDE / 2), P(SIDE / 2, -SIDE / 2), P(SIDE / 2, SIDE / 2), P(-SIDE / 2, SIDE / 2)]
    d.polygon(corners, fill=(255, 255, 255))
    # exit marker
    ei, ej = MZ["exit"]; ex = ei * CS - SIDE / 2 + CS / 2; ey = ej * CS - SIDE / 2 + CS / 2
    off = {"N": (0, -CS / 2), "S": (0, CS / 2), "W": (-CS / 2, 0), "E": (CS / 2, 0)}[MZ["side"]]
    q = P(ex + off[0] * 1.6, ey + off[1] * 1.6)
    d.ellipse([q[0] - 34 * S, q[1] - 34 * S, q[0] + 34 * S, q[1] + 34 * S], fill=(40, 175, 90))
    d.text(q, "EXIT", font=BLACK(20 * S), fill=(255, 255, 255), anchor="mm")
    for (x0, y0, x1, y1) in MZ["segs"]:
        m = ((x0 + x1) / 2, (y0 + y1) / 2); r = math.hypot(*m) / (SIDE * 0.72)
        col = mix((60, 90, 200), (150, 70, 210), min(1, r))
        d.line([P(x0, y0), P(x1, y1)], fill=col, width=8 * S)
        for x, y in ((x0, y0), (x1, y1)):
            c = P(x, y); d.ellipse([c[0] - 4 * S, c[1] - 4 * S, c[0] + 4 * S, c[1] + 4 * S], fill=col)
    for k in range(4):
        for j in range(5, 0, -1):
            if k_ - j < 0: continue
            (px, py), aj = FR[k_ - j][0][k], FR[k_ - j][1]
            X = CX + px * math.cos(aj) + py * math.sin(aj); Y = CY - px * math.sin(aj) + py * math.cos(aj)
            c = ((X - box[0]) * S, (Y - box[1]) * S); r = RB * (0.5 + 0.4 * (1 - j / 6)) * S
            d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=mix((255, 255, 255), LCOL[k], 0.35 * (1 - j / 6)))
        c = P(*pos[k]); r = RB * S
        d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=LCOL[k], outline=mix(LCOL[k], (0, 0, 0), 0.4), width=3 * S)
    img = BGI.copy(); img.paste(im.reduce(S), (box[0], box[1])); d = ImageDraw.Draw(img)
    title(d, TH, "BALL", "")
    tt_ = min(t, T_WIN)
    pp = FR[min(int(tt_ * FPS), len(FR) - 1)][0]
    items = [(NAMES[k][0], LCOL[k], NAMES[k], True) for k in range(4)]
    chips(d, TH, items, t, T_WIN, WIN, y=300, per_row=4, w=232, gap=14)
    if t < T_WIN: d.text((W / 2, 1590), "First ball out of the maze wins", font=SEMI(34), fill=TH["muted"], anchor="mm")
    banner(d, TH, t, T_WIN, 1010, [(NAMES[WIN], LCOL[WIN])], "ESCAPED FIRST!", LCOL[WIN], T_END + 0.3)
    if T_WIN <= t:
        (px, py), aw = FR[min(int(T_WIN * FPS), len(FR) - 1)][0][WIN], FR[min(int(T_WIN * FPS), len(FR) - 1)][1]
        burst(d, t, T_WIN, CX + px * math.cos(aw) + py * math.sin(aw), CY - px * math.sin(aw) + py * math.cos(aw), 9,
              [LCOL[WIN], (255, 196, 0), (60, 140, 255), (40, 175, 90)], 50)
    if t >= T_END:
        end_card(img, TH, t, t - T_END, [(NAMES[WIN] + "?", LCOL[WIN])], "Comment your ball", "gravity in a spinning maze:",
                 ["gx, gy = G * sin(angle), G * cos(angle)", "vx += gx * dt; vy += gy * dt", "for wall in nearby_walls:", "    if dist(ball, wall) < r:", "        bounce(ball, wall.normal)"])
    return np.asarray(img)

def audio():
    m = Mixer(TOTAL); m.beat(122, hot_from=T_WIN - 4, roots=(110.0, 87.31, 130.81, 98.0))
    scale = [523.3, 587.3, 659.3, 784.0, 880.0, 1046.5]; last = -1; n = 0
    for (th, k) in HITS:
        if th > T_WIN: break
        if th - last < 0.06: continue
        m.ping(th, scale[(n + k) % 6], 0.045); last = th; n += 1
    m.win(T_WIN, T_END)
    return m.final()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "still": stills(render, sys.argv[2:]); sys.exit()
    encode(render, TOTAL, audio(), "maze_escape.mp4"); print("done", round(TOTAL, 1))
