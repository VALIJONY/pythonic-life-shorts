"""Pythonic Life — "PICK A BALL 1-6: the floor is crumbling" (YouTube Shorts 1080x1920, ~14 s, loops)
Six balls bounce on a floor of tiles. Every landing cracks a tile; the third hit breaks it. Fall through and you are out.
Last ball on the floor wins. Real gravity, elastic ball-to-ball collisions.
Usage: python3 crumbling_floor.py  |  python3 crumbling_floor.py still 0 5
"""
import math, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
from race_kit import *

V2 = os.environ.get("V2") == "1"                    # variant 2: light theme, another seed
TH = LIGHT if V2 else DARK
AX0, AX1 = 60, 1020
FLOOR_Y = 1300
NT = 14; TW = (AX1 - AX0) / NT                      # tiles
HP = 4
N, RB = 6, 40
G, JUMP = 2600.0, 1350.0                            # gravity, bounce speed (fixed bounce height)
SUB = 8
COLS = [(255, 99, 99), (255, 170, 60), (255, 212, 59), (63, 200, 90), (70, 150, 255), (190, 120, 255)]

def simulate(seed, T=15.0):
    rnd = random.Random(seed)
    tiles = [HP] * NT
    balls = [dict(x=AX0 + (k + 0.5) * (AX1 - AX0) / N, y=FLOOR_Y - RB - rnd.uniform(80, 520), vx=rnd.choice([-1, 1]) * rnd.uniform(140, 380), vy=0.0, out=None)
             for k in range(N)]
    frames, events = [], []; dt = 1 / (FPS * SUB); winner = t_win = None
    for f in range(int(T * FPS)):
        for s in range(SUB):
            t = (f * SUB + s) * dt
            for k, b in enumerate(balls):
                b["vy"] += G * dt; b["x"] += b["vx"] * dt; b["y"] += b["vy"] * dt
                if b["x"] < AX0 + RB: b["x"], b["vx"] = AX0 + RB, abs(b["vx"])
                if b["x"] > AX1 - RB: b["x"], b["vx"] = AX1 - RB, -abs(b["vx"])
                if b["out"] is None and b["vy"] > 0 and FLOOR_Y - RB <= b["y"] <= FLOOR_Y - RB + 40:
                    i = int((b["x"] - AX0) // TW)
                    if 0 <= i < NT and tiles[i] > 0:                     # land on a tile: crack it
                        b["y"] = FLOOR_Y - RB; b["vy"] = -JUMP * (1 + rnd.uniform(-0.04, 0.04))
                        tiles[i] -= 0 if winner is not None else 1; events.append(("crack" if tiles[i] else "break", t, k, i))
                if b["out"] is None and b["y"] > FLOOR_Y + 60:
                    b["out"] = t; events.append(("out", t, k, None))
            for i in range(N):
                for j in range(i + 1, N):
                    p, q = balls[i], balls[j]
                    if p["out"] is not None or q["out"] is not None: continue
                    dx, dy = q["x"] - p["x"], q["y"] - p["y"]; d = math.hypot(dx, dy)
                    if 0 < d < 2 * RB:
                        nx, ny = dx / d, dy / d; rel = (p["vx"] - q["vx"]) * nx + (p["vy"] - q["vy"]) * ny
                        if rel > 0: p["vx"] -= rel * nx; p["vy"] -= rel * ny; q["vx"] += rel * nx; q["vy"] += rel * ny
                        o = (2 * RB - d) / 2; p["x"] -= nx * o; p["y"] -= ny * o; q["x"] += nx * o; q["y"] += ny * o
            alive = [k for k, b in enumerate(balls) if b["out"] is None]
            if winner is None and len(alive) == 1: winner, t_win = alive[0], t
            if winner is None and not alive: return None
        frames.append(([(b["x"], b["y"]) for b in balls], tiles[:]))
        if t_win is not None and t > t_win + 5: break
    if winner is None: return None
    return frames, events, winner, t_win

def pick_seed():
    for seed in (range(300, 3000) if V2 else range(1, 2000)):
        r = simulate(seed, 12.0)
        if r is None: continue
        fr, ev, w, tw = r
        outs = [e[1] for e in ev if e[0] == "out"]
        if not 8.3 <= tw <= 10.0 or len(outs) != N - 1: continue
        if outs[0] < 1.0 or tw - outs[-2] < 1.0 or min(b - a for a, b in zip(outs, outs[1:])) < 0.25: continue
        return seed
    raise RuntimeError("no seed")

SEED = pick_seed()
FR, EV, WIN, T_WIN = simulate(SEED, 16.0)
T_END = T_WIN + 1.1
TOTAL = T_END + 3.3
OUT = {e[2]: e[1] for e in EV if e[0] == "out"}
print(f"seed {SEED}: ball {WIN + 1} wins at {T_WIN:.2f} s, total {TOTAL:.1f} s", flush=True)
BGI = background(TH)
TILEC = [(120, 128, 146), (190, 90, 70), (210, 140, 80), (200, 190, 150), (200, 210, 225)]          # by hp left: 0..3

def render(fi):
    t = fi / FPS; k_ = min(fi, len(FR) - 1)
    pos, tiles = FR[k_]
    S = 2; box = (0, 380, W, 1560)
    im = BGI.crop(box).resize(((box[2] - box[0]) * S, (box[3] - box[1]) * S), Image.NEAREST); d = ImageDraw.Draw(im)
    P = lambda x, y: ((x - box[0]) * S, (y - box[1]) * S)
    # walls + lava pit
    x0, y0 = P(AX0 - 14, 420); x1, y1 = P(AX1 + 14, 1540)
    d.rounded_rectangle([x0, y0, x1, y1], radius=24 * S, fill=TH["arena"], outline=TH["border"], width=4 * S)
    for k in range(40):
        lx = AX0 + k * (AX1 - AX0) / 40; h = 18 + 10 * math.sin(t * 6 + k * 0.9)
        a, b = P(lx, 1540 - h); c, e = P(lx + (AX1 - AX0) / 40 + 1, 1540)
        d.rectangle([a, b, c, e], fill=(255, 90 + int(40 * math.sin(t * 5 + k)), 40))
    a, b = P(AX0, 1490); c, e = P(AX1, 1540); d.rectangle([a, b, c, e], fill=(220, 70, 30))
    # tiles
    for i in range(NT):
        hp = tiles[i]
        tx0, ty0 = P(AX0 + i * TW + 3, FLOOR_Y); tx1, ty1 = P(AX0 + (i + 1) * TW - 3, FLOOR_Y + 60)
        if hp > 0:
            d.rounded_rectangle([tx0, ty0, tx1, ty1], radius=8 * S, fill=TILEC[hp], outline=mix(TILEC[hp], (0, 0, 0), 0.35), width=3 * S)
            rr = random.Random(i)
            for c_ in range(HP - hp):                                       # cracks
                px = rr.uniform(tx0 + 10 * S, tx1 - 10 * S); pts = [(px, ty0)]
                for _ in range(4): pts.append((pts[-1][0] + rr.uniform(-12, 12) * S, pts[-1][1] + 12 * S))
                d.line(pts, fill=(30, 30, 36), width=3 * S)
    # falling tile debris
    for e in EV:
        if e[0] == "break" and 0 <= t - e[1] < 0.8:
            i = e[3]; dt = t - e[1]; rr = random.Random(int(e[1] * 1e4))
            for _ in range(5):
                px = AX0 + (i + rr.random()) * TW; py = FLOOR_Y + 20 + 1400 * dt * dt + rr.uniform(0, 30)
                a, b = P(px, py); s_ = rr.uniform(8, 16) * S
                d.rectangle([a - s_, b - s_ / 2, a + s_, b + s_ / 2], fill=TILEC[1])
    # balls
    for k in range(N):
        x, y = pos[k]
        if k in OUT and t > OUT[k] + 0.4: continue
        for j in range(5, 0, -1):
            if k_ - j < 0: continue
            px, py = FR[k_ - j][0][k]; a_ = (1 - j / 6) * 0.4; r = RB * (0.5 + 0.4 * (1 - j / 6)) * S; q = P(px, py)
            d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=mix(TH["arena"], COLS[k], a_))
        q = P(x, y); r = RB * S
        d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=COLS[k], outline=mix(COLS[k], (0, 0, 0), 0.45), width=3 * S)
        d.ellipse([q[0] - r * 0.55, q[1] - r * 0.62, q[0] - r * 0.05, q[1] - r * 0.25], fill=mix(COLS[k], (255, 255, 255), 0.55))
        d.text((q[0], q[1] + 2 * S), str(k + 1), font=BLACK(38 * S), fill=(20, 22, 28) if k in (2, 3) else (255, 255, 255), anchor="mm")
    img = BGI.copy(); img.paste(im.reduce(S), (box[0], box[1])); d = ImageDraw.Draw(img)
    title(d, TH, "BALL", " 1-6")
    items = [(str(k + 1), COLS[k], "", OUT.get(k, 1e9) > t) for k in range(N)]
    chips(d, TH, items, t, T_WIN, WIN, y=290, per_row=6, w=150, gap=14)
    for k, to in OUT.items():
        x, _ = FR[min(int(to * FPS), len(FR) - 1)][0][k]
        pop_text(d, TH, t, to, x, 1440, "OUT!", COLS[k])
    if t < T_WIN: d.text((W / 2, 1600), "4 hits and the tile breaks", font=SEMI(34), fill=TH["muted"], anchor="mm")
    banner(d, TH, t, T_WIN, 820, [("BALL ", TH["text"]), (str(WIN + 1), COLS[WIN])], "LAST ONE STANDING!", COLS[WIN], T_END + 0.3)
    if T_WIN <= t: x, y = FR[min(int(T_WIN * FPS), len(FR) - 1)][0][WIN]; burst(d, t, T_WIN, x, y, 9, [YEL, COLS[WIN], (255, 255, 255), BLUE], 44)
    if t >= T_END:
        end_card(img, TH, t, t - T_END, [("BALL ", TH["text"]), (str(WIN + 1) + "?", COLS[WIN])], "Comment your ball", "the floor, in Python:",
                 ["if ball.vy > 0 and ball.bottom >= FLOOR:", "    i = int(ball.x // TILE_W)", "    if tiles[i] > 0:", "        ball.vy = -JUMP", "        tiles[i] -= 1   # 0 = gone"])
    return np.asarray(img)

def audio():
    m = Mixer(TOTAL); m.beat(126, hot_from=T_WIN - 4)
    scale = [392.0, 440.0, 523.3, 587.3, 659.3, 784.0]
    for e in EV:
        if e[1] > T_WIN: break
        if e[0] == "crack": m.thud(e[1], 0.16); m.crack(e[1], 0.10)
        elif e[0] == "break": m.crack(e[1], 0.35); m.thud(e[1], 0.3)
        elif e[0] == "out": m.whoosh(e[1] - 0.2, 0.35, 0.10, int(e[1] * 10)); m.pop(e[1] + 0.1, 0.5)
    n = 0
    for e in EV:
        if e[0] == "crack" and e[1] < T_WIN: m.ping(e[1], scale[n % 6], 0.04); n += 1
    m.win(T_WIN, T_END)
    return m.final()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "still": stills(render, sys.argv[2:]); sys.exit()
    encode(render, TOTAL, audio(), "crumbling_floor_v2.mp4" if V2 else "crumbling_floor.mp4"); print("done", round(TOTAL, 1))
