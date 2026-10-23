"""Pythonic Life — "PICK A BALL 1-8: last ball standing" (YouTube Shorts 1080x1920, ~14 s, loops)
Eight balls bounce inside a circle with a spinning spiked blade in the middle. The blade spins faster and grows.
Touch it and you pop. Last ball standing wins. Real elastic collisions.
Usage: python3 last_ball.py  |  python3 last_ball.py still 0 5
"""
import math, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
from race_kit import *

TH = DARK
CX, CY, RA = 540, 1010, 450                # arena circle
N, RB = 8, 30
COLS = [(255, 99, 99), (255, 159, 67), (255, 212, 59), (63, 200, 90), (60, 200, 200), (70, 150, 255), (170, 120, 255), (255, 110, 200)]
SPEED, SUB = 600.0, 8

def blade(t):
    """angle and arm length of the spinning blade"""
    return 0.4 + 1.6 * t + 0.12 * t * t, min(RA - 40, 70 + 36 * t)

def seg_dist(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay; L = vx * vx + vy * vy
    u = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / L)) if L else 0.0
    return math.hypot(px - ax - u * vx, py - ay - u * vy)

def simulate(seed, T=16.0):
    rnd = random.Random(seed); balls = []
    while len(balls) < N:
        a = rnd.uniform(0, 2 * math.pi); r = rnd.uniform(260, RA - RB - 10); x, y = CX + r * math.cos(a), CY + r * math.sin(a)
        if all(math.hypot(x - b["x"], y - b["y"]) > 2 * RB + 6 for b in balls):
            va = rnd.uniform(0, 2 * math.pi); balls.append(dict(x=x, y=y, vx=SPEED * math.cos(va), vy=SPEED * math.sin(va), dead=None))
    frames, pops, hits = [], [], []; dt = 1 / (FPS * SUB); winner = None; t_win = None
    for f in range(int(T * FPS)):
        for s in range(SUB):
            t = (f * SUB + s) * dt; ang, L = blade(t)
            for k, b in enumerate(balls):
                if b["dead"] is not None: continue
                b["x"] += b["vx"] * dt; b["y"] += b["vy"] * dt
                dx, dy = b["x"] - CX, b["y"] - CY; d = math.hypot(dx, dy)
                if d > RA - RB:                                            # circle wall
                    nx, ny = dx / d, dy / d; dot = b["vx"] * nx + b["vy"] * ny
                    if dot > 0: b["vx"] -= 2 * dot * nx; b["vy"] -= 2 * dot * ny; hits.append((t, k))
                    b["x"], b["y"] = CX + nx * (RA - RB), CY + ny * (RA - RB)
                for arm in (0, math.pi):                                   # the blade: touch it and you pop
                    ex, ey = CX + L * math.cos(ang + arm), CY + L * math.sin(ang + arm)
                    if seg_dist(b["x"], b["y"], CX, CY, ex, ey) < RB + 10 and winner is None:
                        b["dead"] = t; pops.append((t, k)); break
            for i in range(N):
                for j in range(i + 1, N):
                    p, q = balls[i], balls[j]
                    if p["dead"] is not None or q["dead"] is not None: continue
                    dx, dy = q["x"] - p["x"], q["y"] - p["y"]; d = math.hypot(dx, dy)
                    if 0 < d < 2 * RB:
                        nx, ny = dx / d, dy / d; rel = (p["vx"] - q["vx"]) * nx + (p["vy"] - q["vy"]) * ny
                        if rel > 0: p["vx"] -= rel * nx; p["vy"] -= rel * ny; q["vx"] += rel * nx; q["vy"] += rel * ny
                        o = (2 * RB - d) / 2; p["x"] -= nx * o; p["y"] -= ny * o; q["x"] += nx * o; q["y"] += ny * o
            alive = [k for k, b in enumerate(balls) if b["dead"] is None]
            if winner is None and len(alive) == 1: winner, t_win = alive[0], t
        frames.append([(b["x"], b["y"], b["dead"]) for b in balls])
        if t_win is not None and t > t_win + 5: break
    return frames, pops, hits, winner, t_win

def pick_seed():
    for seed in range(1, 900):
        fr, pops, hits, w, tw = simulate(seed, 12.0)
        if w is None or not 8.0 <= tw <= 10.0 or len(pops) != N - 1: continue
        ts = [p[0] for p in pops]
        if ts[0] < 0.4 or ts[-1] - ts[-2] < 1.0 or min(b - a for a, b in zip(ts, ts[1:])) < 0.15: continue
        return seed
    raise RuntimeError("no seed")

SEED = pick_seed()
FR, POPS, HITS, WIN, T_WIN = simulate(SEED, 16.0)
T_END = T_WIN + 1.1
TOTAL = T_END + 3.3
DEAD = {k: t for t, k in POPS}
print(f"seed {SEED}: ball {WIN + 1} wins at {T_WIN:.2f} s, total {TOTAL:.1f} s", flush=True)
BGI = background(TH)

def draw_arena(t):
    S = 2; box = (CX - RA - 20, CY - RA - 20, CX + RA + 20, CY + RA + 20)
    w = (box[2] - box[0]) * S; im = BGI.crop(box).resize((w, w), Image.NEAREST); d = ImageDraw.Draw(im)
    P = lambda x, y: ((x - box[0]) * S, (y - box[1]) * S)
    c = P(CX, CY)
    d.ellipse([c[0] - RA * S, c[1] - RA * S, c[0] + RA * S, c[1] + RA * S], fill=TH["arena"], outline=(255, 212, 59), width=8 * S)
    tb = min(t, T_WIN); ang, L = blade(tb) if t < T_WIN else blade(T_WIN)
    for arm in (0, math.pi):
        a = ang + arm; ex, ey = P(CX + L * math.cos(a), CY + L * math.sin(a))
        d.line([c, (ex, ey)], fill=(200, 206, 220), width=16 * S)
        n = int(L / 28)
        for k in range(1, n + 1):                                          # spikes
            r = k * L / n; px, py = P(CX + r * math.cos(a), CY + r * math.sin(a))
            for sgn in (-1, 1):
                nx, ny = -math.sin(a) * sgn, math.cos(a) * sgn
                d.polygon([(px - math.cos(a) * 8 * S, py - math.sin(a) * 8 * S), (px + math.cos(a) * 8 * S, py + math.sin(a) * 8 * S),
                           (px + nx * 22 * S, py + ny * 22 * S)], fill=(240, 80, 80))
    d.ellipse([c[0] - 30 * S, c[1] - 30 * S, c[0] + 30 * S, c[1] + 30 * S], fill=(90, 96, 112), outline=(200, 206, 220), width=4 * S)
    fi = min(int(t * FPS), len(FR) - 1)
    for k in range(N):
        x, y, dead = FR[fi][k]
        if dead is not None and t > dead: continue
        for j in range(5, 0, -1):
            if fi - j < 0: continue
            px, py, _ = FR[fi - j][k]; a = (1 - j / 6) * 0.4; r = RB * (0.5 + 0.4 * (1 - j / 6)) * S; q = P(px, py)
            d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=mix(TH["arena"], COLS[k], a))
        q = P(x, y); r = RB * S
        d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=COLS[k], outline=mix(COLS[k], (0, 0, 0), 0.45), width=3 * S)
        d.ellipse([q[0] - r * 0.55, q[1] - r * 0.62, q[0] - r * 0.05, q[1] - r * 0.25], fill=mix(COLS[k], (255, 255, 255), 0.55))
        d.text((q[0], q[1] + 2 * S), str(k + 1), font=BLACK(30 * S), fill=(20, 22, 28) if k in (2, 3, 4) else (255, 255, 255), anchor="mm")
    return im.reduce(S), box

CODE = ["for ball in balls:",
        "    d = dist_to_blade(ball, angle, length)",
        "    if d < ball.r + blade_width:",
        "        ball.pop()",
        "angle += speed * dt; speed += 0.3 * dt"]

def render(fi):
    t = fi / FPS
    img = BGI.copy(); im, box = draw_arena(t); img.paste(im, (box[0], box[1])); d = ImageDraw.Draw(img)
    title(d, TH, "BALL", " 1-8")
    items = [(str(k + 1), COLS[k], "", DEAD.get(k, 1e9) > t) for k in range(N)]
    chips(d, TH, items, t, T_WIN, WIN, y=290, per_row=8, w=112, gap=12)
    fi_ = min(fi, len(FR) - 1)
    for k, td in DEAD.items():
        x, y, _ = FR[min(int(td * FPS), len(FR) - 1)][k]
        burst(d, t, td, x, y, k + 5, [COLS[k], (255, 255, 255)], 24, 0.6)
        pop_text(d, TH, t, td, x, y - 40, "POP!", COLS[k])
    if t < T_WIN: d.text((W / 2, 1530), "Touch the blade and you pop", font=SEMI(34), fill=TH["muted"], anchor="mm")
    banner(d, TH, t, T_WIN, CY, [("BALL ", TH["text"]), (str(WIN + 1), COLS[WIN])], "LAST ONE STANDING!", COLS[WIN], T_END + 0.3)
    if T_WIN <= t: burst(d, t, T_WIN, CX, CY, 9, [YEL, COLS[WIN], (255, 255, 255), BLUE], 44)
    if t >= T_END: end_card(img, TH, t, t - T_END, [("BALL ", TH["text"]), (str(WIN + 1) + "?", COLS[WIN])], "Comment your ball", "the blade check, in Python:", CODE)
    return np.asarray(img)

def audio():
    m = Mixer(TOTAL); m.beat(126, hot_from=T_WIN - 4)
    scale = [523.3, 587.3, 659.3, 784.0, 880.0, 1046.5]; last = -1; n = 0
    for t, k in HITS:
        if t > T_WIN: break
        if t - last < 0.07: continue
        m.ping(t, scale[n % 6] * (1 if t < T_WIN - 3 else 1.5), 0.06); last = t; n += 1
    for t, k in POPS: m.pop(t, 0.8); m.thud(t, 0.25)
    m.win(T_WIN, T_END)
    return m.final()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "still": stills(render, sys.argv[2:]); sys.exit()
    encode(render, TOTAL, audio(), "last_ball.mp4"); print("done", round(TOTAL, 1))
