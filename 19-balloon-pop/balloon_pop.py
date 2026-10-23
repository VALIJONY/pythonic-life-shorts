"""Pythonic Life — "PICK A BALLOON: which one pops first?" (YouTube Shorts 1080x1920, ~14 s, loops, light theme)
Six balloons, a ball bouncing inside each. Every hit inflates the balloon a little; the first to reach its limit pops.
Usage: python3 balloon_pop.py  |  python3 balloon_pop.py still 0 5
"""
import math, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
from race_kit import *

TH = LIGHT
CENT = [(300, 650), (760, 650), (300, 1020), (760, 1020), (300, 1390), (760, 1390)]
R0, RPOP, RB = 125.0, 178.0, 17
SPEED, SUB = 600.0, 6
COLS = [(240, 70, 80), (255, 150, 30), (240, 190, 20), (40, 175, 90), (40, 130, 240), (160, 90, 230)]

def simulate(seed, T=14.0):
    rnd = random.Random(seed)
    st = []
    for k, (cx, cy) in enumerate(CENT):
        a = rnd.uniform(0, 2 * math.pi)
        st.append(dict(x=cx, y=cy, vx=SPEED * math.cos(a), vy=SPEED * math.sin(a), r=R0, rnd=random.Random(seed * 7 + k)))
    frames, hits, winner, t_win = [], [], None, None; dt = 1 / (FPS * SUB)
    for f in range(int(T * FPS)):
        for s in range(SUB):
            t = (f * SUB + s) * dt
            for k, b in enumerate(st):
                cx, cy = CENT[k]
                b["x"] += b["vx"] * dt; b["y"] += b["vy"] * dt
                dx, dy = b["x"] - cx, b["y"] - cy; d = math.hypot(dx, dy)
                if d > b["r"] - RB:                                         # hit the balloon skin: bounce and inflate
                    nx, ny = dx / d, dy / d; dot = b["vx"] * nx + b["vy"] * ny
                    if dot > 0:
                        b["vx"] -= 2 * dot * nx; b["vy"] -= 2 * dot * ny
                        a = math.atan2(b["vy"], b["vx"]) + math.radians(b["rnd"].uniform(-8, 8))
                        b["vx"], b["vy"] = SPEED * math.cos(a), SPEED * math.sin(a)
                        if winner is None:
                            b["r"] += b["rnd"].uniform(1.4, 3.0); hits.append((t, k, b["r"], math.atan2(ny, nx)))
                            if b["r"] >= RPOP: winner, t_win = k, t
                    b["x"], b["y"] = cx + nx * (b["r"] - RB), cy + ny * (b["r"] - RB)
        frames.append([(b["x"], b["y"], b["r"]) for b in st])
        if t_win is not None and t > t_win + 5: break
    return frames, hits, winner, t_win

def radius_at(fr, k, t): return fr[min(int(t * FPS), len(fr) - 1)][k][2]

def pick_seed():
    for seed in range(1, 3000):
        fr, hits, w, tw = simulate(seed, 12.0)
        if w is None or not 8.6 <= tw <= 10.0: continue
        rs = sorted(fr[min(int(tw * FPS), len(fr) - 1)][k][2] for k in range(6))
        r3 = [radius_at(fr, k, tw - 3) for k in range(6)]
        if rs[-2] > RPOP - 6 and max(range(6), key=lambda k: r3[k]) != w: return seed
    raise RuntimeError("no seed")

SEED = pick_seed()
FR, HITS, WIN, T_WIN = simulate(SEED, 16.0)
T_END = T_WIN + 1.1
TOTAL = T_END + 3.3
print(f"seed {SEED}: balloon {WIN + 1} pops first at {T_WIN:.2f} s, total {TOTAL:.1f} s", flush=True)
BGI = background(TH)

def render(fi):
    t = fi / FPS; k_ = min(fi, len(FR) - 1)
    S = 2; box = (40, 430, 1040, 1600)
    im = BGI.crop(box).resize(((box[2] - box[0]) * S, (box[3] - box[1]) * S), Image.NEAREST); d = ImageDraw.Draw(im)
    P = lambda x, y: ((x - box[0]) * S, (y - box[1]) * S)
    for k, (cx, cy) in enumerate(CENT):
        x, y, r = FR[k_][k]
        popped = k == WIN and t >= T_WIN
        c = P(cx, cy)
        if popped:
            dt = t - T_WIN
            for j in range(10):                                            # shreds
                a = j * 0.63 + 0.3; rr = (r + 400 * dt) * S
                px, py = c[0] + math.cos(a) * rr, c[1] + math.sin(a) * rr + 600 * dt * dt * S
                if dt < 0.7: d.polygon([(px, py), (px + 14 * S, py + 6 * S), (px + 4 * S, py + 16 * S)], fill=COLS[k])
            continue
        wob = 1 + 0.012 * math.sin(t * 9 + k)
        rr = r * S * wob
        d.line([(c[0], c[1] + rr), (c[0] + 8 * S * math.sin(t * 3 + k), c[1] + rr + 60 * S)], fill=(150, 156, 168), width=3 * S)
        d.polygon([(c[0] - 12 * S, c[1] + rr + 12 * S), (c[0] + 12 * S, c[1] + rr + 12 * S), (c[0], c[1] + rr - 4 * S)], fill=COLS[k])
        tension = (r - R0) / (RPOP - R0)
        fill = mix((255, 255, 255), COLS[k], 0.25 + 0.25 * tension)
        d.ellipse([c[0] - rr, c[1] - rr, c[0] + rr, c[1] + rr], fill=fill, outline=COLS[k], width=int((6 + 4 * tension) * S))
        d.ellipse([c[0] - rr * 0.62, c[1] - rr * 0.7, c[0] - rr * 0.25, c[1] - rr * 0.35], fill=mix(fill, (255, 255, 255), 0.75))
        d.text((c[0], c[1]), str(k + 1), font=BLACK(int(r * 0.9) * S), fill=mix(fill, COLS[k], 0.35), anchor="mm")
        # hit flashes on the skin
        for (th, kk, rh, ang) in HITS:
            if th > t: break
            if kk == k and t - th < 0.12:
                a0 = math.degrees(ang); d.arc([c[0] - rr, c[1] - rr, c[0] + rr, c[1] + rr], a0 - 14, a0 + 14, fill=(255, 255, 255), width=10 * S)
        q = P(x, y); br = RB * S
        d.ellipse([q[0] - br, q[1] - br, q[0] + br, q[1] + br], fill=(30, 34, 44), outline=(255, 255, 255), width=3 * S)
    img = BGI.copy(); img.paste(im.reduce(S), (box[0], box[1])); d = ImageDraw.Draw(img)
    title(d, TH, "BALLOON", "")
    tt_ = min(t, T_WIN)
    items = [(str(k + 1), COLS[k], f"{int(min(100, 100 * (radius_at(FR, k, tt_) - R0) / (RPOP - R0)))}%", True) for k in range(6)]
    chips(d, TH, items, t, T_WIN, WIN, y=290, per_row=6, w=150, gap=12)
    if t < T_WIN: d.text((W / 2, 1600), "Every hit inflates it. 100% = POP", font=SEMI(34), fill=TH["muted"], anchor="mm")
    if T_WIN <= t:
        cx, cy = CENT[WIN]
        burst(d, t, T_WIN, cx, cy, 9, [COLS[WIN], (255, 196, 0), (60, 140, 255), (40, 175, 90)], 60)
        pop_text(d, TH, t, T_WIN, cx, cy - 20, "POP!", COLS[WIN], 0.9, 90)
    banner(d, TH, t, T_WIN + 0.25, 1020, [("BALLOON ", TH["text"]), (str(WIN + 1), COLS[WIN])], "POPPED FIRST!", COLS[WIN], T_END + 0.3)
    if t >= T_END:
        end_card(img, TH, t, t - T_END, [("BALLOON ", TH["text"]), (str(WIN + 1) + "?", COLS[WIN])], "Comment your balloon", "every hit, in Python:",
                 ["if dist(ball, center) > r - ball_r:", "    reflect(ball, normal)", "    r += random.uniform(1.4, 3.0)", "    if r >= POP_R:", "        pop()"])
    return np.asarray(img)

def audio():
    m = Mixer(TOTAL); m.beat(122, hot_from=T_WIN - 4, roots=(130.81, 98.0, 110.0, 87.31))
    last = -1
    for (th, k, r, ang) in HITS:
        if th > T_WIN: break
        if th - last < 0.05: continue
        f = 330 * 2 ** (2.0 * (r - R0) / (RPOP - R0))                  # pitch rises as the balloon inflates
        m.ping(th, f, 0.06); last = th
    m.whoosh(T_WIN - 1.5, 1.5, 0.12, 3)
    m.pop(T_WIN, 1.6); m.impact(T_WIN, 1.0); m.chord(T_WIN + 0.1, [523.3, 659.3, 784.0, 1046.5], 0.06)
    m.impact(T_END, 0.5); m.chord(T_END + 0.5, [1318.5, 1975.5], 0.05, 3)
    return m.final()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "still": stills(render, sys.argv[2:]); sys.exit()
    encode(render, TOTAL, audio(), "balloon_pop.mp4"); print("done", round(TOTAL, 1))
