"""Pythonic Life — "PICK A LANE: which ball breaks through first?" (YouTube Shorts 1080x1920, ~20 s, loops)
Four lanes, the same wall of bricks in each, one ball per lane. Every hit breaks a brick; the first ball that
breaks out through the top wins. Real ball physics (axis-separated moves, brick-grid collisions).
The seed is picked so the race ends at ~14 s and the lane that broke the most bricks 4 s before the end loses.
Usage: python3 brick_race.py            -> brick_race.mp4
       python3 brick_race.py still 0 8   -> PNG stills in ./chk
Needs: pip install numpy pillow ; ffmpeg
"""
import math, os, random, re, subprocess, sys, wave
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SR = 1080, 1920, 30, 44100
AX0, AX1, AY0, AY1 = 30, 1050, 340, 1560            # arena
LANES = 4
WALL = 8
LW = (AX1 - AX0 - WALL * (LANES - 1)) / LANES       # lane width
COLS, ROWS = 5, 32
BH = 34                                             # brick height
BTOP = AY0 + 6                                      # bricks start right under the exit
R = 12                                              # ball radius
SPEED = 1700.0
SUB = 6
SPLIT, MAXB = 12, 6                                # one extra ball every SPLIT bricks, at most MAXB balls
LCOL = [(255, 99, 99), (255, 184, 64), (80, 210, 120), (90, 160, 255)]
BG, PANEL, PANEL2, BORDER = (13, 17, 23), (22, 27, 34), (30, 36, 44), (48, 54, 61)
TEXT, MUTED, GREEN, YEL, BLUE, RED = (230, 237, 243), (139, 148, 158), (63, 185, 80), (255, 212, 59), (88, 166, 255), (248, 81, 73)

def lane_x(l): return AX0 + l * (LW + WALL)

def brick_rgb(row):
    h = row / ROWS                                                  # rainbow from top to bottom
    k = [(255, 80, 110), (255, 150, 60), (255, 214, 64), (90, 210, 110), (70, 190, 230), (120, 120, 255), (190, 110, 240)]
    p = h * (len(k) - 1); i = min(int(p), len(k) - 2); f = p - i
    return tuple(int(k[i][j] + (k[i + 1][j] - k[i][j]) * f) for j in range(3))


# ------------------------------------------------------------------ physics
def simulate(seed, T=24.0):
    rnd = random.Random(seed)
    bw = LW / COLS
    grid = [[[True] * COLS for _ in range(ROWS)] for _ in range(LANES)]
    def new_ball(l, x, y, r_):
        a = math.radians(-90 + r_.uniform(-40, 40))
        return dict(x=x, y=y, vx=SPEED * math.cos(a), vy=SPEED * math.sin(a))
    lanes = [dict(balls=[new_ball(l, lane_x(l) + LW / 2, AY1 - 40, rnd)], rnd=random.Random(seed * 10 + l), broken=0) for l in range(LANES)]
    frames, hits, spawns, winner, t_win = [], [], [], None, None
    dt = 1 / (FPS * SUB)

    def brick_at(l, x, y):
        c = int((x - lane_x(l)) // bw); r = int((y - BTOP) // BH)
        if 0 <= r < ROWS and 0 <= c < COLS and grid[l][r][c]: return r, c
        return None

    def break_brick(l, hit, t):
        grid[l][hit[0]][hit[1]] = False; hits.append((t, l, hit[0])); L = lanes[l]; L["broken"] += 1
        if L["broken"] % SPLIT == 0 and len(L["balls"]) < MAXB:          # every SPLIT bricks: one more ball
            L["balls"].append(new_ball(l, lane_x(l) + LW / 2, AY1 - 40, L["rnd"])); spawns.append((t, l))

    for f in range(int(T * FPS)):
        for s in range(SUB):
            t = (f * SUB + s) * dt
            for l, L in enumerate(lanes):
                for b in L["balls"]:
                    x, y, vx, vy = b["x"], b["y"], b["vx"], b["vy"]
                    x += vx * dt
                    hit = brick_at(l, x + R * math.copysign(1, vx), y)
                    if hit: vx = -vx; break_brick(l, hit, t)
                    y += vy * dt
                    hit = brick_at(l, x, y + R * math.copysign(1, vy))
                    if hit: vy = -vy; break_brick(l, hit, t)
                    x0, x1 = lane_x(l) + R, lane_x(l) + LW - R
                    if x < x0: x, vx = x0, abs(vx)
                    if x > x1: x, vx = x1, -abs(vx)
                    if y > AY1 - R:                                     # floor: tiny random deflection avoids loops
                        y = AY1 - R; a = math.atan2(-abs(vy), vx) + math.radians(L["rnd"].uniform(-6, 6))
                        a = max(math.radians(-160), min(math.radians(-20), a)); vx, vy = SPEED * math.cos(a), SPEED * math.sin(a)
                    if y < AY0 + R and winner is None:                  # broke out through the top
                        winner, t_win = l, t
                    if y < AY0 + R and winner is not None: vy = abs(vy) if l != winner else vy
                    b["x"], b["y"], b["vx"], b["vy"] = x, y, vx, vy
        frames.append(([[(b["x"], b["y"]) for b in L["balls"]] for L in lanes], [[row[:] for row in g] for g in grid]))
        if t_win is not None and t > t_win + 6: break
    return frames, hits, winner, t_win, spawns

def broken_at(hits, l, t): return sum(1 for h in hits if h[1] == l and h[0] <= t)

def pick_seed():
    for seed in range(1, 800):
        fr, hits, w, tw, _ = simulate(seed, 17.0)
        if w is None or not 13.0 <= tw <= 15.5: continue
        b4 = [broken_at(hits, l, tw - 4) for l in range(LANES)]
        if b4[w] < max(b4) and sorted(b4)[-1] - b4[w] >= 8: return seed
    raise RuntimeError("no seed")

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "probe":
    for s in range(1, 40):
        fr, hits, w, tw, _ = simulate(s, 30.0)
        print(s, w, None if tw is None else round(tw, 1), [broken_at(hits, l, 30) for l in range(LANES)])
    sys.exit()

SEED = pick_seed()
FR, HITS, WIN, T_WIN, SPAWNS = simulate(SEED, 24.0)
T_END = T_WIN + 1.4
TOTAL = T_END + 4.2
T_LOOP = TOTAL - 0.6
NF = int(TOTAL * FPS)
NB = ROWS * COLS
print(f"seed {SEED}: lane {WIN + 1} wins at {T_WIN:.2f} s, total {TOTAL:.1f} s, {len(HITS)} bricks", flush=True)


# ------------------------------------------------------------------ drawing
F = "/usr/share/fonts/opentype/inter/"
@lru_cache(None)
def font(n, s): return ImageFont.truetype(F + n, max(1, int(s)))
BLACK = lambda s: font("Inter-Black.otf", s)
BOLD = lambda s: font("Inter-Bold.otf", s)
SEMI = lambda s: font("Inter-SemiBold.otf", s)
@lru_cache(None)
def mono(s): return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", max(1, int(s)))
def ease_out(x): x = max(0.0, min(1.0, x)); return 1 - (1 - x) ** 3
def ease_back(x): x = max(0.0, min(1.0, x)); return 1 + 2.70158 * (x - 1) ** 3 + 1.70158 * (x - 1) ** 2
def mix(a, b, k): return tuple(int(a[i] + (b[i] - a[i]) * k) for i in range(3))

S2 = 2
BW = LW / COLS
@lru_cache(None)
def brick_img(row):
    w, h = int(BW * S2), int(BH * S2); c = brick_rgb(row)
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle([2 * S2, 2 * S2, w - 2 * S2, h - 2 * S2], radius=5 * S2, fill=c)
    d.rounded_rectangle([5 * S2, 4 * S2, w - 5 * S2, 9 * S2], radius=2 * S2, fill=mix(c, (255, 255, 255), 0.45))
    d.rounded_rectangle([4 * S2, h - 8 * S2, w - 4 * S2, h - 4 * S2], radius=2 * S2, fill=mix(c, (0, 0, 0), 0.3))
    return im

def draw_arena(t):
    fi = min(int(t * FPS), len(FR) - 1)
    pos, grid = FR[fi]
    w, h = (AX1 - AX0) * S2, (AY1 - AY0) * S2
    im = Image.new("RGBA", (w, h), (18, 23, 30, 255)); d = ImageDraw.Draw(im)
    for l in range(LANES):
        lx = (lane_x(l) - AX0) * S2
        d.rectangle([lx, 0, lx + LW * S2, h], fill=mix((18, 23, 30), LCOL[l], 0.06))
        if l > 0: d.rectangle([lx - WALL * S2, 0, lx, h], fill=(40, 46, 56))
        # exit marker at the top of each lane
        d.rectangle([lx, 0, lx + LW * S2, 4 * S2], fill=mix(LCOL[l], (255, 255, 255), 0.3))
        for r in range(ROWS):
            for c in range(COLS):
                if grid[l][r][c]:
                    im.alpha_composite(brick_img(r), (int(lx + c * BW * S2), int((BTOP - AY0 + r * BH) * S2)))
    # shards from recent hits
    for (th, l, r) in HITS:
        if th > t: break
        if t - th < 0.35:
            k = (t - th) / 0.35; rnd = random.Random(int(th * 1e5))
            bx = lane_x(l) + LW / 2; by = BTOP + (r + 0.5) * BH
            for _ in range(5):
                a = rnd.uniform(0, 2 * math.pi); v = rnd.uniform(60, 220)
                x = (bx + rnd.uniform(-LW / 3, LW / 3) + math.cos(a) * v * k - AX0) * S2; y = (by + math.sin(a) * v * k + 300 * k * k - AY0) * S2
                s = (6 - 4 * k) * S2; d.rectangle([x - s, y - s, x + s, y + s], fill=brick_rgb(r) + (int(255 * (1 - k)),))
    # balls with trails
    for l in range(LANES):
        for bi in range(len(pos[l])):
            for j in range(5, 0, -1):
                if fi - j < 0 or bi >= len(FR[fi - j][0][l]): continue
                x, y = FR[fi - j][0][l][bi]; a = 1 - j / 6; rr = R * (0.4 + 0.5 * a) * S2
                d.ellipse([(x - AX0) * S2 - rr, (y - AY0) * S2 - rr, (x - AX0) * S2 + rr, (y - AY0) * S2 + rr], fill=mix((18, 23, 30), (255, 255, 255), 0.35 * a) + (255,))
            x, y = pos[l][bi]; rr = R * S2
            if y < AY0 - R: continue
            d.ellipse([(x - AX0) * S2 - rr, (y - AY0) * S2 - rr, (x - AX0) * S2 + rr, (y - AY0) * S2 + rr], fill=(255, 255, 255), outline=LCOL[l], width=3 * S2)
    return im.reduce(S2).convert("RGB")

def burst(d, t, t0, cx, cy, seed, cols, n=40):
    dt = t - t0
    if not 0 <= dt < 1.0: return
    rnd = random.Random(seed)
    for _ in range(n):
        a = rnd.uniform(0, 2 * math.pi); v = rnd.uniform(300, 900); s = rnd.uniform(5, 12)
        x = cx + math.cos(a) * v * dt; y = cy + math.sin(a) * v * dt + 900 * dt * dt; k = 1 - dt
        d.ellipse([x - s * k, y - s * k, x + s * k, y + s * k], fill=rnd.choice(cols))

def text2(d, y, parts, fnt, cx=W / 2):
    tot = sum(d.textlength(s, font=fnt) for s, _ in parts); x = cx - tot / 2
    for s, c in parts:
        d.text((x, y), s, font=fnt, fill=c, anchor="lm", stroke_width=5, stroke_fill=(8, 10, 16)); x += d.textlength(s, font=fnt)

CODE = ["x += vx * dt",
        "if brick_at(x + r * sign(vx), y):",
        "    vx = -vx; break_brick()",
        "y += vy * dt",
        "if brick_at(x, y + r * sign(vy)):",
        "    vy = -vy; break_brick()"]

def render(fi):
    t = fi / FPS
    img = Image.new("RGB", (W, H), BG)
    img.paste(draw_arena(t), (AX0, AY0))
    d = ImageDraw.Draw(img)
    d.rectangle([AX0 - 3, AY0 - 3, AX1 + 2, AY1 + 2], outline=BORDER, width=3)
    text2(d, 196, [("PICK A ", TEXT), ("LANE", YEL), (" 1-4", TEXT)], BLACK(92))
    for l in range(LANES):
        cx = lane_x(l) + LW / 2; cy = 286
        pct = int(round(100 * broken_at(HITS, l, min(t, T_WIN)) / NB))
        win = t >= T_WIN and l == WIN; kk = ease_back((t - T_WIN) / 0.4) if win else 0
        lost = t >= T_WIN and l != WIN
        col = mix(LCOL[l], BG, 0.55) if lost else LCOL[l]
        wd, hh = 228 * (1 + 0.1 * kk), 64 * (1 + 0.1 * kk)
        d.rounded_rectangle([cx - wd / 2, cy - hh / 2, cx + wd / 2, cy + hh / 2], radius=hh / 2, fill=mix(BG, col, 0.22), outline=YEL if win else col, width=5 if win else 3)
        d.ellipse([cx - wd / 2 + 12, cy - 22, cx - wd / 2 + 56, cy + 22], fill=col)
        d.text((cx - wd / 2 + 34, cy + 1), str(l + 1), font=BLACK(30), fill=(20, 22, 28), anchor="mm")
        d.text((cx + wd / 2 - 18, cy), f"{pct}%", font=mono(30), fill=(255, 255, 255) if not lost else MUTED, anchor="rm")
        nb = len(FR[min(int(t * FPS), len(FR) - 1)][0][l])
        d.text((cx - 4, cy), f"x{nb}", font=BOLD(24), fill=col if not lost else MUTED, anchor="mm")
    for (ts, l) in SPAWNS:
        if 0 <= t - ts < 0.7 and ts < T_WIN:
            a = 1 - (t - ts) / 0.7
            d.text((lane_x(l) + LW / 2, AY1 - 70 - 60 * (1 - a)), "+1 BALL", font=BLACK(34), fill=mix(BG, LCOL[l], a), anchor="mm", stroke_width=4, stroke_fill=(8, 10, 16))
    if t < T_WIN:
        d.text((W / 2, 1600), "First ball out of the top wins", font=SEMI(34), fill=MUTED, anchor="mm")
    if T_WIN <= t < T_END + 0.4:
        kb = ease_back((t - T_WIN) / 0.35)
        if kb > 0.02:
            cy = (AY0 + AY1) / 2
            d.rounded_rectangle([W / 2 - 360 * kb, cy - 100 * kb, W / 2 + 360 * kb, cy + 100 * kb], radius=30 * kb, fill=(16, 20, 26), outline=LCOL[WIN], width=6)
            if kb > 0.5:
                text2(d, cy - 30, [("LANE ", TEXT), (str(WIN + 1), LCOL[WIN])], BLACK(92 * kb))
                d.text((W / 2, cy + 52), "BROKE THROUGH!", font=BLACK(52 * kb), fill=YEL, anchor="mm", stroke_width=4, stroke_fill=(8, 10, 16))
        burst(d, t, T_WIN, lane_x(WIN) + LW / 2, AY0 + 10, 9, [YEL, LCOL[WIN], (255, 255, 255), BLUE])
    if t >= T_END: end_card(img, t, t - T_END)
    return np.asarray(img)

def end_card(img, t, et):
    k = min(1.0, et / 0.3)
    img.paste(Image.blend(img, Image.new("RGB", img.size, (8, 10, 14)), 0.82 * k))
    d = ImageDraw.Draw(img)
    kp = ease_out(et / 0.35); y0 = 360 + (1 - kp) * 120
    d.rounded_rectangle([40, y0, 940, y0 + 1200], radius=34, fill=(16, 20, 26), outline=BORDER, width=3)
    if et < 0.12: return
    k1 = ease_back((et - 0.1) / 0.4)
    d.text((490, y0 + 100), "DID YOU PICK", font=BLACK(84 * k1), fill=TEXT, anchor="mm")
    text2(d, y0 + 210, [("LANE ", TEXT), (str(WIN + 1) + "?", LCOL[WIN])], BLACK(110 * k1), cx=490)
    k3 = ease_out((et - 0.7) / 0.3)
    if k3 > 0:
        d.text((490, y0 + 360), "Comment your lane", font=BLACK(60), fill=mix((16, 20, 26), YEL, k3), anchor="mm")
        ay = y0 + 425 + 10 * math.sin(t * 8)
        d.polygon([(470, ay), (510, ay), (490, ay + 26)], fill=mix((16, 20, 26), YEL, k3))
    k4 = ease_out((et - 1.1) / 0.3)
    if k4 > 0:
        bx0, by0 = 70, y0 + 500
        d.rounded_rectangle([bx0, by0, 910, by0 + 380], radius=18, fill=(10, 13, 18), outline=BORDER, width=2)
        d.text((bx0 + 24, by0 + 28), "the bounce, in Python:", font=SEMI(26), fill=MUTED, anchor="lm")
        for li, ln in enumerate(CODE):
            x = bx0 + 24; y = by0 + 80 + li * 48
            for m in re.finditer(r'(\bif\b)|(\w+(?=\())|(.)', ln):
                s = m.group(0); col = (255, 123, 114) if m.group(1) else (210, 168, 255) if m.group(2) else TEXT
                d.text((x, y), s, font=mono(26), fill=mix((10, 13, 18), col, k4), anchor="lm"); x += d.textlength(s, font=mono(26))
    k5 = ease_out((et - 1.6) / 0.3)
    if k5 > 0:
        d.text((490, y0 + 970), "Full code: link in profile", font=BOLD(40), fill=mix((16, 20, 26), TEXT, k5), anchor="mm")
        text2(d, y0 + 1085, [("PYTHONIC ", mix((16, 20, 26), BLUE, k5)), ("LIFE", mix((16, 20, 26), YEL, k5))], BLACK(68), cx=490)


# ------------------------------------------------------------------ audio
def synth_audio():
    NS = int((TOTAL + 1) * SR); mus = np.zeros(NS, np.float32); sfx = np.zeros(NS, np.float32)
    def put(buf, t0, sig):
        s = int(t0 * SR); n = min(len(sig), NS - s)
        if n > 0 and s >= 0: buf[s:s + n] += sig[:n].astype(np.float32)
    def tt(dur): return np.arange(int(dur * SR)) / SR
    def noise(n, seed): return np.random.default_rng(seed).standard_normal(n)
    def lp(x, a): return np.convolve(x, np.ones(a) / a, mode="same")
    def ping(t0, f, amp):
        x = tt(0.35); ph = 2 * np.pi * f * x
        put(sfx, t0, (np.sin(ph) + 0.4 * np.sin(2 * ph) + 0.15 * np.sin(3 * ph)) * np.exp(-x * 14) * np.minimum(1, x * 1500) * amp)
        n = tt(0.012); put(sfx, t0, noise(len(n), int(t0 * 1e4)) * np.exp(-n * 300) * amp * 0.6)
    def kick(t0, amp):
        x = tt(0.3); put(mus, t0, np.sin(2 * np.pi * np.cumsum(140 * np.exp(-x * 30) + 46) / SR) * np.exp(-x * 9) * amp)
    def snare(t0, amp):
        x = tt(0.2); put(mus, t0, (0.7 * noise(len(x), int(t0 * 1000)) * np.exp(-x * 22) + 0.5 * np.sin(2 * np.pi * 200 * x) * np.exp(-x * 30)) * amp)
    def hat(t0, amp):
        x = tt(0.04); put(mus, t0, np.diff(noise(len(x) + 1, int(t0 * 991))) * np.exp(-x * 90) * amp)
    def bass(t0, f, dur, amp):
        x = tt(dur); ph = 2 * np.pi * f * x
        put(mus, t0, (np.sin(ph) + 0.2 * np.sign(np.sin(ph))) * np.minimum(1, x * 300) * np.exp(-x * 4) * amp)
    def impact(t0, amp):
        x = tt(1.0)
        put(sfx, t0, np.sin(2 * np.pi * np.cumsum(90 * np.exp(-x * 12) + 40) / SR) * np.exp(-x * 4) * 0.6 * amp)
        put(sfx, t0, lp(noise(len(x), 5), 5) * 2.2 * np.exp(-x * 10) * 0.3 * amp)
    def chord(t0, notes, amp, dec=2.5):
        x = tt(1.6); put(sfx, t0, sum(np.sin(2 * np.pi * f * x) + 0.3 * np.sin(4 * np.pi * f * x) for f in notes) * np.exp(-x * dec) * np.minimum(1, x * 500) * amp)
    def whoosh(t0, dur, amp, seed):
        x = tt(dur); n = noise(len(x), seed); k = x / dur
        put(sfx, t0, (lp(n, 16) * 3.5 * (1 - k) + (n - lp(n, 5)) * k) * np.sin(np.pi * k) ** 1.5 * amp)
    B = 60 / 124; t0 = 0.0; roots = [110.0, 87.31, 130.81, 98.0]; bar = 0
    while t0 < TOTAL + 0.5:
        hot = t0 >= T_WIN - 5
        for b_ in range(4):
            kick(t0 + b_ * B, 0.34 if hot else 0.28)
            if b_ in (1, 3): snare(t0 + b_ * B, 0.16 if hot else 0.11)
            for h in range(4 if hot else 2): hat(t0 + b_ * B + h * B / (4 if hot else 2), 0.045 if h % 2 else 0.028)
        for e in range(8): bass(t0 + e * B / 2, roots[bar % 4], B / 2, 0.12)
        t0 += 4 * B; bar += 1
    # brick breaks: pentatonic notes, pitch rises as the lane gets closer to the top
    scale = [440.0, 523.3, 587.3, 659.3, 784.0, 880.0, 1046.5, 1174.7, 1318.5, 1568.0]
    last = -1
    for th, l, r in HITS:
        if th > T_WIN: break
        if th - last < 0.045: continue
        idx = min(len(scale) - 1, int((ROWS - 1 - r) / ROWS * len(scale)))
        ping(th, scale[idx], 0.09); last = th
    for ts, l in SPAWNS:
        if ts < T_WIN:
            x = tt(0.2); put(sfx, ts, np.sin(2 * np.pi * np.cumsum(600 + 1400 * x / 0.2) / SR) * np.exp(-x * 12) * 0.08)
    whoosh(T_WIN - 1.2, 1.2, 0.12, 8)
    impact(T_WIN, 1.3); chord(T_WIN + 0.05, [440.0, 554.4, 659.3, 880.0], 0.06)
    impact(T_END, 0.6); chord(T_END + 0.7, [1318.5, 1975.5], 0.05, 3)
    a = (mus + sfx)[: int(TOTAL * SR)]
    a = np.tanh(a / (np.percentile(np.abs(a), 99.7) + 1e-6) * 1.1) * 0.9
    nf = int(0.1 * SR); a[-nf:] *= np.linspace(1, 0.3, nf)
    return a


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "still":
        os.makedirs("chk", exist_ok=True)
        for ts in sys.argv[2:]: Image.fromarray(render(int(float(ts) * FPS))).save(f"chk/still_{ts}.png")
        sys.exit()
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                             "-c:v", "libx264", "-preset", "slow", "-crf", "15", "-maxrate", "10M", "-bufsize", "20M", "-pix_fmt", "yuv420p",
                             "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "brick_noaudio.mp4"], stdin=subprocess.PIPE)
    F0 = render(0).astype(np.float32)
    for f in range(NF):
        t = f / FPS; fr = render(f)
        if t > T_LOOP:
            a = min(1.0, (t - T_LOOP) / (TOTAL - T_LOOP - 1 / FPS)); fr = (fr * (1 - a) + F0 * a).astype(np.uint8)
        proc.stdin.write(np.ascontiguousarray(fr).tobytes())
    proc.stdin.close(); proc.wait()
    a = synth_audio()
    with wave.open("brick_audio.wav", "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR); wf.writeframes((a * 32767).astype(np.int16).tobytes())
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "brick_noaudio.mp4", "-i", "brick_audio.wav", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", "brick_race.mp4"], check=True)
    print("done", round(TOTAL, 1))
