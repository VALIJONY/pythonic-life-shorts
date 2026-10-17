"""Pythonic Life — "PICK A BALL: who escapes 3 rings first?" (YouTube Shorts 1080x1920, ~20 s, loops)
Six balls bounce inside three rotating rings, each ring has a gap. First ball out of all three wins.
Real physics (gravity, elastic wall and ball collisions), every wall bounce plays the next note of a melody.
The seed is chosen so the race is close and the winner comes from behind (printed when rendering).
Usage: python3 ring_escape.py            -> ring_escape.mp4
       python3 ring_escape.py still 0 9   -> PNG stills in ./chk
Needs: pip install numpy pillow ; ffmpeg
"""
import math, os, random, re, subprocess, sys, wave
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SR = 1080, 1920, 30, 44100
CX, CY = 540, 860                                   # ring centre
RINGS = [200, 290, 380]                             # inner -> outer
TH = 12                                             # ring thickness
RB = 22                                             # ball radius
HG = 0.18                                           # gap half-width (rad)
SPIN = [(1.05, 0.020), (-0.85, -0.030), (0.70, 0.035)]   # angular speed, acceleration per ring
G = 1100.0
VMIN, VMAX = 700.0, 1350.0
SUB = 8
BALLC = [(248, 81, 73), (255, 159, 67), (255, 212, 59), (63, 185, 80), (88, 166, 255), (188, 140, 255)]
RINGC = [(255, 212, 59), (88, 166, 255), (255, 128, 92)]
BG, PANEL, PANEL2, BORDER = (13, 17, 23), (22, 27, 34), (30, 36, 44), (48, 54, 61)
TEXT, MUTED, GREEN, YEL, BLUE = (230, 237, 243), (139, 148, 158), (63, 185, 80), (255, 212, 59), (88, 166, 255)


def ring_angle(i, t):
    w, a = SPIN[i]; return [0.3, 2.4, 4.4][i] + w * t + 0.5 * a * t * t

def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi


# ------------------------------------------------------------------ physics
def in_gap(i, x, y, t, hg):
    a = math.atan2(y - CY, x - CX)
    return abs(wrap(a - ring_angle(i, t))) < hg - RB / RINGS[i]

def bounce(b, nx, ny, sign):
    """reflect velocity on a wall with normal (nx, ny); sign=+1 inner wall of a ring, -1 outer side"""
    dot = b["vx"] * nx + b["vy"] * ny
    if dot * sign > 0:
        b["vx"], b["vy"] = b["vx"] - 2 * dot * nx, b["vy"] - 2 * dot * ny
        s = math.hypot(b["vx"], b["vy"]); k = min(VMAX, max(VMIN, s)) / (s + 1e-9)
        b["vx"] *= k; b["vy"] *= k
        return True
    return False

def simulate(seed, T=22.0):
    rnd = random.Random(seed)
    balls = []
    for k in range(6):
        a = k * math.pi / 3 + rnd.uniform(-0.2, 0.2); sp = rnd.uniform(650, 950); va = rnd.uniform(0, 2 * math.pi)
        balls.append(dict(x=CX + 80 * math.cos(a), y=CY + 80 * math.sin(a), vx=sp * math.cos(va), vy=sp * math.sin(va), level=0, passing=None, out_t=None))
    frames, events = [], []
    winner, t_win = None, None
    dt = 1 / (FPS * SUB)
    for f in range(int(T * FPS)):
        for s in range(SUB):
            t = (f * SUB + s) * dt
            hg = HG if t_win is None else HG * max(0.0, 1 - (t - t_win) / 0.3)     # gaps close after the win
            for k, b in enumerate(balls):
                b["vy"] += G * dt; b["x"] += b["vx"] * dt; b["y"] += b["vy"] * dt
                if b["level"] >= 3: continue
                x, y = b["x"], b["y"]; cx, cy = CX, CY
                d = math.hypot(x - cx, y - cy); i = b["level"]; R = RINGS[i]
                # inner side of the ring that contains the ball
                if b["passing"] is None:
                    if d > R - TH / 2 - RB:
                        if in_gap(i, x, y, t, hg): b["passing"] = i
                        else:
                            nx, ny = (x - cx) / d, (y - cy) / d
                            if bounce(b, nx, ny, +1): events.append(("bounce", t, k, i, math.atan2(ny, nx)))
                            b["x"], b["y"] = cx + nx * (R - TH / 2 - RB), cy + ny * (R - TH / 2 - RB)
                else:
                    if d > R + TH / 2 + RB:
                        b["passing"] = None; b["level"] += 1; events.append(("pass", t, k, b["level"]))
                        if b["level"] == 3 and winner is None: winner, t_win = k, t; events.append(("win", t, k))
                    elif d < R - TH / 2 - RB - 1: b["passing"] = None
                    elif not in_gap(i, x, y, t, hg + RB / R):                       # gap rotated away: blocked
                        b["passing"] = None; nx, ny = (x - cx) / d, (y - cy) / d
                        if bounce(b, nx, ny, +1): events.append(("bounce", t, k, i, math.atan2(ny, nx)))
                        b["x"], b["y"] = cx + nx * (R - TH / 2 - RB), cy + ny * (R - TH / 2 - RB)
                # outer side of the ring the ball already left
                if b["level"] >= 1 and b["level"] < 3:
                    j = b["level"] - 1; r0 = RINGS[j] + TH / 2 + RB
                    d = math.hypot(b["x"] - cx, b["y"] - cy)
                    if d < r0:
                        nx, ny = (b["x"] - cx) / d, (b["y"] - cy) / d
                        if bounce(b, nx, ny, -1): events.append(("bounce", t, k, j, math.atan2(ny, nx)))
                        b["x"], b["y"] = cx + nx * r0, cy + ny * r0
            # ball - ball (same level only)
            for a_ in range(6):
                for c_ in range(a_ + 1, 6):
                    p, q = balls[a_], balls[c_]
                    if p["level"] != q["level"] or p["level"] >= 3: continue
                    dx, dy = q["x"] - p["x"], q["y"] - p["y"]; d = math.hypot(dx, dy)
                    if 0 < d < 2 * RB:
                        nx, ny = dx / d, dy / d; rel = (p["vx"] - q["vx"]) * nx + (p["vy"] - q["vy"]) * ny
                        if rel > 0:
                            p["vx"] -= rel * nx; p["vy"] -= rel * ny; q["vx"] += rel * nx; q["vy"] += rel * ny
                            events.append(("clack", t, a_, c_))
                        o = (2 * RB - d) / 2; p["x"] -= nx * o; p["y"] -= ny * o; q["x"] += nx * o; q["y"] += ny * o
        frames.append([(b["x"], b["y"], b["level"]) for b in balls])
    return frames, events, winner, t_win


def pick_seed():
    for seed in range(1, 400):
        fr, ev, w, tw = simulate(seed, 17.0)
        if w is None or not (12.8 <= tw <= 15.0): continue
        f0 = int((tw - 3.5) * FPS)
        lv = [fr[f0][k][2] for k in range(6)]
        if lv[w] < max(lv[k] for k in range(6) if k != w):              # winner came from behind
            return seed
    raise RuntimeError("no seed")


# ------------------------------------------------------------------ fonts / helpers
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

GAP = 54
def make_bg():
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    for y in range(0, H, GAP):
        for x in range(0, W, GAP): d.ellipse([x - 2.5, y - 2.5, x + 2.5, y + 2.5], fill=(26, 33, 44))
    return im
BGIMG = make_bg()

# ------------------------------------------------------------------ build the race
SEED = pick_seed()
FR, EV, WIN, T_WIN = simulate(SEED, 24.0)
T_BANNER = T_WIN
T_END = T_WIN + 1.4
TOTAL = T_END + 4.2
T_LOOP = TOTAL - 0.6
NF = int(TOTAL * FPS)
BOUNCES = [e for e in EV if e[0] == "bounce"]
PASSES = [e for e in EV if e[0] == "pass"]
print(f"seed {SEED}: ball {WIN + 1} wins at {T_WIN:.2f} s, total {TOTAL:.1f} s, {len(BOUNCES)} bounces", flush=True)

def level_at(k, t):
    return FR[min(int(t * FPS), len(FR) - 1)][k][2]

# ------------------------------------------------------------------ drawing
BOX = (CX - 420, CY - 420, CX + 420, CY + 420)                  # ring area, drawn at 2x
S2 = 2

def draw_arena(t):
    w, h = (BOX[2] - BOX[0]) * S2, (BOX[3] - BOX[1]) * S2
    im = Image.new("RGB", (w, h), BG); d = ImageDraw.Draw(im)
    ox, oy = BOX[0], BOX[1]
    P = lambda x, y: ((x - ox) * S2, (y - oy) * S2)
    # faint dots copied from the background so the panel blends in
    im.paste(BGIMG.crop(BOX).resize((w, h), Image.NEAREST))
    hg = HG if t < T_WIN else HG * max(0.0, 1 - (t - T_WIN) / 0.3)
    for i, R in enumerate(RINGS):
        th = ring_angle(i, t); col = RINGC[i]
        bb = [P(CX - R, CY - R), P(CX + R, CY + R)]
        bb = [bb[0][0], bb[0][1], bb[1][0], bb[1][1]]
        a0, a1 = math.degrees(th + hg), math.degrees(th - hg + 2 * math.pi)
        d.arc(bb, a0, a1, fill=col, width=TH * S2)
        if hg > 0.01:
            for a in (th + hg, th - hg):
                ex, ey = P(CX + R * math.cos(a), CY + R * math.sin(a)); r = TH * S2 / 2
                d.ellipse([ex - r, ey - r, ex + r, ey + r], fill=col)
    # bounce flashes on the rings
    for e in BOUNCES:
        if 0 <= t - e[1] < 0.14:
            k = 1 - (t - e[1]) / 0.14; i = e[3]; R = RINGS[i]; a = e[4]
            bb = [P(CX - R, CY - R), P(CX + R, CY + R)]; bb = [bb[0][0], bb[0][1], bb[1][0], bb[1][1]]
            d.arc(bb, math.degrees(a - 0.16), math.degrees(a + 0.16), fill=mix(RINGC[i], (255, 255, 255), 0.8), width=int((TH + 10 * k) * S2))
        elif e[1] > t: break
    # trails then balls
    fi = min(int(t * FPS), len(FR) - 1)
    for k in range(6):
        for j in range(6, 0, -1):
            if fi - j < 0: continue
            x, y, _ = FR[fi - j][k]; a = (1 - j / 7) * 0.45; r = RB * (0.45 + 0.4 * (1 - j / 7))
            c = mix(BG, BALLC[k], a); px, py = P(x, y)
            d.ellipse([px - r * S2, py - r * S2, px + r * S2, py + r * S2], fill=c)
    for k in range(6):
        x, y, lv = FR[fi][k]; px, py = P(x, y); r = RB * S2
        if lv >= 3 and (x < BOX[0] - 40 or x > BOX[2] + 40 or y > BOX[3] + 40): continue
        d.ellipse([px - r, py - r, px + r, py + r], fill=BALLC[k], outline=mix(BALLC[k], (0, 0, 0), 0.45), width=3 * S2)
        d.ellipse([px - r * 0.55, py - r * 0.62, px - r * 0.05, py - r * 0.25], fill=mix(BALLC[k], (255, 255, 255), 0.55))
        d.text((px, py + 2 * S2), str(k + 1), font=BLACK(25 * S2), fill=(20, 22, 28) if k == 2 else (255, 255, 255), anchor="mm")
    return im.reduce(S2)

def burst(d, t, t0, cx, cy, seed):
    dt = t - t0
    if not 0 <= dt < 1.0: return
    rnd = random.Random(seed)
    for _ in range(40):
        a = rnd.uniform(0, 2 * math.pi); v = rnd.uniform(380, 900); s = rnd.uniform(6, 13)
        col = rnd.choice([YEL, BLUE, (255, 128, 92), GREEN, (255, 255, 255), BALLC[WIN]])
        x = cx + math.cos(a) * v * dt; y = cy + math.sin(a) * v * dt + 900 * dt * dt; k = 1 - dt
        d.ellipse([x - s * k, y - s * k, x + s * k, y + s * k], fill=col)

def text2(d, y, parts, fnt, cx=W / 2):
    tot = sum(d.textlength(s, font=fnt) for s, _ in parts); x = cx - tot / 2
    for s, c in parts:
        d.text((x, y), s, font=fnt, fill=c, anchor="lm", stroke_width=5, stroke_fill=(8, 10, 16)); x += d.textlength(s, font=fnt)

CODE = ["d = math.hypot(x - cx, y - cy)",
        "if d > R - r and not in_gap(x, y):",
        "    nx, ny = (x - cx) / d, (y - cy) / d",
        "    dot = vx * nx + vy * ny",
        "    vx, vy = vx - 2*dot*nx, vy - 2*dot*ny"]

def render(fi):
    t = fi / FPS
    img = BGIMG.copy(); img.paste(draw_arena(t), (BOX[0], BOX[1])); d = ImageDraw.Draw(img)
    text2(d, 196, [("PICK A ", TEXT), ("BALL", YEL), (" 1-6", TEXT)], BLACK(100))
    d.text((W / 2, 282), "Who escapes all 3 rings first?", font=SEMI(42), fill=MUTED, anchor="mm")
    tt = min(t, T_WIN); s = f"{tt:05.2f} s"
    col = YEL if t < T_WIN else GREEN
    d.rounded_rectangle([W / 2 - 110, 336, W / 2 + 110, 386], radius=25, fill=mix(BG, col, 0.2), outline=col, width=3)
    d.text((W / 2, 361), s, font=mono(32), fill=(255, 255, 255), anchor="mm")
    # leaderboard chips: rings escaped per ball
    for k in range(6):
        x = 135 + k * 138; y = 1352; lv = level_at(k, t)
        win = t >= T_WIN and k == WIN; kk = ease_back((t - T_WIN) / 0.4) if win else 0
        r = 46 * (1 + 0.28 * kk)
        if win: d.ellipse([x - r - 8, y - r - 8, x + r + 8, y + r + 8], outline=YEL, width=6)
        lost = t >= T_WIN and k != WIN
        c = mix(BALLC[k], BG, 0.55) if lost else BALLC[k]
        d.ellipse([x - r, y - r, x + r, y + r], fill=c)
        d.text((x, y + 2), str(k + 1), font=BLACK(48 * (1 + 0.28 * kk)), fill=(20, 22, 28) if k == 2 and not lost else (255, 255, 255), anchor="mm")
        for j in range(3):
            dx = x - 26 + j * 26; dy = 1428 + 10 * kk
            on = lv > j
            d.ellipse([dx - 8, dy - 8, dx + 8, dy + 8], fill=RINGC[j] if on else PANEL2, outline=BORDER if not on else None)
    # recent ring pass pop text
    for e in PASSES:
        if 0 <= t - e[1] < 0.7 and e[3] < 3:
            k = e[2]; x = 135 + k * 138; a = 1 - (t - e[1]) / 0.7
            d.text((x, 1278 - 30 * (1 - a)), f"+{e[3]}", font=BLACK(34), fill=mix(BG, RINGC[e[3] - 1], a), anchor="mm")
    if t < T_WIN:
        d.text((W / 2, 1510), "Pick one now. Watch the dots.", font=SEMI(34), fill=MUTED, anchor="mm")
    # winner banner
    if T_WIN <= t < T_END + 0.4:
        k = ease_back((t - T_WIN) / 0.35)
        if k > 0.02:
            d.rounded_rectangle([W / 2 - 330 * k, CY - 95 * k, W / 2 + 330 * k, CY + 95 * k], radius=30 * k, fill=(16, 20, 26), outline=BALLC[WIN], width=5)
            if k > 0.5:
                text2(d, CY - 30, [("BALL ", TEXT), (str(WIN + 1), BALLC[WIN])], BLACK(84 * k))
                d.text((W / 2, CY + 48), "ESCAPED!", font=BLACK(56 * k), fill=YEL, anchor="mm", stroke_width=4, stroke_fill=(8, 10, 16))
        fw = min(int(T_WIN * FPS), len(FR) - 1); ex, ey, _ = FR[fw][WIN]
        burst(d, t, T_WIN, ex, ey, 5)
    if t >= T_END:
        end_card(img, t, t - T_END)
    return np.asarray(img)

def end_card(img, t, et):
    k = min(1.0, et / 0.3)
    img.paste(Image.blend(img, Image.new("RGB", img.size, (8, 10, 14)), 0.8 * k))
    d = ImageDraw.Draw(img)
    kp = ease_out(et / 0.35); y0 = 340 + (1 - kp) * 120
    d.rounded_rectangle([40, y0, 940, y0 + 1220], radius=34, fill=(16, 20, 26), outline=BORDER, width=3)
    if et < 0.12: return
    k1 = ease_back((et - 0.1) / 0.4)
    d.text((490, y0 + 110), "DID YOU PICK", font=BLACK(84 * k1), fill=TEXT, anchor="mm")
    cx, cy, r = 490, y0 + 290, 96 * k1
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=BALLC[WIN], outline=YEL, width=7)
    d.text((cx, cy + 4), str(WIN + 1), font=BLACK(120 * k1), fill=(20, 22, 28) if WIN == 2 else (255, 255, 255), anchor="mm")
    k3 = ease_out((et - 0.8) / 0.3)
    if k3 > 0:
        d.text((490, y0 + 470), "Comment your ball", font=BLACK(60), fill=mix((16, 20, 26), YEL, k3), anchor="mm")
        ay = y0 + 535 + 10 * math.sin(t * 8)
        d.polygon([(470, ay), (510, ay), (490, ay + 26)], fill=mix((16, 20, 26), YEL, k3))
    k4 = ease_out((et - 1.2) / 0.3)
    if k4 > 0:
        bx0, by0 = 70, y0 + 610
        d.rounded_rectangle([bx0, by0, 910, by0 + 300], radius=18, fill=(10, 13, 18), outline=BORDER, width=2)
        d.text((bx0 + 24, by0 + 28), "the bounce, in Python:", font=SEMI(26), fill=MUTED, anchor="lm")
        for li, ln in enumerate(CODE):
            x = bx0 + 24; y = by0 + 80 + li * 46
            for m in re.finditer(r'(\bif\b|\band\b|\bnot\b)|(\w+(?=\())|(.)', ln):
                s = m.group(0); col = (255, 123, 114) if m.group(1) else (210, 168, 255) if m.group(2) else TEXT
                d.text((x, y), s, font=mono(26), fill=mix((10, 13, 18), col, k4), anchor="lm"); x += d.textlength(s, font=mono(26))
    k5 = ease_out((et - 1.7) / 0.3)
    if k5 > 0:
        d.text((490, y0 + 990), "Full code: link in profile", font=BOLD(40), fill=mix((16, 20, 26), TEXT, k5), anchor="mm")
        text2(d, y0 + 1100, [("PYTHONIC ", mix((16, 20, 26), BLUE, k5)), ("LIFE", mix((16, 20, 26), YEL, k5))], BLACK(68), cx=490)


# ------------------------------------------------------------------ audio
def synth_audio():
    NS = int((TOTAL + 1) * SR); mus = np.zeros(NS, np.float32); sfx = np.zeros(NS, np.float32)
    def put(buf, t0, sig):
        s = int(t0 * SR); n = min(len(sig), NS - s)
        if n > 0 and s >= 0: buf[s:s + n] += sig[:n].astype(np.float32)
    def tt(dur): return np.arange(int(dur * SR)) / SR
    def noise(n, seed): return np.random.default_rng(seed).standard_normal(n)
    def lp(x, a): return np.convolve(x, np.ones(a) / a, mode="same")
    def pluck(buf, t0, f, amp, dec=7.0):
        x = tt(0.9); ph = 2 * np.pi * f * x
        put(buf, t0, (np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.12 * np.sin(3 * ph + 0.4) + 0.2 * np.sin(ph * 1.003)) * np.exp(-x * dec) * np.minimum(1, x * 900) * amp)
    def kick(t0, amp):
        x = tt(0.3); put(mus, t0, np.sin(2 * np.pi * np.cumsum(130 * np.exp(-x * 30) + 45) / SR) * np.exp(-x * 9) * amp)
    def hat(t0, amp):
        x = tt(0.05); put(mus, t0, np.diff(noise(len(x) + 1, int(t0 * 991))) * np.exp(-x * 80) * amp)
    def pad(t0, notes, dur, amp):
        x = tt(dur); env = np.minimum(1, x / 0.4) * np.minimum(1, (dur - x) / 0.6)
        put(mus, t0, sum(np.sin(2 * np.pi * f * x) + 0.3 * np.sin(4 * np.pi * f * x + 0.5) for f in notes) * env * amp)
    def whoosh(t0, dur, amp, seed):
        x = tt(dur); n = noise(len(x), seed); k = x / dur
        put(sfx, t0, (lp(n, 16) * 3.5 * (1 - k) + (n - lp(n, 5)) * k) * np.sin(np.pi * k) ** 1.5 * amp)
    def impact(t0, amp):
        x = tt(1.0)
        put(sfx, t0, np.sin(2 * np.pi * np.cumsum(90 * np.exp(-x * 12) + 40) / SR) * np.exp(-x * 4) * 0.6 * amp)
        put(sfx, t0, lp(noise(len(x), 7), 5) * 2.2 * np.exp(-x * 10) * 0.3 * amp)

    # the melody: every wall bounce plays its next note (A minor pentatonic, original phrase)
    A = {"A3": 220.0, "C4": 261.6, "D4": 293.7, "E4": 329.6, "G4": 392.0, "A4": 440.0, "C5": 523.3, "D5": 587.3, "E5": 659.3, "G5": 784.0, "A5": 880.0}
    MEL = "A4 C5 E5 D5 C5 A4 G4 A4 C5 D5 E5 G5 E5 D5 C5 D5 E5 A5 G5 E5 D5 C5 A4 C5 D5 E5 D5 C5 A4 G4 E4 A4".split()
    last, n = -1.0, 0
    for e in BOUNCES:
        if e[1] >= T_WIN + 0.35: break
        if e[1] - last < 0.085: continue
        f = A[MEL[n % len(MEL)]] * (2 if e[1] > T_WIN - 4 else 1) ** 0                 # same octave, it gets busier on its own
        pluck(sfx, e[1], f, 0.10 + 0.03 * e[3], 7.5); last = e[1]; n += 1
    for e in EV:                                                                           # ball-ball clacks, quiet
        if e[0] == "clack" and e[1] < T_WIN:
            x = tt(0.03); put(sfx, e[1], noise(len(x), int(e[1] * 1e4)) * np.exp(-x * 220) * 0.05)
    for e in PASSES:                                                                       # ring passed: rising chime + whoosh
        if e[3] >= 3: continue
        base = [523.3, 659.3, 784.0][e[3] - 1]
        for j, m in enumerate((1, 1.25, 1.5)): pluck(sfx, e[1] + 0.05 * j, base * m, 0.09, 6)
        whoosh(e[1] - 0.05, 0.3, 0.07, int(e[1] * 100))
    whoosh(T_WIN - 1.6, 1.6, 0.13, 99)                                                     # riser into the win
    impact(T_WIN, 1.3)
    for j, f in enumerate((440.0, 554.4, 659.3, 880.0)): pluck(sfx, T_WIN + 0.06 * j, f, 0.13, 2.5)   # A major lift
    impact(T_END, 0.6)
    for j, f in enumerate((1318.5, 1975.5)): pluck(sfx, T_END + 0.8 + 0.08 * j, f, 0.08, 3)
    # soft bed: Am - F - C - G pads, light pulse
    B = 0.5; bar = 4 * B; ch = [[220.0, 261.6, 329.6], [174.6, 220.0, 261.6], [196.0, 261.6, 329.6], [196.0, 246.9, 293.7]]
    tb, ci = 0.0, 0
    while tb < TOTAL + 0.5:
        pad(tb, ch[ci % 4], bar + 0.3, 0.020)
        for b_ in range(4):
            kick(tb + b_ * B, 0.20 if tb < T_WIN - 4 else 0.30)
            hat(tb + b_ * B + B / 2, 0.035 if tb < T_WIN - 4 else 0.055)
        tb += bar; ci += 1
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
                             "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "ring_noaudio.mp4"], stdin=subprocess.PIPE)
    F0 = render(0).astype(np.float32)
    for f in range(NF):
        t = f / FPS; fr = render(f)
        if t > T_LOOP:
            a = min(1.0, (t - T_LOOP) / (TOTAL - T_LOOP - 1 / FPS)); fr = (fr * (1 - a) + F0 * a).astype(np.uint8)
        proc.stdin.write(np.ascontiguousarray(fr).tobytes())
    proc.stdin.close(); proc.wait()
    a = synth_audio()
    with wave.open("ring_audio.wav", "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR); wf.writeframes((a * 32767).astype(np.int16).tobytes())
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "ring_noaudio.mp4", "-i", "ring_audio.wav", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", "ring_escape.mp4"], check=True)
    print("done", round(TOTAL, 1))
