"""Pythonic Life — "PICK A SNAKE: last one alive wins" (YouTube Shorts 1080x1920, ~20 s, loops)
Four AI snakes race for apples on a 34x40 grid while the zone closes in. Crash into a wall, a body or the zone and
you are out. The AI is a few lines of Python (greedy to the nearest apple + a flood-fill check for free space).
The seed is picked so the race ends at ~14 s and the winner is not the snake that was leading.
Usage: python3 snake_battle.py            -> snake_battle.mp4
       python3 snake_battle.py still 0 8   -> PNG stills in ./chk
Needs: pip install numpy pillow ; ffmpeg
"""
import math, os, random, re, subprocess, sys, wave
from collections import deque
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SR = 1080, 1920, 30, 44100
GW, GH, C = 25, 30, 40                       # grid and cell size
X0, Y0 = 40, 340                             # arena top-left
SPS = 18                                     # snake steps per second
ZONE_T0, ZONE_DT, ZONE_MIN = 7.0, 0.75, 3    # zone starts closing, seconds per ring, rings left open at least
N_APPLES = 8
NAMES = ["GREEN", "BLUE", "YELLOW", "PINK"]
SCOL = [(63, 200, 90), (70, 150, 255), (255, 205, 50), (240, 100, 200)]
BG, PANEL, PANEL2, BORDER = (13, 17, 23), (22, 27, 34), (30, 36, 44), (48, 54, 61)
TEXT, MUTED, GREEN, YEL, BLUE, RED = (230, 237, 243), (139, 148, 158), (63, 185, 80), (255, 212, 59), (88, 166, 255), (248, 81, 73)
DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


# ------------------------------------------------------------------ the game (pure logic)
def zone_at(step):
    t = step / SPS
    return 0 if t < ZONE_T0 else min(int((t - ZONE_T0) / ZONE_DT) + 1, min(GW, GH) // 2 - ZONE_MIN)

def inside(x, y, z): return z <= x < GW - z and z <= y < GH - z

def dist(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])

def space(start, blocked, z, cap):
    """flood fill: how many free cells can the snake reach from here (stops at cap)"""
    seen, q = {start}, deque([start])
    while q and len(seen) < cap:
        x, y = q.popleft()
        for dx, dy in DIRS:
            n = (x + dx, y + dy)
            if n not in seen and n not in blocked and inside(*n, z): seen.add(n); q.append(n)
    return len(seen)

def choose(s, snakes, apples, z, rnd):
    """the whole AI: go to the nearest apple, but never into a wall, a body or a dead end"""
    hx, hy = s["body"][0]
    blocked = {p for o in snakes if o["alive"] for p in o["body"]}
    blocked.discard(s["body"][-1])
    apple = min(apples, key=lambda a: dist(a, (hx, hy))) if apples else (GW // 2, GH // 2)
    best, best_score = s["dir"], -1e9
    for d in DIRS:
        if d == (-s["dir"][0], -s["dir"][1]): continue
        nx, ny = hx + d[0], hy + d[1]
        if (nx, ny) in blocked or not inside(nx, ny, z): continue
        room = space((nx, ny), blocked, z, len(s["body"]) * 2 + 4)
        score = -dist((nx, ny), apple) + (0 if room > len(s["body"]) else -60) + rnd.uniform(0, s["chaos"])
        if score > best_score: best, best_score = d, score
    return best

def simulate(seed, max_t=24.0):
    rnd = random.Random(seed)
    starts = [((7, 6), (1, 0)), ((17, 6), (-1, 0)), ((7, 23), (1, 0)), ((17, 23), (-1, 0))]
    snakes = []
    for i, ((x, y), d) in enumerate(starts):
        body = [(x - d[0] * k, y) for k in range(6)]
        snakes.append(dict(body=body, dir=d, alive=True, grow=0, chaos=rnd.uniform(1.0, 3.5), died=None))
    def free_cell():
        occ = {p for s in snakes if s["alive"] for p in s["body"]} | set(apples)
        for _ in range(200):
            c = (rnd.randrange(1, GW - 1), rnd.randrange(1, GH - 1))
            z = zone_at(step)
            if c not in occ and inside(*c, z + 1): return c
        return None
    apples, step = [], 0
    while len(apples) < N_APPLES: apples.append(free_cell())
    hist, events = [], []
    hist.append(([list(s["body"]) for s in snakes], [s["alive"] for s in snakes], list(apples), 0))
    winner = None
    while step < max_t * SPS:
        step += 1; z = zone_at(step)
        for s in snakes:
            if s["alive"]: s["dir"] = choose(s, snakes, apples, z, rnd)
        heads = {}
        for i, s in enumerate(snakes):
            if not s["alive"]: continue
            hx, hy = s["body"][0]; nh = (hx + s["dir"][0], hy + s["dir"][1])
            s["body"].insert(0, nh)
            if nh in apples:
                apples.remove(nh); s["grow"] += 1; events.append(("eat", step, i))
            if s["grow"] > 0: s["grow"] -= 1
            else: s["body"].pop()
            heads[i] = nh
        dead = set()
        for i, nh in heads.items():
            if not inside(*nh, z): dead.add(i); continue
            for j, o in enumerate(snakes):
                if not o["alive"]: continue
                body = o["body"][1:] if j == i else o["body"][1:]
                if nh in body: dead.add(i)
                if j != i and j in heads and heads[j] == nh: dead.add(i)
        # the zone cuts off the part of a body that is outside (only the head being outside kills)
        for i, s in enumerate(snakes):
            if s["alive"] and i not in dead:
                for j, p in enumerate(s["body"]):
                    if j > 0 and not inside(*p, z): s["body"] = s["body"][:j]; break
        for i in dead:
            s = snakes[i]; s["alive"] = False; s["died"] = step; events.append(("die", step, i))
            for p in s["body"][::2]:                                   # a dead snake turns into food
                if inside(*p, z) and p not in apples and len(apples) < 40: apples.append(p)
        while len(apples) < N_APPLES:
            c = free_cell()
            if c is None: break
            apples.append(c)
        apples = [a for a in apples if inside(*a, z)]
        hist.append(([list(s["body"]) for s in snakes], [s["alive"] for s in snakes], list(apples), z))
        alive = [i for i, s in enumerate(snakes) if s["alive"]]
        if len(alive) <= 1:
            winner = alive[0] if alive else None
            break
    return hist, events, winner, step, snakes

def pick_seed():
    for seed in range(1, 600):
        hist, ev, w, st, sn = simulate(seed)
        if w is None: continue
        tw = st / SPS
        if not 13.0 <= tw <= 15.5: continue
        dies = sorted(e[1] / SPS for e in ev if e[0] == "die")
        if len(dies) != 3 or dies[0] < 3.5 or min(b - a for a, b in zip(dies, dies[1:])) < 1.2: continue
        k = int((tw - 4) * SPS); bodies, alive = hist[k][0], hist[k][1]
        lens = [len(b) if a else 0 for b, a in zip(bodies, alive)]
        if lens[w] < max(lens): return seed                      # the leader 4 s before the end did NOT win
    raise RuntimeError("no seed")


# ------------------------------------------------------------------ build the match
SEED = pick_seed()
HIST, EV, WIN, WSTEP, _ = simulate(SEED)
T_WIN = WSTEP / SPS
T_END = T_WIN + 1.4
TOTAL = T_END + 4.2
T_LOOP = TOTAL - 0.6
NF = int(TOTAL * FPS)
DIES = {e[2]: e[1] / SPS for e in EV if e[0] == "die"}
EATS = [(e[1] / SPS, e[2]) for e in EV if e[0] == "eat"]
print(f"seed {SEED}: {NAMES[WIN]} wins at {T_WIN:.2f} s, deaths {[(NAMES[k], round(v, 1)) for k, v in sorted(DIES.items(), key=lambda kv: kv[1])]}, total {TOTAL:.1f} s", flush=True)


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
AW, AH = GW * C, GH * C
def make_board():
    im = Image.new("RGB", (AW * S2, AH * S2), BG); d = ImageDraw.Draw(im)
    for y in range(GH):
        for x in range(GW):
            c = (20, 26, 34) if (x + y) % 2 == 0 else (24, 31, 40)
            d.rectangle([x * C * S2, y * C * S2, (x + 1) * C * S2 - 1, (y + 1) * C * S2 - 1], fill=c)
    return im
BOARD = make_board()

def cell_xy(p): return ((p[0] + 0.5) * C * S2, (p[1] + 0.5) * C * S2)

def body_at(i, t):
    """smooth snake: interpolate every segment between two steps"""
    k = min(int(t * SPS), len(HIST) - 1); fr = t * SPS - k if k + 1 < len(HIST) else 0.0
    if not HIST[k][1][i]: fr = 0.0
    a = HIST[k][0][i]; b = HIST[min(k + 1, len(HIST) - 1)][0][i] if HIST[k][1][i] else a
    pts = []
    for j in range(len(b)):
        pa = a[j] if j < len(a) else a[-1]; pb = b[j]
        if abs(pa[0] - pb[0]) + abs(pa[1] - pb[1]) > 1: pa = pb
        pts.append((pa[0] + (pb[0] - pa[0]) * fr, pa[1] + (pb[1] - pa[1]) * fr))
    return pts

def draw_arena(t):
    im = BOARD.copy(); d = ImageDraw.Draw(im)
    k = min(int(t * SPS), len(HIST) - 1)
    bodies, alive, apples, z = HIST[k]
    # zone
    if z > 0:
        zc = (70, 18, 24); x0, y0, x1, y1 = z * C * S2, z * C * S2, (GW - z) * C * S2, (GH - z) * C * S2
        for rect in ([0, 0, AW * S2, y0], [0, y1, AW * S2, AH * S2], [0, y0, x0, y1], [x1, y0, AW * S2, y1]):
            d.rectangle(rect, fill=zc)
        for s in range(-AH * S2, AW * S2, 36 * S2):                                        # warning stripes
            d.line([(s, 0), (s + AH * S2, AH * S2)], fill=(90, 24, 30), width=8 * S2)
        d.rectangle([x0, y0, x1, y1], fill=None, outline=mix(RED, (255, 255, 255), 0.3 + 0.3 * math.sin(t * 12)), width=4 * S2)
        # re-draw the free area on top of the stripes
        im.paste(BOARD.crop((x0 + 4 * S2, y0 + 4 * S2, x1 - 4 * S2, y1 - 4 * S2)), (x0 + 4 * S2, y0 + 4 * S2))
        d = ImageDraw.Draw(im)
    # apples
    for a in apples:
        cx, cy = cell_xy(a); r = C * S2 * 0.36 * (1 + 0.06 * math.sin(t * 8 + a[0]))
        d.ellipse([cx - r, cy - r * 0.92, cx + r, cy + r], fill=(235, 60, 60))
        d.ellipse([cx - r * 0.5, cy - r * 0.6, cx - r * 0.1, cy - r * 0.2], fill=(255, 150, 140))
        d.line([(cx, cy - r * 0.9), (cx + r * 0.2, cy - r * 1.3)], fill=(110, 70, 40), width=3 * S2)
        d.ellipse([cx + r * 0.15, cy - r * 1.35, cx + r * 0.7, cy - r * 1.0], fill=(80, 190, 90))
    # snakes
    for i in range(4):
        died = DIES.get(i)
        if died is not None and t > died + 0.6: continue
        pts = body_at(i, t if died is None or t < died else died)
        col = SCOL[i]; n = len(pts)
        flash = died is not None and t >= died
        for j in range(n - 1, -1, -1):
            cx, cy = cell_xy(pts[j]); g = 1 - 0.35 * j / max(1, n - 1)
            c = mix(col, (0, 0, 0), 1 - g)
            if flash:
                ft = (t - died) / 0.6; c = mix((255, 255, 255), (60, 60, 70), ft)
            r = C * S2 * (0.46 if j == 0 else 0.42)
            d.rounded_rectangle([cx - r, cy - r, cx + r, cy + r], radius=r * 0.55, fill=c)
        if not flash and n:
            hx, hy = cell_xy(pts[0]); hb = HIST[min(int(t * SPS), len(HIST) - 1)][0][i]
            dx, dy = (hb[0][0] - hb[1][0], hb[0][1] - hb[1][1]) if len(hb) > 1 else (1, 0)
            if abs(dx) + abs(dy) != 1: dx, dy = 1, 0
            px, py = -dy, dx
            for s in (-1, 1):
                ex = hx + dx * 5 * S2 + px * s * 6 * S2; ey = hy + dy * 5 * S2 + py * s * 6 * S2
                d.ellipse([ex - 4.5 * S2, ey - 4.5 * S2, ex + 4.5 * S2, ey + 4.5 * S2], fill=(255, 255, 255))
                d.ellipse([ex + dx * 1.5 * S2 - 2.2 * S2, ey + dy * 1.5 * S2 - 2.2 * S2, ex + dx * 1.5 * S2 + 2.2 * S2, ey + dy * 1.5 * S2 + 2.2 * S2], fill=(10, 10, 14))
    return im.reduce(S2)

def burst(d, t, t0, cx, cy, seed, cols, n=34):
    dt = t - t0
    if not 0 <= dt < 0.9: return
    rnd = random.Random(seed)
    for _ in range(n):
        a = rnd.uniform(0, 2 * math.pi); v = rnd.uniform(300, 850); s = rnd.uniform(5, 12)
        x = cx + math.cos(a) * v * dt; y = cy + math.sin(a) * v * dt + 800 * dt * dt; k = 1 - dt / 0.9
        d.ellipse([x - s * k, y - s * k, x + s * k, y + s * k], fill=rnd.choice(cols))

def text2(d, y, parts, fnt, cx=W / 2):
    tot = sum(d.textlength(s, font=fnt) for s, _ in parts); x = cx - tot / 2
    for s, c in parts:
        d.text((x, y), s, font=fnt, fill=c, anchor="lm", stroke_width=5, stroke_fill=(8, 10, 16)); x += d.textlength(s, font=fnt)

def head_px(i, t):
    pts = body_at(i, t)
    return X0 + (pts[0][0] + 0.5) * C, Y0 + (pts[0][1] + 0.5) * C

CODE = ["for d in moves:",
        "    nx, ny = hx + d[0], hy + d[1]",
        "    if (nx, ny) in blocked: continue",
        "    score = -dist((nx, ny), apple)",
        "    score += 0 if space(nx, ny) > len(body) else -60"]

def render(fi):
    t = fi / FPS
    img = Image.new("RGB", (W, H), BG)
    sh = (0, 0)
    for i, td in DIES.items():
        if 0 <= t - td < 0.25: a = 9 * (1 - (t - td) / 0.25); sh = (a * math.sin(t * 90), a * math.cos(t * 70))
    img.paste(draw_arena(t), (int(X0 + sh[0]), int(Y0 + sh[1])))
    d = ImageDraw.Draw(img)
    d.rectangle([X0 - 3, Y0 - 3, X0 + AW + 2, Y0 + AH + 2], outline=BORDER, width=3)
    text2(d, 196, [("PICK A ", TEXT), ("SNAKE", (110, 225, 130))], BLACK(92))
    # chips with live length
    k = min(int(t * SPS), len(HIST) - 1)
    for i in range(4):
        cx = 150 + i * 260; cy = 286
        alive = HIST[k][1][i]; ln = len(HIST[k][0][i])
        win = t >= T_WIN and i == WIN; kk = ease_back((t - T_WIN) / 0.4) if win else 0
        wd, hh = 240 * (1 + 0.12 * kk), 64 * (1 + 0.12 * kk)
        col = SCOL[i] if alive or win else mix(SCOL[i], BG, 0.65)
        d.rounded_rectangle([cx - wd / 2, cy - hh / 2, cx + wd / 2, cy + hh / 2], radius=hh / 2, fill=mix(BG, col, 0.22), outline=YEL if win else col, width=5 if win else 3)
        d.ellipse([cx - wd / 2 + 14, cy - 14, cx - wd / 2 + 42, cy + 14], fill=col)
        d.text((cx - wd / 2 + 54, cy), NAMES[i], font=BLACK(26), fill=TEXT if alive or win else MUTED, anchor="lm")
        d.text((cx + wd / 2 - 18, cy), str(ln), font=mono(28), fill=(255, 255, 255) if alive or win else MUTED, anchor="rm")
        if not alive and not win:
            d.line([(cx - wd / 2 + 10, cy), (cx + wd / 2 - 10, cy)], fill=RED, width=4)
    # zone warning
    if ZONE_T0 - 0.2 <= t < ZONE_T0 + 1.6:
        kz = ease_back((t - ZONE_T0 + 0.2) / 0.3) * (1 if t < ZONE_T0 + 1.3 else max(0, (ZONE_T0 + 1.6 - t) / 0.3))
        if kz > 0.02:
            wd = 420 * kz
            d.rounded_rectangle([W / 2 - wd / 2, Y0 + 30, W / 2 + wd / 2, Y0 + 30 + 66 * kz], radius=33 * kz, fill=(60, 16, 20), outline=RED, width=4)
            if kz > 0.5: d.text((W / 2, Y0 + 30 + 33 * kz), "ZONE CLOSING!", font=BLACK(34 * kz), fill=(255, 150, 146), anchor="mm")
    # deaths: name pops at the head
    for i, td in DIES.items():
        if 0 <= t - td < 1.0:
            hx, hy = head_px(i, td); hx = min(max(hx, 110), 970); a = 1 - (t - td)
            d.text((hx, hy - 40 - 60 * (1 - a)), "OUT!", font=BLACK(44), fill=mix(BG, SCOL[i], a), anchor="mm", stroke_width=4, stroke_fill=(8, 10, 16))
            burst(d, t, td, hx, hy, i + 3, [SCOL[i], (255, 255, 255), (120, 120, 130)], 22)
    if t < T_WIN:
        d.text((W / 2, 1585), "Last snake alive wins", font=SEMI(34), fill=MUTED, anchor="mm")
    if T_WIN <= t < T_END + 0.4:
        kb = ease_back((t - T_WIN) / 0.35)
        if kb > 0.02:
            cy = Y0 + AH / 2
            d.rounded_rectangle([W / 2 - 360 * kb, cy - 100 * kb, W / 2 + 360 * kb, cy + 100 * kb], radius=30 * kb, fill=(16, 20, 26), outline=SCOL[WIN], width=6)
            if kb > 0.5:
                d.text((W / 2, cy - 32), NAMES[WIN], font=BLACK(92 * kb), fill=SCOL[WIN], anchor="mm", stroke_width=5, stroke_fill=(8, 10, 16))
                d.text((W / 2, cy + 52), "WINS!", font=BLACK(58 * kb), fill=YEL, anchor="mm", stroke_width=4, stroke_fill=(8, 10, 16))
        hx, hy = head_px(WIN, T_WIN)
        burst(d, t, T_WIN, hx, hy, 9, [YEL, SCOL[WIN], (255, 255, 255), BLUE], 44)
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
    d.text((490, y0 + 205), NAMES[WIN] + "?", font=BLACK(110 * k1), fill=SCOL[WIN], anchor="mm", stroke_width=5, stroke_fill=(8, 10, 16))
    k3 = ease_out((et - 0.7) / 0.3)
    if k3 > 0:
        d.text((490, y0 + 360), "Comment your snake", font=BLACK(60), fill=mix((16, 20, 26), YEL, k3), anchor="mm")
        ay = y0 + 425 + 10 * math.sin(t * 8)
        d.polygon([(470, ay), (510, ay), (490, ay + 26)], fill=mix((16, 20, 26), YEL, k3))
    k4 = ease_out((et - 1.1) / 0.3)
    if k4 > 0:
        bx0, by0 = 70, y0 + 500
        d.rounded_rectangle([bx0, by0, 910, by0 + 330], radius=18, fill=(10, 13, 18), outline=BORDER, width=2)
        d.text((bx0 + 24, by0 + 28), "the snake AI, in Python:", font=SEMI(26), fill=MUTED, anchor="lm")
        for li, ln in enumerate(CODE):
            x = bx0 + 24; y = by0 + 80 + li * 48
            for m in re.finditer(r'(\bfor\b|\bin\b|\bif\b|\belse\b|\bcontinue\b)|(\w+(?=\())|(\d+)|(.)', ln):
                s = m.group(0); col = (255, 123, 114) if m.group(1) else (210, 168, 255) if m.group(2) else (121, 192, 255) if m.group(3) else TEXT
                d.text((x, y), s, font=mono(23), fill=mix((10, 13, 18), col, k4), anchor="lm"); x += d.textlength(s, font=mono(23))
    k5 = ease_out((et - 1.6) / 0.3)
    if k5 > 0:
        d.text((490, y0 + 940), "Full code: link in profile", font=BOLD(40), fill=mix((16, 20, 26), TEXT, k5), anchor="mm")
        text2(d, y0 + 1060, [("PYTHONIC ", mix((16, 20, 26), BLUE, k5)), ("LIFE", mix((16, 20, 26), YEL, k5))], BLACK(68), cx=490)


# ------------------------------------------------------------------ audio
def synth_audio():
    NS = int((TOTAL + 1) * SR); mus = np.zeros(NS, np.float32); sfx = np.zeros(NS, np.float32)
    def put(buf, t0, sig):
        s = int(t0 * SR); n = min(len(sig), NS - s)
        if n > 0 and s >= 0: buf[s:s + n] += sig[:n].astype(np.float32)
    def tt(dur): return np.arange(int(dur * SR)) / SR
    def noise(n, seed): return np.random.default_rng(seed).standard_normal(n)
    def lp(x, a): return np.convolve(x, np.ones(a) / a, mode="same")
    def blip(t0, f, amp):
        x = tt(0.14); ph = 2 * np.pi * np.cumsum(f * (1 + 0.5 * np.exp(-x * 60))) / SR
        put(sfx, t0, (np.sin(ph) + 0.3 * np.sign(np.sin(ph))) * np.exp(-x * 28) * np.minimum(1, x * 900) * amp)
    def kick(t0, amp):
        x = tt(0.3); put(mus, t0, np.sin(2 * np.pi * np.cumsum(140 * np.exp(-x * 30) + 46) / SR) * np.exp(-x * 9) * amp)
    def snare(t0, amp):
        x = tt(0.2); put(mus, t0, (0.7 * noise(len(x), int(t0 * 1000)) * np.exp(-x * 22) + 0.5 * np.sin(2 * np.pi * 200 * x) * np.exp(-x * 30)) * amp)
    def hat(t0, amp):
        x = tt(0.04); put(mus, t0, np.diff(noise(len(x) + 1, int(t0 * 991))) * np.exp(-x * 90) * amp)
    def bass(t0, f, dur, amp):
        x = tt(dur); ph = 2 * np.pi * f * x
        put(mus, t0, (np.sin(ph) + 0.4 * np.sign(np.sin(ph)) * 0.5) * np.minimum(1, x * 300) * np.exp(-x * 4) * amp)
    def crash(t0, amp):
        x = tt(0.7)
        put(sfx, t0, lp(noise(len(x), int(t0 * 77)), 3) * 1.6 * np.exp(-x * 9) * 0.35 * amp)
        put(sfx, t0, np.sin(2 * np.pi * np.cumsum(220 * np.exp(-x * 4) + 50) / SR) * np.exp(-x * 5) * 0.45 * amp)
    def impact(t0, amp):
        x = tt(1.0)
        put(sfx, t0, np.sin(2 * np.pi * np.cumsum(90 * np.exp(-x * 12) + 40) / SR) * np.exp(-x * 4) * 0.6 * amp)
        put(sfx, t0, lp(noise(len(x), 5), 5) * 2.2 * np.exp(-x * 10) * 0.3 * amp)
    def chord(t0, notes, amp, dec=2.5):
        x = tt(1.6)
        put(sfx, t0, sum(np.sin(2 * np.pi * f * x) + 0.3 * np.sin(4 * np.pi * f * x) for f in notes) * np.exp(-x * dec) * np.minimum(1, x * 500) * amp)
    def whoosh(t0, dur, amp, seed):
        x = tt(dur); n = noise(len(x), seed); k = x / dur
        put(sfx, t0, (lp(n, 16) * 3.5 * (1 - k) + (n - lp(n, 5)) * k) * np.sin(np.pi * k) ** 1.5 * amp)

    # music: 128 bpm, E minor drive; gets busier when the zone starts
    B = 60 / 128; t0 = 0.0; roots = [82.41, 65.41, 73.42, 61.74]; bar = 0
    while t0 < TOTAL + 0.5:
        hot = t0 >= ZONE_T0
        for b_ in range(4):
            kick(t0 + b_ * B, 0.42 if hot else 0.34)
            if b_ in (1, 3): snare(t0 + b_ * B, 0.20 if hot else 0.14)
            for h in range(4 if hot else 2): hat(t0 + b_ * B + h * B / (4 if hot else 2), 0.05 if h % 2 else 0.03)
        for e in range(8): bass(t0 + e * B / 2, roots[bar % 4], B / 2, 0.14)
        t0 += 4 * B; bar += 1
    # apple pops: each snake its own scale
    scales = [[523.3, 659.3, 784.0, 1046.5], [587.3, 740.0, 880.0, 1174.7], [659.3, 830.6, 987.8, 1318.5], [440.0, 554.4, 659.3, 880.0]]
    cnt = [0, 0, 0, 0]; last = -1
    for te, i in EATS:
        if te > T_WIN or te - last < 0.04: continue
        blip(te, scales[i][cnt[i] % 4], 0.10); cnt[i] += 1; last = te
    # zone shrinks: low pulses
    k = 0
    while ZONE_T0 + k * ZONE_DT < T_WIN:
        tz = ZONE_T0 + k * ZONE_DT; x = tt(0.25)
        put(sfx, tz, np.sin(2 * np.pi * 55 * x) * np.exp(-x * 10) * 0.25); k += 1
    whoosh(ZONE_T0 - 0.3, 0.5, 0.12, 4)
    for i, td in DIES.items(): crash(td, 1.0)
    whoosh(T_WIN - 1.2, 1.2, 0.12, 8)
    impact(T_WIN, 1.3); chord(T_WIN + 0.05, [329.6, 415.3, 493.9, 659.3], 0.06)
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
                             "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "snake_noaudio.mp4"], stdin=subprocess.PIPE)
    F0 = render(0).astype(np.float32)
    for f in range(NF):
        t = f / FPS; fr = render(f)
        if t > T_LOOP:
            a = min(1.0, (t - T_LOOP) / (TOTAL - T_LOOP - 1 / FPS)); fr = (fr * (1 - a) + F0 * a).astype(np.uint8)
        proc.stdin.write(np.ascontiguousarray(fr).tobytes())
    proc.stdin.close(); proc.wait()
    a = synth_audio()
    with wave.open("snake_audio.wav", "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR); wf.writeframes((a * 32767).astype(np.int16).tobytes())
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "snake_noaudio.mp4", "-i", "snake_audio.wav", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", "snake_battle.mp4"], check=True)
    print("done", round(TOTAL, 1))
