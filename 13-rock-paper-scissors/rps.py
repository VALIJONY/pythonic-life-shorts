"""Pythonic Life — Rock Paper Scissors vs Python
One file, two uses:
  python3 rps.py              play the game in a tkinter window (click a card or press 1 / 2 / 3)
  python3 rps.py video        render the YouTube Short (1080x1920, 20 s, loops) -> rps_vs_python.mp4
  python3 rps.py still 0 3.2  PNG stills of the Short in ./chk
The Short is drawn by the same renderer as the game, and Python's moves come from random.choice.
Game needs: pip install pillow   (tkinter ships with Python; on Debian/Ubuntu: sudo apt install python3-tk)
Video also needs: numpy and ffmpeg
"""
import math, os, random, subprocess, sys, time
from functools import lru_cache
from PIL import Image, ImageChops, ImageDraw, ImageFont

W, H = 1080, 1920
KINDS = ["rock", "paper", "scissors"]
BEATS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
BG, PANEL, PANEL2, BORDER = (13, 17, 23), (22, 27, 34), (30, 36, 44), (48, 54, 61)
TEXT, MUTED = (230, 237, 243), (139, 148, 158)
GREEN, RED, YEL, BLUE = (63, 185, 80), (248, 81, 73), (255, 212, 59), (88, 166, 255)
KCOL = {"rock": (176, 186, 204), "paper": (88, 166, 255), "scissors": (255, 128, 92)}
RESCOL = {"WIN": GREEN, "LOSE": RED, "DRAW": MUTED}
CX = [180, 480, 780]                      # your cards (inside the Shorts safe zone)
PY_C, YOU_C = (540, 640), 1240            # python card centre, your cards centre y
WORD_Y = 958


def outcome(you, py):
    return "DRAW" if you == py else "WIN" if BEATS[you] == py else "LOSE"


# ------------------------------------------------------------------ fonts (Inter if installed, else system fallbacks)
_DIRS = ["/usr/share/fonts/opentype/inter/", "/usr/share/fonts/truetype/inter/", os.path.expanduser("~/.local/share/fonts/"),
         os.path.expanduser("~/.fonts/"), "C:/Windows/Fonts/", "/Library/Fonts/", "/System/Library/Fonts/Supplemental/"]
_NAMES = {"black": ["Inter-Black.otf", "Inter-Black.ttf", "ariblk.ttf", "Arial Black.ttf"],
          "bold": ["Inter-Bold.otf", "Inter-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"],
          "semi": ["Inter-SemiBold.otf", "Inter-SemiBold.ttf", "arialbd.ttf", "Arial Bold.ttf"]}

@lru_cache(None)
def font(weight, size):
    size = max(1, int(size))
    for dr in _DIRS:
        for n in _NAMES[weight]:
            if os.path.exists(dr + n): return ImageFont.truetype(dr + n, size)
    for n in ("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"):
        try: return ImageFont.truetype(n, size)
        except OSError: pass
    return ImageFont.load_default(size)

@lru_cache(None)
def mono(size):
    for n in ("DejaVuSansMono-Bold.ttf", "consolab.ttf", "Menlo.ttc"):
        try: return ImageFont.truetype(n, max(1, int(size)))
        except OSError: pass
    return font("bold", size)

def ease_out(x): x = max(0.0, min(1.0, x)); return 1 - (1 - x) ** 3
def ease_back(x): x = max(0.0, min(1.0, x)); return 1 + 2.70158 * (x - 1) ** 3 + 1.70158 * (x - 1) ** 2
def mix(a, b, k): return tuple(int(a[i] + (b[i] - a[i]) * k) for i in range(3))


# ------------------------------------------------------------------ icons (drawn at 4x, reduced = smooth edges)
def _blade(d, tip, base, w0, w1, k, fill, edge):
    (x0, y0), (x1, y1) = tip, base
    L = math.hypot(x1 - x0, y1 - y0); nx, ny = -(y1 - y0) / L, (x1 - x0) / L
    pts = [(x0 + nx * w0, y0 + ny * w0), (x1 + nx * w1, y1 + ny * w1), (x1 - nx * w1, y1 - ny * w1), (x0 - nx * w0, y0 - ny * w0)]
    d.polygon([(x * k, y * k) for x, y in pts], fill=fill, outline=edge, width=int(1.6 * k))

@lru_cache(None)
def icon(kind, size):
    S = 4; k = size * S / 100
    im = Image.new("RGBA", (size * S, size * S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    P = lambda pts: [(x * k, y * k) for x, y in pts]
    if kind == "rock":
        d.ellipse(P([(14, 80), (86, 92)]), fill=(0, 0, 0, 70))                                        # shadow
        body = [(16, 62), (22, 38), (40, 22), (64, 20), (84, 34), (90, 58), (80, 78), (52, 86), (26, 80)]
        d.polygon(P(body), fill=(118, 126, 144), outline=(66, 72, 88), width=int(2.4 * k))
        d.polygon(P([(22, 38), (40, 22), (64, 20), (58, 44), (34, 50)]), fill=(160, 168, 186))
        d.polygon(P([(64, 20), (84, 34), (90, 58), (66, 56), (58, 44)]), fill=(138, 146, 164))
        d.polygon(P([(26, 80), (52, 86), (80, 78), (70, 62), (44, 66)]), fill=(92, 98, 116))
        d.polygon(P(body), outline=(66, 72, 88), width=int(2.4 * k))
        d.ellipse(P([(38, 28), (52, 35)]), fill=(214, 220, 232))
    elif kind == "paper":
        sh = Image.new("RGBA", im.size, (0, 0, 0, 0)); e = ImageDraw.Draw(sh)
        e.polygon(P([(22, 12), (62, 12), (78, 28), (78, 88), (22, 88)]), fill=(246, 248, 252), outline=(168, 178, 196), width=int(2 * k))
        e.polygon(P([(62, 12), (78, 28), (62, 28)]), fill=(196, 206, 222), outline=(168, 178, 196), width=int(1.5 * k))
        for j, y in enumerate((38, 48, 58, 68, 78)):
            e.line(P([(30, y), (70 - (14 if j == 4 else 0), y)]), fill=(132, 172, 236), width=int(2.6 * k))
        e.line(P([(30, 22), (52, 22)]), fill=(255, 128, 92), width=int(3 * k))
        im = sh.rotate(-8, resample=Image.BICUBIC); d = ImageDraw.Draw(im)
    else:
        steel, edge, ring = (214, 220, 232), (120, 128, 146), (255, 128, 92)
        _blade(d, (20, 12), (62, 70), 1.2, 6.5, k, steel, edge)
        _blade(d, (80, 12), (38, 70), 1.2, 6.5, k, (232, 236, 244), edge)
        for cx in (30, 70):
            d.ellipse(P([(cx - 14, 66), (cx + 14, 94)]), outline=ring, width=int(7 * k))
        d.ellipse(P([(45, 43), (55, 53)]), fill=(84, 90, 106))
        d.ellipse(P([(48, 46), (52, 50)]), fill=(200, 206, 220))
    return im.resize((size, size), Image.LANCZOS)


# ------------------------------------------------------------------ cards
@lru_cache(None)
def card_img(face, w, h, border=None):
    S = 2; im = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    r = 30 * S
    if face == "back":
        d.rounded_rectangle([0, 0, w * S - 1, h * S - 1], radius=r, fill=(30, 66, 110))
        for k in range(-h, w + h, 34):                                               # diagonal stripes
            d.line([(k * S, 0), ((k + h) * S, h * S)], fill=(36, 76, 124), width=12 * S)
        mask = Image.new("L", im.size, 0); ImageDraw.Draw(mask).rounded_rectangle([0, 0, w * S - 1, h * S - 1], radius=r, fill=255)
        im.putalpha(mask); d = ImageDraw.Draw(im)
        d.rounded_rectangle([6 * S, 6 * S, (w - 6) * S, (h - 6) * S], radius=r - 6 * S, outline=border or YEL, width=6 * S)
        d.text((w * S / 2, h * S * 0.44), "?", font=font("black", 170 * S), fill=YEL, anchor="mm")
        d.text((w * S / 2, h * S * 0.84), "PYTHON", font=font("black", 34 * S), fill=(255, 255, 255), anchor="mm")
    else:
        bc = border or KCOL[face]
        d.rounded_rectangle([0, 0, w * S - 1, h * S - 1], radius=r, fill=PANEL, outline=bc, width=6 * S)
        d.rounded_rectangle([6 * S, 6 * S, (w - 6) * S, int(h * 0.74) * S], radius=r - 6 * S, fill=PANEL2)
        isz = int(min(w, h) * 0.66)
        ic = icon(face, isz * S)
        im.alpha_composite(ic, (int((w * S - isz * S) / 2), int((h * 0.40 - isz / 2) * S)))
        d.text((w * S / 2, h * S * 0.87), face.upper(), font=font("black", int(w * 0.13) * S), fill=bc, anchor="mm")
    return im.resize((w, h), Image.LANCZOS)

def paste_card(img, card, cx, cy, sx=1.0, sy=1.0, rot=0.0, alpha=1.0):
    if sx < 0.02 or sy < 0.02 or alpha < 0.02: return
    c = card
    if abs(sx - 1) > 1e-3 or abs(sy - 1) > 1e-3: c = c.resize((max(1, int(card.width * sx)), max(1, int(card.height * sy))), Image.BILINEAR)
    if abs(rot) > 0.05: c = c.rotate(rot, resample=Image.BICUBIC, expand=True)
    if alpha < 0.999:
        a = c.getchannel("A").point(lambda v: int(v * alpha)); c = c.copy(); c.putalpha(a)
    img.alpha_composite(c, (int(cx - c.width / 2), int(cy - c.height / 2)))


# ------------------------------------------------------------------ background with drifting dots
GAP = 54
@lru_cache(None)
def _bg_tile():
    im = Image.new("RGBA", (W + GAP, H + GAP), BG + (255,)); d = ImageDraw.Draw(im)
    for y in range(0, H + GAP, GAP):
        for x in range(0, W + GAP, GAP): d.ellipse([x - 2.5, y - 2.5, x + 2.5, y + 2.5], fill=(28, 35, 46))
    return im

def bg_frame(t):
    o = int(t * 18) % GAP
    return _bg_tile().crop((o, o, o + W, o + H))


# ------------------------------------------------------------------ the renderer shared by the game and the Short
def burst(d, t, t0, cx, cy, seed):
    dt = t - t0
    if not 0 <= dt < 0.9: return
    rnd = random.Random(seed)
    for _ in range(34):
        a = rnd.uniform(0, 2 * math.pi); v = rnd.uniform(420, 900); s = rnd.uniform(6, 13)
        col = rnd.choice([YEL, BLUE, (255, 128, 92), GREEN, (255, 255, 255)])
        x = cx + math.cos(a) * v * dt; y = cy + math.sin(a) * v * dt + 900 * dt * dt
        k = 1 - dt / 0.9
        if rnd.random() < 0.5: d.ellipse([x - s * k, y - s * k, x + s * k, y + s * k], fill=col)
        else: d.rectangle([x - s * k, y - s * 0.5 * k, x + s * k, y + s * 0.5 * k], fill=col)

def text2(d, y, parts, fnt):
    """centered text built from (string, colour) parts"""
    tot = sum(d.textlength(s, font=fnt) for s, _ in parts); x = W / 2 - tot / 2
    for s, c in parts:
        d.text((x, y), s, font=fnt, fill=c, anchor="lm", stroke_width=4, stroke_fill=(8, 10, 16)); x += d.textlength(s, font=fnt)

def draw(S, t):
    """S = scene state (a dict). Returns an RGB PIL image 1080x1920."""
    img = bg_frame(t); d = ImageDraw.Draw(img)
    text2(d, 222, [("BEAT ", TEXT), ("PYTHON", YEL)], font("black", 104))
    d.text((W / 2, 300), "at Rock  ·  Paper  ·  Scissors", font=font("semi", 38), fill=MUTED, anchor="mm")
    pw = d.textlength(S["pill"], font=font("black", 30)) + 64
    d.rounded_rectangle([W / 2 - pw / 2, 352, W / 2 + pw / 2, 404], radius=26, fill=PANEL2, outline=BORDER, width=3)
    d.text((W / 2, 378), S["pill"], font=font("black", 30), fill=TEXT, anchor="mm")
    # python's card
    paste_card(img, card_img(S["py_face"], 300, 380, S.get("py_border")), PY_C[0], PY_C[1] + S["pump"], S.get("py_sx", 1.0), 1.0, S.get("py_rot", 0.0))
    # the word in the middle
    if S.get("word"):
        k = ease_back(S.get("word_k", 1.0)); size = S.get("word_size", 96) * (0.45 + 0.55 * k)
        d.text((W / 2, WORD_Y), S["word"], font=font("black", size), fill=S.get("word_col", TEXT), anchor="mm", stroke_width=6, stroke_fill=(8, 10, 16))
    # your cards
    d.text((W / 2, 1020), S.get("you_label", "YOU"), font=font("black", 28), fill=MUTED, anchor="mm")
    for i, kd in enumerate(KINDS):
        sc = S["scale"][i]
        paste_card(img, card_img(kd, 270, 330, S["border"][i]), CX[i], YOU_C + S["pump"] + S["dy"][i], sc, sc, S["rot"][i], S["alpha"][i])
        ch = S["chips"][i]
        if ch:
            res, kk = ch; kk = ease_back(kk)
            if kk > 0.02:
                col = RESCOL[res]; cw = (170 if res != "DRAW" else 180) * kk; chh = 56 * kk
                d.rounded_rectangle([CX[i] - cw / 2, 1452 - chh / 2, CX[i] + cw / 2, 1452 + chh / 2], radius=chh / 2, fill=mix(BG, col, 0.28), outline=col, width=3)
                if kk > 0.5: d.text((CX[i], 1452), res, font=font("black", 32 * kk), fill=col, anchor="mm")
    if S.get("hint"): d.text((W / 2, 1548), S["hint"], font=font("semi", 34), fill=S.get("hint_col", MUTED), anchor="mm")
    for b in S.get("bursts", ()): burst(d, t, *b)
    if S.get("end_k", 0) > 0: end_card(img, t, S["end_k"], S.get("end_t", 0))
    out = img.convert("RGB")
    if S.get("shake"):
        sx, sy = S["shake"]; out = ImageChops.offset(out, int(sx), int(sy))
    return out

CODE_LINES = ['import random', 'python = random.choice(', '    ["rock", "paper", "scissors"])']
def end_card(img, t, k, et):
    """et = seconds since the end card started"""
    ov = Image.new("RGBA", img.size, (8, 10, 14, int(200 * k))); img.alpha_composite(ov)
    d = ImageDraw.Draw(img)
    kp = ease_out(et / 0.35); y0 = 340 + (1 - kp) * 120
    d.rounded_rectangle([40, y0, 940, y0 + 1240], radius=34, fill=(16, 20, 26), outline=BORDER, width=3)
    if et < 0.12: return
    k1 = ease_back((et - 0.1) / 0.4)
    d.text((490, y0 + 120), "HOW MANY", font=font("black", 92 * k1), fill=TEXT, anchor="mm")
    d.text((490, y0 + 222), "DID YOU WIN?", font=font("black", 92 * k1), fill=YEL, anchor="mm")
    for j in range(4):
        kj = ease_back((et - 0.45 - j * 0.1) / 0.35)
        if kj <= 0.02: continue
        cx = 205 + j * 190; cy = y0 + 400 + 10 * math.sin(t * 6 + j); r = 70 * kj
        col = [RED, MUTED, BLUE, GREEN][j]
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=mix(BG, col, 0.25), outline=col, width=5)
        d.text((cx, cy), str(j), font=font("black", 84 * kj), fill=(255, 255, 255), anchor="mm")
    k3 = ease_out((et - 1.0) / 0.3)
    if k3 > 0:
        d.text((490, y0 + 560), "Comment your score", font=font("black", 58), fill=mix((16, 20, 26), YEL, k3), anchor="mm")
        ay = y0 + 628 + 10 * math.sin(t * 8)
        d.polygon([(470, ay), (510, ay), (490, ay + 26)], fill=mix((16, 20, 26), YEL, k3))
    k4 = ease_out((et - 1.4) / 0.3)
    if k4 > 0:
        bx0, by0 = 80, y0 + 700
        d.rounded_rectangle([bx0, by0, 900, by0 + 200], radius=18, fill=(10, 13, 18), outline=BORDER, width=2)
        d.text((bx0 + 24, by0 + 26), "python's move:", font=font("semi", 26), fill=MUTED, anchor="lm")
        for li, ln in enumerate(CODE_LINES):
            x = bx0 + 24; y = by0 + 74 + li * 46
            for m in __import__("re").finditer(r'("[^"]*")|(\bimport\b)|(\w+(?=\())|(.)', ln):
                s = m.group(0); col = (165, 214, 255) if m.group(1) else (255, 123, 114) if m.group(2) else (210, 168, 255) if m.group(3) else TEXT
                d.text((x, y), s, font=mono(30), fill=mix((10, 13, 18), col, k4), anchor="lm"); x += d.textlength(s, font=mono(30))
    k5 = ease_out((et - 1.8) / 0.3)
    if k5 > 0:
        d.text((490, y0 + 990), "Full game code: link in profile", font=font("bold", 40), fill=mix((16, 20, 26), TEXT, k5), anchor="mm")
        text2(d, y0 + 1110, [("PYTHONIC ", mix((16, 20, 26), BLUE, k5)), ("LIFE", mix((16, 20, 26), YEL, k5))], font("black", 70))


def base_state(pill):
    return dict(pill=pill, py_face="back", pump=0.0, scale=[1.0] * 3, dy=[0.0] * 3, rot=[0.0] * 3, alpha=[1.0] * 3,
                border=[None] * 3, chips=[None] * 3, bursts=[])


# ================================================================== the YouTube Short
BPM = 120; B = 60 / BPM
ROUND_BEATS, END_BEATS = 10, 10
TOTAL = (3 * ROUND_BEATS + END_BEATS) * B                       # 20.0 s, exactly 10 bars: the music loops cleanly
T_END = 3 * ROUND_BEATS * B
FPS, SR = 30, 44100

V2 = "--v2" in sys.argv                                         # second upload: a different set of moves
def python_moves():
    """Python's three moves: random.choice with the first seed that gives three different moves (every viewer pick gets a mix)"""
    s = 1; first = None
    while True:
        r = random.Random(s); m = [r.choice(KINDS) for _ in range(3)]
        if len(set(m)) == 3:
            if not V2: return s, m
            if first is None: first = m
            elif m != first and m[0] != first[0]: return s, m
        s += 1
SEED, PY_MOVES = python_moves()
WORDS = [("ROCK", KCOL["rock"], 120), ("PAPER", KCOL["paper"], 120), ("SCISSORS", KCOL["scissors"], 112), ("SHOOT!", YEL, 150)]

def short_state(t):
    S = base_state("ROUND 1 / 3")
    rnd = min(2, int(t / (ROUND_BEATS * B)))
    u = (t - rnd * ROUND_BEATS * B) / B                         # beats into this round
    py = PY_MOVES[rnd]; S["pill"] = f"ROUND {rnd + 1} / 3"
    # idle motion
    for i in range(3):
        S["dy"][i] = 6 * math.sin(t * 3.2 + i * 1.7); S["rot"][i] = 1.2 * math.sin(t * 2.1 + i)
    S["py_rot"] = 1.5 * math.sin(t * 2.6)
    # pump on every count word
    if 3 <= u < 7:
        bt = math.floor(u); S["pump"] = 26 * math.exp(-(u - bt) * B * 9)
    if u < 3:                                                   # pick phase: highlight sweeps the cards
        hl = int(u * 2) % 3
        for i in range(3):
            if i == hl: S["scale"][i] = 1.07; S["border"][i] = (255, 255, 255); S["dy"][i] -= 14
        S["word"], S["word_col"], S["word_size"] = ("PICK ONE!" if rnd == 0 else "PICK AGAIN!"), YEL, 92
        S["word_k"] = 1.0 if rnd == 0 and t < 0.05 else min(1.0, (u % 1.0) * B / 0.18 + 0.35)
        S["hint"] = "Pick in your head. Python picks too."
        S["py_rot"] = 4 * math.sin(t * 9)                       # python "shuffling"
        if rnd > 0 and u < 0.6:                                 # flip python's card back
            p = u / 0.6; S["py_sx"] = abs(math.cos(math.pi * p)); S["py_face"] = PY_MOVES[rnd - 1] if p < 0.5 else "back"
    elif u < 6:
        w = WORDS[int(u) - 3]
        S["word"], S["word_col"], S["word_size"], S["word_k"] = w[0], w[1], w[2], (u - int(u)) * B / 0.18
        S["hint"] = "Lock in your pick!"
    else:
        p = (u - 6) * B / 0.28                                  # flip python's card
        if p < 1:
            S["py_sx"] = abs(math.cos(math.pi * p)); S["py_face"] = "back" if p < 0.5 else py
            S["word"], S["word_col"], S["word_size"], S["word_k"] = WORDS[3][0], YEL, 150, (u - 6) * B / 0.18
        else:
            S["py_face"] = py; S["py_border"] = YEL
            S["word"], S["word_col"], S["word_size"] = f"PYTHON: {py.upper()}", KCOL[py], 74
            S["word_k"] = (u - 6.56) * B / 0.25
        S["bursts"] = [(rnd * ROUND_BEATS * B + 6 * B + 0.14, PY_C[0], PY_C[1], 7 + rnd)]
        if u < 6.6:
            sh = 10 * math.exp(-(u - 6) * B * 10); S["shake"] = (sh * math.sin(t * 90), sh * math.cos(t * 77))
        fade = 1.0 if rnd == 2 or u < 9.4 else max(0.0, 1 - (u - 9.4) / 0.6)
        for i, kd in enumerate(KINDS):
            res = outcome(kd, py); kk = (u - 6.7 - 0.3 * i) * B / 0.3
            if kk > 0:
                S["chips"][i] = (res, kk * fade if fade < 1 else kk)
                if res == "WIN": S["border"][i] = GREEN; S["scale"][i] = 1.0 + 0.07 * ease_back(min(1, kk)) * fade; S["dy"][i] -= 18 * fade
                elif res == "LOSE": S["alpha"][i] = 1 - 0.45 * min(1, kk) * fade; S["border"][i] = RED
        S["hint"] = "Did you beat Python?"; S["hint_col"] = TEXT
    if t >= T_END:
        S["end_k"] = min(1.0, (t - T_END) / 0.3); S["end_t"] = t - T_END; S["hint"] = None
    return S


def synth_audio():
    import numpy as np
    NS = int((TOTAL + 0.5) * SR); mus = np.zeros(NS, np.float32); sfx = np.zeros(NS, np.float32)
    def put(buf, t0, sig):
        s = int(t0 * SR); n = min(len(sig), NS - s)
        if n > 0 and s >= 0: buf[s:s + n] += sig[:n].astype(np.float32)
    def tt(dur): return np.arange(int(dur * SR)) / SR
    def noise(n, seed): return np.random.default_rng(seed).standard_normal(n)
    def lowpass(x, a): return np.convolve(x, np.ones(a) / a, mode="same")
    def kick(t0, amp=0.55):
        x = tt(0.35); put(mus, t0, np.sin(2 * np.pi * np.cumsum(140 * np.exp(-x * 30) + 46) / SR) * np.exp(-x * 9) * amp)
    def snare(t0, amp=0.22):
        x = tt(0.22); n = noise(len(x), int(t0 * 1000))
        put(mus, t0, (0.7 * n * np.exp(-x * 22) + 0.5 * np.sin(2 * np.pi * 190 * x) * np.exp(-x * 30)) * amp)
    def hat(t0, amp=0.05):
        x = tt(0.05); put(mus, t0, np.diff(noise(len(x) + 1, int(t0 * 997))) * np.exp(-x * 80) * amp)
    def bass(t0, f, dur, amp=0.16):
        x = tt(dur); ph = 2 * np.pi * f * x
        put(mus, t0, (np.sin(ph) + 0.3 * np.sin(2 * ph) + 0.15 * np.sign(np.sin(ph))) * np.minimum(1, x * 200) * np.exp(-x * 5) * amp)
    def pluck(buf, t0, f, amp, dec=9):
        x = tt(0.5); put(buf, t0, (np.sin(2 * np.pi * f * x) + 0.25 * np.sin(4 * np.pi * f * x)) * np.exp(-x * dec) * np.minimum(1, x * 600) * amp)
    def whoosh(t0, dur, amp, up=True, seed=0):
        x = tt(dur); n = noise(len(x), seed)
        a = lowpass(n, 18) * 4; b = n - lowpass(n, 6)
        k = x / dur if up else 1 - x / dur
        put(sfx, t0, (a * (1 - k) + b * k) * np.sin(np.pi * np.clip(x / dur, 0, 1)) ** 1.5 * amp)
    def impact(t0, amp=1.0):
        x = tt(0.9)
        put(sfx, t0, np.sin(2 * np.pi * np.cumsum(90 * np.exp(-x * 14) + 42) / SR) * np.exp(-x * 4.5) * 0.65 * amp)
        n = noise(len(x), int(t0 * 31)); put(sfx, t0, lowpass(n, 5) * 2.2 * np.exp(-x * 11) * 0.30 * amp)
        for f in (220.0, 261.6, 329.6, 440.0):                                   # A-minor stab
            ph = 2 * np.pi * f * x
            put(sfx, t0, (np.sin(ph) + 0.4 * np.sin(2 * ph) + 0.2 * np.sin(3 * ph)) * np.exp(-x * 5) * 0.055 * amp)
    def tom(t0, f, amp=0.62):
        x = tt(0.4); put(sfx, t0, np.sin(2 * np.pi * np.cumsum(f * (1 + 0.8 * np.exp(-x * 25))) / SR) * np.exp(-x * 9) * amp)
        put(sfx, t0, noise(int(0.012 * SR), int(f)) * np.exp(-tt(0.012) * 300) * 0.25)
    def tick(t0, f=1900, amp=0.17):
        x = tt(0.06); put(sfx, t0, (np.sin(2 * np.pi * f * x) + 0.5 * np.sin(2 * np.pi * f * 1.5 * x)) * np.exp(-x * 70) * amp)
    def bell(t0, f, amp=0.12):
        x = tt(1.6); put(sfx, t0, sum(np.sin(2 * np.pi * f * m * x) * a for m, a in ((1, 1), (2.76, 0.4), (5.4, 0.2))) * np.exp(-x * 3) * amp)

    # music: 120 bpm, Am - F - C - G, four on the floor, exactly 10 bars
    roots = [110.0, 87.31, 130.81, 98.0]; arp = [[440, 523.3, 659.3, 523.3], [349.2, 440, 523.3, 440], [392, 523.3, 659.3, 523.3], [392, 493.9, 587.3, 493.9]]
    for bar in range(10):
        t0 = bar * 4 * B; ri = bar % 4; loud = bar >= 7
        for b_ in range(4):
            kick(t0 + b_ * B, 0.33 if not loud else 0.4)
            if b_ in (1, 3): snare(t0 + b_ * B, 0.2 if not loud else 0.26)
            hat(t0 + b_ * B, 0.03); hat(t0 + b_ * B + B / 2, 0.06)
        for e in range(8): bass(t0 + e * B / 2, roots[ri], B / 2 + 0.02, 0.15 if e % 2 == 0 else 0.10)
        for s in range(16): pluck(mus, t0 + s * B / 4, arp[ri][s % 4] * (2 if s % 8 >= 4 else 1), 0.022 if not loud else 0.03)
    # sfx per round
    for r in range(3):
        R = r * ROUND_BEATS * B
        if r > 0: whoosh(R - 0.05, 0.35, 0.10, up=False, seed=r)                 # card flips back
        for k in range(6): tick(R + k * B / 2, 1900 if k % 3 else 2500)          # highlight sweep
        for j, f in enumerate((150.0, 175.0, 205.0)): tom(R + (3 + j) * B, f)     # ROCK PAPER SCISSORS
        whoosh(R + 6 * B - 0.18, 0.22, 0.14, up=True, seed=10 + r)              # swish into the flip
        impact(R + 6 * B, 1.5 if r < 2 else 1.7)                                # SHOOT!
        for i, f in enumerate((880.0, 1108.7, 1318.5)): pluck(sfx, R + (6.7 + 0.3 * i) * B, f, 0.11, 18)   # result chips
    whoosh(T_END - 0.55, 0.55, 0.12, up=True, seed=99)                           # riser into the end card
    impact(T_END, 1.1)
    for j in range(4): tick(T_END + 0.45 + j * 0.1, 1500 + 200 * j, 0.12)
    bell(T_END + 1.0, 1318.5, 0.11); bell(T_END + 1.0, 1975.5, 0.05)
    a = mus * 0.85 + sfx
    a = a[: int(TOTAL * SR)]
    a = np.tanh(a / (np.percentile(np.abs(a), 99.7) + 1e-6) * 1.1) * 0.9
    return a

def render_video(out="rps_vs_python.mp4"):
    import numpy as np, wave
    print(f"python's moves (seed {SEED}): {PY_MOVES}", flush=True)
    NF = int(round(TOTAL * FPS)); T_LOOP = TOTAL - 0.6
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                             "-c:v", "libx264", "-preset", "slow", "-crf", "15", "-maxrate", "10M", "-bufsize", "20M", "-pix_fmt", "yuv420p",
                             "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "rps_noaudio.mp4"], stdin=subprocess.PIPE)
    F0 = np.asarray(draw(short_state(0.0), 0.0)).astype(np.float32)
    for f in range(NF):
        t = f / FPS; fr = np.asarray(draw(short_state(t), t))
        if t > T_LOOP:
            a = min(1.0, (t - T_LOOP) / (TOTAL - T_LOOP - 1 / FPS)); fr = (fr * (1 - a) + F0 * a).astype(np.uint8)
        proc.stdin.write(np.ascontiguousarray(fr).tobytes())
    proc.stdin.close(); proc.wait()
    a = synth_audio()
    with wave.open("rps_audio.wav", "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR); wf.writeframes((a * 32767).astype(np.int16).tobytes())
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "rps_noaudio.mp4", "-i", "rps_audio.wav", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", out], check=True)
    print("done", out, round(TOTAL, 1), "s")


# ================================================================== the tkinter game
class Game:
    SCALE = 2                                                     # window = 540 x 960
    STEP = 0.36                                                   # seconds between ROCK / PAPER / SCISSORS / SHOOT

    def __init__(self):
        import tkinter as tk
        from PIL import ImageTk
        self.tk, self.ImageTk = tk, ImageTk
        self.root = tk.Tk(); self.root.title("Rock Paper Scissors vs Python"); self.root.resizable(False, False)
        self.cv = tk.Canvas(self.root, width=W // self.SCALE, height=H // self.SCALE, highlightthickness=0, bg="#0d1117"); self.cv.pack()
        self.item = self.cv.create_image(0, 0, anchor="nw")
        self.cv.bind("<Button-1>", self.click); self.cv.bind("<Motion>", self.motion)
        for key, i in (("1", 0), ("2", 1), ("3", 2), ("r", 0), ("p", 1), ("s", 2)): self.root.bind(key, lambda e, i=i: self.choose(i))
        self.score = [0, 0, 0]                                    # you, python, draws
        self.phase, self.hover, self.you, self.py, self.t0 = "pick", None, None, None, time.time()
        self.start = time.time(); self.tick()

    def card_at(self, x, y):
        x, y = x * self.SCALE, y * self.SCALE
        for i, cx in enumerate(CX):
            if abs(x - cx) < 135 and abs(y - YOU_C) < 165: return i
        return None

    def motion(self, e): self.hover = self.card_at(e.x, e.y)
    def click(self, e):
        if self.phase == "pick":
            i = self.card_at(e.x, e.y)
            if i is not None: self.choose(i)
        elif self.phase == "result" and time.time() - self.t0 > 0.6: self.phase = "pick"

    def choose(self, i):
        if self.phase != "pick": return
        self.you, self.py = KINDS[i], random.choice(KINDS)       # python's move
        self.phase, self.t0 = "count", time.time()

    def state(self, t):
        w, l, dr = self.score
        S = base_state(f"YOU {w}  :  {l} PYTHON" + (f"   ·   {dr} draws" if dr else ""))
        for i in range(3): S["dy"][i] = 5 * math.sin(t * 3.2 + i * 1.7)
        S["you_label"] = "YOU"
        el = time.time() - self.t0
        if self.phase == "pick":
            S["word"], S["word_col"], S["word_size"] = "PICK ONE!", YEL, 92
            S["hint"] = "Click a card or press 1 · 2 · 3"; S["py_rot"] = 2 * math.sin(t * 5)
            if self.hover is not None:
                i = self.hover; S["scale"][i] = 1.07; S["border"][i] = (255, 255, 255); S["dy"][i] -= 14
        else:
            ci = KINDS.index(self.you)
            for i in range(3):
                if i != ci: S["alpha"][i] = 0.35
            S["border"][ci] = (255, 255, 255); S["scale"][ci] = 1.07
            u = el / self.STEP
            if self.phase == "count":
                bt = math.floor(u); S["pump"] = 24 * math.exp(-(u - bt) * self.STEP * 9)
                if u < 3:
                    w_ = WORDS[int(u)]; S["word"], S["word_col"], S["word_size"], S["word_k"] = w_[0], w_[1], w_[2], (u - int(u)) * self.STEP / 0.15
                else:
                    p = (u - 3) * self.STEP / 0.26
                    S["word"], S["word_col"], S["word_size"], S["word_k"] = "SHOOT!", YEL, 150, (u - 3) * self.STEP / 0.15
                    S["py_sx"] = abs(math.cos(math.pi * min(p, 1))); S["py_face"] = "back" if p < 0.5 else self.py
                    if p >= 1:
                        res = outcome(self.you, self.py)
                        self.score[{"WIN": 0, "LOSE": 1, "DRAW": 2}[res]] += 1
                        self.phase, self.t0, self.res = "result", time.time(), res
            if self.phase == "result":
                el = time.time() - self.t0
                S["py_face"] = self.py; S["py_border"] = YEL; S["pump"] = 0
                msg = {"WIN": ("YOU WIN!", GREEN), "LOSE": ("PYTHON WINS", RED), "DRAW": ("DRAW", MUTED)}[self.res]
                S["word"], S["word_col"], S["word_size"], S["word_k"] = msg[0], msg[1], 100, el / 0.25
                S["chips"][ci] = (self.res, el / 0.3); S["border"][ci] = RESCOL[self.res]
                S["bursts"] = [(self.start_res, PY_C[0], PY_C[1], 3)] if self.res == "WIN" else []
                S["hint"] = "Click anywhere to play again"
                if el > 2.6: self.phase = "pick"
        return S

    def tick(self):
        t = time.time() - self.start
        if self.phase == "result" and not hasattr(self, "_res_mark"): self._res_mark = True; self.start_res = t
        if self.phase != "result" and hasattr(self, "_res_mark"): del self._res_mark
        if not hasattr(self, "start_res"): self.start_res = -9
        img = draw(self.state(t), t).reduce(self.SCALE)
        self.photo = self.ImageTk.PhotoImage(img); self.cv.itemconfig(self.item, image=self.photo)
        self.root.after(16, self.tick)

    def run(self): self.root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "video": render_video("rps_vs_python_v2.mp4" if V2 else "rps_vs_python.mp4")
    elif len(sys.argv) > 1 and sys.argv[1] == "still":
        os.makedirs("chk", exist_ok=True)
        for ts in sys.argv[2:]: draw(short_state(float(ts)), float(ts)).save(f"chk/still_{ts}.png")
    else: Game().run()
