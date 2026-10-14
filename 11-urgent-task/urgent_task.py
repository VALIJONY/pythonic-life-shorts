"""Pythonic Life — "BOSS: SORT 1,200 FILES IN 10 MINUTES" (YouTube Shorts 1080x1920, ~18 s, loops)
A boss message, a messy Downloads folder and the real Python code that fixes it.
The code shown on screen is executed for real: the timing and the folder counts in the video are measured.
Crisp flat UI (no glow / haze), YouTube Shorts safe zones respected, encoded at high quality.
Usage: python3 urgent_task.py            (python3 urgent_task.py still 0 5 9 11 15 -> PNG stills to check the layout)
"""
import math, os, shutil, subprocess, sys, tempfile, time, wave, statistics
from functools import lru_cache
import re
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SR = 1080, 1920, 30, 44100
N_FILES = 1200

# ------------------------------------------------------------------ the code that is shown AND executed
CODE = '''import shutil, pathlib as p
K = {"pdf":"PDF","xlsx":"Excel","docx":"Word",
     "jpg":"Images","mp4":"Video","zip":"Zips"}
for f in p.Path("Downloads").iterdir():
    d = p.Path("Sorted") / K[f.suffix[1:]]
    d.mkdir(exist_ok=True, parents=True)
    shutil.move(f, d / f.name)'''
EXTS = ["pdf", "xlsx", "docx", "jpg", "mp4", "zip"]
FOLDERS = ["PDF", "Excel", "Word", "Images", "Video", "Zips"]
rng = np.random.default_rng(11)
TYPES = rng.choice(6, N_FILES, p=[0.22, 0.17, 0.20, 0.25, 0.06, 0.10])


def bench():
    """create 1,200 real files, run CODE for real 7 times, return median seconds and per-folder counts"""
    times, counts = [], None
    cwd = os.getcwd()
    for rep in range(7):
        tmp = tempfile.mkdtemp()
        try:
            os.chdir(tmp); os.makedirs("Downloads")
            for i, t in enumerate(TYPES):
                open(f"Downloads/{['report','budget','notes','photo','clip','backup'][t]}_{i:04d}.{EXTS[t]}", "w").close()
            t0 = time.perf_counter(); exec(CODE, {}); dt = time.perf_counter() - t0
            if rep > 0: times.append(dt)                      # first run warms the cache
            counts = [len(os.listdir(f"Sorted/{f}")) for f in FOLDERS]
        finally:
            os.chdir(cwd); shutil.rmtree(tmp, ignore_errors=True)
    return statistics.median(times), counts


REAL_T, COUNTS = bench()
assert sum(COUNTS) == N_FILES
MANUAL_S = N_FILES * 10                                      # 10 s per file by hand
SPEEDUP = MANUAL_S / REAL_T
print(f"real run: {REAL_T:.3f} s, counts {COUNTS}, manual {MANUAL_S / 3600:.2f} h, speed-up {SPEEDUP:,.0f}x", flush=True)

# ------------------------------------------------------------------ timeline (seconds)
T_HAND, T_DRAG0, T_DRAG1 = 1.2, 1.6, 3.2
TY0, TY1 = 3.4, 7.3
T_RUN, T_RUN_END = 7.9, 9.7
T_DONE, T_RES, T_CTA = 9.9, 10.3, 14.2
TOTAL = 18.2
T_LOOP = TOTAL - 0.6
NF = int(TOTAL * FPS)

# ------------------------------------------------------------------ style
BG, PANEL, PANEL2, BORDER = (13, 17, 23), (22, 27, 34), (30, 36, 44), (48, 54, 61)
TEXT, MUTED = (230, 237, 243), (139, 148, 158)
RED, GREEN, BLUE, YEL, PUR, GRAY = (248, 81, 73), (63, 185, 80), (88, 166, 255), (210, 153, 34), (188, 140, 255), (139, 148, 158)
TCOL = [RED, GREEN, BLUE, YEL, PUR, GRAY]                    # pdf xlsx docx jpg mp4 zip
F = "/usr/share/fonts/opentype/inter/"
@lru_cache(None)
def font(n, s): return ImageFont.truetype(F + n, max(1, int(s)))
@lru_cache(None)
def mono(s): return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", max(1, int(s)))
BLACK = lambda s: font("Inter-Black.otf", s)
BOLD = lambda s: font("Inter-Bold.otf", s)
SEMI = lambda s: font("Inter-SemiBold.otf", s)
def ease_out(x): x = max(0.0, min(1.0, x)); return 1 - (1 - x) ** 3
def ease_io(x): x = max(0.0, min(1.0, x)); return 3 * x * x - 2 * x ** 3
def ease_back(x): x = max(0.0, min(1.0, x)); return 1 + 2.70158 * (x - 1) ** 3 + 1.70158 * (x - 1) ** 2
def tc(c, a=1.0): return tuple(int(min(255, max(0, v * a))) for v in c)
def mix(a, b, k): return tuple(int(a[i] + (b[i] - a[i]) * k) for i in range(3))

# ------------------------------------------------------------------ layout
GX0, GX1, GY0, GY1 = 40, 1040, 410, 985                      # file grid
COLS_, ROWS_ = 48, 25
IW, IH = 16, 20                                               # icon size
TX0, TX1, TY_0, TY_1 = 30, 930, 1015, 1343                     # terminal
CY0, CY1 = 1362, 1602                                         # folder cards
CW, CGAP = 137, 15.6
CARD_X = [30 + i * (CW + CGAP) for i in range(6)]
LINE_H, CODE_Y0 = 34, TY_0 + 62

# ------------------------------------------------------------------ files (grid, flights)
gx = np.linspace(GX0 + IW, GX1 - IW, COLS_); gy = np.linspace(GY0 + IH, GY1 - IH, ROWS_)
GRID = np.array([(gx[c], gy[r]) for r in range(ROWS_) for c in range(COLS_)])[:N_FILES]
order = rng.permutation(N_FILES)
GRID = GRID[order] + rng.uniform(-2.5, 2.5, (N_FILES, 2))      # messy positions
PHASE = rng.uniform(0, 6.28, N_FILES)

HAND = int(np.argmin([(GRID[i, 0] - 150) ** 2 + (GRID[i, 1] - 880) ** 2 if TYPES[i] == 0 else 1e12 for i in range(N_FILES)]))
END = np.zeros((N_FILES, 2)); DELAY = np.zeros(N_FILES); CTRL = np.zeros((N_FILES, 2))
for i in range(N_FILES):
    j = TYPES[i]; cx = CARD_X[j] + CW / 2
    END[i] = (cx + rng.uniform(-14, 14), CY0 + 70 + rng.uniform(-8, 8))
    DELAY[i] = (1 - (GRID[i, 1] - GY0) / (GY1 - GY0)) * 0.55 + rng.uniform(0, 0.55)   # bottom rows leave first
    CTRL[i] = (GRID[i, 0] + (END[i, 0] - GRID[i, 0]) * 0.2 + rng.uniform(-30, 30), GRID[i, 1] - 70 - rng.uniform(0, 50))
FLIGHT = 0.55
LAUNCH = T_RUN + DELAY                                         # departure time of each file
ARRIVE = LAUNCH + FLIGHT
LAUNCH[HAND] = 1e9; ARRIVE[HAND] = T_DRAG1                     # the hand-moved file is separate


def pos_fly(i, t):
    u = ease_io((t - LAUNCH[i]) / FLIGHT)
    p0, p1, p2 = GRID[i], CTRL[i], END[i]
    return ((1 - u) ** 2 * p0[0] + 2 * (1 - u) * u * p1[0] + u * u * p2[0],
            (1 - u) ** 2 * p0[1] + 2 * (1 - u) * u * p1[1] + u * u * p2[1])


def counts_at(t):
    c = np.zeros(6, int)
    done = ARRIVE <= t
    for j in range(6): c[j] = int(np.sum(done & (TYPES == j)))
    return c


def departed_at(t):
    return int(np.sum(LAUNCH <= t)) + (1 if t >= T_DRAG0 else 0)


# ------------------------------------------------------------------ sprites
def make_icon(col):
    s = 6; w, h = IW * s, IH * s
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    fold = 5 * s
    d.polygon([(0, 2 * s), (2 * s, 0), (w - fold, 0), (w, fold), (w, h - 2 * s), (w - 2 * s, h), (2 * s, h), (0, h - 2 * s)], fill=col + (255,))
    d.polygon([(w - fold, 0), (w - fold, fold), (w, fold)], fill=tuple(int(v * 0.55) for v in col) + (255,))
    for k in range(3):
        d.rectangle([3 * s, (9 + 3 * k) * s, (w - 4 * s) - (k == 2) * 3 * s, (10 + 3 * k) * s], fill=(255, 255, 255, 190))
    im = im.resize((IW, IH), Image.LANCZOS)
    a = np.asarray(im).astype(np.float32)
    return a[..., :3], a[..., 3:4] / 255.0
ICONS = [make_icon(c) for c in TCOL]


def blit(arr, spr, x, y):
    rgb, a = spr; h, w = a.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1: return
    sa = a[y0 - y:y1 - y, x0 - x:x1 - x]; sr = rgb[y0 - y:y1 - y, x0 - x:x1 - x]
    reg = arr[y0:y1, x0:x1]
    reg[:] = (reg + (sr - reg) * sa).astype(np.uint8)


# ------------------------------------------------------------------ static base
def folder_glyph(d, cx, cy, col, s=1.0):
    w, h = 78 * s, 54 * s
    d.rounded_rectangle([cx - w / 2, cy - h / 2 - 8 * s, cx - w / 2 + 34 * s, cy - h / 2 + 6 * s], radius=6 * s, fill=tc(col, 0.8))
    d.rounded_rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], radius=9 * s, fill=col)
    d.rounded_rectangle([cx - w / 2, cy - h / 2 + 12 * s, cx + w / 2, cy + h / 2], radius=9 * s, fill=tc(col, 0.88))


def make_base():
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    # boss bubble
    d.ellipse([40, 148, 124, 232], fill=RED)
    d.text((82, 192), "B", font=BLACK(50), fill=(255, 255, 255), anchor="mm")
    d.rounded_rectangle([146, 140, 1040, 310], radius=34, fill=PANEL2, outline=BORDER, width=2)
    d.polygon([(146, 185), (120, 200), (146, 218)], fill=PANEL2)
    d.text((184, 120), "BOSS", font=BOLD(26), fill=RED, anchor="lm")
    # grid frame
    d.rounded_rectangle([GX0 - 14, GY0 - 14, GX1 + 14, GY1 + 14], radius=22, fill=PANEL, outline=BORDER, width=2)
    # terminal
    d.rounded_rectangle([TX0, TY_0, TX1, TY_1], radius=18, fill=(10, 13, 18), outline=BORDER, width=2)
    d.rounded_rectangle([TX0, TY_0, TX1, TY_0 + 40], radius=18, fill=PANEL2)
    d.rectangle([TX0, TY_0 + 22, TX1, TY_0 + 40], fill=PANEL2)
    for k, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([TX0 + 18 + k * 26, TY_0 + 12, TX0 + 34 + k * 26, TY_0 + 28], fill=c)
    d.text((TX0 + 120, TY_0 + 20), "sort.py", font=SEMI(22), fill=MUTED, anchor="lm")
    # folder cards
    for j in range(6):
        x0 = CARD_X[j]
        d.rounded_rectangle([x0, CY0, x0 + CW, CY1], radius=20, fill=PANEL, outline=tc(TCOL[j], 0.55), width=3)
        folder_glyph(d, x0 + CW / 2, CY0 + 70, TCOL[j])
        d.text((x0 + CW / 2, CY0 + 140), FOLDERS[j], font=BOLD(26), fill=TEXT, anchor="mm")
    d.text((W / 2, 1640), "PYTHONIC LIFE", font=BLACK(30), fill=(70, 78, 90), anchor="mm")
    return np.asarray(im).copy()
BASE = make_base()

# syntax colours for the code
CODE_COL = [TEXT] * len(CODE)
for m in re.finditer(r'(".*?")|\b(import|for|in|as)\b|(\b\w+(?=\())|(\b\w+(?=\.\w+\())', CODE):
    col = (165, 214, 255) if m.group(1) else (255, 123, 114) if m.group(2) else (210, 168, 255)
    for k in range(m.start(), m.end()): CODE_COL[k] = col
NCH = len(CODE)
w_ = np.array([4.0 if c == "\n" else 1.0 for c in CODE]); cum = np.cumsum(w_) / w_.sum()
rj = np.random.default_rng(5).uniform(-0.002, 0.002, NCH)
CH_T = TY0 + (TY1 - TY0) * np.clip(cum + rj, 0, 1)             # time every character is typed
CH_T = np.maximum.accumulate(CH_T)

# ------------------------------------------------------------------ audio
NS = int((TOTAL + 1) * SR)
audio = np.zeros(NS, np.float32)
def put(t0, sig):
    s = int(t0 * SR); n = min(len(sig), NS - s)
    if n > 0 and s >= 0: audio[s:s + n] += sig[:n].astype(np.float32)
def tone(t0, f, dur, amp, decay=7.0):
    tt = np.arange(int(dur * SR)) / SR
    put(t0, (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt)) * np.exp(-tt * decay) * np.minimum(1, tt * 800) * amp)
def click(t0, amp, f=1700, seed=None):
    tt = np.arange(int(0.025 * SR)) / SR
    nz = np.random.default_rng(seed if seed is not None else int(t0 * 1e4) % 99991).standard_normal(len(tt))
    put(t0, (nz * 0.5 + np.sin(2 * np.pi * f * tt)) * np.exp(-tt * 160) * amp)
def kick(t0, amp=0.40):
    tt = np.arange(int(0.25 * SR)) / SR
    put(t0, np.sin(2 * np.pi * np.cumsum(130 * np.exp(-tt * 26) + 46) / SR) * np.exp(-tt * 10) * amp)
def hat(t0, a=0.07):
    tt = np.arange(int(0.05 * SR)) / SR
    nz = np.diff(np.random.default_rng(int(t0 * 1e3) % 99991).standard_normal(len(tt) + 1))
    put(t0, nz * np.exp(-tt * 70) * a)
def bass(t0, f, dur, amp=0.15):
    tt = np.arange(int(dur * SR)) / SR; ph = (f * tt) % 1.0
    put(t0, ((2 * ph - 1) * 0.6 + np.sin(2 * np.pi * f * tt)) * np.exp(-tt * 5) * np.minimum(1, tt * 500) * amp)

tone(0.0, 1320, 0.25, 0.30, 12); tone(0.12, 1760, 0.35, 0.28, 9)                       # boss message ping
for k in range(0, 8): click(0.55 + k * 1.0, 0.10, 2600, seed=k)                         # ticking deadline clock
tone(T_HAND + 0.05, 233, 0.5, 0.18, 4); tone(T_HAND + 0.55, 208, 0.7, 0.18, 3)          # "by hand" sad notes
for k in range(NCH):                                                                    # keyboard
    if CODE[k] != " ": click(CH_T[k], 0.07 + 0.04 * (k % 3), 1200 + 600 * (k % 4), seed=100 + k)
tone(T_RUN - 0.05, 392, 0.12, 0.25, 15); tone(T_RUN + 0.02, 587, 0.2, 0.25, 12)         # run button
n_w = int((T_RUN_END - T_RUN + 0.5) * SR); tw = np.arange(n_w) / SR                      # swarm whoosh
nz = np.random.default_rng(3).standard_normal(n_w)
nz = np.convolve(nz, np.ones(14) / 14, mode="same") * 14 ** 0.5
put(T_RUN, nz * np.sin(np.pi * np.clip(tw / (n_w / SR), 0, 1)) ** 1.5 * 0.13)
PENTA = [523.3, 587.3, 659.3, 784.0, 880.0, 1046.5]
arr_sorted = np.sort(ARRIVE[ARRIVE < 1e8])
for k in range(0, len(arr_sorted), 6):
    tone(arr_sorted[k], PENTA[(k // 6) % 6] * (1 + (k // 36) * 0.0), 0.12, 0.035, 22)   # tiny landing ticks (rain)
for k_, fq in enumerate([523.3, 659.3, 784.0, 1046.5]): tone(T_DONE + 0.07 * k_, fq, 1.3, 0.28, 3.5)   # success
tone(T_RES + 0.05, 1320, 0.25, 0.30, 12); tone(T_RES + 0.17, 1760, 0.35, 0.28, 9)       # boss reply ping
beat = 60 / 126.0; BASS = [55.0, 55.0, 82.41, 55.0, 65.41, 55.0, 98.0, 82.41]
tb, bi = T_RUN, 0
while tb < T_CTA + 0.3:
    kick(tb); hat(tb + beat / 2)
    for h_ in range(2): bass(tb + h_ * beat / 2, BASS[(bi * 2 + h_) % 8], beat / 2)
    tb += beat; bi += 1
for k_, fq in enumerate([261.6, 329.6, 392.0, 523.3]): tone(T_CTA + 0.1 * k_, fq, 2.0, 0.2, 2.5)


# ------------------------------------------------------------------ frame renderer
def fmt_t(sec):
    sec = max(0.0, sec); return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


def draw_text_pop(d, xy, txt, fnt, col, k, anchor="mm"):
    if k > 0.01: d.text(xy, txt, font=font(fnt[0], fnt[1] * k), fill=col, anchor=anchor)


def render(f):
    t = f / FPS
    arr = BASE.copy()
    # ---- files in the grid ----
    tr = 1.3 if t < T_RUN else 0.0                                     # nervous shaking before the run
    for i in range(N_FILES):
        if i == HAND:
            if t < T_DRAG0: pass
            else: continue
        if LAUNCH[i] <= t: continue
        x = GRID[i, 0] + tr * math.sin(t * 13 + PHASE[i]); y = GRID[i, 1] + tr * math.cos(t * 11 + PHASE[i] * 1.3)
        blit(arr, ICONS[TYPES[i]], int(round(x - IW / 2)), int(round(y - IH / 2)))
    # ---- hand-dragged file ----
    cur = None
    if T_DRAG0 <= t < T_DRAG1 + 0.15:
        u = ease_io((t - T_DRAG0) / (T_DRAG1 - T_DRAG0))
        sx, sy = GRID[HAND]; ex, ey = CARD_X[0] + CW / 2, CY0 + 70
        x = sx + (ex - sx) * u; y = sy + (ey - sy) * u
        if t < T_DRAG1: blit(arr, ICONS[0], int(round(x - IW / 2)), int(round(y - IH / 2))); cur = (x + 6, y + 8)
    # ---- flying files ----
    if T_RUN <= t < T_RUN_END + FLIGHT + 0.2:
        for i in range(N_FILES):
            if LAUNCH[i] <= t < ARRIVE[i]:
                x, y = pos_fly(i, t)
                blit(arr, ICONS[TYPES[i]], int(round(x - IW / 2)), int(round(y - IH / 2)))
    img = Image.fromarray(arr); d = ImageDraw.Draw(img)
    if cur is not None:                                                # mouse cursor
        cx, cy = cur
        d.polygon([(cx, cy), (cx, cy + 30), (cx + 8, cy + 23), (cx + 15, cy + 36), (cx + 21, cy + 33), (cx + 14, cy + 21), (cx + 24, cy + 21)],
                  fill=(255, 255, 255), outline=(0, 0, 0))
    # ---- boss message ----
    if t < T_RES:
        l1, l2 = "Sort ALL 1,200 files by type.", "You have 10 minutes."
        d.text((184, 200), l1, font=BOLD(46), fill=TEXT, anchor="lm")
        d.text((184, 262), l2, font=BLACK(46), fill=RED, anchor="lm")
    else:
        k = ease_back((t - T_RES) / 0.35)
        d.rounded_rectangle([146, 140, 1040, 310], radius=34, fill=PANEL2, outline=BORDER, width=2)
        d.polygon([(146, 185), (120, 200), (146, 218)], fill=PANEL2)
        d.text((184, 200), "Perfect. Now do 1,000,000.", font=BOLD(max(1, 46 * min(k, 1.0))), fill=TEXT, anchor="lm")
        d.text((184, 262), "Same deadline.", font=BLACK(46), fill=RED, anchor="lm")
    # ---- timer pill / stopwatch ----
    if t < T_RUN:
        left = 600 - t
        blink = int(t * 2) % 2 == 0
        d.rounded_rectangle([700, 328, 1040, 388], radius=30, fill=(60, 20, 22), outline=RED, width=3)
        d.ellipse([722, 350, 740, 368], fill=RED if blink else (110, 40, 40))
        d.text((752, 358), "DEADLINE", font=BOLD(22), fill=(255, 150, 146), anchor="lm")
        d.text((1018, 358), fmt_t(left), font=mono(30), fill=(255, 255, 255), anchor="rm")
    else:
        run_p = min(1.0, max(0.0, (t - T_RUN) / (T_RUN_END - T_RUN)))
        val = REAL_T * run_p
        fin = t >= T_RUN_END
        col = GREEN if fin else YEL
        d.rounded_rectangle([700, 328, 1040, 388], radius=30, fill=(18, 40, 24) if fin else (44, 36, 14), outline=col, width=3)
        d.text((722, 358), "SLOW-MO", font=BOLD(22), fill=col, anchor="lm")
        d.text((1018, 358), f"{val:0.2f} s" if REAL_T >= 0.095 else f"{val:0.3f} s", font=mono(30), fill=(255, 255, 255), anchor="rm")
        if not fin: d.text((1040, 405 - 4), "SLOW MOTION", font=BOLD(20), fill=MUTED, anchor="rm") if False else None
    # ---- grid header ----
    left_files = N_FILES - departed_at(t) + (1 if t >= T_DRAG0 and False else 0)
    left_files = max(0, N_FILES - departed_at(t))
    d.text((44, 358), "Downloads", font=BLACK(34), fill=TEXT, anchor="lm")
    d.text((262, 358), f"{left_files:,} files", font=BOLD(30), fill=MUTED if left_files else GREEN, anchor="lm")
    # ---- by-hand pill ----
    if T_HAND <= t < TY0 + 0.2:
        k = ease_back((t - T_HAND) / 0.35) * (1.0 if t < TY0 - 0.1 else max(0.0, 1 - (t - (TY0 - 0.1)) / 0.3))
        if k > 0.02:
            cx, cy = (GX0 + GX1) / 2, 700
            wd, hd = 760 * min(k, 1.05), 170 * min(k, 1.05)
            d.rounded_rectangle([cx - wd / 2, cy - hd / 2, cx + wd / 2, cy + hd / 2], radius=26, fill=(40, 14, 16), outline=RED, width=4)
            if k > 0.7:
                d.text((cx, cy - 34), "BY HAND", font=BOLD(34), fill=(255, 150, 146), anchor="mm")
                d.text((cx, cy + 22), "3 h 20 min", font=BLACK(86), fill=(255, 255, 255), anchor="mm")
                d.text((cx, cy + 68), "10 seconds per file", font=SEMI(24), fill=(255, 150, 146), anchor="mm")
    # ---- terminal ----
    typed = int(np.searchsorted(CH_T, t, side="right")) if t >= TY0 else 0
    cwid = mono(30).getlength("M")
    ln, col_ = 0, 0
    for k in range(typed):
        ch = CODE[k]
        if ch == "\n": ln += 1; col_ = 0; continue
        if ch != " ": d.text((TX0 + 22 + col_ * cwid, CODE_Y0 + ln * LINE_H), ch, font=mono(30), fill=CODE_COL[k], anchor="lm")
        col_ += 1
    if typed < NCH and t >= TY0 - 0.4 and int(t * 3) % 2 == 0:
        d.rectangle([TX0 + 22 + col_ * cwid, CODE_Y0 + ln * LINE_H - 13, TX0 + 22 + col_ * cwid + 13, CODE_Y0 + ln * LINE_H + 13], fill=(200, 205, 212))
    if t >= TY0 - 0.4 and t < TY0 + 0.05 and typed == 0 and False: pass
    # run button
    if t >= TY1 - 0.1:
        pulse = 1.0 + 0.08 * math.sin(t * 12) if t < T_RUN else 1.0
        press = t >= T_RUN
        d.rounded_rectangle([TX1 - 150, TY_0 + 6, TX1 - 14, TY_0 + 34], radius=14, fill=GREEN if not press else (35, 110, 50))
        d.text((TX1 - 82, TY_0 + 20), "RUN", font=BLACK(int(20 * pulse)), fill=(6, 28, 12), anchor="mm")
    if t >= T_DONE:
        k = ease_out((t - T_DONE) / 0.25)
        d.text((TX0 + 22, CODE_Y0 + 7 * LINE_H), f"Sorted {N_FILES:,} files in {REAL_T:0.2f} s" if REAL_T >= 0.095 else f"Sorted {N_FILES:,} files in {REAL_T:0.3f} s",
               font=mono(30), fill=tc(GREEN, k), anchor="lm")
    # ---- folder counters ----
    cnt = counts_at(t)
    for j in range(6):
        x0 = CARD_X[j]; n = int(cnt[j])
        recent = np.sum((ARRIVE > t - 0.12) & (ARRIVE <= t) & (TYPES == j)) > 0
        d.text((x0 + CW / 2, CY0 + 192), f"{n}", font=BLACK(44), fill=mix(TEXT, TCOL[j], 0.75) if recent else TEXT, anchor="mm")
    # ---- result card ----
    if T_RES <= t < T_CTA:
        rt = t - T_RES; k0 = ease_out(rt / 0.35)
        x0, y0, x1, y1 = 30, 400, 1050, 1350
        d.rounded_rectangle([x0, y0, x1, y1], radius=30, fill=(16, 20, 26), outline=BORDER, width=3)
        if k0 > 0.4:
            d.text((540, 470), "RESULT", font=BOLD(30), fill=MUTED, anchor="mm")
            bw = 900
            kb = ease_out((rt - 0.15) / 0.6)
            d.text((90, 570), "BY HAND", font=BLACK(40), fill=(255, 150, 146), anchor="lm")
            d.text((990, 570), "3 h 20 min", font=mono(36), fill=TEXT, anchor="rm")
            d.rounded_rectangle([90, 610, 90 + bw * kb, 670], radius=16, fill=RED)
            d.text((90, 790), "PYTHON", font=BLACK(40), fill=(140, 230, 160), anchor="lm")
            d.text((990, 790), f"{REAL_T:0.2f} s" if REAL_T >= 0.095 else f"{REAL_T:0.3f} s", font=mono(36), fill=TEXT, anchor="rm")
            d.rounded_rectangle([90, 830, 90 + max(10, bw * (REAL_T / MANUAL_S)) * kb, 890], radius=16, fill=GREEN)
            ks = ease_back((rt - 0.9) / 0.5)
            if ks > 0.01:
                sp_txt = f"{round(SPEEDUP, -3):,.0f}x"; sp_fs = min(190, 190 * 900 / BLACK(190).getlength(sp_txt))
                d.text((540, 1040), sp_txt, font=BLACK(sp_fs * ks), fill=YEL, anchor="mm")
                d.text((540, 1170), "FASTER", font=BLACK(88 * ks), fill=(255, 255, 255), anchor="mm")
            if rt > 1.6:
                d.text((540, 1270), "real run, measured by the code in this video", font=SEMI(26), fill=MUTED, anchor="mm")
    # ---- call to action ----
    if t >= T_CTA:
        ct = t - T_CTA
        x0, y0, x1, y1 = 30, 400, 1050, 1350
        d.rounded_rectangle([x0, y0, x1, y1], radius=30, fill=(16, 20, 26), outline=BORDER, width=3)
        k1 = ease_back(ct / 0.4); k2 = ease_out((ct - 0.5) / 0.4); k3 = ease_back((ct - 1.0) / 0.5)
        if k1 > 0.01:
            d.text((540, 560), "WHAT BORING TASK", font=BLACK(84 * k1), fill=TEXT, anchor="mm")
            d.text((540, 660), "SHOULD PYTHON DO", font=BLACK(84 * k1), fill=TEXT, anchor="mm")
            d.text((540, 760), "NEXT?", font=BLACK(130 * k1), fill=YEL, anchor="mm")
        if k2 > 0: d.text((540, 890), "Comment it below", font=BOLD(46), fill=tc((255, 120, 116), k2), anchor="mm")
        if k3 > 0.01:
            d.text((540, 1050), "PYTHONIC", font=BLACK(110 * k3), fill=BLUE, anchor="mm")
            d.text((540, 1170), "LIFE", font=BLACK(110 * k3), fill=YEL, anchor="mm")
            d.text((540, 1262), "100% Python  •  Follow", font=BOLD(34), fill=tc(MUTED, min(1, k3)), anchor="mm")
    return np.asarray(img)


# ------------------------------------------------------------------ main
if len(sys.argv) > 1 and sys.argv[1] == "still":
    os.makedirs("chk", exist_ok=True)
    for ts in sys.argv[2:]:
        Image.fromarray(render(int(float(ts) * FPS))).save(f"chk/still_{ts}.png")
    sys.exit()

proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "15",
                         "-maxrate", "10M", "-bufsize", "20M", "-pix_fmt", "yuv420p", "-colorspace", "bt709",
                         "-color_primaries", "bt709", "-color_trc", "bt709", "urgent_noaudio.mp4"], stdin=subprocess.PIPE)
F0 = render(0)
for f in range(NF):
    t = f / FPS
    out = F0 if f == 0 else render(f)
    if t > T_LOOP:
        a = min(1.0, (t - T_LOOP) / (TOTAL - T_LOOP - 1 / FPS))
        out = cv2.addWeighted(out, 1 - a, F0, a, 0)
    proc.stdin.write(np.ascontiguousarray(out).tobytes())
proc.stdin.close(); proc.wait()

a = audio[: int(TOTAL * SR)].copy()
a = np.tanh(a / (np.percentile(np.abs(a), 99.9) + 1e-6) * 1.1) * 0.92
nfd = int(0.1 * SR); a[-nfd:] *= np.linspace(1, 0.2, nfd)
with wave.open("urgent_audio.wav", "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR); wf.writeframes((a * 32767).astype(np.int16).tobytes())
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "urgent_noaudio.mp4", "-i", "urgent_audio.wav", "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k", "-shortest", "pythonic_life_urgent_task.mp4"], check=True)
print("done", TOTAL)
