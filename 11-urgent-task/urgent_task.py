"""Pythonic Life — "BOSS: SORT 1,200 FILES IN 10 MINUTES" v2 (YouTube Shorts 1080x1920, ~26 s, loops)
A boss, a stressed worker, two Piper TTS voices, word-by-word captions and the real Python code that fixes it.
The code shown on screen is executed for real: the timing and the folder counts in the video are measured.
Crisp flat art (no glow / haze), YouTube Shorts safe zones respected, high quality encode.
Usage: python3 urgent_task.py /path/en-us-lessac-medium.onnx /path/en-us-ryan-high.onnx
       python3 urgent_task.py <worker.onnx> <boss.onnx> still 0 3 9 -> PNG stills in ./chk
Voices: Piper v0.0.2 release (github.com/rhasspy/piper). pip install piper-tts
"""
import math, os, re, shutil, statistics, subprocess, sys, tempfile, time, wave
from functools import lru_cache
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SR = 1080, 1920, 30, 44100
N_FILES = 1200
VOICE_ME = sys.argv[1] if len(sys.argv) > 1 else "en-us-lessac-medium.onnx"
VOICE_BOSS = sys.argv[2] if len(sys.argv) > 2 else "en-us-ryan-high.onnx"

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
T_TXT = f"{REAL_T:0.2f} s" if REAL_T >= 0.095 else f"{REAL_T:0.3f} s"
print(f"real run: {REAL_T:.3f} s, counts {COUNTS}, manual {MANUAL_S / 3600:.2f} h, speed-up {SPEEDUP:,.0f}x", flush=True)

# ------------------------------------------------------------------ voices
def tts(text, voice, path, length_scale=1.0, pitch=1.0, gain=1.0):
    r = subprocess.run([sys.executable, "-m", "piper", "-m", voice, "-f", path, "--length-scale", str(length_scale),
                        "--noise-scale", "0.75", "--noise-w-scale", "0.9", "--sentence-silence", "0.12"],
                       input=text.encode(), capture_output=True)
    if r.returncode != 0: raise RuntimeError(r.stderr.decode()[-400:])
    with wave.open(path) as wf:
        sr0 = wf.getframerate(); x = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float32) / 32768
    n = int(len(x) * SR / sr0 / pitch)                       # resample to 44.1 kHz (pitch<1 = deeper and slower)
    x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    nz = np.where(np.abs(x) > 0.02 * np.abs(x).max())[0]     # trim silence
    x = x[max(0, nz[0] - int(0.03 * SR)): nz[-1] + int(0.06 * SR)]
    x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9) * 0.11 * gain   # equal loudness
    return x.astype(np.float32)

DONE_TXT = "Done. Under a tenth of a second." if REAL_T < 0.1 else f"Done. {REAL_T:0.1f} seconds."
# who, text, voice settings
LINES = [
    ("boss", "Twelve hundred files. Sorted by type. Ten minutes!", dict(length_scale=0.9, pitch=0.93, gain=1.15)),
    ("me", "Ten minutes? By hand, that's three hours!", dict(length_scale=0.88)),
    ("me", "Or... I use Python. It's very easy.", dict(length_scale=1.05)),
    ("me", "Seven lines. Run!", dict(length_scale=1.0)),
    ("me", DONE_TXT, dict(length_scale=1.0)),
    ("boss", "Great. Now do a million.", dict(length_scale=0.95, pitch=0.93, gain=1.1)),
    ("me", "Very easy.", dict(length_scale=1.12)),
    ("me", "What should Python do next? Comment below!", dict(length_scale=1.0)),
]
os.makedirs("tts_tmp", exist_ok=True)
WAVS = [tts(txt, VOICE_BOSS if who == "boss" else VOICE_ME, f"tts_tmp/l{i}.wav", **kw) for i, (who, txt, kw) in enumerate(LINES)]
DUR = [len(w) / SR for w in WAVS]
print("voice lines:", [round(d, 2) for d in DUR], flush=True)

# ------------------------------------------------------------------ timeline (seconds)
START = [0.0] * len(LINES)
START[0] = 0.05
START[1] = START[0] + DUR[0] + 0.28
START[2] = START[1] + DUR[1] + 0.30
START[3] = START[2] + DUR[2] + 0.18
T_RUN = START[3] + DUR[3] - 0.30                              # the "Run!" word
T_RUN_END = T_RUN + 1.8
START[4] = T_RUN + 1.55
START[5] = START[4] + DUR[4] + 0.35
START[6] = START[5] + DUR[5] + 0.22
START[7] = START[6] + DUR[6] + 0.30
T_DONE = START[4] - 0.05
T_RES = START[4] + 0.25
T_CTA = START[7] - 0.15
TOTAL = START[7] + DUR[7] + 1.0
T_LOOP = TOTAL - 0.6
NF = int(TOTAL * FPS)
T_HAND, T_DRAG0, T_DRAG1 = START[1] + 0.35, START[1] + 0.3, START[1] + DUR[1] + 0.25
TY0, TY1 = START[2] + 0.9, START[3] + DUR[3] * 0.55
print(f"total {TOTAL:.1f} s, run at {T_RUN:.1f}", flush=True)

# mouth openness per frame from the voice envelope
def make_env(w):
    n = int(len(w) / SR * FPS) + 2; e = np.zeros(n)
    for k in range(n):
        seg = w[int(k / FPS * SR): int((k + 1) / FPS * SR)]
        e[k] = np.sqrt(np.mean(seg ** 2)) if len(seg) else 0
    e = e / (np.percentile(e, 95) + 1e-9)
    return np.clip(np.convolve(e, [0.25, 0.5, 0.25], mode="same"), 0, 1)
ENV = [make_env(w) for w in WAVS]

def mouth(t, who):
    for i, (wh, _, _) in enumerate(LINES):
        if wh == who and START[i] <= t < START[i] + DUR[i]:
            k = int((t - START[i]) * FPS)
            return float(ENV[i][min(k, len(ENV[i]) - 1)])
    return 0.0

def speaking_line(t):
    for i in range(len(LINES)):
        if START[i] - 0.02 <= t < START[i] + DUR[i] + 0.35: return i
    return None

# word timings for captions
WORDS = []
for i, (who, txt, _) in enumerate(LINES):
    ws = txt.split(); wts = np.array([len(re.sub(r"\W", "", x)) + (3 if x[-1] in ".?!" else 1 if x[-1] == "," else 0) + 1 for x in ws], float)
    cum = np.concatenate([[0], np.cumsum(wts)]) / wts.sum()
    WORDS.append([(ws[k], START[i] + cum[k] * DUR[i], START[i] + cum[k + 1] * DUR[i]) for k in range(len(ws))])

# ------------------------------------------------------------------ style
BG, PANEL, PANEL2, BORDER = (13, 17, 23), (22, 27, 34), (30, 36, 44), (48, 54, 61)
TEXT, MUTED = (230, 237, 243), (139, 148, 158)
RED, GREEN, BLUE, YEL, PUR, GRAY = (248, 81, 73), (63, 185, 80), (88, 166, 255), (210, 153, 34), (188, 140, 255), (139, 148, 158)
TCOL = [RED, GREEN, BLUE, YEL, PUR, GRAY]
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
SC_Y, SW, SH = 100, 1080, 430                                  # scene strip
CAP_Y = 590                                                    # caption centre
GRID_FRAME = (26, 712, 1054, 1108)
GX0, GX1, GY0, GY1 = 52, 992, 732, 1088
COLS_, ROWS_ = 50, 24
IW, IH = 12, 15
TX0, TX1, TY_0 = 30, 930, 1122
LINE_H = 28; CODE_Y0 = TY_0 + 34 + 15; TY_1 = TY_0 + 34 + 8 * LINE_H + 18
CY0, CY1 = 1408, 1596
CW, CGAP = 137, 15.6
CARD_X = [30 + i * (CW + CGAP) for i in range(6)]
CARD_CX = 515

# ------------------------------------------------------------------ files
gx = np.linspace(GX0, GX1, COLS_); gy = np.linspace(GY0, GY1, ROWS_)
GRID = np.array([(gx[c], gy[r]) for r in range(ROWS_) for c in range(COLS_)])[:N_FILES]
GRID = GRID[rng.permutation(N_FILES)] + rng.uniform(-2.0, 2.0, (N_FILES, 2))
PHASE = rng.uniform(0, 6.28, N_FILES)
HAND = int(np.argmin([(GRID[i, 0] - 80) ** 2 + (GRID[i, 1] - 1060) ** 2 if TYPES[i] == 0 else 1e12 for i in range(N_FILES)]))
END = np.zeros((N_FILES, 2)); DELAY = np.zeros(N_FILES); CTRL = np.zeros((N_FILES, 2))
for i in range(N_FILES):
    j = TYPES[i]; cx = CARD_X[j] + CW / 2
    END[i] = (cx + rng.uniform(-14, 14), CY0 + 62 + rng.uniform(-8, 8))
    DELAY[i] = (1 - (GRID[i, 1] - GY0) / (GY1 - GY0)) * 0.5 + rng.uniform(0, 0.5)
    CTRL[i] = (GRID[i, 0] + (END[i, 0] - GRID[i, 0]) * 0.2 + rng.uniform(-30, 30), GRID[i, 1] - 50 - rng.uniform(0, 40))
FLIGHT = 0.5
LAUNCH = T_RUN + DELAY; ARRIVE = LAUNCH + FLIGHT
LAUNCH[HAND] = 1e9; ARRIVE[HAND] = T_DRAG1

def pos_fly(i, t):
    u = ease_io((t - LAUNCH[i]) / FLIGHT)
    p0, p1, p2 = GRID[i], CTRL[i], END[i]
    return ((1 - u) ** 2 * p0[0] + 2 * (1 - u) * u * p1[0] + u * u * p2[0], (1 - u) ** 2 * p0[1] + 2 * (1 - u) * u * p1[1] + u * u * p2[1])

def counts_at(t):
    done = ARRIVE <= t
    return np.array([int(np.sum(done & (TYPES == j))) for j in range(6)])

def departed_at(t):
    return int(np.sum(LAUNCH <= t)) + (1 if t >= T_DRAG0 else 0)

def make_icon(col):
    s = 6; w, h = IW * s, IH * s
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im); fold = 4 * s
    d.polygon([(0, 2 * s), (2 * s, 0), (w - fold, 0), (w, fold), (w, h - 2 * s), (w - 2 * s, h), (2 * s, h), (0, h - 2 * s)], fill=col + (255,))
    d.polygon([(w - fold, 0), (w - fold, fold), (w, fold)], fill=tuple(int(v * 0.55) for v in col) + (255,))
    for k in range(2): d.rectangle([2.5 * s, (8 + 3 * k) * s, w - 3 * s, (9 + 3 * k) * s], fill=(255, 255, 255, 190))
    a = np.asarray(im.resize((IW, IH), Image.LANCZOS)).astype(np.float32)
    return a[..., :3], a[..., 3:4] / 255.0
ICONS = [make_icon(c) for c in TCOL]

def blit(arr, spr, x, y):
    rgb, a = spr; h, w = a.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1: return
    reg = arr[y0:y1, x0:x1]
    reg[:] = (reg + (rgb[y0 - y:y1 - y, x0 - x:x1 - x] - reg) * a[y0 - y:y1 - y, x0 - x:x1 - x]).astype(np.uint8)

# ------------------------------------------------------------------ office scene (drawn at 2x, then reduced = anti-aliased)
class Dr:
    def __init__(s, img): s.d = ImageDraw.Draw(img)
    def ell(s, cx, cy, rx, ry, fill, outline=None, w=0):
        s.d.ellipse([(cx - rx) * 2, (cy - ry) * 2, (cx + rx) * 2, (cy + ry) * 2], fill=fill, outline=outline, width=int(w * 2))
    def poly(s, pts, fill): s.d.polygon([(x * 2, y * 2) for x, y in pts], fill=fill)
    def rr(s, x0, y0, x1, y1, r, fill, outline=None, w=0):
        s.d.rounded_rectangle([x0 * 2, y0 * 2, x1 * 2, y1 * 2], radius=r * 2, fill=fill, outline=outline, width=int(w * 2))
    def line(s, pts, fill, w):
        s.d.line([(x * 2, y * 2) for x, y in pts], fill=fill, width=max(1, int(w * 2)), joint="curve")
        for x, y in pts: s.ell(x, y, w / 2, w / 2, fill)

SKIN_W, SKIN_B = (231, 175, 143), (235, 192, 160)
HAIR_W, SUIT, SHIRT_W = (64, 40, 36), (46, 54, 74), (88, 166, 255)

def make_scene_bg():
    h2 = SH * 2
    yy = np.linspace(0, 1, h2)[:, None, None]
    top, bot = np.array([22, 30, 52]), np.array([36, 48, 78])
    wall = (top + (bot - top) * yy) * np.ones((1, SW * 2, 1))
    img = Image.fromarray(wall.astype(np.uint8)); d = Dr(img)
    # floor
    d.d.rectangle([0, 345 * 2, SW * 2, SH * 2], fill=(34, 30, 44)); d.d.rectangle([0, 340 * 2, SW * 2, 348 * 2], fill=(52, 48, 66))
    # window with a night skyline
    d.rr(466, 20, 704, 204, 8, (176, 184, 200))
    sky = Image.fromarray((np.stack([np.linspace(16, 70, 160)[:, None] * np.ones((1, 220)), np.linspace(24, 50, 160)[:, None] * np.ones((1, 220)),
                                     np.linspace(62, 120, 160)[:, None] * np.ones((1, 220))], 2)).astype(np.uint8)).resize((440, 320))
    img.paste(sky, (478 * 2, 32 * 2)); d = Dr(img)
    sr = np.random.default_rng(2)
    for k in range(9):
        bx = 478 + k * 25 + sr.integers(-3, 3); bh = int(sr.integers(40, 110))
        d.d.rectangle([bx * 2, (192 - bh) * 2, (bx + 22) * 2, 192 * 2], fill=(14, 18, 34))
        for wy in range(192 - bh + 6, 188, 11):
            for wx in range(bx + 4, bx + 20, 8):
                if sr.random() < 0.45: d.d.rectangle([wx * 2, wy * 2, (wx + 3) * 2, (wy + 5) * 2], fill=(250, 214, 120))
    d.d.rectangle([(587 - 2) * 2, 32 * 2, (587 + 2) * 2, 192 * 2], fill=(176, 184, 200)); d.d.rectangle([478 * 2, (112 - 2) * 2, 696 * 2, (112 + 2) * 2], fill=(176, 184, 200))
    # clock face
    d.ell(150, 84, 46, 46, (60, 66, 84)); d.ell(150, 84, 41, 41, (240, 242, 246))
    for k in range(12):
        a = k * math.pi / 6; d.line([(150 + 33 * math.sin(a), 84 - 33 * math.cos(a)), (150 + 38 * math.sin(a), 84 - 38 * math.cos(a))], (60, 66, 84), 2.5)
    return img
SCENE_BG = make_scene_bg()

def make_desk_layer():
    img = Image.new("RGBA", (SW * 2, SH * 2), (0, 0, 0, 0)); d = Dr(img)
    d.rr(0, 296, 650, 316, 4, (156, 112, 72)); d.d.rectangle([0, 316 * 2, 650 * 2, SH * 2], fill=(104, 72, 46))
    d.d.rectangle([0, 316 * 2, 650 * 2, 321 * 2], fill=(86, 58, 38))
    for x0 in (30, 240, 450): d.rr(x0, 336, x0 + 170, 410, 6, (118, 84, 56), (86, 58, 38), 3); d.rr(x0 + 62, 366, x0 + 108, 374, 4, (190, 150, 100))
    # laptop lid seen from the back + sticker
    d.rr(206, 226, 394, 296, 7, (78, 86, 104)); d.rr(210, 230, 390, 292, 5, (92, 100, 120))
    d.ell(300, 262, 15, 15, (255, 212, 59)); d.ell(300, 262, 9, 9, (55, 160, 235))
    d.rr(190, 292, 410, 300, 3, (70, 76, 92))
    # paper stack + plant
    for k in range(4): d.rr(96 + k * 2, 284 - k * 3, 168 + k * 2, 296 - k * 3, 2, (236, 238, 244), (170, 176, 190), 1.5)
    d.rr(34, 262, 70, 296, 4, (180, 96, 70)); d.rr(30, 258, 74, 268, 3, (200, 112, 82))
    for ang, ln in ((-50, 38), (-20, 46), (15, 44), (48, 36), (0, 52)):
        a = math.radians(ang - 90); d.ell(52 + math.cos(a) * ln * 0.5, 262 + math.sin(a) * ln * 0.5, 9, ln * 0.5, (60, 160, 90))
    return img
DESK = make_desk_layer()

def eyes(d, cx, cy, sep, kind, t, look=(0, 0), blink_phase=0.0):
    blink = (t + blink_phase) % 3.1 < 0.11
    for s in (-1, 1):
        ex = cx + s * sep
        if blink or kind == "closed":
            d.line([(ex - 9, cy), (ex, cy + 3), (ex + 9, cy)], (50, 30, 30), 2.5)
        elif kind == "happy":
            d.line([(ex - 9, cy + 3), (ex, cy - 5), (ex + 9, cy + 3)], (50, 30, 30), 3)
        elif kind == "wide":
            d.ell(ex, cy, 13, 15, (255, 255, 255), (60, 40, 40), 1.5); d.ell(ex + look[0], cy + look[1], 3.5, 3.5, (30, 20, 20))
        else:
            d.ell(ex, cy, 10, 11, (255, 255, 255), (60, 40, 40), 1.2); d.ell(ex + look[0], cy + look[1], 5, 5.5, (30, 20, 20)); d.ell(ex + look[0] + 1.5, cy + look[1] - 2, 1.6, 1.6, (255, 255, 255))

def mouth_draw(d, x, y, a, mood, wide=1.0):
    if mood == "smile" and a < 0.12:
        pts = [(x + k * 3.3 * wide, y - 2 + (k * k) * 0.18) for k in range(-6, 7)]
        d.line(pts, (150, 50, 60), 3)
    elif mood == "smirk" and a < 0.12:
        d.line([(x - 14, y + 3), (x + 4, y + 5), (x + 16, y - 3)], (140, 50, 50), 3.5)
    elif mood == "o":
        d.ell(x, y + 4, 12, 15, (70, 20, 30)); d.ell(x, y + 10, 7, 4, (200, 90, 100))
    else:
        rx, ry = (11 + 7 * a) * wide, 2.5 + 15 * a
        d.ell(x, y + ry * 0.5, rx, ry, (70, 20, 30))
        if a > 0.3: d.d.rectangle([(x - rx * 0.7) * 2, (y - 1) * 2, (x + rx * 0.7) * 2, (y + 4) * 2], fill=(246, 246, 246)); d.ell(x, y + ry * 0.9, rx * 0.5, ry * 0.35, (200, 90, 100))

def draw_worker(img, t, mood, ma):
    d = Dr(img)
    shake = 2.6 * math.sin(t * 38) if mood == "panic" else 0.0
    bob = 2 * math.sin(t * 3) if mood in ("sip", "happy") else 0
    wx, wy = 300 + shake, 150 + bob
    d.rr(246, 190, 354, 310, 18, (36, 42, 62))                                   # chair back
    cloth = SHIRT_W
    d.poly([(wx - 72, wy + 62), (wx + 72, wy + 62), (wx + 88, 300), (wx - 88, 300)], cloth)   # torso
    d.rr(wx - 15, wy + 38, wx + 15, wy + 68, 8, SKIN_W)                          # neck
    d.ell(wx, wy - 2, 56, 62, HAIR_W)                                            # back hair
    d.ell(wx, wy - 66, 22, 22, HAIR_W)                                           # bun
    d.ell(wx, wy + 4, 46, 52, SKIN_W)                                            # face
    d.poly([(wx - 47, wy - 6), (wx - 40, wy - 40), (wx, wy - 54), (wx + 42, wy - 38), (wx + 47, wy - 6), (wx + 22, wy - 24), (wx - 4, wy - 18), (wx - 26, wy - 26)], HAIR_W)
    # arms
    sl = (70, 140, 230)
    if mood == "panic":
        for s in (-1, 1):
            d.line([(wx + s * 66, wy + 72), (wx + s * 108, wy + 120), (wx + s * 54, wy + 10)], sl, 22)
            d.ell(wx + s * 52, wy + 8, 13, 13, SKIN_W)
    elif mood == "happy":
        for s in (-1, 1):
            d.line([(wx + s * 66, wy + 72), (wx + s * 112, wy + 20), (wx + s * 120, wy - 52 + 5 * math.sin(t * 14 + s))], sl, 22)
            d.ell(wx + s * 120, wy - 58 + 5 * math.sin(t * 14 + s), 13, 13, SKIN_W)
    elif mood == "sip":
        d.line([(wx - 66, wy + 72), (wx - 82, wy + 140), (wx - 40, wy + 150)], sl, 22)
        d.line([(wx + 66, wy + 72), (wx + 104, wy + 118), (wx + 50, wy + 52 + 2 * math.sin(t * 2))], sl, 22)
        mx, my = wx + 44, wy + 46 + 2 * math.sin(t * 2)
        d.ell(mx + 6, my + 4, 12, 12, SKIN_W); d.rr(mx - 8, my - 16, mx + 20, my + 16, 4, (248, 248, 250), (190, 190, 200), 1.5)
        d.ell(mx + 24, my, 8, 9, (0, 0, 0, 0) if False else (248, 248, 250), (190, 190, 200), 3)
        for k in range(3): d.line([(mx + 2 + k * 6, my - 24), (mx + 6 + k * 6, my - 34 - 3 * math.sin(t * 4 + k)), (mx + 2 + k * 6, my - 44)], (230, 232, 240), 2)
    else:
        jx = 4 * math.sin(t * 22) if mood == "confident" else 0
        for s in (-1, 1):
            d.line([(wx + s * 66, wy + 72), (wx + s * 90, wy + 130), (wx + s * (44 + jx * s), 282)], sl, 22)
    # face
    look = (3, 1) if mood == "confident" else (0, -3) if mood == "idea" else (0, 0)
    kind = {"panic": "wide", "happy": "happy", "sip": "closed"}.get(mood, "open")
    eyes(d, wx, wy + 0, 20, kind, t, look)
    if mood == "panic":
        d.line([(wx - 32, wy - 22), (wx - 10, wy - 16)], HAIR_W, 3.5); d.line([(wx + 32, wy - 22), (wx + 10, wy - 16)], HAIR_W, 3.5)
    elif mood == "idea":
        d.line([(wx - 32, wy - 20), (wx - 10, wy - 24)], HAIR_W, 3.5); d.line([(wx + 32, wy - 24), (wx + 10, wy - 28)], HAIR_W, 3.5)
    else:
        d.line([(wx - 32, wy - 17), (wx - 10, wy - 19)], HAIR_W, 3.5); d.line([(wx + 32, wy - 17), (wx + 10, wy - 19)], HAIR_W, 3.5)
    if mood in ("happy", "sip", "confident"): d.ell(wx - 30, wy + 22, 8, 5, (240, 140, 140)); d.ell(wx + 30, wy + 22, 8, 5, (240, 140, 140))
    mouth_draw(d, wx, wy + 33, ma if mood != "panic" or ma > 0.05 else 0.12 + 0.1 * math.sin(t * 30), "smile" if mood in ("happy", "sip", "idea") else "smirk" if mood == "confident" else "open")
    if mood == "panic":                                                             # sweat drops
        for k in range(3):
            ph = (t * 1.3 + k * 0.37) % 1.0
            sx_ = wx + (-62 if k % 2 else 64) + k * 3; sy_ = wy - 40 + ph * 80
            d.poly([(sx_, sy_ - 10), (sx_ - 6, sy_ + 2), (sx_ + 6, sy_ + 2)], (110, 190, 255)); d.ell(sx_, sy_ + 3, 6, 6, (110, 190, 255))
    if mood == "idea":                                                              # light bulb
        pulse = 0.6 + 0.4 * math.sin(t * 14)
        d.ell(wx, wy - 118, 30 + 6 * pulse, 30 + 6 * pulse, (255, 230, 120, 60) if False else (92, 80, 40))
        d.ell(wx, wy - 118, 20, 20, (255, 225, 90)); d.rr(wx - 8, wy - 100, wx + 8, wy - 90, 3, (160, 160, 170))
        for k in range(8):
            a = k * math.pi / 4 + t; d.line([(wx + 28 * math.cos(a), wy - 118 + 28 * math.sin(a)), (wx + 38 * math.cos(a), wy - 118 + 38 * math.sin(a))], (255, 225, 90), 3)

def draw_boss(img, t, mood, ma):
    d = Dr(img)
    shake = 2.2 * math.sin(t * 33) if mood == "rage" and ma > 0.05 else 0.0
    lean = 10 if mood == "shock" else 0
    bx, by = 840 + shake + lean, 118
    d.poly([(bx - 78, by + 70), (bx + 78, by + 70), (bx + 104, SH), (bx - 104, SH)], SUIT)                 # suit
    d.poly([(bx - 26, by + 66), (bx + 26, by + 66), (bx, by + 140)], (240, 242, 246))                        # shirt
    d.poly([(bx - 8, by + 72), (bx + 8, by + 72), (bx + 14, by + 150), (bx, by + 172), (bx - 14, by + 150)], RED)   # tie
    d.line([(bx - 26, by + 66), (bx - 4, by + 142)], (30, 36, 54), 5); d.line([(bx + 26, by + 66), (bx + 4, by + 142)], (30, 36, 54), 5)
    # arms: left (towards worker) pointing / limp, right on hip
    if mood == "shock":
        d.line([(bx - 78, by + 82), (bx - 100, by + 150), (bx - 96, by + 220)], SUIT, 28); d.ell(bx - 96, by + 224, 13, 13, SKIN_B)
    else:
        j = 5 * math.sin(t * 26) if mood == "rage" and ma > 0.05 else 0
        d.line([(bx - 78, by + 84), (bx - 150, by + 94 + j), (bx - 214, by + 82 + j)], SUIT, 28)
        d.ell(bx - 222, by + 82 + j, 14, 13, SKIN_B); d.line([(bx - 228, by + 80 + j), (bx - 258, by + 76 + j)], SKIN_B, 9)
    d.line([(bx + 78, by + 84), (bx + 124, by + 150), (bx + 70, by + 196)], SUIT, 28); d.ell(bx + 66, by + 198, 13, 13, SKIN_B)
    # head
    d.ell(bx - 48, by + 4, 12, 22, (150, 150, 158)); d.ell(bx + 48, by + 4, 12, 22, (150, 150, 158))
    d.ell(bx, by + 6, 50, 56, SKIN_B)
    d.ell(bx - 3, by - 38, 22, 9, (255, 226, 205))                                                            # shiny bald spot
    kind = "wide" if mood == "shock" else "open"
    eyes(d, bx, by + 4, 21, kind, t, (-3, 0), 1.3)
    if mood == "rage":
        d.line([(bx - 38, by - 20), (bx - 9, by - 8)], (70, 70, 80), 5.5); d.line([(bx + 38, by - 20), (bx + 9, by - 8)], (70, 70, 80), 5.5)
    elif mood == "shock":
        d.line([(bx - 36, by - 28), (bx - 10, by - 30)], (70, 70, 80), 5); d.line([(bx + 36, by - 28), (bx + 10, by - 30)], (70, 70, 80), 5)
    else:
        d.line([(bx - 36, by - 16), (bx - 9, by - 17)], (70, 70, 80), 5.5); d.line([(bx + 36, by - 24), (bx + 9, by - 17)], (70, 70, 80), 5.5)
    d.ell(bx - 9, by + 28, 12, 6, (90, 90, 100)); d.ell(bx + 9, by + 28, 12, 6, (90, 90, 100))             # moustache
    mouth_draw(d, bx, by + 40, 0.95 if mood == "shock" and False else ma, "o" if mood == "shock" else "smirk" if mood == "smug" else "open", 1.15)
    if mood == "rage" and ma > 0.05:                                                                         # anger marks and steam
        pulse = 1 + 0.25 * math.sin(t * 22)
        for s in (-1, 1):
            cx_, cy_ = bx + s * 60, by - 56
            for a in (0.6, 2.2): d.line([(cx_ + math.cos(a) * 4 * pulse, cy_ + math.sin(a) * 4 * pulse), (cx_ + math.cos(a) * 15 * pulse, cy_ + math.sin(a) * 15 * pulse)], (255, 70, 60), 4)
        for k in range(3):
            ph = (t * 1.6 + k * 0.33) % 1.0
            d.ell(bx - 70 + k * 70, by - 76 - ph * 36, 7 + ph * 6, 7 + ph * 6, (255, 255, 255))
    if mood == "shock":
        d.poly([(bx + 66, by - 30), (bx + 60, by - 18), (bx + 72, by - 18)], (110, 190, 255)); d.ell(bx + 66, by - 15, 6, 6, (110, 190, 255))

def render_scene(t, wm, bm, wma, bma):
    img = SCENE_BG.copy()
    d = Dr(img)
    # second hand of the clock (urgency)
    a = (t % 60) / 60 * 2 * math.pi * 6 if False else t * 2 * math.pi / 2.0
    ang_m = math.radians(-6 + (0 if True else 0)); ang_h = math.radians(-35)
    d.line([(150, 84), (150 + 24 * math.sin(ang_h), 84 - 24 * math.cos(ang_h))], (40, 40, 52), 4)
    d.line([(150, 84), (150 + 34 * math.sin(ang_m), 84 - 34 * math.cos(ang_m))], (40, 40, 52), 3)
    d.line([(150, 84), (150 + 36 * math.sin(a), 84 - 36 * math.cos(a))], RED, 2)
    draw_boss(img, t, bm, bma)
    draw_worker(img, t, wm, wma)
    img = img.convert("RGBA"); img.alpha_composite(DESK); img = img.convert("RGB")
    if wm not in ("sip",):                                                                                  # mug on the desk
        d2 = Dr(img); d2.rr(540, 270, 574, 298, 4, (248, 248, 250), (190, 190, 200), 1.5); d2.ell(578, 283, 8, 9, (248, 248, 250), (190, 190, 200), 3)
        d2.d.rectangle([541 * 2, 272 * 2, 573 * 2, 276 * 2], fill=(110, 70, 50))
    return np.asarray(img.resize((SW, SH), Image.LANCZOS))

def moods(t):
    wm = "panic"
    if t >= START[2]: wm = "idea"
    if t >= START[2] + 1.5: wm = "confident"
    if t >= START[4] - 0.1: wm = "happy"
    if t >= START[6] - 0.05: wm = "sip"
    bm = "rage"
    if t >= T_RUN: bm = "shock"
    if t >= START[5] - 0.1: bm = "smug"
    return wm, bm

# ------------------------------------------------------------------ static base for the "screen" part
def folder_glyph(d, cx, cy, col, s=1.0):
    w, h = 78 * s, 54 * s
    d.rounded_rectangle([cx - w / 2, cy - h / 2 - 8 * s, cx - w / 2 + 34 * s, cy - h / 2 + 6 * s], radius=6 * s, fill=tc(col, 0.8))
    d.rounded_rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], radius=9 * s, fill=col)
    d.rounded_rectangle([cx - w / 2, cy - h / 2 + 12 * s, cx + w / 2, cy + h / 2], radius=9 * s, fill=tc(col, 0.88))

def make_base():
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.rounded_rectangle(GRID_FRAME, radius=20, fill=PANEL, outline=BORDER, width=2)
    d.rounded_rectangle([TX0, TY_0, TX1, TY_1], radius=16, fill=(10, 13, 18), outline=BORDER, width=2)
    d.rounded_rectangle([TX0, TY_0, TX1, TY_0 + 34], radius=16, fill=PANEL2); d.rectangle([TX0, TY_0 + 18, TX1, TY_0 + 34], fill=PANEL2)
    for k, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]): d.ellipse([TX0 + 16 + k * 24, TY_0 + 11, TX0 + 30 + k * 24, TY_0 + 25], fill=c)
    d.text((TX0 + 112, TY_0 + 17), "sort.py", font=SEMI(20), fill=MUTED, anchor="lm")
    for j in range(6):
        x0 = CARD_X[j]
        d.rounded_rectangle([x0, CY0, x0 + CW, CY1], radius=18, fill=PANEL, outline=tc(TCOL[j], 0.55), width=3)
        folder_glyph(d, x0 + CW / 2, CY0 + 62, TCOL[j], 0.85)
        d.text((x0 + CW / 2, CY0 + 124), FOLDERS[j], font=BOLD(24), fill=TEXT, anchor="mm")
    return np.asarray(im).copy()
BASE = make_base()

CODE_COL = [TEXT] * len(CODE)
for m in re.finditer(r'(".*?")|\b(import|for|in|as)\b|(\b\w+(?=\())|(\b\w+(?=\.\w+\())', CODE):
    col = (165, 214, 255) if m.group(1) else (255, 123, 114) if m.group(2) else (210, 168, 255)
    for k in range(m.start(), m.end()): CODE_COL[k] = col
NCH = len(CODE)
w_ = np.array([4.0 if c == "\n" else 1.0 for c in CODE]); cum = np.cumsum(w_) / w_.sum()
CH_T = np.maximum.accumulate(TY0 + (TY1 - TY0) * np.clip(cum + np.random.default_rng(5).uniform(-0.002, 0.002, NCH), 0, 1))

# ------------------------------------------------------------------ audio
NS = int((TOTAL + 1) * SR)
sfx = np.zeros(NS, np.float32); music = np.zeros(NS, np.float32); vo = np.zeros(NS, np.float32)
def put(buf, t0, sig):
    s = int(t0 * SR); n = min(len(sig), NS - s)
    if n > 0 and s >= 0: buf[s:s + n] += sig[:n].astype(np.float32)
def tone(buf, t0, f, dur, amp, decay=7.0, harm=0.3):
    tt = np.arange(int(dur * SR)) / SR
    put(buf, t0, (np.sin(2 * np.pi * f * tt) + harm * np.sin(4 * np.pi * f * tt)) * np.exp(-tt * decay) * np.minimum(1, tt * 800) * amp)
def click(t0, amp, f=1700, seed=None):
    tt = np.arange(int(0.025 * SR)) / SR
    nz = np.random.default_rng(seed if seed is not None else int(t0 * 1e4) % 99991).standard_normal(len(tt))
    put(sfx, t0, (nz * 0.5 + np.sin(2 * np.pi * f * tt)) * np.exp(-tt * 160) * amp)
def kick(t0, amp=0.30):
    tt = np.arange(int(0.25 * SR)) / SR
    put(music, t0, np.sin(2 * np.pi * np.cumsum(120 * np.exp(-tt * 26) + 44) / SR) * np.exp(-tt * 10) * amp)
def hat(t0, a=0.05):
    tt = np.arange(int(0.05 * SR)) / SR
    nz = np.diff(np.random.default_rng(int(t0 * 1e3) % 99991).standard_normal(len(tt) + 1))
    put(music, t0, nz * np.exp(-tt * 70) * a)
def pad(t0, notes, dur, amp):
    tt = np.arange(int(dur * SR)) / SR
    env = np.minimum(1, tt / 0.35) * np.minimum(1, (dur - tt) / 0.5)
    sig = sum(np.sin(2 * np.pi * f * tt) + 0.35 * np.sin(2 * np.pi * 2 * f * tt + 0.7) + 0.12 * np.sin(2 * np.pi * 3 * f * tt) for f in notes)
    put(music, t0, sig * env * amp)

for i, w in enumerate(WAVS): put(vo, START[i], w)
# gentle lo-fi music bed: Am - F - C - G, soft pulse and a plucked arpeggio
bpm = 92; beat = 60 / bpm; bar = 4 * beat
CHORDS = [([220.0, 261.6, 329.6], [440.0, 523.3, 659.3, 523.3]), ([174.6, 220.0, 261.6], [349.2, 440.0, 523.3, 440.0]),
          ([196.0, 261.6, 329.6], [392.0, 523.3, 659.3, 523.3]), ([196.0, 246.9, 293.7], [392.0, 493.9, 587.3, 493.9])]
tb, ci = 0.0, 0
while tb < TOTAL + 0.5:
    notes, arp = CHORDS[ci % 4]
    pad(tb, notes, bar + 0.2, 0.028)
    for k in range(8):
        tone(music, tb + k * beat / 2, arp[k % 4] * (2 if k % 8 >= 4 else 1), 0.5, 0.030, 6.5, 0.2)
    for k in range(4):
        kick(tb + k * beat, 0.26 if tb < T_RUN else 0.34); hat(tb + k * beat + beat / 2, 0.035 if tb < T_RUN else 0.055)
    if tb >= T_RUN:
        for k in range(8): hat(tb + k * beat / 2 + beat / 4, 0.03)
    tb += bar; ci += 1
for k in range(0, int(T_RUN) + 1): click(0.3 + k * 1.0, 0.07, 2600, seed=k)                              # deadline clock
tone(sfx, T_HAND, 233, 0.5, 0.10, 4); tone(sfx, T_HAND + 0.5, 208, 0.7, 0.10, 3)
for k in range(NCH):
    if CODE[k] != " ": click(CH_T[k], 0.05 + 0.03 * (k % 3), 1200 + 600 * (k % 4), seed=100 + k)
tone(sfx, T_RUN - 0.03, 587, 0.2, 0.18, 12)
n_w = int((T_RUN_END - T_RUN + 0.5) * SR); tw = np.arange(n_w) / SR
nz = np.convolve(np.random.default_rng(3).standard_normal(n_w), np.ones(14) / 14, mode="same") * 14 ** 0.5
put(sfx, T_RUN, nz * np.sin(np.pi * np.clip(tw / (n_w / SR), 0, 1)) ** 1.5 * 0.10)
PENTA = [523.3, 587.3, 659.3, 784.0, 880.0, 1046.5]
arr_sorted = np.sort(ARRIVE[ARRIVE < 1e8])
for k in range(0, len(arr_sorted), 7): tone(sfx, arr_sorted[k], PENTA[(k // 7) % 6], 0.12, 0.025, 22)
for k_, fq in enumerate([523.3, 659.3, 784.0, 1046.5]): tone(sfx, T_DONE + 0.07 * k_, fq, 1.3, 0.16, 3.5)
tone(sfx, START[5] - 0.2, 196, 0.5, 0.12, 6)                                                            # boss sting
for k_, fq in enumerate([261.6, 329.6, 392.0, 523.3]): tone(sfx, T_CTA + 0.1 * k_, fq, 2.0, 0.10, 2.5)
# duck the music (and sfx a little) under the voice
env_v = np.abs(vo); win = int(0.06 * SR)
env_v = np.convolve(env_v, np.ones(win) / win, mode="same")
duck = 1 - 0.72 * np.clip(env_v / 0.06, 0, 1)
duck = np.convolve(duck, np.ones(int(0.12 * SR)) / int(0.12 * SR), mode="same")
audio = vo * 1.0 + music * duck * 1.0 + sfx * (0.55 + 0.45 * duck)

# ------------------------------------------------------------------ frame renderer
def fmt_t(sec):
    sec = max(0.0, sec); return f"{int(sec // 60):02d}:{int(sec % 60):02d}"

def wrap_words(items, fnt, maxw, d):
    lines, cur, w = [], [], 0
    for it in items:
        ww = d.textlength(it[0] + " ", font=fnt)
        if cur and w + ww > maxw: lines.append(cur); cur, w = [], 0
        cur.append(it); w += ww
    if cur: lines.append(cur)
    return lines

def render(f):
    t = f / FPS
    arr = BASE.copy()
    wm, bm = moods(t)
    cur_i = speaking_line(t)
    wma = mouth(t, "me"); bma = mouth(t, "boss")
    arr[SC_Y:SC_Y + SH, 0:SW] = render_scene(t, wm, bm, wma, bma)
    # ---- files ----
    tr = 1.2 if t < T_RUN else 0.0
    for i in range(N_FILES):
        if i == HAND and t >= T_DRAG0: continue
        if LAUNCH[i] <= t: continue
        x = GRID[i, 0] + tr * math.sin(t * 13 + PHASE[i]); y = GRID[i, 1] + tr * math.cos(t * 11 + PHASE[i] * 1.3)
        blit(arr, ICONS[TYPES[i]], int(round(x - IW / 2)), int(round(y - IH / 2)))
    cur = None
    if T_DRAG0 <= t < T_DRAG1 + 0.1:
        u = ease_io((t - T_DRAG0) / (T_DRAG1 - T_DRAG0))
        sx, sy = GRID[HAND]; ex, ey = CARD_X[0] + CW / 2, CY0 + 62
        x = sx + (ex - sx) * u; y = sy + (ey - sy) * u
        if t < T_DRAG1: blit(arr, ICONS[0], int(round(x - IW / 2)), int(round(y - IH / 2))); cur = (x + 4, y + 6)
    if T_RUN <= t < T_RUN_END + FLIGHT + 0.2:
        for i in range(N_FILES):
            if LAUNCH[i] <= t < ARRIVE[i]:
                x, y = pos_fly(i, t); blit(arr, ICONS[TYPES[i]], int(round(x - IW / 2)), int(round(y - IH / 2)))
    img = Image.fromarray(arr); d = ImageDraw.Draw(img)
    if cur is not None:
        cx, cy = cur
        d.polygon([(cx, cy), (cx, cy + 28), (cx + 7, cy + 21), (cx + 14, cy + 33), (cx + 19, cy + 30), (cx + 13, cy + 19), (cx + 22, cy + 19)], fill=(255, 255, 255), outline=(0, 0, 0))
    # ---- captions (word by word) ----
    if cur_i is not None:
        who = LINES[cur_i][0]; items = WORDS[cur_i]
        fnt = BLACK(46); lines = wrap_words(items, fnt, 960, d)
        y0 = CAP_Y - (len(lines) - 1) * 28
        tag, tagc = ("BOSS", (255, 128, 118)) if who == "boss" else ("YOU", (120, 190, 255))
        tx = 835 if who == "boss" else 300
        d.rounded_rectangle([tx - 56, 114, tx + 56, 148], radius=17, fill=tc(tagc, 0.28), outline=tagc, width=3)
        d.text((tx, 131), tag, font=BLACK(22), fill=tagc, anchor="mm")
        for li, ln in enumerate(lines):
            tot = sum(d.textlength(w_[0] + " ", font=fnt) for w_ in ln); x = SW / 2 - tot / 2
            for (wd, ws, we) in ln:
                col = (255, 214, 64) if ws <= t < we else (255, 255, 255) if t >= we else (122, 130, 146)
                d.text((x, y0 + li * 56), wd, font=fnt, fill=col, anchor="lm", stroke_width=5, stroke_fill=(8, 10, 16))
                x += d.textlength(wd + " ", font=fnt)
    # ---- header row: Downloads + timer ----
    left_files = max(0, N_FILES - departed_at(t))
    d.text((44, 676), "Downloads", font=BLACK(32), fill=TEXT, anchor="lm")
    d.text((252, 676), f"{left_files:,} files", font=BOLD(28), fill=MUTED if left_files else GREEN, anchor="lm")
    if t < T_RUN:
        blink = int(t * 2) % 2 == 0
        d.rounded_rectangle([690, 648, 1040, 704], radius=28, fill=(60, 20, 22), outline=RED, width=3)
        d.ellipse([710, 667, 728, 685], fill=RED if blink else (110, 40, 40))
        d.text((740, 676), "DEADLINE", font=BOLD(22), fill=(255, 150, 146), anchor="lm")
        d.text((1018, 676), fmt_t(600 - t), font=mono(30), fill=(255, 255, 255), anchor="rm")
    else:
        p = min(1.0, max(0.0, (t - T_RUN) / (T_RUN_END - T_RUN))); fin = t >= T_RUN_END; col = GREEN if fin else YEL
        d.rounded_rectangle([690, 648, 1040, 704], radius=28, fill=(18, 40, 24) if fin else (44, 36, 14), outline=col, width=3)
        d.text((712, 676), "SLOW-MO", font=BOLD(22), fill=col, anchor="lm")
        d.text((1018, 676), f"{REAL_T * p:0.2f} s" if REAL_T >= 0.095 else f"{REAL_T * p:0.3f} s", font=mono(30), fill=(255, 255, 255), anchor="rm")
    # ---- by-hand pill ----
    if T_HAND <= t < TY0:
        k = ease_back((t - T_HAND) / 0.35) * (1.0 if t < TY0 - 0.3 else max(0.0, (TY0 - t) / 0.3))
        if k > 0.02:
            cx, cy = 515, 910; wd, hd = 700 * min(k, 1.05), 160 * min(k, 1.05)
            d.rounded_rectangle([cx - wd / 2, cy - hd / 2, cx + wd / 2, cy + hd / 2], radius=26, fill=(40, 14, 16), outline=RED, width=4)
            if k > 0.7:
                d.text((cx, cy - 32), "BY HAND", font=BOLD(32), fill=(255, 150, 146), anchor="mm")
                d.text((cx, cy + 20), "3 h 20 min", font=BLACK(80), fill=(255, 255, 255), anchor="mm")
                d.text((cx, cy + 62), "10 seconds per file", font=SEMI(22), fill=(255, 150, 146), anchor="mm")
    # ---- terminal ----
    typed = int(np.searchsorted(CH_T, t, side="right")) if t >= TY0 else 0
    cwid = mono(25).getlength("M"); ln, col_ = 0, 0
    for k in range(typed):
        ch = CODE[k]
        if ch == "\n": ln += 1; col_ = 0; continue
        if ch != " ": d.text((TX0 + 20 + col_ * cwid, CODE_Y0 + ln * LINE_H), ch, font=mono(25), fill=CODE_COL[k], anchor="lm")
        col_ += 1
    if typed < NCH and t >= TY0 - 0.4 and int(t * 3) % 2 == 0:
        d.rectangle([TX0 + 20 + col_ * cwid, CODE_Y0 + ln * LINE_H - 12, TX0 + 20 + col_ * cwid + 12, CODE_Y0 + ln * LINE_H + 12], fill=(200, 205, 212))
    if t >= TY1 - 0.1:
        pulse = 1.0 + 0.08 * math.sin(t * 12) if t < T_RUN else 1.0; press = t >= T_RUN
        d.rounded_rectangle([TX1 - 140, TY_0 + 5, TX1 - 12, TY_0 + 29], radius=12, fill=GREEN if not press else (35, 110, 50))
        d.text((TX1 - 76, TY_0 + 17), "RUN", font=BLACK(int(18 * pulse)), fill=(6, 28, 12), anchor="mm")
    if t >= T_DONE:
        k = ease_out((t - T_DONE) / 0.25)
        d.text((TX0 + 20, CODE_Y0 + 7 * LINE_H), f"Sorted {N_FILES:,} files in {T_TXT}", font=mono(25), fill=tc(GREEN, k), anchor="lm")
    # ---- folder counters ----
    cnt = counts_at(t)
    for j in range(6):
        recent = np.sum((ARRIVE > t - 0.12) & (ARRIVE <= t) & (TYPES == j)) > 0
        d.text((CARD_X[j] + CW / 2, CY0 + 162), f"{int(cnt[j])}", font=BLACK(40), fill=mix(TEXT, TCOL[j], 0.75) if recent else TEXT, anchor="mm")
    # ---- result card ----
    if T_RES <= t < T_CTA:
        rt = t - T_RES; k0 = ease_out(rt / 0.35)
        x0, y0, x1, y1 = 26, 712, 1054, TY_1
        d.rounded_rectangle([x0, y0, x1, y1], radius=26, fill=(16, 20, 26), outline=BORDER, width=3)
        if k0 > 0.4:
            d.text((540, 756), "RESULT", font=BOLD(28), fill=MUTED, anchor="mm")
            bw = 940; kb = ease_out((rt - 0.15) / 0.6)
            d.text((80, 820), "BY HAND", font=BLACK(36), fill=(255, 150, 146), anchor="lm"); d.text((1000, 820), "3 h 20 min", font=mono(32), fill=TEXT, anchor="rm")
            d.rounded_rectangle([80, 850, 80 + bw * kb, 900], radius=14, fill=RED)
            d.text((80, 950), "PYTHON", font=BLACK(36), fill=(140, 230, 160), anchor="lm"); d.text((1000, 950), T_TXT, font=mono(32), fill=TEXT, anchor="rm")
            d.rounded_rectangle([80, 980, 80 + max(10, bw * (REAL_T / MANUAL_S)) * kb, 1030], radius=14, fill=GREEN)
            ks = ease_back((rt - 0.9) / 0.5)
            if ks > 0.01:
                sp_txt = f"{round(SPEEDUP, -3):,.0f}x"; sp_fs = min(170, 170 * 900 / BLACK(170).getlength(sp_txt))
                d.text((540, 1160), sp_txt, font=BLACK(sp_fs * ks), fill=YEL, anchor="mm")
                d.text((540, 1270), "FASTER", font=BLACK(80 * ks), fill=(255, 255, 255), anchor="mm")
            if rt > 1.6: d.text((540, 1352), "real run, measured by the code in this video", font=SEMI(24), fill=MUTED, anchor="mm")
    # ---- call to action ----
    if t >= T_CTA:
        ct = t - T_CTA
        d.rounded_rectangle([26, 712, 1054, TY_1], radius=26, fill=(16, 20, 26), outline=BORDER, width=3)
        k1 = ease_back(ct / 0.4); k2 = ease_out((ct - 0.5) / 0.4); k3 = ease_back((ct - 1.0) / 0.5)
        if k1 > 0.01:
            d.text((540, 810), "WHAT BORING TASK", font=BLACK(78 * k1), fill=TEXT, anchor="mm")
            d.text((540, 900), "SHOULD PYTHON DO", font=BLACK(78 * k1), fill=TEXT, anchor="mm")
            d.text((540, 1000), "NEXT?", font=BLACK(120 * k1), fill=YEL, anchor="mm")
        if k2 > 0: d.text((540, 1090), "Comment it below", font=BOLD(42), fill=tc((255, 120, 116), k2), anchor="mm")
        if k3 > 0.01:
            d.text((540, 1195), "PYTHONIC", font=BLACK(96 * k3), fill=BLUE, anchor="mm")
            d.text((540, 1295), "LIFE", font=BLACK(96 * k3), fill=YEL, anchor="mm")
            d.text((540, 1362), "100% Python  •  Follow", font=BOLD(30), fill=tc(MUTED, min(1, k3)), anchor="mm")
    return np.asarray(img)

# ------------------------------------------------------------------ main
if len(sys.argv) > 3 and sys.argv[3] == "still":
    os.makedirs("chk", exist_ok=True)
    for ts in sys.argv[4:]:
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
        a = min(1.0, (t - T_LOOP) / (TOTAL - T_LOOP - 1 / FPS)); out = cv2.addWeighted(out, 1 - a, F0, a, 0)
    proc.stdin.write(np.ascontiguousarray(out).tobytes())
proc.stdin.close(); proc.wait()

a = audio[: int(TOTAL * SR)].copy()
a = np.tanh(a / (np.percentile(np.abs(a), 99.8) + 1e-6) * 1.15) * 0.92
nfd = int(0.1 * SR); a[-nfd:] *= np.linspace(1, 0.2, nfd)
with wave.open("urgent_audio.wav", "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR); wf.writeframes((a * 32767).astype(np.int16).tobytes())
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "urgent_noaudio.mp4", "-i", "urgent_audio.wav", "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k", "-shortest", "pythonic_life_urgent_task.mp4"], check=True)
print("done", round(TOTAL, 1))
