"""Pythonic Life — "THE CLIENT SENT A 5,000-ROW EXCEL FILE" (YouTube Shorts 1080x1920, ~21 s, loops)
Scene 1: boss and worker argue (avatars) above a messy spreadsheet.  Scene 2: avatar-free full-screen code window,
a real before/after clean-up and a measured result.  Scene 3: the avatars come back, the boss says "Wow."
The 5,000 messy rows are generated, the code on screen is executed for real (fresh Python process, pandas import
included, median of 5 runs) and the before/after rows come from that run.
Usage: python3 excel_cleanup.py <worker.onnx> <boss.onnx>          (Piper placeholder voices)
       python3 excel_cleanup.py <worker.onnx> <boss.onnx> still 0 5 14   -> PNG stills in ./chk
Better voices: put ElevenLabs files in audio/l0.mp3 ... audio/l5.mp3 (they replace Piper).  Music: music/*.mp3 (optional).
"""
import datetime as dt, glob, math, os, re, shutil, statistics, subprocess, sys, tempfile, time, wave
from functools import lru_cache
import numpy as np, cv2, pandas as pd
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SR = 1080, 1920, 30, 44100
HERE = os.path.dirname(os.path.abspath(__file__))
VOICE_ME = sys.argv[1] if len(sys.argv) > 1 else "en-us-lessac-medium.onnx"
VOICE_BOSS = sys.argv[2] if len(sys.argv) > 2 else "en-us-ryan-high.onnx"

# ------------------------------------------------------------------ the code that is shown AND executed
CODE = '''import pandas as pd
df = pd.read_excel("clients.xlsx")
df = df.drop_duplicates()
df.date = pd.to_datetime(df.date, format="mixed")
df.phone = (df.phone.astype(str)
    .str.replace(r"\\D", "", regex=True))
df.to_excel("clean.xlsx", index=False)'''
SHOWN_CODE = CODE.replace("\\\\", "\\")

# ------------------------------------------------------------------ 5,000 messy rows
rng = np.random.default_rng(12)
FIRST = ["Aziz", "Malika", "Jasur", "Dilnoza", "Bobur", "Nigora", "Sardor", "Zilola", "Otabek", "Madina", "Rustam", "Feruza",
         "Timur", "Sevara", "Akmal", "Shahlo", "Davron", "Gulnora", "Sherzod", "Kamola", "Jamshid", "Umida", "Farhod", "Lola"]
LAST = ["Karimov", "Rahimova", "Tursunov", "Yusupova", "Aliyev", "Nazarova", "Ergashev", "Sodiqova", "Mirzayev", "Abdullayeva",
        "Hasanov", "Ismoilova", "Qodirov", "Saidova", "Normatov", "Ahmedova", "Bekmurodov", "Xolmatova"]
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
N_BASE, N_DUP = 4400, 600
seen, base = set(), []
while len(base) < N_BASE:
    ds = "998" + str(rng.choice([90, 91, 93, 94, 95, 97, 99, 88, 33])) + "".join(str(x) for x in rng.integers(0, 10, 7))
    if ds in seen: continue
    seen.add(ds)
    fmt = int(rng.choice(3, p=[0.4, 0.3, 0.3]))
    d0 = dt.date(2023, 1, 1) + dt.timedelta(days=int(rng.integers(0, 1095)))
    if fmt == 1 and d0.day < 13: d0 = d0.replace(day=int(rng.integers(13, 29)))
    dstr = [d0.isoformat(), f"{d0.day:02d}.{d0.month:02d}.{d0.year}", f"{MON[d0.month - 1]} {d0.day}, {d0.year}"][fmt]
    pf = int(rng.choice(3, p=[0.3, 0.35, 0.35]))
    a, b, c = ds[5:8], ds[8:10], ds[10:12]
    phone = [ds, f"+998 {ds[3:5]} {a}-{b}-{c}", f"+998 ({ds[3:5]}) {a} {b} {c}"][pf]
    base.append((f"{rng.choice(FIRST)} {rng.choice(LAST)}", dstr, phone, d0.isoformat(), ds))
keys = [(i, i) for i in range(N_BASE)]
for i in rng.choice(N_BASE, N_DUP, replace=False): keys.append((int(i) + float(rng.uniform(0.2, 5)), int(i)))
keys.sort(key=lambda k: k[0])
rows = [base[k[1]] for k in keys]
RAW = pd.DataFrame([r[:3] for r in rows], columns=["name", "date", "phone"])
TRUE_ISO = [r[3] for r in rows]
DUPF = RAW.duplicated().values
BADD = ~RAW.date.str.match(r"\d{4}-\d\d-\d\d$").values
BADP = ~RAW.phone.str.fullmatch(r"\d+").values
assert len(RAW) == N_BASE + N_DUP and DUPF.sum() == N_DUP


def bench():
    """run CODE for real in a fresh python process (pandas import included): median of 5 after a warm-up"""
    tmp = tempfile.mkdtemp(); cwd = os.getcwd()
    try:
        RAW.to_excel(os.path.join(tmp, "clients.xlsx"), index=False)
        def run():
            t0 = time.perf_counter(); r = subprocess.run([sys.executable, "-c", CODE], cwd=tmp, capture_output=True)
            d_ = time.perf_counter() - t0
            if r.returncode: raise RuntimeError(r.stderr.decode()[-600:])
            return d_
        run(); ts = [run() for _ in range(5)]
        os.chdir(tmp); g = {}; exec(CODE, g); clean = g["df"].copy()
    finally:
        os.chdir(cwd); shutil.rmtree(tmp, ignore_errors=True)
    return statistics.median(ts), clean

REAL_T, CLEAN = bench()
assert len(CLEAN) == N_BASE, len(CLEAN)
CLEAN_ISO = CLEAN.date.dt.strftime("%Y-%m-%d").tolist()
UNIQ = RAW[~DUPF].reset_index(drop=True)
assert CLEAN_ISO == [TRUE_ISO[i] for i in np.where(~DUPF)[0]], "date clean-up differs from the truth"
assert (CLEAN.phone.astype(str).values == UNIQ.phone.str.replace(r"\D", "", regex=True).values).all()
UBADD, UBADP = BADD[~DUPF], BADP[~DUPF]
N_DATE, N_PHONE = int(UBADD.sum()), int(UBADP.sum())
MANUAL_S = (N_BASE + N_DUP) * 3                                  # estimate: 3 s per row by hand = 4 h 10 min
SPEEDUP = MANUAL_S / REAL_T
SP_TXT = f"{int(float(f'{SPEEDUP:.2g}')):,}x"
T_TXT = f"{REAL_T:0.1f} s"
print(f"real run {REAL_T:.2f} s, dup {N_DUP}, dates fixed {N_DATE}, phones fixed {N_PHONE}, speed-up {SPEEDUP:,.0f}x ({SP_TXT})", flush=True)

# ------------------------------------------------------------------ voices
def finish(x, gain=1.0):
    nz = np.where(np.abs(x) > 0.02 * np.abs(x).max())[0]
    x = x[max(0, nz[0] - int(0.03 * SR)): nz[-1] + int(0.06 * SR)]
    return (x / (np.sqrt(np.mean(x ** 2)) + 1e-9) * 0.11 * gain).astype(np.float32)

def tts(text, voice, path, length_scale=1.0, pitch=1.0, gain=1.0):
    r = subprocess.run([sys.executable, "-m", "piper", "-m", voice, "-f", path, "--length-scale", str(length_scale),
                        "--noise-scale", "0.75", "--noise-w-scale", "0.9", "--sentence-silence", "0.12"], input=text.encode(), capture_output=True)
    if r.returncode != 0: raise RuntimeError(r.stderr.decode()[-400:])
    with wave.open(path) as wf:
        sr0 = wf.getframerate(); x = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float32) / 32768
    n = int(len(x) * SR / sr0 / pitch)
    return finish(np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x), gain)

def load_ext(path, gain=1.0):
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True)
    return finish(np.frombuffer(r.stdout, np.float32).copy(), gain)

LINES = [
    ("boss", "The client sent a five-thousand-row Excel file. Clean it. Five minutes!", dict(length_scale=0.92, pitch=0.93, gain=1.15)),
    ("me", "Five minutes? Duplicates, broken dates, bad phone numbers... that's hours by hand!", dict(length_scale=0.88)),
    ("me", "Or... I use Python. It's very easy.", dict(length_scale=1.05)),
    ("me", "Seven lines. Run!", dict(length_scale=1.0)),
    ("me", "Done. Five thousand rows, in seconds!", dict(length_scale=1.0)),
    ("boss", "Wow.", dict(length_scale=1.1, pitch=0.93, gain=1.1)),
]
os.makedirs("tts_tmp", exist_ok=True)
WAVS, EXT = [], []
for i, (who, txt, kw) in enumerate(LINES):
    ext = [p for p in glob.glob(os.path.join(HERE, "audio", f"l{i}.*")) if p.lower().endswith((".mp3", ".wav", ".m4a"))]
    EXT.append(bool(ext))
    WAVS.append(load_ext(ext[0], kw.get("gain", 1.0)) if ext else tts(txt, VOICE_BOSS if who == "boss" else VOICE_ME, f"tts_tmp/l{i}.wav", **kw))
DUR = [len(w) / SR for w in WAVS]
print("voice lines:", [round(d, 2) for d in DUR], "external:", EXT, flush=True)

# ------------------------------------------------------------------ timeline (seconds)
START = [0.0] * 6
START[0] = 0.0
START[1] = START[0] + DUR[0] + 0.22
START[2] = START[1] + DUR[1] + 0.22
T_S2 = START[2] + DUR[2] + 0.22                                # scene 2: avatar-free code window
TY0 = T_S2 + 0.30; TY1 = TY0 + 2.3                             # typing the code
START[3] = T_S2 + 0.9
T_RUN = max(START[3] + DUR[3] - 0.25, TY1 + 0.35)
T_ROWS = T_RUN + 0.15                                          # rows start to be cleaned
T_RUN_END = T_RUN + 1.5                                        # slow-mo timer reaches the measured time
START[4] = T_RUN + 1.35
T_DONE = START[4] - 0.05
T_RES = START[4] + 0.30
T_S3 = START[4] + DUR[4] + 0.35                                # scene 3: avatars come back
START[5] = T_S3 + 0.3
TOTAL = START[5] + DUR[5] + 1.0
T_LOOP = TOTAL - 0.6
NF = int(TOTAL * FPS)
FD = 0.15                                                      # half length of the scene cross-fade
print(f"total {TOTAL:.1f} s, scene2 at {T_S2:.1f}, run at {T_RUN:.1f}, scene3 at {T_S3:.1f}", flush=True)

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
    for i in reversed(range(len(LINES))):
        if START[i] - 0.02 <= t < START[i] + DUR[i] + 0.35: return i
    return None

WORDS = []
for i, (who, txt, _) in enumerate(LINES):
    ws = txt.split(); wts = np.array([len(re.sub(r"\W", "", x)) + (3 if x[-1] in ".?!" else 1 if x[-1] == "," else 0) + 1 for x in ws], float)
    cum = np.concatenate([[0], np.cumsum(wts)]) / wts.sum()
    WORDS.append([(ws[k], START[i] + cum[k] * DUR[i], START[i] + cum[k + 1] * DUR[i]) for k in range(len(ws))])

def moods(t):
    wm = "panic"
    if t >= START[2]: wm = "idea"
    if t >= START[2] + 1.2: wm = "confident"
    if t >= T_S3 - 0.05: wm = "sip"
    bm = "rage" if t < T_S2 else "shock"
    return wm, bm

SW, SH = 1080, 430
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
    return np.asarray(img.crop((240, 0, 2000, 860)).resize((1080, 527), Image.LANCZOS))

def wrap_words(items, fnt, maxw, d):
    lines, cur, w = [], [], 0
    for it in items:
        ww = d.textlength(it[0] + " ", font=fnt)
        if cur and w + ww > maxw: lines.append(cur); cur, w = [], 0
        cur.append(it); w += ww
    if cur: lines.append(cur)
    return lines

# ------------------------------------------------------------------ layout
SCENE_Y, SCENE_H = 150, 527
CAP_Y = 830
PX0, PX1 = 30, 930
SHEET_Y0, SHEET_HDR, SHEET_COLH = 930, 46, 40
ROWS_Y0 = SHEET_Y0 + SHEET_HDR + SHEET_COLH
ROWS_Y1 = 1570
VIEW = ROWS_Y1 - ROWS_Y0
RH = 56
SCROLL = 52                                                     # px per second
BASE_IMG = Image.new("RGB", (W, H), BG)
GREEN_T = (140, 230, 160)

# ------------------------------------------------------------------ spreadsheet rows (pre-rendered strips)
def strip(name, date, phone, bg, tcol=TEXT, dbg=None, dcol=TEXT, pbg=None, pcol=TEXT):
    im = Image.new("RGB", (900, RH), bg); d = ImageDraw.Draw(im)
    if dbg: d.rounded_rectangle([326, 7, 560, RH - 7], radius=8, fill=dbg)
    if pbg: d.rounded_rectangle([566, 7, 884, RH - 7], radius=8, fill=pbg)
    d.text((20, RH // 2), name, font=SEMI(24), fill=tcol, anchor="lm")
    d.text((340, RH // 2), date, font=mono(24), fill=dcol, anchor="lm")
    d.text((580, RH // 2), phone, font=mono(24), fill=pcol, anchor="lm")
    d.line([0, RH - 1, 900, RH - 1], fill=(34, 40, 48))
    return im

ALT = [(22, 27, 34), (26, 32, 40)]
def raw_strip(i):
    n, dte, ph = tuple(RAW.iloc[i])
    if DUPF[i]: return strip(n, dte, ph, (58, 22, 24), (255, 170, 166), None, (255, 170, 166), None, (255, 170, 166))
    return strip(n, dte, ph, ALT[i % 2], TEXT,
                 (66, 52, 12) if BADD[i] else None, (255, 214, 100) if BADD[i] else TEXT,
                 (50, 34, 74) if BADP[i] else None, (210, 180, 255) if BADP[i] else TEXT)

def clean_strip_vals(n, iso, ph, fd, fp, k=0):
    return strip(n, iso, ph, ALT[k % 2], TEXT, (16, 56, 30) if fd else None, GREEN_T if fd else TEXT,
                 (16, 56, 30) if fp else None, GREEN_T if fp else TEXT)

def removed_strip(i):
    n = RAW.name.iloc[i]
    im = Image.new("RGB", (900, RH), (34, 24, 26)); d = ImageDraw.Draw(im)
    d.text((20, RH // 2), n, font=SEMI(24), fill=(150, 96, 98), anchor="lm")
    wd = d.textlength(n, font=SEMI(24)); d.line([18, RH // 2, 22 + wd, RH // 2], fill=(200, 90, 90), width=3)
    d.text((340, RH // 2), "duplicate removed", font=BOLD(24), fill=(255, 130, 124), anchor="lm")
    d.line([0, RH - 1, 900, RH - 1], fill=(34, 40, 48))
    return im

def tall(strips):
    im = Image.new("RGB", (900, RH * len(strips)))
    for k, s in enumerate(strips): im.paste(s, (0, k * RH))
    return np.asarray(im)

NROW = 52
TALL1 = tall([raw_strip(i) for i in range(NROW)])
TALL3 = tall([clean_strip_vals(UNIQ.name[j], CLEAN_ISO[j], CLEAN.phone.astype(str).iloc[j], UBADD[j], UBADP[j], j) for j in range(NROW)])

def pick_window():
    for s in range(0, 400):
        idx = range(s, s + 6)
        dj = [j for j in idx if DUPF[j]]
        if len(dj) != 1 or sum(BADD[j] for j in idx) < 2 or sum(BADP[j] for j in idx) < 2: continue
        j = dj[0]
        if any(tuple(RAW.iloc[k]) == tuple(RAW.iloc[j]) for k in range(s, j)): return s
    raise RuntimeError("no sample window")
WIN = pick_window()
KEY2ISO = {(UNIQ.name[j], CLEAN.phone.astype(str).iloc[j]): CLEAN_ISO[j] for j in range(len(UNIQ))}
BEFORE = [np.asarray(raw_strip(i)).astype(np.float32) for i in range(WIN, WIN + 6)]
AFTER = []
for k, i in enumerate(range(WIN, WIN + 6)):
    n, dte, ph = tuple(RAW.iloc[i])
    if DUPF[i]: AFTER.append(np.asarray(removed_strip(i)).astype(np.float32))
    else:
        dg = re.sub(r"\D", "", ph)
        AFTER.append(np.asarray(clean_strip_vals(n, KEY2ISO[(n, dg)], dg, BADD[i], BADP[i], i)).astype(np.float32))

# ------------------------------------------------------------------ code colours
CODE_COL = [TEXT] * len(CODE)
for m in re.finditer(r'(r?".*?")|\b(import|as|True)\b|(\b\w+(?=\())', CODE):
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
def whoosh(t0, dur=0.45, amp=0.12):
    n = int(dur * SR); tw = np.arange(n) / SR
    nz = np.convolve(np.random.default_rng(int(t0 * 100)).standard_normal(n), np.ones(14) / 14, mode="same") * 14 ** 0.5
    put(sfx, t0, nz * np.sin(np.pi * np.clip(tw / dur, 0, 1)) ** 1.5 * amp)

for i, w in enumerate(WAVS): put(vo, START[i], w)
MUS = os.environ.get("MUSIC") or next(iter(sorted(glob.glob(os.path.join(HERE, "music", "*.mp3")))), None)
if MUS:                                                          # the user's own track, a loud-enough segment, ducked under the voice
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", "35", "-t", str(TOTAL + 1), "-i", MUS, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, check=True)
    m_ = np.frombuffer(r.stdout, np.float32).copy(); m_ = m_ / (np.sqrt(np.mean(m_ ** 2)) + 1e-9) * 0.075
    put(music, 0, m_ * np.minimum(1, np.arange(len(m_)) / SR / 0.4))
    print("music:", os.path.basename(MUS), flush=True)
else:
    bar = 4 * 60 / 92
    for b in range(int(TOTAL / bar) + 2):
        for f in [220.0, 261.6, 329.6]: tone(music, b * bar, f, bar + 0.2, 0.03, 0.8, 0.3)
for k in range(0, int(T_S2)): click(0.3 + k * 1.0, 0.06, 2600, seed=k)                  # deadline clock
whoosh(T_S2 - 0.25); whoosh(T_S3 - 0.25, amp=0.09)
for k in range(NCH):
    if CODE[k] != " " and CODE[k] != "\n": click(CH_T[k], 0.05 + 0.03 * (k % 3), 1200 + 600 * (k % 4), seed=100 + k)
tone(sfx, T_RUN - 0.03, 587, 0.2, 0.18, 12)
PENTA = [523.3, 587.3, 659.3, 784.0, 880.0, 1046.5]
for k in range(6): tone(sfx, T_ROWS + k * 0.17 + 0.1, PENTA[k], 0.15, 0.05, 18)
for k_, fq in enumerate([523.3, 659.3, 784.0, 1046.5]): tone(sfx, T_DONE + 0.07 * k_, fq, 1.3, 0.14, 3.5)
tone(sfx, START[5] - 0.15, 196, 0.5, 0.12, 6)
env_v = np.convolve(np.abs(vo), np.ones(int(0.06 * SR)) / int(0.06 * SR), mode="same")
duck = 1 - 0.70 * np.clip(env_v / 0.06, 0, 1)
duck = np.convolve(duck, np.ones(int(0.12 * SR)) / int(0.12 * SR), mode="same")
audio = vo + music * duck + sfx * (0.55 + 0.45 * duck)

# ------------------------------------------------------------------ drawing helpers
def caption(d, i, t, y, size=52, maxw=980, cx=540):
    fnt = BLACK(size); lines = wrap_words(WORDS[i], fnt, maxw, d)
    lh = int(size * 1.2); y0 = y - (len(lines) - 1) * lh / 2
    for li, ln in enumerate(lines):
        tot = sum(d.textlength(w[0] + " ", font=fnt) for w in ln); x = cx - tot / 2
        for (wd, ws, we) in ln:
            col = (255, 214, 64) if ws <= t < we else (255, 255, 255) if t >= we else (122, 130, 146)
            d.text((x, y0 + li * lh), wd, font=fnt, fill=col, anchor="lm", stroke_width=6, stroke_fill=(8, 10, 16))
            x += d.textlength(wd + " ", font=fnt)

def chip(d, cx, cy, text, col, fs=22, pad=22):
    wd = d.textlength(text, font=BLACK(fs)) + pad * 2
    d.rounded_rectangle([cx - wd / 2, cy - 18, cx + wd / 2, cy + 18], radius=18, fill=tc(col, 0.25), outline=col, width=3)
    d.text((cx, cy), text, font=BLACK(fs), fill=col, anchor="mm")

def sheet_panel(d, img, tallarr, off, title, chips, colh=("NAME", "DATE", "PHONE")):
    d.rounded_rectangle([PX0, SHEET_Y0, PX1, ROWS_Y1 + 6], radius=18, fill=PANEL, outline=BORDER, width=2)
    d.text((PX0 + 22, SHEET_Y0 + 24), title, font=BLACK(26), fill=TEXT, anchor="lm")
    x = PX1 - 14
    for txt, col in reversed(chips):
        wd = d.textlength(txt, font=BOLD(19)) + 26
        d.rounded_rectangle([x - wd, SHEET_Y0 + 8, x, SHEET_Y0 + 40], radius=16, fill=tc(col, 0.22), outline=col, width=2)
        d.text((x - wd / 2, SHEET_Y0 + 24), txt, font=BOLD(19), fill=col, anchor="mm"); x -= wd + 8
    d.rectangle([PX0 + 2, SHEET_Y0 + SHEET_HDR, PX1 - 2, ROWS_Y0], fill=PANEL2)
    for xx, lab in zip((PX0 + 20, PX0 + 340, PX0 + 580), colh): d.text((xx, SHEET_Y0 + SHEET_HDR + SHEET_COLH / 2), lab, font=BOLD(18), fill=MUTED, anchor="lm")
    off = int(off) % (tallarr.shape[0] - VIEW)
    img.paste(Image.fromarray(tallarr[off: off + VIEW]), (PX0, ROWS_Y0))
    d.rounded_rectangle([PX0, SHEET_Y0, PX1, ROWS_Y1 + 6], radius=18, outline=BORDER, width=2)

def avatar_frame(t, scene):
    img = BASE_IMG.copy()
    wm, bm = moods(t)
    img.paste(Image.fromarray(render_scene(t, wm, bm, mouth(t, "me"), mouth(t, "boss"))), (0, SCENE_Y))
    d = ImageDraw.Draw(img)
    d.rectangle([0, SCENE_Y + SCENE_H, W, SCENE_Y + SCENE_H + 3], fill=BORDER)
    li = speaking_line(t)
    if li is not None and li in (0, 1, 2, 5):
        who = LINES[li][0]
        chip(d, 540, SCENE_Y + SCENE_H + 40, "BOSS" if who == "boss" else "YOU", (255, 128, 118) if who == "boss" else (120, 190, 255))
        caption(d, li, t, CAP_Y)
    if scene == 1:
        s = 300 - t * 8.5; blink = int(t * 2) % 2 == 0
        d.rounded_rectangle([30, 166, 372, 210], radius=22, fill=(60, 20, 22), outline=RED, width=3)
        d.ellipse([46, 180, 62, 196], fill=RED if blink else (110, 40, 40))
        d.text((74, 188), "DEADLINE", font=BOLD(20), fill=(255, 150, 146), anchor="lm")
        d.text((354, 188), f"{int(s // 60):02d}:{int(s % 60):02d}", font=mono(26), fill=(255, 255, 255), anchor="rm")
        sheet_panel(d, img, TALL1, 20 + t * SCROLL, f"clients.xlsx   {N_BASE + N_DUP:,} rows",
                    [(f"{N_DUP} duplicates", (255, 120, 114)), ("broken dates", (255, 205, 90)), ("bad phones", (200, 160, 255))])
    else:
        sheet_panel(d, img, TALL3, 20 + max(0, t - T_S3) * 40, "clean.xlsx   " + f"{N_BASE:,} rows",
                    [("no duplicates", (110, 220, 140)), ("dates fixed", (110, 220, 140)), ("phones fixed", (110, 220, 140))])
    return np.asarray(img)

def code_frame(t):
    img = BASE_IMG.copy(); d = ImageDraw.Draw(img)
    # --- code window
    X0, X1, Y0, Y1 = 30, 930, 130, 474
    d.rounded_rectangle([X0, Y0, X1, Y1], radius=16, fill=(10, 13, 18), outline=BORDER, width=2)
    d.rounded_rectangle([X0, Y0, X1, Y0 + 34], radius=16, fill=PANEL2); d.rectangle([X0, Y0 + 18, X1, Y0 + 34], fill=PANEL2)
    for k, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]): d.ellipse([X0 + 16 + k * 24, Y0 + 11, X0 + 30 + k * 24, Y0 + 25], fill=c)
    d.text((X0 + 112, Y0 + 17), "clean.py", font=SEMI(20), fill=MUTED, anchor="lm")
    typed = int(np.searchsorted(CH_T, t, side="right")) if t >= TY0 else 0
    cw = mono(28).getlength("M"); ln, col_ = 0, 0; LY = Y0 + 34 + 34
    for k in range(typed):
        ch = CODE[k]
        if ch == "\n": ln += 1; col_ = 0; continue
        if ch != " ": d.text((X0 + 20 + col_ * cw, LY + ln * 40), ch, font=mono(28), fill=CODE_COL[k], anchor="lm")
        col_ += 1
    if typed < NCH and t >= TY0 - 0.3 and int(t * 3) % 2 == 0:
        d.rectangle([X0 + 20 + col_ * cw, LY + ln * 40 - 15, X0 + 20 + col_ * cw + 13, LY + ln * 40 + 15], fill=(200, 205, 212))
    if t >= TY1 - 0.1:
        pulse = 1.0 + 0.08 * math.sin(t * 12) if t < T_RUN else 1.0; press = t >= T_RUN
        d.rounded_rectangle([X1 - 140, Y0 + 5, X1 - 12, Y0 + 29], radius=12, fill=GREEN if not press else (35, 110, 50))
        d.text((X1 - 76, Y0 + 17), "RUN", font=BLACK(int(18 * pulse)), fill=(6, 28, 12), anchor="mm")
    # --- timer pill
    if t < T_RUN: col, lab, val = MUTED, "READY", "0.00 s"
    else:
        p = min(1.0, (t - T_RUN) / (T_RUN_END - T_RUN)); fin = p >= 1.0
        col, lab, val = (GREEN, "PYTHON", T_TXT) if fin else (YEL, "SLOW-MO", f"{REAL_T * p:0.2f} s")
    d.rounded_rectangle([30, 490, 400, 536], radius=23, fill=tc(col, 0.2), outline=col, width=3)
    d.text((52, 513), lab, font=BOLD(22), fill=col, anchor="lm"); d.text((380, 513), val, font=mono(30), fill=(255, 255, 255), anchor="rm")
    done_rows = t >= T_ROWS + 5 * 0.17 + 0.2
    d.text((430, 513), f"{N_BASE + N_DUP:,} rows" + (f"  >  {N_BASE:,} rows" if t >= T_RUN else ""), font=BOLD(24), fill=GREEN_T if t >= T_RUN else MUTED, anchor="lm")
    # --- before / after table
    TX0, TY_0, TX1, TY_1 = 30, 560, 930, 1010
    d.rounded_rectangle([TX0, TY_0, TX1, TY_1], radius=18, fill=PANEL, outline=BORDER, width=2)
    d.text((TX0 + 22, TY_0 + 26), "clients.xlsx", font=BLACK(26), fill=(255, 205, 90) if t < T_RUN else MUTED, anchor="lm")
    d.text((TX0 + 270, TY_0 + 26), ">", font=BLACK(26), fill=MUTED, anchor="lm")
    d.text((TX0 + 310, TY_0 + 26), "clean.xlsx", font=BLACK(26), fill=GREEN_T if t >= T_RUN else MUTED, anchor="lm")
    d.rectangle([TX0 + 2, TY_0 + 52, TX1 - 2, TY_0 + 92], fill=PANEL2)
    for xx, lab in zip((TX0 + 20, TX0 + 340, TX0 + 580), ("NAME", "DATE", "PHONE")): d.text((xx, TY_0 + 72), lab, font=BOLD(18), fill=MUTED, anchor="lm")
    for k in range(6):
        kk = ease_io((t - (T_ROWS + k * 0.17)) / 0.2)
        row = BEFORE[k] * (1 - kk) + AFTER[k] * kk
        img.paste(Image.fromarray(row.astype(np.uint8)), (TX0, TY_0 + 92 + k * RH))
    d.rounded_rectangle([TX0, TY_0, TX1, TY_1], radius=18, outline=BORDER, width=2)
    # --- stat cards
    kc = ease_out((t - (T_RUN + 0.25)) / 1.2) if t >= T_RUN else 0.0
    for j, (val, lab, c) in enumerate([(N_DUP, "duplicates removed", (255, 120, 114)), (N_DATE, "dates fixed", (255, 205, 90)), (N_PHONE, "phones fixed", (200, 160, 255))]):
        x0 = 30 + j * 305
        d.rounded_rectangle([x0, 1034, x0 + 290, 1190], radius=18, fill=PANEL, outline=tc(c, 0.7 if kc > 0 else 0.3), width=3)
        d.text((x0 + 145, 1092), f"{int(val * kc):,}", font=BLACK(70), fill=c if kc > 0 else MUTED, anchor="mm")
        d.text((x0 + 145, 1156), lab, font=SEMI(24), fill=TEXT, anchor="mm")
    # --- result card
    if T_RES <= t:
        rt = t - T_RES; k0 = ease_out(rt / 0.35)
        d.rounded_rectangle([30, 560, 930, 1200], radius=26, fill=(16, 20, 26), outline=BORDER, width=3)
        if k0 > 0.4:
            d.text((480, 606), "RESULT", font=BOLD(28), fill=MUTED, anchor="mm")
            bw = 820; kb = ease_out((rt - 0.15) / 0.6)
            d.text((70, 672), "BY HAND", font=BLACK(34), fill=(255, 150, 146), anchor="lm"); d.text((890, 672), "~4 h 10 min", font=mono(30), fill=TEXT, anchor="rm")
            d.rounded_rectangle([70, 700, 70 + bw * kb, 748], radius=14, fill=RED)
            d.text((70, 802), "PYTHON", font=BLACK(34), fill=(140, 230, 160), anchor="lm"); d.text((890, 802), T_TXT, font=mono(30), fill=TEXT, anchor="rm")
            d.rounded_rectangle([70, 830, 70 + max(10, bw * (REAL_T / MANUAL_S)) * kb, 878], radius=14, fill=GREEN)
            ks = ease_back((rt - 0.9) / 0.5)
            if ks > 0.01:
                fs = min(160, 160 * 820 / BLACK(160).getlength(SP_TXT))
                d.text((480, 970), SP_TXT, font=BLACK(fs * ks), fill=YEL, anchor="mm")
                d.text((480, 1112), "FASTER", font=BLACK(70 * ks), fill=(255, 255, 255), anchor="mm")
            if rt > 1.6: d.text((480, 1165), "by hand = estimate, 3 s per row. Python = measured", font=SEMI(22), fill=MUTED, anchor="mm")
    # --- voice-over captions
    li = speaking_line(t)
    if li in (3, 4): caption(d, li, t, 1310, size=52, maxw=860, cx=480)
    return np.asarray(img)

def frame_at(t):
    if t < T_S2 - FD: return avatar_frame(t, 1)
    if t < T_S2 + FD:
        k = ease_io((t - (T_S2 - FD)) / (2 * FD)); return cv2.addWeighted(avatar_frame(t, 1), 1 - k, code_frame(t), k, 0)
    if t < T_S3 - FD: return code_frame(t)
    if t < T_S3 + FD:
        k = ease_io((t - (T_S3 - FD)) / (2 * FD)); return cv2.addWeighted(code_frame(t), 1 - k, avatar_frame(t, 3), k, 0)
    return avatar_frame(t, 3)

# ------------------------------------------------------------------ main
if len(sys.argv) > 3 and sys.argv[3] == "still":
    os.makedirs("chk", exist_ok=True)
    for ts in sys.argv[4:]: Image.fromarray(frame_at(float(ts))).save(f"chk/still_{ts}.png")
    sys.exit()

proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "15",
                         "-maxrate", "10M", "-bufsize", "20M", "-pix_fmt", "yuv420p", "-colorspace", "bt709",
                         "-color_primaries", "bt709", "-color_trc", "bt709", "excel_noaudio.mp4"], stdin=subprocess.PIPE)
F0 = frame_at(0.0)
for f in range(NF):
    t = f / FPS
    out = F0 if f == 0 else frame_at(t)
    if t > T_LOOP:
        a = min(1.0, (t - T_LOOP) / (TOTAL - T_LOOP - 1 / FPS)); out = cv2.addWeighted(out, 1 - a, F0, a, 0)
    proc.stdin.write(np.ascontiguousarray(out).tobytes())
proc.stdin.close(); proc.wait()

a = audio[: int(TOTAL * SR)].copy()
a = np.tanh(a / (np.percentile(np.abs(a), 99.8) + 1e-6) * 1.15) * 0.92
nfd = int(0.1 * SR); a[-nfd:] *= np.linspace(1, 0.2, nfd)
with wave.open("excel_audio.wav", "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR); wf.writeframes((a * 32767).astype(np.int16).tobytes())
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "excel_noaudio.mp4", "-i", "excel_audio.wav", "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k", "-shortest", "pythonic_life_excel_cleanup.mp4"], check=True)
print("done", round(TOTAL, 1))
