"""Pythonic Life — PICK A MARBLE 1-6 (YouTube Shorts 1080x1920, ~19 s, loops back to its first frame)
Marble physics in plain Python + NumPy (marble-marble, pegs, ramps, spinning bars, funnel, bumpers),
rendered with Pillow + OpenCV, sounds synthesized in NumPy, encoded with ffmpeg.
Usage: python3 marble_race.py [seed]        (python3 marble_race.py probe -> scan seeds)
"""
import math, subprocess, sys, wave
from functools import lru_cache
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
import marble_sim as S
from marble_sim import NC, R, XL, XR, Y_FIN, Y_FLOOR, SUBS, PEGS, PEG_R, SEGS, SEG_R, SPIN, SPIN_R, BUMP, BUMP_R

XL, XR = int(XL), int(XR)
W, H, FPS, SR = 1080, 1920, 30, 44100
COLS = [(255, 62, 84), (72, 132, 255), (62, 232, 122), (255, 208, 44), (255, 104, 202), (64, 226, 236)]
TOP_PAD, BOT_PAD = 560, 240
WORLD_H = TOP_PAD + int(Y_FLOOR) + BOT_PAD
LEAD_SCREEN_Y = 1100            # where the leading marble sits on screen while the camera scrolls

# ------------------------------------------------------------------ probe
if len(sys.argv) > 1 and sys.argv[1] == "probe":
    for sd in range(1, 61):
        st = S.stats(S.simulate(sd))
        good = 13.0 <= st["t_win"] <= 16.2 and st["margin"] < 0.6 and st["lc"] >= 5 and st["worst"] >= 4
        print(f"seed {sd:2d} win {st['win'] + 1} t={st['t_win']:5.2f} margin={st['margin']:4.2f} lc={st['lc']:2d} "
              f"worst={st['worst'] + 1} {'<== good' if good else ''}", flush=True)
    sys.exit()

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 15
SIM = S.simulate(SEED)
ST = S.stats(SIM)
HIST, EV, SHIFT = SIM["hist"], SIM["ev"], SIM["shift"]
WIN, MARGIN, T_FIN = ST["win"], ST["margin"], ST["t_win"]
FIN = ST["fin"]
ORDER_F = [int(i) for i in np.argsort(FIN)]
TOTAL = T_FIN + 4.4
NF = int(TOTAL * FPS)
T_RES = T_FIN + 0.35                       # result card appears
T_LOOP = TOTAL - 0.7                       # crossfade back to the first frame
print(f"seed {SEED}: winner {WIN + 1} at {T_FIN:.2f}s margin {MARGIN:.2f}s lead changes {ST['lc']} total {TOTAL:.1f}s", flush=True)


def state(t):
    """marble positions and angles at video time t (interpolated between physics steps)"""
    k = t * SUBS; k0 = int(min(max(k, 0), len(HIST) - 1)); k1 = min(k0 + 1, len(HIST) - 1); a = min(max(k - k0, 0), 1)
    h = HIST[k0] * (1 - a) + HIST[k1] * a
    return h[:NC], h[NC:2 * NC], h[2 * NC:]


# lead-change events and the moment the winner was last (for the "from last place" tag)
Y_ALL = HIST[:, NC:2 * NC]
POP = []; lead = int(np.argmax(Y_ALL[0])); cand = None; cnt = 0; last_pop = -9
for f in range(int(T_FIN * FPS)):
    yy = Y_ALL[min(int(f / FPS * SUBS), len(Y_ALL) - 1)]
    ld = int(np.argmax(yy)); t = f / FPS
    if ld != lead and yy[ld] > yy[lead] + 35:
        cnt = cnt + 1 if cand == ld else 1; cand = ld
        if cnt >= 3 and t > 1.8 and t - last_pop > 1.3 and t < T_FIN - 1.0:
            POP.append((t, ld)); last_pop = t; lead = ld; cnt = 0
    else:
        cand = None; cnt = 0
        if ld != lead and yy[ld] > yy[lead]: lead = ld
LAST_T = next((k / SUBS for k in range(int(2.5 * SUBS), int(T_FIN * SUBS), 8)
               if int(np.sum(Y_ALL[k] > Y_ALL[k][WIN])) == NC - 1), None)

# ------------------------------------------------------------------ helpers
F = "/usr/share/fonts/opentype/inter/"
@lru_cache(None)
def font(n, s): return ImageFont.truetype(F + n, max(1, int(s)))
BLACK = lambda s: font("Inter-Black.otf", s)
BOLD = lambda s: font("Inter-Bold.otf", s)
SEMI = lambda s: font("Inter-SemiBold.otf", s)
def ease_out(x): x = max(0, min(1, x)); return 1 - (1 - x) ** 3
def ease_back(x): x = max(0, min(1, x)); return 1 + 2.70158 * (x - 1) ** 3 + 1.70158 * (x - 1) ** 2
def tc(c, a=1.0): return tuple(int(min(255, max(0, v * a))) for v in c)
STROKE = dict(stroke_width=6, stroke_fill=(5, 6, 14))

def glow(a, s=0.30):
    sm = cv2.resize(a, (a.shape[1] // 4, a.shape[0] // 4), interpolation=cv2.INTER_AREA)
    b1 = cv2.GaussianBlur(sm, (0, 0), 3); b2 = cv2.GaussianBlur(sm, (0, 0), 10)
    sz = (a.shape[1], a.shape[0])
    return cv2.add(cv2.addWeighted(a, 1, cv2.resize(b1, sz), s, 0), (cv2.resize(b2, sz) * 0.2).astype(np.uint8))

# ------------------------------------------------------------------ static world
def make_world():
    keys = [(-TOP_PAD, (12, 12, 32)), (900, (20, 14, 48)), (2200, (12, 26, 54)), (3200, (26, 16, 50)), (Y_FLOOR + BOT_PAD, (34, 14, 38))]
    ys = np.arange(WORLD_H) - TOP_PAD
    col = np.stack([np.interp(ys, [k[0] for k in keys], [k[1][c] for k in keys]) for c in range(3)], 1)
    img = np.repeat(col[:, None, :], W, 1).astype(np.float32)
    xx = np.arange(W)[None, :]
    img *= (0.82 + 0.18 * np.cos((xx - W / 2) / (W / 2) * 1.2))[..., None]          # soft vignette across
    img = img.astype(np.uint8)
    # faint horizontal depth marks
    for y in range(0, int(Y_FIN) + 1, 250):
        cv2.line(img, (XL, y + TOP_PAD), (XR, y + TOP_PAD), (28, 30, 62), 2, cv2.LINE_AA)
    # side panels + neon walls
    img[:, :XL] = (6, 7, 18); img[:, XR:] = (6, 7, 18)
    for x in (XL, XR): cv2.line(img, (x, 0), (x, WORLD_H), (255, 200, 80), 6, cv2.LINE_AA)
    # ramps and funnel
    for k, (x0, y0, x1, y1) in enumerate(SEGS):
        c = (255, 170, 60) if k >= 3 else (90, 225, 255)
        p0, p1 = (int(x0), int(y0) + TOP_PAD), (int(x1), int(y1) + TOP_PAD)
        cv2.line(img, p0, p1, c, int(SEG_R * 2), cv2.LINE_AA)
        cv2.circle(img, p0, int(SEG_R), c, -1, cv2.LINE_AA); cv2.circle(img, p1, int(SEG_R), c, -1, cv2.LINE_AA)
    # pegs
    for (px, py) in PEGS:
        cv2.circle(img, (int(px), int(py) + TOP_PAD), int(PEG_R), (255, 235, 225), -1, cv2.LINE_AA)
        cv2.circle(img, (int(px), int(py) + TOP_PAD), int(PEG_R) - 5, (255, 255, 255), -1, cv2.LINE_AA)
    # start gate
    cv2.rectangle(img, (XL, 52 + TOP_PAD), (XR, 70 + TOP_PAD), (120, 130, 170), -1, cv2.LINE_AA)
    # finish: checkered band
    for i, x in enumerate(range(XL, XR, 30)):
        for j in range(3):
            c = (255, 255, 255) if (i + j) % 2 == 0 else (20, 20, 30)
            cv2.rectangle(img, (x, int(Y_FIN) - 45 + j * 30 + TOP_PAD), (min(x + 30, XR), int(Y_FIN) - 15 + j * 30 + TOP_PAD), c, -1)
    pil = Image.fromarray(img); d = ImageDraw.Draw(pil)
    d.text((W / 2, Y_FIN + 140 + TOP_PAD), "FINISH", font=BLACK(110), fill=(255, 255, 255), anchor="mm", **STROKE)
    d.text((W / 2, 20 + TOP_PAD), "START", font=BLACK(30), fill=(160, 170, 210), anchor="mm")
    out = np.asarray(pil)
    return glow(np.ascontiguousarray(out), 0.45)

WORLD = make_world()

# ------------------------------------------------------------------ marble sprites
SPR_R = int(R) + 4
SZ = SPR_R * 2
SS = 5                                                       # supersampling for decoration and mask
yy_, xx_ = np.mgrid[0:SZ, 0:SZ].astype(np.float32); cxs = (SZ - 1) / 2
nx_, ny_ = (xx_ - cxs) / R, (yy_ - cxs) / R
rr_ = nx_ ** 2 + ny_ ** 2; nz_ = np.sqrt(np.clip(1 - rr_, 0, 1))
Lv = np.array([-0.45, -0.55, 0.70]); Lv /= np.linalg.norm(Lv)
DIFF = np.clip(nx_ * Lv[0] + ny_ * Lv[1] + nz_ * Lv[2], 0, 1)
SHADE = (0.38 + 0.80 * DIFF)[..., None]
Hh = Lv + np.array([0, 0, 1.0]); Hh /= np.linalg.norm(Hh)
SPEC = (np.clip(nx_ * Hh[0] + ny_ * Hh[1] + nz_ * Hh[2], 0, 1) ** 45 * 0.85)[..., None] * 255
RIM = np.clip(1 - rr_ * 0.25, 0.6, 1)[..., None]

def make_deco(i):
    s = SZ * SS; im = Image.new("RGBA", (s, s), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    c = COLS[i]; r = R * SS
    d.ellipse([s / 2 - r, s / 2 - r, s / 2 + r, s / 2 + r], fill=c + (255,))
    dark = tuple(int(v * 0.62) for v in c)
    d.rectangle([s / 2 - r, s / 2 - r * 0.17, s / 2 + r, s / 2 + r * 0.17], fill=dark + (255,))         # stripe
    d.ellipse([s / 2 - r * 0.50, s / 2 - r * 0.50, s / 2 + r * 0.50, s / 2 + r * 0.50], fill=(255, 255, 255, 255))
    d.text((s / 2, s / 2 + r * 0.03), str(i + 1), font=BLACK(r * 0.78), fill=(18, 20, 40, 255), anchor="mm")
    mask = Image.new("L", (s, s), 0); ImageDraw.Draw(mask).ellipse([s / 2 - r, s / 2 - r, s / 2 + r, s / 2 + r], fill=255)
    im.putalpha(mask)
    return np.asarray(im)
DECO = [make_deco(i) for i in range(NC)]
DECO_BG = [cv2.cvtColor(d, cv2.COLOR_RGBA2BGRA) for d in DECO]

def marble_sprite(i, ang):
    M = cv2.getRotationMatrix2D((SZ * SS / 2 - 0.5, SZ * SS / 2 - 0.5), -math.degrees(ang), 1.0)
    rot = cv2.warpAffine(DECO[i], M, (SZ * SS, SZ * SS), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    sm = cv2.resize(rot, (SZ, SZ), interpolation=cv2.INTER_AREA).astype(np.float32)
    rgb = np.clip(sm[..., :3] * SHADE * RIM + SPEC * (sm[..., 3:4] / 255), 0, 255)
    return rgb, sm[..., 3:4] / 255

def paste_sprite(arr, rgb, a, cx, cy):
    x0, y0 = int(round(cx)) - SPR_R, int(round(cy)) - SPR_R
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + SZ, arr.shape[1]), min(y0 + SZ, arr.shape[0])
    if xa >= xb or ya >= yb: return
    sl = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
    reg = arr[ya:yb, xa:xb].astype(np.float32)
    arr[ya:yb, xa:xb] = (reg * (1 - a[sl]) + rgb[sl] * a[sl]).astype(np.uint8)

# ------------------------------------------------------------------ audio
NS = int((TOTAL + 1) * SR)
audio = np.zeros(NS, np.float32)
def put(t0, sig):
    s = int(t0 * SR); n = min(len(sig), NS - s)
    if n > 0 and s >= 0: audio[s:s + n] += sig[:n].astype(np.float32)
def tone(t0, f, dur, amp, decay=7.0):
    tt = np.arange(int(dur * SR)) / SR
    put(t0, (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt)) * np.exp(-tt * decay) * np.minimum(1, tt * 800) * amp)
def kick(t0, amp=0.42):
    tt = np.arange(int(0.25 * SR)) / SR
    put(t0, np.sin(2 * np.pi * np.cumsum(130 * np.exp(-tt * 26) + 46) / SR) * np.exp(-tt * 10) * amp)
def hat(t0, a=0.08):
    tt = np.arange(int(0.05 * SR)) / SR
    nz = np.diff(np.random.default_rng(int(t0 * 1e3) % 99991).standard_normal(len(tt) + 1))
    put(t0, nz * np.exp(-tt * 70) * a)
def bass(t0, f, dur, amp=0.16):
    tt = np.arange(int(dur * SR)) / SR; ph = (f * tt) % 1.0
    put(t0, ((2 * ph - 1) * 0.6 + np.sin(2 * np.pi * f * tt)) * np.exp(-tt * 5) * np.minimum(1, tt * 500) * amp)
def noise_burst(t0, dur, amp, seed=1, lp=1):
    n = int(dur * SR); nz = np.random.default_rng(seed).standard_normal(n)
    if lp > 1: nz = np.convolve(nz, np.ones(lp) / lp, mode="same") * lp ** 0.5
    put(t0, nz * np.sin(np.pi * np.arange(n) / n) ** 2 * amp)
def click(t0, amp):
    tt = np.arange(int(0.03 * SR)) / SR
    nz = np.random.default_rng(int(t0 * 1e4) % 99991).standard_normal(len(tt))
    put(t0, (nz * 0.5 + np.sin(2 * np.pi * 1700 * tt)) * np.exp(-tt * 140) * amp)

beat = 60 / 132.0
BASS = [55.0, 55.0, 82.41, 55.0, 65.41, 55.0, 98.0, 82.41]
PENTA = [523.3, 587.3, 659.3, 784.0, 880.0, 1046.5, 1174.7, 1318.5]
tb, bi = 0.0, 0
while tb < T_FIN + 0.2:
    kick(tb); hat(tb + beat / 2)
    for h_ in range(2): bass(tb + h_ * beat / 2, BASS[(bi * 2 + h_) % 8], beat / 2)
    if tb > 5.5: hat(tb + beat / 4, 0.05); hat(tb + 3 * beat / 4, 0.05)
    tb += beat; bi += 1
last_ev = {}; n_ev = {}
for (te, kind, i, st_) in EV:
    if te > TOTAL - 0.5: continue
    q = 0.35 if te > T_FIN + 0.2 else 1.0
    key = (kind, i)
    if te - last_ev.get(key, -9) < 0.05: continue
    last_ev[key] = te
    if kind == "peg": tone(te, PENTA[(i * 2 + int(te * 7)) % len(PENTA)], 0.22, 0.10 * (0.4 + st_) * q, 16)
    elif kind == "ramp": click(te, 0.10 * (0.4 + st_) * q)
    elif kind == "bar": tone(te, 196 + 60 * i, 0.30, 0.20 * (0.4 + st_) * q, 9); click(te, 0.06 * q)
    elif kind == "bump": tone(te, 1568, 0.14, 0.16 * q, 18); tone(te, 784, 0.25, 0.13 * q, 10)
    elif kind == "clack": click(te, 0.18 * (0.4 + st_) * q)
for (te, ld) in POP:
    tone(te, 880, 0.12, 0.22, 20); tone(te + 0.08, 1320, 0.16, 0.22, 16); tone(te + 0.16, 1760, 0.3, 0.18, 9)
# riser into the finish, crowd and chord at the end
t_r = max(0.0, T_FIN - 2.4); n_r = int(2.4 * SR); tt_r = np.arange(n_r) / SR
put(t_r, np.sin(2 * np.pi * np.cumsum(300 + 900 * (tt_r / 2.4) ** 2) / SR) * (tt_r / 2.4) ** 2 * 0.10)
noise_burst(T_FIN - 0.2, 2.4, 0.14, seed=7, lp=3)
for k_, fq in enumerate([523.3, 659.3, 784.0, 1046.5]): tone(T_FIN + 0.04 * k_, fq, 1.8, 0.26, 3)
for k_, fq in enumerate([261.6, 329.6, 392.0]): tone(T_RES + 0.9 + 0.12 * k_, fq, 1.6, 0.14, 2.5)

# ------------------------------------------------------------------ camera
def cam_for(t):
    return None
CAM = np.zeros(NF + 1)
cam = -TOP_PAD
for f in range(NF + 1):
    t = min(f / FPS, T_FIN + 1.2)
    x, y, a = state(t)
    target = float(np.max(y)) - LEAD_SCREEN_Y
    if f > 0: cam += (target - cam) * 0.14
    cam = float(np.clip(cam, -TOP_PAD, WORLD_H - TOP_PAD - H))
    CAM[f] = cam if f > 0 else -TOP_PAD
CAM[0] = -TOP_PAD

# ------------------------------------------------------------------ frames
proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                         "-maxrate", "8M", "-bufsize", "16M", "-pix_fmt", "yuv420p", "marble_noaudio.mp4"], stdin=subprocess.PIPE)
chip_x = np.array([130 + i * 164 for i in range(NC)], float)
confetti = []; crng = np.random.default_rng(3)
trail = {}
F0 = None
DIM_T0 = 1.6                                              # title shrinks after this

def render(f, final_pass=True):
    global confetti
    t = f / FPS
    c0 = int(round(CAM[f])); r0 = c0 + TOP_PAD
    arr = WORLD[r0:r0 + H].copy()
    x, y, ang = state(t)
    # spinning bars + bumpers (dynamic)
    sp = S.spin_ends(t, SHIFT)
    lay = np.zeros_like(arr)
    for b in range(len(SPIN)):
        p0 = (int(sp[b, 0]), int(sp[b, 1]) - c0); p1 = (int(sp[b, 2]), int(sp[b, 3]) - c0)
        if max(p0[1], p1[1]) < -50 or min(p0[1], p1[1]) > H + 50: continue
        cv2.line(lay, p0, p1, (200, 80, 255), int(SPIN_R * 2), cv2.LINE_AA)
        cv2.circle(lay, p0, int(SPIN_R), (200, 80, 255), -1, cv2.LINE_AA); cv2.circle(lay, p1, int(SPIN_R), (200, 80, 255), -1, cv2.LINE_AA)
        cv2.circle(lay, (int(SPIN[b, 0]), int(SPIN[b, 1]) - c0), 15, (255, 255, 255), -1, cv2.LINE_AA)
    for k, (bx, by) in enumerate(BUMP):
        near = np.min(np.hypot(x - bx, y - by)) - (BUMP_R + R)
        pul = 1.0 + 0.20 * max(0.0, 1 - max(near, 0) / 14)
        cy_ = int(by) - c0
        if -100 < cy_ < H + 100:
            cv2.circle(lay, (int(bx), cy_), int(BUMP_R * pul), (60, 210, 255), -1, cv2.LINE_AA)
            cv2.circle(lay, (int(bx), cy_), int(BUMP_R * pul * 0.72), (24, 30, 70), -1, cv2.LINE_AA)
            cv2.circle(lay, (int(bx), cy_), int(BUMP_R * pul * 0.30), (60, 210, 255), -1, cv2.LINE_AA)
    arr = np.where(lay > 0, lay, arr)
    # trails
    tl = np.zeros_like(arr)
    for i in range(NC):
        for k in range(1, 9):
            tx_, ty_, _ = state(max(0.0, t - k * 0.018))
            cv2.circle(tl, (int(tx_[i]), int(ty_[i]) - c0), int(R * (1 - k / 11)), tuple(int(v * 0.5 * (1 - k / 9)) for v in COLS[i]), -1, cv2.LINE_AA)
    arr = cv2.add(arr, cv2.GaussianBlur(tl, (0, 0), 4))
    # marbles (leaders on top)
    order = np.argsort(y)
    for i in order:
        cy_ = y[i] - c0
        if -60 < cy_ < H + 60:
            cv2.ellipse(arr, (int(x[i] + 6), int(cy_ + 10)), (int(R * 0.95), int(R * 0.7)), 0, 0, 360, (4, 4, 10), -1, cv2.LINE_AA)
    for i in order:
        cy_ = y[i] - c0
        if -60 < cy_ < H + 60:
            rgb, al = marble_sprite(i, ang[i]); paste_sprite(arr, rgb, al, x[i], cy_)
    # confetti
    if t >= T_FIN and not confetti and f > 0:
        for _ in range(140):
            confetti.append([crng.uniform(XL + 40, XR - 40), (Y_FIN - c0) - 80, crng.uniform(-420, 420), crng.uniform(-1100, -300),
                             COLS[WIN] if crng.random() < 0.5 else (255, 255, 255) if crng.random() < 0.5 else COLS[int(crng.integers(0, NC))]])
    keep = []
    for q in confetti:
        q[0] += q[2] / FPS; q[1] += q[3] / FPS; q[3] += 1500 / FPS
        if q[1] < H + 40:
            cv2.rectangle(arr, (int(q[0]), int(q[1])), (int(q[0]) + 9, int(q[1]) + 14), q[4], -1); keep.append(q)
    confetti = keep

    img = Image.fromarray(arr); d = ImageDraw.Draw(img)
    # ---- ranking from y ----
    rank = np.argsort(-y)                                   # rank[0] = leader
    pos = np.empty(NC, int); pos[rank] = np.arange(NC)
    lead = int(rank[0])
    # ---- title ----
    k_t = ease_out((t - DIM_T0) / 0.5)
    ty = 150 - 70 * k_t; fs = 118 - 46 * k_t
    d.text((W / 2, ty), "PICK A MARBLE", font=BLACK(fs), fill=(255, 255, 255), anchor="mm", **STROKE)
    d.text((W / 2, ty + 112 - 36 * k_t), "1 – 6", font=BLACK(150 - 70 * k_t), fill=(255, 212, 59), anchor="mm", **STROKE)
    # ---- ranking chips (slide to their rank) ----
    for i in range(NC):
        chip_x[i] += ((130 + pos[i] * 164 if t > 0.05 else 130 + i * 164) - chip_x[i]) * 0.28
    for i in sorted(range(NC), key=lambda c: -pos[c]):
        cx = chip_x[i]; top_ = 330 - (8 if (pos[i] == 0 and t < T_FIN) else 0)
        d.rounded_rectangle([cx - 66, top_, cx + 66, top_ + 120], radius=22, fill=COLS[i],
                            outline=(255, 255, 255) if pos[i] == 0 and t < T_FIN else None, width=5)
        d.text((cx, top_ + 48), str(i + 1), font=BLACK(70), fill=(14, 18, 36), anchor="mm")
        lab = "1ST" if pos[i] == 0 else "2ND" if pos[i] == 1 else "3RD" if pos[i] == 2 else f"{pos[i] + 1}TH"
        d.text((cx, top_ + 98), lab, font=BOLD(26), fill=(14, 18, 36), anchor="mm")
    # ---- progress bar on the right panel ----
    bx0, by0, by1 = 1045, 560, 1750
    d.rounded_rectangle([bx0 - 4, by0, bx0 + 4, by1], radius=4, fill=(40, 44, 76))
    for i in sorted(range(NC), key=lambda c: -pos[c]):
        p = min(1.0, max(0.0, y[i] / Y_FIN)); yy = by0 + (by1 - by0) * p
        d.ellipse([bx0 - 11, yy - 11, bx0 + 11, yy + 11], fill=COLS[i], outline=(10, 10, 20), width=2)
    # ---- popups ----
    for (te, ld) in POP:
        age = t - te
        if 0 <= age < 1.3:
            a = min(1.0, age / 0.12) * (1.0 if age < 0.9 else 1 - (age - 0.9) / 0.4)
            parts = [(f"MARBLE {ld + 1}", COLS[ld]), (" TAKES THE LEAD!", (255, 255, 255))]
            fnt = BLACK(58)
            tw = sum(d.textlength(p[0], font=fnt) for p in parts); xx0 = W / 2 - tw / 2
            for txt, c in parts:
                d.text((xx0, 520), txt, font=fnt, fill=tc(c, a), anchor="lm", stroke_width=6, stroke_fill=tc((5, 6, 14), a)); xx0 += d.textlength(txt, font=fnt)
    if LAST_T is not None and abs(t - (LAST_T + 0.6)) < 0.0 : pass
    if y.max() > Y_FIN - 800 and t < T_FIN - 0.1:
        a = 0.5 + 0.5 * math.sin(t * 10)
        d.text((W / 2, 640), "FINAL STRETCH", font=BLACK(54), fill=tc((255, 92, 122), 0.5 + 0.5 * a), anchor="mm", **STROKE)
    d.text((W / 2, 1855), "COMMENT YOUR NUMBER  •  PYTHONIC LIFE", font=SEMI(30), fill=(150, 158, 196), anchor="mm", **STROKE)

    # ---- finish / result ----
    if T_FIN <= t < T_RES + 0.1:
        fl = max(0.0, 1 - (t - T_FIN) / 0.4)
        if fl > 0:
            ov = Image.new("RGBA", (W, H), (255, 255, 255, int(190 * fl))); img.paste(ov, (0, 0), ov); d = ImageDraw.Draw(img)
        d.text((W / 2, 900), "FINISH!", font=BLACK(190 * max(ease_back((t - T_FIN) / 0.35), .01)), fill=(255, 212, 59), anchor="mm", **STROKE)
    if t >= T_RES:
        rtt = t - T_RES
        ov = Image.new("RGBA", (W, H), (5, 6, 14, int(238 * min(1, rtt / 0.3)))); img.paste(ov, (0, 0), ov); d = ImageDraw.Draw(img)
        d.text((W / 2, 560), f"MARBLE {WIN + 1}", font=BLACK(170 * max(ease_back(rtt / 0.45), .01)), fill=COLS[WIN], anchor="mm", **STROKE)
        d.text((W / 2, 730), "WINS!", font=BLACK(140 * max(ease_back((rtt - 0.1) / 0.45), .01)), fill=(255, 255, 255), anchor="mm", **STROKE)
        k2 = ease_out((rtt - 0.6) / 0.4)
        if k2 > 0:
            d.text((W / 2, 860), f"WON BY {MARGIN:.2f} SECONDS", font=BLACK(54), fill=tc((255, 212, 59), k2), anchor="mm")
            if LAST_T is not None:
                d.text((W / 2, 940), f"WAS LAST PLACE AT {LAST_T:.0f}s", font=BOLD(42), fill=tc((255, 92, 122), k2), anchor="mm")
        for rk in range(3):
            k3 = ease_out((rtt - 0.9 - rk * 0.2) / 0.35)
            if k3 <= 0: continue
            i = ORDER_F[rk]; yy = 1080 + rk * 92
            d.ellipse([260, yy - 24, 308, yy + 24], fill=tc(COLS[i], k3))
            d.text((335, yy), f"{rk + 1}   MARBLE {i + 1}", font=BLACK(50), fill=tc((255, 255, 255), k3), anchor="lm")
        k4 = ease_back((rtt - 1.6) / 0.4)
        if k4 > 0:
            d.text((W / 2, 1450), "DID YOU PICK IT?", font=BLACK(80 * max(k4, .01)), fill=(255, 255, 255), anchor="mm", **STROKE)
            d.text((W / 2, 1560), "Comment your number", font=BOLD(46), fill=tc((255, 92, 122), min(1, k4)), anchor="mm")
        k5 = ease_out((rtt - 2.1) / 0.4)
        if k5 > 0:
            d.text((W / 2, 1690), "PYTHONIC LIFE  •  100% Python", font=BOLD(40), fill=tc((255, 212, 59), k5), anchor="mm")

    out = glow(np.asarray(img))
    return out

proc_frames = 0
for f in range(NF):
    t = f / FPS
    out = render(f)
    if f == 0: F0 = out.copy()
    if t > T_LOOP:                                         # crossfade into frame 0 so the video loops seamlessly
        a = (t - T_LOOP) / (TOTAL - T_LOOP - 1 / FPS)
        out = cv2.addWeighted(out, 1 - min(1, a), F0, min(1, a), 0)
    proc.stdin.write(out.tobytes())
proc.stdin.close(); proc.wait()

a = audio[: int(TOTAL * SR)].copy()
a = np.tanh(a / (np.percentile(np.abs(a), 99.9) + 1e-6) * 1.1) * 0.92
nfd = int(0.12 * SR); a[-nfd:] *= np.linspace(1, 0.15, nfd)
with wave.open("marble_audio.wav", "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR); wf.writeframes((a * 32767).astype(np.int16).tobytes())
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "marble_noaudio.mp4", "-i", "marble_audio.wav", "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k", "-shortest", "pythonic_life_marble_race.mp4"], check=True)
print("done", round(TOTAL, 1))
