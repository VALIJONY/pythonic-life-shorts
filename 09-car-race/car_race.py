"""Pythonic Life — PICK A CAR: 6 cars, 3 laps, 1 winner (YouTube Shorts 1080x1920)
Top-down racing simulation in plain Python + NumPy: cornering speed limits, braking, slipstream, boost,
spin-outs and a light rubber-band catch-up so the race stays close. Rendered with Pillow + OpenCV,
engines and music synthesized in NumPy, encoded with ffmpeg.
Usage: python3 car_race.py [seed]        (python3 car_race.py probe -> scan seeds)
"""
import math, subprocess, sys, wave
from functools import lru_cache
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SR = 1080, 1920, 30, 44100
LAPS, NC = 3, 6
NAMES = ["RED", "BLUE", "GREEN", "YELLOW", "PINK", "CYAN"]
COLS = [(255, 62, 84), (72, 132, 255), (62, 232, 122), (255, 208, 44), (255, 104, 202), (64, 226, 236)]
TRACK_W = 128
CAR_L, CAR_W = 70, 36
SUBSTEPS = 4
DT = 1.0 / (FPS * SUBSTEPS)
RACE_TARGET = 21.0                  # seconds the winner needs for 3 laps (the sim is calibrated to it)
ASPHALT = (52, 56, 74)

# ---------------- track: a closed, smooth, never self-crossing loop ----------------
def build_track():
    th = np.linspace(0, 2 * np.pi, 8000, endpoint=False)
    r = (1 + 0.17 * np.cos(2 * th + 0.6) + 0.13 * np.cos(3 * th + 2.1)
         + 0.07 * np.cos(5 * th + 0.4) + 0.035 * np.cos(7 * th + 1.0))
    x, y = r * np.cos(th), r * np.sin(th)
    x0, x1, y0, y1 = 100, 980, 520, 1450                       # board area for the centre line
    x = x0 + (x - x.min()) / (x.max() - x.min()) * (x1 - x0)
    y = y0 + (y - y.min()) / (y.max() - y.min()) * (y1 - y0)
    px, py = np.append(x, x[0]), np.append(y, y[0])
    cum = np.concatenate([[0], np.cumsum(np.hypot(np.diff(px), np.diff(py)))])
    L = cum[-1]; n = int(L // 2.0); dse = L / n
    s = np.arange(n) * dse
    TX, TY = np.interp(s, cum, px), np.interp(s, cum, py)
    tx = np.roll(TX, -1) - np.roll(TX, 1); ty = np.roll(TY, -1) - np.roll(TY, 1)
    nr = np.hypot(tx, ty); tx, ty = tx / nr, ty / nr
    head = np.arctan2(ty, tx)
    dh = np.angle(np.exp(1j * (np.roll(head, -3) - np.roll(head, 3)))) / (6 * dse)
    k = np.convolve(np.concatenate([dh[-8:], dh, dh[:8]]), np.ones(15) / 15, mode="same")[8:-8]
    kabs = np.abs(k)
    klook = kabs.copy()
    for j in range(1, 45): klook = np.maximum(klook, np.roll(kabs, -j))      # brake for what is coming
    return TX, TY, tx, ty, kabs, klook, L, n, dse

TX, TY, TTX, TTY, KAP, KLOOK, L, N, DSE = build_track()

def at(s):
    u = (s / DSE) % N
    i = int(u); fr = u - i; j = (i + 1) % N; i %= N
    x = TX[i] * (1 - fr) + TX[j] * fr; y = TY[i] * (1 - fr) + TY[j] * fr
    tx = TTX[i] * (1 - fr) + TTX[j] * fr; ty = TTY[i] * (1 - fr) + TTY[j] * fr
    nr = math.hypot(tx, ty)
    return x, y, tx / nr, ty / nr

# ---------------- race simulation ----------------
def simulate(seed, f=1.0):
    rnd = np.random.default_rng(seed)
    top = 1 + rnd.normal(0, 0.010, NC); cor = 1 + rnd.normal(0, 0.012, NC)
    boosts = [sorted(rnd.uniform(2.5, 17.0, int(rnd.integers(1, 3)))) for _ in range(NC)]
    mist = [float(rnd.uniform(3.0, 17.0)) if rnd.random() < 0.6 else None for _ in range(NC)]
    VTOP, ALAT, ACC, BRK = 470 * f, 720 * f * f, 300 * f * f, 640 * f * f
    grid = rnd.permutation(NC)
    D = np.zeros(NC); home = np.zeros(NC)
    for slot, car in enumerate(grid):
        D[car] = -(slot // 2 + 1) * 80.0 - (slot % 2) * 32.0
        home[car] = -30.0 if slot % 2 == 0 else 30.0
    lat = home.copy(); v = np.zeros(NC); form = np.zeros(NC)
    D0, lat0 = D.copy(), lat.copy()
    t = 0.0; frames = []; finish = [None] * NC
    order = [int(i) for i in np.argsort(-D)]
    prev_pos = np.argsort(np.argsort(-D)); spun = [False] * NC
    while True:
        for _ in range(SUBSTEPS):
            Dlead = D.max()
            for i in range(NC):
                kk = KLOOK[int((D[i] % L) / DSE) % N]
                vt = min(VTOP * top[i], math.sqrt(ALAT / max(kk, 1e-4)) * cor[i])
                form[i] += -form[i] / 2.5 * DT + 0.02 * math.sqrt(2 * DT / 2.5) * rnd.normal()
                m = 1.0 + form[i]
                for bt in boosts[i]:
                    if bt <= t < bt + 1.2: m *= 1.07
                if mist[i] is not None and mist[i] <= t < mist[i] + 0.9: m *= 0.6
                for j in range(NC):                                          # slipstream
                    g = D[j] - D[i]
                    if j != i and 0 < g < 105 and abs(lat[j] - lat[i]) < 36:
                        m *= 1 + 0.02 * (1 - g / 105); break
                m *= 1 + min(0.012, (Dlead - D[i]) / 12000)                  # light rubber band
                vt *= m
                if v[i] < vt: v[i] = min(vt, v[i] + ACC * DT)
                else: v[i] = max(vt, v[i] - BRK * DT)
            D += v * DT; t += DT
            for i in range(NC):
                if finish[i] is None and D[i] >= LAPS * L:
                    finish[i] = t - (D[i] - LAPS * L) / max(v[i], 1.0)
        for i in range(NC):                                                  # lanes / overtaking
            tgt = home[i]; best = 1e9; jn = -1
            for j in range(NC):
                g = D[j] - D[i]
                if j != i and 0 < g < 98 and abs(lat[j] - lat[i]) < 32 and g < best: best, jn = g, j
            if jn >= 0:
                tgt = lat[jn] + 42 if lat[jn] <= 0 else lat[jn] - 42
                tgt = max(-44.0, min(44.0, tgt))
            lat[i] += max(-80 / FPS, min(80 / FPS, tgt - lat[i]))
        spin = np.zeros(NC); boost = np.zeros(NC)
        for i in range(NC):
            if mist[i] is not None and mist[i] <= t < mist[i] + 0.9:
                u = (t - mist[i]) / 0.9; spin[i] = 2 * math.pi * u * u * (3 - 2 * u)
            boost[i] = any(bt <= t < bt + 1.2 for bt in boosts[i])
        fin_mask = [finish[i] is not None for i in range(NC)]
        Drk = np.array([1e6 - finish[i] * 1000 if fin_mask[i] else D[i] for i in range(NC)])
        for _ in range(NC):                       # sticky order: a car must be 18 px ahead to count as a pass
            changed = False
            for k in range(NC - 1):
                a_, b_ = order[k], order[k + 1]
                if Drk[b_] - Drk[a_] > 18.0: order[k], order[k + 1] = b_, a_; changed = True
            if not changed: break
        pos = np.empty(NC, int); pos[order] = np.arange(NC)
        evs = []
        if t > 1.5:
            for i in range(NC):
                if pos[i] < prev_pos[i] and not fin_mask[i]:
                    for j in range(NC):
                        if pos[j] > prev_pos[j] and prev_pos[j] == pos[i] and not fin_mask[j]:
                            evs.append(("pass", i, j, int(pos[i])))
        for i in range(NC):
            if mist[i] is not None and not spun[i] and t >= mist[i]:
                spun[i] = True; evs.append(("spin", i, -1, int(pos[i])))
        prev_pos = pos
        frames.append(dict(t=t, D=D.copy(), v=v.copy(), lat=lat.copy(), pos=pos.copy(), spin=spin, boost=boost, ev=evs))
        done = all(x is not None for x in finish)
        if (done and t > max(finish) + 1.0) or t > 60: break
    winner = int(np.argmin(finish)); fs = sorted(finish)
    passes = sum(1 for fr in frames for e in fr["ev"] if e[0] == "pass" and e[3] <= 2)
    lead_changes = sum(1 for fr in frames for e in fr["ev"] if e[0] == "pass" and e[3] == 0)
    return dict(frames=frames, finish=finish, winner=winner, margin=fs[1] - fs[0], twin=fs[0], passes=passes,
                lead_changes=lead_changes, D0=D0, lat0=lat0, alat=ALAT, f=f)

def run(seed):
    r = simulate(seed, 1.0)
    return simulate(seed, r["twin"] / RACE_TARGET)

if len(sys.argv) > 1 and sys.argv[1] == "probe":
    print("lap length", round(L), "px, min corner radius", round(1 / KAP.max()), "px")
    for s in range(1, 61):
        r = run(s); order = sorted(range(NC), key=lambda i: r["finish"][i])
        print(s, "win", NAMES[r["winner"]], "margin %.3f" % r["margin"], "t %.1f" % r["twin"],
              "passes", r["passes"], "lead-changes", r["lead_changes"], "order", [NAMES[i][:3] for i in order])
    sys.exit()

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
R = run(SEED)
FR, FIN, WINNER, TWIN, MARGIN = R["frames"], R["finish"], R["winner"], R["twin"], R["margin"]
print("seed", SEED, "winner", NAMES[WINNER], "margin %.3f" % MARGIN, "t %.1f" % TWIN,
      "passes", R["passes"], "lead changes", R["lead_changes"])
ORDER_F = sorted(range(NC), key=lambda i: FIN[i])

T_CD, T0 = 2.4, 4.2
TW = T0 + TWIN
T_RES, T_OUT = TW + 1.3, TW + 5.0
TOTAL = T_OUT + 3.6
NF = int(TOTAL * FPS)
INIT = dict(t=0.0, D=R["D0"], v=np.zeros(NC), lat=R["lat0"], spin=np.zeros(NC), boost=np.zeros(NC),
            pos=np.argsort(np.argsort(-R["D0"])), ev=[])
def state_at(rt):
    return INIT if rt < 0 else FR[min(int(rt * FPS), len(FR) - 1)]

# ---------------- helpers ----------------
F = "/usr/share/fonts/opentype/inter/"
@lru_cache(None)
def font(n, s): return ImageFont.truetype(F + n, max(1, int(s)))
@lru_cache(None)
def mono(s): return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", max(1, int(s)))
BLACK = lambda s: font("Inter-Black.otf", s)
BOLD = lambda s: font("Inter-Bold.otf", s)
SEMI = lambda s: font("Inter-SemiBold.otf", s)
def ease_out(x): x = max(0, min(1, x)); return 1 - (1 - x) ** 3
def ease_back(x): x = max(0, min(1, x)); return 1 + 2.70158 * (x - 1) ** 3 + 1.70158 * (x - 1) ** 2
def tc(c, a=1.0): return tuple(int(min(255, max(0, v * a))) for v in c)
STROKE = dict(stroke_width=6, stroke_fill=(5, 6, 14))

def glow(a, s=0.3):
    sm = cv2.resize(a, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    b1 = cv2.GaussianBlur(sm, (0, 0), 3); b2 = cv2.GaussianBlur(sm, (0, 0), 10)
    return cv2.add(cv2.addWeighted(a, 1, cv2.resize(b1, (W, H)), s, 0), (cv2.resize(b2, (W, H)) * 0.2).astype(np.uint8))

# ---------------- static scene ----------------
def make_base():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    img = np.zeros((H, W, 3), np.float32); img[...] = (8, 10, 22)
    img += (np.exp(-(((xx - W / 2) / 650) ** 2 + ((yy - 980) / 700) ** 2)) * 22)[..., None] * np.array([0.5, 0.7, 1.3])
    img[:, ::60] += 4; img[::60, :] += 4
    base = np.clip(img, 0, 255).astype(np.uint8)
    pts = np.stack([TX, TY], 1).astype(np.int32)
    nx, ny = -TTY, TTX
    for side in (1, -1):                                                     # red/white kerbs
        ex = (TX + nx * side * 70).astype(np.int32); ey = (TY + ny * side * 70).astype(np.int32)
        for i in range(0, N, 5):
            j = (i + 5) % N
            cv2.line(base, (ex[i], ey[i]), (ex[j], ey[j]), (235, 235, 240) if (i // 5) % 2 else (230, 50, 70), 12, cv2.LINE_AA)
    cv2.polylines(base, [pts], True, ASPHALT, TRACK_W, cv2.LINE_AA)
    for i in range(0, N, 14):                                                # dashed centre line
        j = min(i + 5, N - 1)
        cv2.line(base, (int(TX[i]), int(TY[i])), (int(TX[j]), int(TY[j])), (96, 102, 126), 3, cv2.LINE_AA)
    x, y, tx, ty = at(0.0); nxx, nyy = -ty, tx                               # start / finish line
    for row in range(2):
        for col in range(8):
            c = np.array([x, y]) + np.array([nxx, nyy]) * ((col - 3.5) * 14) + np.array([tx, ty]) * ((row - 0.5) * 14)
            q = [c + np.array([tx, ty]) * a * 7 + np.array([nxx, nyy]) * b * 7 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            cv2.fillConvexPoly(base, np.array(q, np.int32), (245, 245, 250) if (row + col) % 2 else (14, 14, 22), cv2.LINE_AA)
    for slot in range(NC):                                                   # grid boxes
        d_ = -(slot // 2 + 1) * 80.0 - (slot % 2) * 32.0; la = -30.0 if slot % 2 == 0 else 30.0
        gx, gy, gtx, gty = at(d_); cx, cy = gx - gty * la, gy + gtx * la
        q = [np.array([cx, cy]) + np.array([gtx, gty]) * a * 38 + np.array([-gty, gtx]) * b * 21 for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        cv2.polylines(base, [np.array(q, np.int32)], True, (112, 118, 146), 2, cv2.LINE_AA)
    im = Image.fromarray(base); d = ImageDraw.Draw(im)
    d.text((W / 2, 930), "PYTHONIC", font=BLACK(110), fill=(26, 32, 58), anchor="mm")
    d.text((W / 2, 1040), "LIFE", font=BLACK(110), fill=(34, 38, 60), anchor="mm")
    return np.asarray(im).copy()
BASE = make_base()

def make_sprite(col):
    k = 4
    Wc, Hc = (CAR_L + 12) * k, (CAR_W + 12) * k
    im = Image.new("RGBA", (Wc, Hc), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    ox, oy, L_, W_ = 6 * k, 6 * k, CAR_L * k, CAR_W * k
    dark = tuple(int(c * 0.45) for c in col) + (255,)
    for fx in (0.14, 0.60):                                                  # wheels
        for fy in (-0.08, 0.78):
            d.rounded_rectangle([ox + fx * L_, oy + fy * W_, ox + (fx + 0.24) * L_, oy + (fy + 0.30) * W_], radius=2 * k, fill=(14, 14, 20, 255))
    d.rounded_rectangle([ox + 0.88 * L_, oy + 0.02 * W_, ox + 1.0 * L_, oy + 0.98 * W_], radius=2 * k, fill=dark)   # front wing
    d.rounded_rectangle([ox + 0.0 * L_, oy + 0.04 * W_, ox + 0.12 * L_, oy + 0.96 * W_], radius=2 * k, fill=dark)   # rear wing
    d.rounded_rectangle([ox + 0.08 * L_, oy + 0.14 * W_, ox + 0.92 * L_, oy + 0.86 * W_], radius=int(0.30 * W_),
                        fill=col + (255,), outline=(12, 14, 28, 255), width=int(1.4 * k))
    d.rounded_rectangle([ox + 0.40 * L_, oy + 0.30 * W_, ox + 0.66 * L_, oy + 0.70 * W_], radius=int(0.14 * W_), fill=(18, 22, 40, 255))
    d.rectangle([ox + 0.14 * L_, oy + 0.46 * W_, ox + 0.36 * L_, oy + 0.54 * W_], fill=(255, 255, 255, 215))
    d.rectangle([ox + 0.69 * L_, oy + 0.46 * W_, ox + 0.86 * L_, oy + 0.54 * W_], fill=(255, 255, 255, 215))
    return im.resize((Wc // k * 2, Hc // k * 2), Image.LANCZOS)
SPR = [make_sprite(c) for c in COLS]

def paste_car(img, spr2, cx, cy, ang):
    rot = spr2.rotate(-math.degrees(ang), resample=Image.BICUBIC, expand=True)
    small = rot.resize((max(1, rot.width // 2), max(1, rot.height // 2)), Image.LANCZOS)
    sh = Image.new("RGBA", small.size, (0, 0, 0, 0)); sh.putalpha(small.getchannel("A").point(lambda a: int(a * 0.4)))
    px, py = int(cx - small.width / 2), int(cy - small.height / 2)
    img.paste(sh, (px + 4, py + 5), sh); img.paste(small, (px, py), small)

def multi_text(d, y, parts, fnt, a=1.0):
    ws = [d.textlength(t, font=fnt) for t, _ in parts]; x = W / 2 - sum(ws) / 2
    for (t, c), w in zip(parts, ws):
        d.text((x, y), t, font=fnt, fill=tc(c, a), anchor="lm", **STROKE); x += w

# ---------------- popups + audio events ----------------
POP = []; last_t = -9
for k, fr in enumerate(FR):
    for e in fr["ev"]:
        te = T0 + k / FPS
        if (e[0] == "pass" and e[3] <= 2) or e[0] == "spin":
            if te - last_t > 0.9: POP.append((te, ) + e); last_t = te

# ---------------- audio ----------------
NS = int((TOTAL + 2) * SR)
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

# engines: one sawtooth voice per car, pitch follows the car's speed
nf_tot = NF + 3; tf = np.arange(nf_tot) / FPS; tt_a = np.arange(NS) / SR
vs = np.zeros((nf_tot, NC))
for f in range(nf_tot):
    rt = f / FPS - T0
    if rt >= 0: vs[f] = FR[min(int(rt * FPS), len(FR) - 1)]["v"]
rev = 38 * np.clip(tf - (T0 - 1.0), 0, 1) * (tf < T0)
gain = np.interp(tt_a, [0, 1.0, T0, TW + 1.0, TW + 3.0, T_OUT, T_OUT + 0.8, TOTAL + 2], [0.0, 0.45, 0.8, 1.0, 0.25, 0.2, 0.0, 0.0])
eng = np.zeros(NS)
for i in range(NC):
    fr_f = (58 + rev + 0.17 * vs[:, i]) * (1 + 0.035 * i)
    ph = np.cumsum(np.interp(tt_a, tf, fr_f)) / SR
    w = (2 * (ph % 1.0) - 1) * 0.5 + 0.5 * np.sin(2 * np.pi * 2 * ph)
    eng += np.convolve(w, np.ones(10) / 10, mode="same") * 0.022
audio += (eng * gain).astype(np.float32)

beat = 60 / 132.0
BASS = [55.0, 55.0, 82.41, 55.0, 65.41, 55.0, 98.0, 82.41]
t_lap3 = next((T0 + k / FPS for k, fr in enumerate(FR) if fr["D"].max() >= 2 * L), TW)
tb, bi = T0, 0
while tb < TW + 0.4:
    kick(tb); hat(tb + beat / 2)
    for h_ in range(2):
        bass(tb + h_ * beat / 2, BASS[(bi * 2 + h_) % 8], beat / 2)
    if tb > t_lap3: hat(tb + beat / 4, 0.05); hat(tb + 3 * beat / 4, 0.05)
    tb += beat; bi += 1
for i, tcd in enumerate([T_CD, T_CD + 0.6, T_CD + 1.2]): tone(tcd, 660, 0.25, 0.4, 9)
tone(T0, 1320, 0.6, 0.45, 5); tone(T0, 990, 0.6, 0.3, 5)
for (te, kind, i, j, p) in POP:
    if kind == "pass":
        tone(te, 880, 0.12, 0.22, 20); tone(te + 0.08, 1320, 0.16, 0.22, 16)
        if p == 0: tone(te + 0.18, 1760, 0.3, 0.2, 9)
    else:
        noise_burst(te, 0.8, 0.12, seed=int(te * 100), lp=6)
noise_burst(TW - 0.3, 2.6, 0.14, seed=7, lp=3)                                # crowd swells at the finish
for k_, fq in enumerate([523.3, 659.3, 784.0, 1046.5]): tone(TW + 0.05 * k_, fq, 1.6, 0.25, 3)
for k_, fq in enumerate([261.6, 329.6, 392.0, 523.3]): tone(T_OUT + 0.12 * k_, fq, 2.2, 0.22, 2.5)

# ---------------- render ----------------
proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                         "-maxrate", "8M", "-bufsize", "16M", "-pix_fmt", "yuv420p", "car_noaudio.mp4"], stdin=subprocess.PIPE)
skid = np.zeros((H, W, 3), np.uint8); skid_prev = {}
particles = []; prng = np.random.default_rng(4)
chip_x = np.array([35 + 80 + int(INIT["pos"][i]) * 165 for i in range(NC)], float)
DIM = Image.new("RGBA", (W, H), (0, 0, 0, 110))
for f in range(NF):
    t = f / FPS; rt = t - T0
    fr = state_at(rt)
    arr = cv2.subtract(BASE, skid) if rt > 0 else BASE.copy()
    D, v, lat, spin, boost, pos = fr["D"], fr["v"], fr["lat"], fr["spin"], fr["boost"], fr["pos"]
    cars = []
    for i in range(NC):
        x, y, tx, ty = at(D[i])
        cx, cy = x - ty * lat[i], y + tx * lat[i]
        ang = math.atan2(ty, tx) + spin[i]
        cars.append((i, cx, cy, ang))
        if rt > 0:
            dx, dy = math.cos(ang), math.sin(ang)
            cornering = v[i] ** 2 * KAP[int((D[i] % L) / DSE) % N] > 0.7 * R["alat"]
            for sd in (-1, 1):
                wp = (int(cx - dx * 26 - dy * sd * 12), int(cy - dy * 26 + dx * sd * 12))
                if (cornering and v[i] > 150) or spin[i] > 0:
                    if skid_prev.get((i, sd)) is not None: cv2.line(skid, skid_prev[(i, sd)], wp, (44, 44, 44), 3, cv2.LINE_AA)
                    skid_prev[(i, sd)] = wp
                else:
                    skid_prev[(i, sd)] = None
            if boost[i]:
                for _ in range(2):
                    particles.append([cx - dx * 38, cy - dy * 38, -dx * 90 + prng.normal(0, 40), -dy * 90 + prng.normal(0, 40), 0.0, prng.uniform(0.2, 0.35), "fire"])
            if spin[i] > 0:
                for _ in range(3):
                    particles.append([cx, cy, prng.normal(0, 60), prng.normal(0, 60), 0.0, prng.uniform(0.7, 1.2), "smoke"])
    keep = []; smoke_layer = None
    for q in particles:
        q[4] += 1 / FPS
        if q[4] > q[5]: continue
        q[0] += q[2] / FPS; q[1] += q[3] / FPS
        a = 1 - q[4] / q[5]
        if q[6] == "fire":
            col = tuple(int(ASPHALT[c] * (1 - a) + (255, 150, 40)[c] * a) for c in range(3)); rad = max(1, int(5 * a))
        else:
            if smoke_layer is None: smoke_layer = np.zeros((H, W, 3), np.uint8)
            cv2.circle(smoke_layer, (int(q[0]), int(q[1])), int(7 + 14 * q[4]), (int(70 * a),) * 3, -1)
            keep.append(q); continue
        cv2.circle(arr, (int(q[0]), int(q[1])), rad, col, -1, cv2.LINE_AA)
        keep.append(q)
    particles = keep
    if smoke_layer is not None:
        arr = cv2.add(arr, cv2.GaussianBlur(smoke_layer, (0, 0), 7))

    img = Image.fromarray(arr); d = ImageDraw.Draw(img)
    for (i, cx, cy, ang) in sorted(cars, key=lambda c: -pos[c[0]]):
        paste_car(img, SPR[i], cx, cy, ang)

    # ----- HUD -----
    order = sorted(range(NC), key=lambda i: pos[i]); lead = order[0]
    Dlead = float(D[lead]); lap = int(min(LAPS, max(0, Dlead) // L + 1))
    if t < T0:
        k1 = ease_back(t / 0.5)
        d.text((W / 2, 150), "PICK A CAR", font=BLACK(118 * max(k1, .01)), fill=(255, 255, 255), anchor="mm", **STROKE)
        d.text((W / 2, 270), "ONLY ONE WINS", font=BLACK(84 * max(ease_back((t - 0.25) / 0.5), .01)), fill=(255, 212, 59), anchor="mm", **STROKE)
        k2 = ease_out((t - 0.7) / 0.5)
        d.text((W / 2, 372), "6 cars  •  3 laps  •  no rules", font=SEMI(38), fill=tc((190, 198, 228), k2), anchor="mm", **STROKE)
        if t >= T_CD:
            img.paste(DIM, (0, 0), DIM); d = ImageDraw.Draw(img)
            ci = int((t - T_CD) / 0.6); ph = ((t - T_CD) % 0.6) / 0.6
            d.text((W / 2, 960), ["3", "2", "1"][min(ci, 2)], font=BLACK(380 * (1.25 - 0.25 * ease_out(ph * 2))), fill=(255, 255, 255), anchor="mm", **STROKE)
    elif t < TW:
        d.text((W / 2, 140), f"LAP {lap}/{LAPS}", font=BLACK(108), fill=(255, 255, 255), anchor="mm", **STROKE)
        multi_text(d, 238, [("LEADING: ", (190, 198, 228)), (NAMES[lead], COLS[lead])], BOLD(46))
        prog = max(0.0, min(1.0, Dlead / (LAPS * L)))
        d.rounded_rectangle([160, 395, 920, 409], radius=7, fill=(30, 34, 52))
        d.rounded_rectangle([160, 395, 160 + 760 * prog, 409], radius=7, fill=COLS[lead])
        if rt < 0.7:
            a = 1 - rt / 0.7
            d.text((W / 2, 960), "GO!", font=BLACK(int(300 * (1 + 0.4 * (1 - a)))), fill=tc((140, 255, 160), a), anchor="mm", **STROKE)
        for (te, kind, i, j, p) in POP:
            age = t - te
            if 0 <= age < 1.5:
                a = min(1.0, age / 0.15) * (1.0 if age < 1.1 else 1 - (age - 1.1) / 0.4)
                if kind == "spin": multi_text(d, 330, [(NAMES[i], COLS[i]), (" SPINS OUT!", (255, 255, 255))], BLACK(58), a)
                elif p == 0: multi_text(d, 330, [(NAMES[i], COLS[i]), (" TAKES THE LEAD!", (255, 255, 255))], BLACK(58), a)
                else: multi_text(d, 330, [(NAMES[i], COLS[i]), (" PASSES ", (255, 255, 255)), (NAMES[j], COLS[j])], BOLD(52), a)
    else:
        d.text((W / 2, 140), f"LAP {LAPS}/{LAPS}", font=BLACK(108), fill=(255, 255, 255), anchor="mm", **STROKE)

    # standings chips (slide to their new position)
    for i in range(NC): chip_x[i] += (35 + 80 + int(pos[i]) * 165 - chip_x[i]) * 0.25
    vl = max(float(v[lead]), 150.0)
    for i in sorted(range(NC), key=lambda c: -pos[c]):
        cx = chip_x[i]; top_ = 1590 - (10 if pos[i] == 0 and t >= T0 else 0)
        d.rounded_rectangle([cx - 78, top_, cx + 78, top_ + 142], radius=22, fill=COLS[i],
                            outline=(255, 255, 255) if pos[i] == 0 and t >= T0 else None, width=5)
        d.text((cx, top_ + 38), f"P{int(pos[i]) + 1}" if t >= T0 else "?", font=BLACK(52), fill=(14, 18, 36), anchor="mm")
        d.text((cx, top_ + 86), NAMES[i], font=BOLD(27), fill=(14, 18, 36), anchor="mm")
        if t >= T0:
            if FIN[i] is not None and rt >= FIN[i]: gtxt = "FIN"
            elif pos[i] == 0: gtxt = "LEADER"
            else: gtxt = f"+{max(0.0, (Dlead - D[i]) / vl):.1f}s"
            d.text((cx, top_ + 119), gtxt, font=mono(24), fill=(14, 18, 36), anchor="mm")
    d.text((W / 2, 1810), "PYTHONIC LIFE  •  simulated 100% in Python", font=SEMI(28), fill=(110, 118, 150), anchor="mm")

    # ----- finish / results / outro -----
    if TW <= t < T_RES:
        fl = max(0.0, 1 - (t - TW) / 0.45)
        if fl > 0:
            ov = Image.new("RGBA", (W, H), (255, 255, 255, int(200 * fl))); img.paste(ov, (0, 0), ov); d = ImageDraw.Draw(img)
        d.text((W / 2, 760), "FINISH!", font=BLACK(190 * max(ease_back((t - TW) / 0.4), .01)), fill=(255, 212, 59), anchor="mm", **STROKE)
        if MARGIN < 0.25 and t - TW > 0.3:
            d.text((W / 2, 900), "PHOTO FINISH", font=BLACK(64), fill=(255, 255, 255), anchor="mm", **STROKE)
    if T_RES <= t < T_OUT:
        rtt = t - T_RES
        ov = Image.new("RGBA", (W, H), (5, 6, 14, int(205 * min(1, rtt / 0.3)))); img.paste(ov, (0, 0), ov); d = ImageDraw.Draw(img)
        d.text((W / 2, 520), NAMES[WINNER], font=BLACK(190 * max(ease_back(rtt / 0.45), .01)), fill=COLS[WINNER], anchor="mm", **STROKE)
        d.text((W / 2, 690), "WINS!", font=BLACK(140 * max(ease_back((rtt - 0.1) / 0.45), .01)), fill=(255, 255, 255), anchor="mm", **STROKE)
        k2 = ease_out((rtt - 0.6) / 0.4)
        if k2 > 0:
            d.text((W / 2, 820), f"WON BY {MARGIN:.2f} SECONDS" if MARGIN < 1.0 else f"WON BY {MARGIN:.1f} SECONDS",
                   font=BLACK(54), fill=tc((255, 212, 59), k2), anchor="mm")
        for rk in range(3):
            k3 = ease_out((rtt - 0.9 - rk * 0.25) / 0.35)
            if k3 <= 0: continue
            i = ORDER_F[rk]; y = 960 + rk * 100
            d.ellipse([200, y - 24, 248, y + 24], fill=tc(COLS[i], k3))
            d.text((275, y), f"{rk + 1}   {NAMES[i]}", font=BLACK(54), fill=tc((255, 255, 255), k3), anchor="lm")
            d.text((880, y), f"{FIN[i]:.2f}s" if rk == 0 else f"+{FIN[i] - FIN[ORDER_F[0]]:.2f}s", font=mono(46), fill=tc((200, 205, 230), k3), anchor="rm")
        k4 = ease_out((rtt - 1.9) / 0.4)
        if k4 > 0: d.text((W / 2, 1330), "Did you pick it?", font=BOLD(56), fill=tc((255, 92, 122), k4), anchor="mm")
    if t >= T_OUT:
        ot = t - T_OUT
        ov = Image.new("RGBA", (W, H), (5, 6, 14, int(235 * min(1, ot / 0.4)))); img.paste(ov, (0, 0), ov); d = ImageDraw.Draw(img)
        k = ease_back((ot - 0.15) / 0.5)
        if k > 0:
            d.text((W / 2, 560), "DID YOUR CAR", font=BLACK(86 * max(k, .01)), fill=(255, 255, 255), anchor="mm")
            d.text((W / 2, 670), "WIN?", font=BLACK(130 * max(k, .01)), fill=COLS[WINNER], anchor="mm")
        k2 = ease_out((ot - 0.7) / 0.5)
        if k2 > 0:
            d.text((W / 2, 790), "Comment your color below", font=SEMI(44), fill=tc((220, 220, 240), k2), anchor="mm")
            for i in range(NC):
                cx_ = W / 2 + (i - 2.5) * 100; r_ = 30 * k2
                d.ellipse([cx_ - r_, 880 - r_, cx_ + r_, 880 + r_], fill=COLS[i])
        k3 = ease_back((ot - 1.3) / 0.5)
        if k3 > 0:
            d.text((W / 2, 1090), "PYTHONIC", font=BLACK(120 * max(k3, .01)), fill=(55, 160, 235), anchor="mm")
            d.text((W / 2, 1220), "LIFE", font=BLACK(120 * max(k3, .01)), fill=(255, 212, 59), anchor="mm")
            d.text((W / 2, 1330), "Race simulated 100% in Python  •  Follow", font=BOLD(36), fill=tc((255, 92, 122), min(1, k3)), anchor="mm")

    out = glow(np.asarray(img))
    if t < 0.15: out = (out * (t / 0.15)).astype(np.uint8)
    if t > TOTAL - 0.4: out = (out * max(0, (TOTAL - t) / 0.4)).astype(np.uint8)
    proc.stdin.write(out.tobytes())
proc.stdin.close(); proc.wait()

a = audio[: int(TOTAL * SR)]
a = np.tanh(a / (np.percentile(np.abs(a), 99.9) + 1e-6) * 1.1) * 0.92
with wave.open("car_audio.wav", "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR); wf.writeframes((a * 32767).astype(np.int16).tobytes())
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "car_noaudio.mp4", "-i", "car_audio.wav", "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k", "-shortest", "pythonic_life_car_race.mp4"], check=True)
print("done", round(TOTAL, 1))
