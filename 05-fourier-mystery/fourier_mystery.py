"""Pythonic Life — What Are These Circles Drawing? (YouTube Shorts 1080x1920)
A closed outline is turned into a Fourier series with NumPy's FFT; 180 rotating circles (epicycles) redraw it.
Rendering with OpenCV + Pillow, audio synthesized in NumPy, encoded with ffmpeg.
Usage: python3 fourier_mystery.py
"""
import math, subprocess, wave
from functools import lru_cache
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SR = 1080, 1920, 30, 44100
K = 180                     # number of epicycles
M = 2048                    # outline samples
CX, CY = W / 2, 960
SCALE = 440
T_INTRO, T_DRAW = 0.6, 18.0
T_REVEAL = T_INTRO + T_DRAW
T_BONUS = T_REVEAL + 4.0
T_OUTRO = T_BONUS + 5.0
TOTAL = T_OUTRO + 3.8

# ---------------- the secret shape: a python, built from a sinuous centreline ----------------
def snake_outline():
    s = np.linspace(0, 1, 900)
    x = -1 + 2 * s
    env = 0.35 + 0.65 * np.sin(np.pi * np.clip(s * 1.1, 0, 1)) ** 0.7
    y = 0.34 * np.sin(2.3 * np.pi * s + 0.4) * env * (1 - 0.7 * np.clip((s - 0.86) / 0.14, 0, 1))
    w = 0.078 * np.clip(s / 0.4, 0, 1) ** 0.8
    w += 0.075 * np.exp(-((s - 0.925) / 0.045) ** 2)                       # head bulge
    w *= np.sqrt(np.clip((1 - s) / 0.045, 0, 1))                         # rounded nose
    dx, dy = np.gradient(x), np.gradient(y)
    nl = np.hypot(dx, dy); nx, ny = -dy / nl, dx / nl
    left = np.stack([x + nx * w, y + ny * w], 1)
    right = np.stack([x - nx * w, y - ny * w], 1)
    nose = np.array([x[-1], y[-1]])
    d = np.array([dx[-1], dy[-1]]) / nl[-1]; n = np.array([nx[-1], ny[-1]])
    tip = nose + d * 0.10
    tongue = [nose, tip, tip + d * 0.05 + n * 0.035, tip, tip + d * 0.05 - n * 0.035, tip, nose]
    pts = np.concatenate([left, tongue, right[::-1]])
    eye = np.array([x[-55] + nx[-55] * 0.06, y[-55] + ny[-55] * 0.06])
    return pts, eye

def resample(pts, m):
    seg = np.hypot(*np.diff(np.vstack([pts, pts[:1]]), axis=0).T)
    cum = np.concatenate([[0], np.cumsum(seg)])
    u = np.linspace(0, cum[-1], m, endpoint=False)
    closed = np.vstack([pts, pts[:1]])
    return np.interp(u, cum, closed[:, 0]) + 1j * np.interp(u, cum, closed[:, 1])

pts, EYE = snake_outline()
z = resample(pts, M)
zm = z.mean()
z = z - zm
z = z.real * SCALE + 1j * (-z.imag * SCALE)                                # screen coords (y down)
EYE = ((EYE[0] - zm.real) * SCALE, -(EYE[1] - zm.imag) * SCALE)
coef = np.fft.fft(z) / M
freqs = np.fft.fftfreq(M, 1 / M)
order = np.argsort(-np.abs(coef))
order = order[freqs[order] != 0]

def chain(tau, n):
    """positions of each epicycle centre for the n largest terms at time tau in [0,1)"""
    idx = order[:n]
    terms = coef[idx] * np.exp(2j * np.pi * freqs[idx] * tau)
    return np.concatenate([[0], np.cumsum(terms)]) + (CX + 1j * CY)

def approx_path(n, steps=1200):
    taus = np.linspace(0, 1, steps, endpoint=False)
    idx = order[:n]
    return (coef[idx][None, :] * np.exp(2j * np.pi * freqs[idx][None, :] * taus[:, None])).sum(1)

FULL = approx_path(K, 2400) + (CX + 1j * CY)

# ---------------- render helpers ----------------
F = "/usr/share/fonts/opentype/inter/"
@lru_cache(None)
def font(n, s): return ImageFont.truetype(F + n, max(1, int(s)))
@lru_cache(None)
def mono(s): return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", max(1, int(s)))
BLACK = lambda s: font("Inter-Black.otf", s)
BOLD = lambda s: font("Inter-Bold.otf", s)
SEMI = lambda s: font("Inter-SemiBold.otf", s)
def ease_out(t): t = max(0, min(1, t)); return 1 - (1 - t) ** 3
def ease_back(t): t = max(0, min(1, t)); return 1 + 2.70158 * (t - 1) ** 3 + 1.70158 * (t - 1) ** 2
def tc(c, a=1.0): return tuple(int(min(255, max(0, x * a))) for x in c)
BLUE, YEL = np.array([55, 160, 235.]), np.array([255, 212, 59.])

yy, xx = np.mgrid[0:H, 0:W]
BG = np.zeros((H, W, 3), np.float32); BG[...] = (6, 7, 14)
BG += (np.exp(-(((xx - CX) / 620) ** 2 + ((yy - CY) / 700) ** 2)) * 22)[..., None] * np.array([0.6, 0.7, 1.3])
for gx in range(0, W, 54): BG[:, gx] += 5
for gy in range(0, H, 54): BG[gy, :] += 5
BG = np.clip(BG, 0, 255).astype(np.uint8)

def glow(a, s=0.85):
    sm = cv2.resize(a, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    b1 = cv2.GaussianBlur(sm, (0, 0), 3); b2 = cv2.GaussianBlur(sm, (0, 0), 11)
    return cv2.add(cv2.addWeighted(a, 1, cv2.resize(b1, (W, H)), s, 0), (cv2.resize(b2, (W, H)) * 0.6).astype(np.uint8))

def path_color(u):           # gradient along the drawing: python blue -> python yellow
    return tuple(int(v) for v in BLUE * (1 - u) + YEL * u)

def draw_path(img, pts, upto, thick=4):
    n = int(upto)
    if n < 2: return
    p = np.stack([pts.real[:n], pts.imag[:n]], 1).astype(np.int32)
    segs = 24
    for k in range(segs):
        a, b = k * n // segs, min(n, (k + 1) * n // segs + 1)
        if b - a < 2: continue
        cv2.polylines(img, [p[a:b]], False, path_color(k / segs * n / len(pts)), thick, cv2.LINE_AA)

# ---------------- audio ----------------
audio = np.zeros(int((TOTAL + 2) * SR), np.float32)
def put(t0, sig):
    s = int(t0 * SR); n = min(len(sig), len(audio) - s)
    if n > 0: audio[s:s + n] += sig[:n].astype(np.float32)
def pluck(t0, f, amp=0.2, dur=0.9):
    tt = np.arange(int(dur * SR)) / SR
    w = np.sin(2 * np.pi * f * tt) + 0.4 * np.sin(4 * np.pi * f * tt) * np.exp(-tt * 8) + 0.2 * np.sin(6 * np.pi * f * tt) * np.exp(-tt * 12)
    put(t0, w * np.exp(-tt * 5) * np.minimum(1, tt * 700) * amp)
def sweep(t0, f0, f1, dur, amp):
    tt = np.arange(int(dur * SR)) / SR
    fr = f0 + (f1 - f0) * (tt / dur) ** 2
    put(t0, np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.sin(np.pi * tt / dur) ** 2 * amp)
def kick(t0, amp=0.45):
    tt = np.arange(int(0.3 * SR)) / SR
    put(t0, np.sin(2 * np.pi * np.cumsum(140 * np.exp(-tt * 25) + 45) / SR) * np.exp(-tt * 9) * amp)
PENTA = [220.0, 246.94, 277.18, 329.63, 369.99]
def penta(k): o, d = divmod(k, 5); return PENTA[d] * 2 ** o

# melody follows the height of the pen while drawing (one note per 8th at 120 bpm)
step = 0.25
t = T_INTRO
while t < T_REVEAL:
    tau = (t - T_INTRO) / T_DRAW
    pen = chain(tau, K)[-1]
    k = int(np.clip((CY + 380 - pen.imag) / 760 * 12, 0, 12))
    pluck(t, penta(k + 3), 0.16)
    if int(round((t - T_INTRO) / step)) % 2 == 0: kick(t, 0.35)
    t += step
sweep(T_REVEAL - 1.2, 200, 1400, 1.2, 0.25)
for i, fr in enumerate([440, 554.4, 659.3, 880]): pluck(T_REVEAL + 0.06 * i, fr, 0.3, 2.0)
for i, n in enumerate([3, 12, 50]): pluck(T_BONUS - 3.4 + i * 1.0, penta(5 + 2 * i), 0.3, 1.2)
for i, fr in enumerate([329.6, 440, 554.4, 659.3]): pluck(T_OUTRO + 0.12 * i, fr, 0.25, 2.2)

# ---------------- render ----------------
proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                         "-maxrate", "8M", "-bufsize", "16M", "-pix_fmt", "yuv420p", "fourier_noaudio.mp4"], stdin=subprocess.PIPE)
traced = []
NF = int(TOTAL * FPS)
MINI = {n: approx_path(n, 900) for n in (3, 12, 50)}
snake_poly = np.stack([FULL.real, FULL.imag], 1).astype(np.int32)
eye_xy = (int(CX + EYE[0]), int(CY + EYE[1]))

for f in range(NF):
    t = f / FPS
    img = BG.copy()
    phase_draw = T_INTRO <= t < T_REVEAL
    if t < T_REVEAL:
        tau = max(0.0, (t - T_INTRO) / T_DRAW)
        # sub-sample the pen between frames for a smooth trail
        if phase_draw:
            for s in np.linspace(0, 1, 12, endpoint=False):
                tt_ = tau + s / (T_DRAW * FPS)
                traced.append(chain(min(tt_, 0.99999), K)[-1])
        tr = np.array(traced) if traced else np.array([CX + 1j * CY])
        draw_path(img, tr, len(tr), 5)
        c = chain(tau, K)
        layer = img.copy()
        for i in range(min(K, 70)):
            r = abs(c[i + 1] - c[i])
            if r < 1.5: continue
            cv2.circle(layer, (int(c[i].real), int(c[i].imag)), int(r), (90, 110, 170), 1, cv2.LINE_AA)
        img = cv2.addWeighted(layer, 0.55, img, 0.45, 0)
        p = np.stack([c.real, c.imag], 1).astype(np.int32)
        cv2.polylines(img, [p[:90]], False, (230, 235, 255), 2, cv2.LINE_AA)
        pen = (int(c[-1].real), int(c[-1].imag))
        cv2.circle(img, pen, 10, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, pen, 18, (255, 212, 59), 2, cv2.LINE_AA)
    elif t < T_BONUS:
        rt = t - T_REVEAL
        a = ease_out(rt / 0.8)
        fill = img.copy()
        cv2.fillPoly(fill, [snake_poly], (40, 110, 170), cv2.LINE_AA)
        # python-pattern: yellow blotches clipped to the body
        mask = np.zeros((H, W), np.uint8); cv2.fillPoly(mask, [snake_poly], 255, cv2.LINE_AA)
        blot = fill.copy()
        for k in range(0, len(FULL) // 2, 60):
            q = FULL[k]
            cv2.ellipse(blot, (int(q.real), int(q.imag)), (22, 14), (k * 7) % 180, 0, 360, (240, 200, 60), -1, cv2.LINE_AA)
        fill = np.where(mask[..., None] > 0, blot, fill)
        img = cv2.addWeighted(fill, a, img, 1 - a, 0)
        draw_path(img, FULL, len(FULL), 5)
        if rt > 0.4:
            cv2.circle(img, eye_xy, 11, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, eye_xy, 5, (10, 10, 20), -1, cv2.LINE_AA)
    elif t < T_OUTRO:
        bt = t - T_BONUS
        for i, n in enumerate((3, 12, 50)):
            cy = 520 + i * 400
            pth = MINI[n] * 0.42 + (CX + 1j * cy)
            prog = ease_out((bt - i * 1.0 + 3.4 - 3.4) / 1.0)
            draw_path(img, pth, len(pth) * prog, 3)
        draw_path(img, FULL * 0 + FULL, 0, 3)
    else:
        img = (img * 0.6).astype(np.uint8)

    im = Image.fromarray(img); d = ImageDraw.Draw(im)
    if t < T_REVEAL:
        k1 = ease_back(t / 0.5)
        d.text((W / 2, 170), "WHAT ARE THESE", font=BLACK(88 * max(k1, .01)), fill=(255, 255, 255), anchor="mm")
        d.text((W / 2, 270), "CIRCLES DRAWING?", font=BLACK(88 * max(k1, .01)), fill=(255, 212, 59), anchor="mm")
        k2 = ease_out((t - 0.5) / 0.5)
        d.text((W / 2, 360), f"{K} spinning circles  •  guess before it ends", font=SEMI(36), fill=tc((170, 180, 220), k2), anchor="mm")
        prog = max(0, min(1, (t - T_INTRO) / T_DRAW))
        d.rounded_rectangle([160, 1520, 920, 1536], radius=8, fill=(30, 34, 52))
        d.rounded_rectangle([160, 1520, 160 + 760 * prog, 1536], radius=8, fill=(255, 212, 59))
        d.text((W / 2, 1600), f"{prog * 100:3.0f}% drawn", font=mono(46), fill=(255, 255, 255), anchor="mm")
        if prog > 0.75:
            a = 0.6 + 0.4 * math.sin(t * 7)
            d.text((W / 2, 1440), "Last chance to guess!", font=BOLD(48), fill=tc((255, 92, 122), a), anchor="mm")
    elif t < T_BONUS:
        rt = t - T_REVEAL
        sc = ease_back((rt - 0.3) / 0.5)
        if sc > 0:
            d.text((W / 2, 230), "IT'S A PYTHON!", font=BLACK(110 * max(sc, .01)), fill=(255, 212, 59), anchor="mm")
        k2 = ease_out((rt - 0.9) / 0.5)
        if k2 > 0:
            d.text((W / 2, 1450), f"drawn by {K} rotating circles", font=BOLD(48), fill=tc((230, 230, 255), k2), anchor="mm")
            d.text((W / 2, 1520), "and nothing else", font=SEMI(40), fill=tc((170, 180, 220), k2), anchor="mm")
    elif t < T_OUTRO:
        bt = t - T_BONUS
        d.text((W / 2, 230), "SAME DRAWING WITH", font=BLACK(70), fill=(255, 255, 255), anchor="mm")
        d.text((W / 2, 320), "FEWER CIRCLES", font=BLACK(70), fill=(255, 92, 122), anchor="mm")
        for i, n in enumerate((3, 12, 50)):
            a = ease_out((bt - i * 1.0) / 0.4)
            if a > 0:
                d.text((120, 520 + i * 400), f"{n}", font=BLACK(74), fill=tc((255, 212, 59), a), anchor="mm")
                d.text((120, 580 + i * 400), "circles", font=SEMI(28), fill=tc((170, 180, 220), a), anchor="mm")
    else:
        ot = t - T_OUTRO
        k = ease_back((ot - 0.1) / 0.5)
        if k > 0:
            d.text((W / 2, 620), "THIS IS A", font=BLACK(76 * max(k, .01)), fill=(255, 255, 255), anchor="mm")
            d.text((W / 2, 735), "FOURIER SERIES", font=BLACK(104 * max(k, .01)), fill=(255, 92, 122), anchor="mm")
        k2 = ease_out((ot - 0.6) / 0.5)
        if k2 > 0:
            d.text((W / 2, 850), "Enough circles can draw anything.", font=SEMI(44), fill=tc((220, 220, 240), k2), anchor="mm")
        k3 = ease_back((ot - 1.2) / 0.5)
        if k3 > 0:
            d.text((W / 2, 1060), "PYTHONIC", font=BLACK(120 * max(k3, .01)), fill=(55, 160, 235), anchor="mm")
            d.text((W / 2, 1190), "LIFE", font=BLACK(120 * max(k3, .01)), fill=(255, 212, 59), anchor="mm")
            d.text((W / 2, 1300), "100% Python  •  Follow for more", font=BOLD(40), fill=tc((255, 92, 122), min(1, k3)), anchor="mm")

    out = glow(np.asarray(im), 0.75)
    if t < 0.15: out = (out * (t / 0.15)).astype(np.uint8)
    if t > TOTAL - 0.4: out = (out * max(0, (TOTAL - t) / 0.4)).astype(np.uint8)
    proc.stdin.write(out.tobytes())
proc.stdin.close(); proc.wait()

tt = np.arange(len(audio)) / SR
audio += ((np.sin(2 * np.pi * 55 * tt) + 0.6 * np.sin(2 * np.pi * 82.4 * tt)) * 0.03).astype(np.float32)
a = audio[: int(TOTAL * SR)]
a = np.tanh(a / (np.percentile(np.abs(a), 99.9) + 1e-6) * 1.1) * 0.92
with wave.open("fourier_audio.wav", "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR); wf.writeframes((a * 32767).astype(np.int16).tobytes())
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "fourier_noaudio.mp4", "-i", "fourier_audio.wav", "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k", "-shortest", "pythonic_life_fourier_mystery.mp4"], check=True)
print("done", TOTAL)
