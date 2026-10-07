"""Pythonic Life — 150 Pendulums, 0.0000001° Apart (YouTube Shorts 1080x1920)
Double-pendulum chaos: RK4 physics in NumPy, rendering with OpenCV + Pillow, audio synthesized in NumPy.
Usage: python3 chaos_pendulums.py
"""
import math, subprocess, wave, colorsys
from functools import lru_cache
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SR = 1080, 1920, 30, 44100
N = 150
SUB = 20                                  # RK4 steps per frame
DT = 1 / (FPS * SUB)
G = 9.81
BASE = 2.0                                # starting angle of both arms (rad)
EPS = math.radians(1e-7)                  # difference between neighbours
SIM_T = 28.0
OUTRO = 4.0
TOTAL = SIM_T + OUTRO
PX, PY = W / 2, 820                       # pivot
L = 205                                   # arm length in px

# ---------------- physics ----------------
def deriv(s):
    t1, t2, w1, w2 = s
    d = t1 - t2; den = 3 - np.cos(2 * d)
    a1 = (-3 * G * np.sin(t1) - G * np.sin(t1 - 2 * t2) - 2 * np.sin(d) * (w2 ** 2 + w1 ** 2 * np.cos(d))) / den
    a2 = (2 * np.sin(d) * (2 * w1 ** 2 + 2 * G * np.cos(t1) + w2 ** 2 * np.cos(d))) / den
    return np.array([w1, w2, a1, a2])

state = np.array([np.full(N, BASE), BASE + EPS * np.arange(N), np.zeros(N), np.zeros(N)])
frames = []
split_t = None
for f in range(int(SIM_T * FPS)):
    for _ in range(SUB):
        k1 = deriv(state); k2 = deriv(state + DT / 2 * k1); k3 = deriv(state + DT / 2 * k2); k4 = deriv(state + DT * k3)
        state = state + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    t1, t2 = state[0], state[1]
    x1, y1 = PX + L * np.sin(t1), PY + L * np.cos(t1)
    x2, y2 = x1 + L * np.sin(t2), y1 + L * np.cos(t2)
    spread = float(np.hypot(x2 - x2.mean(), y2 - y2.mean()).max())
    if split_t is None and spread > 30: split_t = f / FPS
    frames.append((x1, y1, x2, y2, spread, state[3].copy()))
print("split at", split_t)

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

HUES = np.array([colorsys.hsv_to_rgb(i / N * 0.85, 0.85, 1.0) for i in range(N)]) * 255

yy, xx = np.mgrid[0:H, 0:W]
BG = np.zeros((H, W, 3), np.float32); BG[...] = (5, 6, 12)
BG += (np.exp(-(((xx - PX) / 600) ** 2 + ((yy - PY) / 650) ** 2)) * 20)[..., None] * np.array([0.6, 0.7, 1.3])
BG = np.clip(BG, 0, 255).astype(np.uint8)

def glow(a, s=0.9):
    sm = cv2.resize(a, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    b1 = cv2.GaussianBlur(sm, (0, 0), 3); b2 = cv2.GaussianBlur(sm, (0, 0), 12)
    return cv2.add(cv2.addWeighted(a, 1, cv2.resize(b1, (W, H)), s, 0), (cv2.resize(b2, (W, H)) * 0.7).astype(np.uint8))

# ---------------- audio ----------------
audio = np.zeros(int((TOTAL + 2) * SR), np.float32)
def put(t0, sig):
    s = int(t0 * SR); n = min(len(sig), len(audio) - s)
    if n > 0: audio[s:s + n] += sig[:n].astype(np.float32)
def chime(t0, f, amp=0.2, dur=1.6):
    tt = np.arange(int(dur * SR)) / SR
    w = np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(2 * np.pi * 2.01 * f * tt) * np.exp(-tt * 5)
    put(t0, w * np.exp(-tt * 3) * np.minimum(1, tt * 500) * amp)
def sweep(t0, f0, f1, dur, amp):
    tt = np.arange(int(dur * SR)) / SR
    fr = f0 + (f1 - f0) * (tt / dur) ** 2
    put(t0, np.sin(2 * np.pi * np.cumsum(fr) / SR) * (tt / dur) ** 2 * amp)
def boom(t0, amp=0.8):
    tt = np.arange(int(1.6 * SR)) / SR
    fr = 90 * np.exp(-tt * 2.5) + 38
    put(t0, np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-tt * 2) * amp)
PENTA = [261.63, 293.66, 329.63, 392.0, 440.0]
def penta(k): o, d = divmod(k, 5); return PENTA[d] * 2 ** o

# "whoosh": noise shaped by the swing speed of the lead pendulum
rng = np.random.default_rng(2)
speed = np.array([abs(fr[5][0]) for fr in frames])
speed = speed / speed.max()
env = np.interp(np.arange(int(SIM_T * SR)) / SR, np.arange(len(speed)) / FPS, speed)
nz = rng.standard_normal(len(env)).astype(np.float32)
nz = np.convolve(nz, np.ones(24) / 24, mode="same")          # low-pass -> airy swish
audio[: len(env)] += nz * env ** 2 * 0.9
# heartbeat-like tension pulse before the split, gets faster
t = 0.0
while split_t and t < split_t:
    tt = np.arange(int(0.25 * SR)) / SR
    put(t, np.sin(2 * np.pi * 55 * tt) * np.exp(-tt * 14) * 0.6)
    t += max(0.32, 1.0 - t / split_t * 0.68)
sweep(split_t - 2.0, 150, 900, 2.0, 0.25)
boom(split_t)
for k in range(10): chime(split_t + 0.08 * k, penta(5 + k), 0.18)

# ---------------- render ----------------
proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                         "-pix_fmt", "yuv420p", "chaos_noaudio.mp4"], stdin=subprocess.PIPE)
trail = np.zeros((H, W, 3), np.float32)
prev_tip = None
NF = int(TOTAL * FPS)
next_chime = split_t + 1.0
ci = 0
for f in range(NF):
    t = f / FPS
    sim = f < len(frames)
    img = BG.copy().astype(np.float32)
    if sim:
        x1, y1, x2, y2, spread, _ = frames[f]
        mix = min(1.0, max(0.0, (spread - 3) / 60))           # 0 = white single pendulum, 1 = rainbow
        cols = 255 * (1 - mix) + HUES * mix
        # fading trails of tips
        trail *= 0.93
        if prev_tip is not None:
            for i in range(N):
                cv2.line(trail, (int(prev_tip[0][i]), int(prev_tip[1][i])), (int(x2[i]), int(y2[i])),
                         tuple(float(c) * 0.9 for c in cols[i]), 3, cv2.LINE_AA)
        prev_tip = (x2, y2)
        img = np.maximum(img, trail)
        layer = img.astype(np.uint8)
        for i in range(N):
            c = tuple(int(v) for v in cols[i])
            cv2.line(layer, (int(PX), int(PY)), (int(x1[i]), int(y1[i])), c, 2, cv2.LINE_AA)
            cv2.line(layer, (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i])), c, 2, cv2.LINE_AA)
        for i in range(N):
            cv2.circle(layer, (int(x2[i]), int(y2[i])), 6, tuple(int(v) for v in cols[i]), -1, cv2.LINE_AA)
        cv2.circle(layer, (int(PX), int(PY)), 9, (230, 230, 240), -1, cv2.LINE_AA)
        if t > next_chime and t < SIM_T - 1:
            chime(t, penta(int(5 + (ci * 7) % 11)), 0.09, 1.2); ci += 1
            next_chime = t + 0.45
    else:
        trail *= 0.9
        layer = np.maximum(img, trail).astype(np.uint8)

    im = Image.fromarray(layer); d = ImageDraw.Draw(im)

    if sim:
        before = split_t is None or t < split_t
        if before:
            k = ease_back(t / 0.5)
            d.text((W / 2, 170), "150 PENDULUMS", font=BLACK(104 * max(k, .01)), fill=(255, 255, 255), anchor="mm")
            k2 = ease_out((t - 0.3) / 0.5)
            d.text((W / 2, 268), "each one starts", font=SEMI(44), fill=tc((170, 180, 220), k2), anchor="mm")
            d.text((W / 2, 340), "0.0000001° apart", font=mono(58), fill=tc((255, 212, 59), k2), anchor="mm")
            if t > 3.0:
                a = 0.6 + 0.4 * math.sin(t * 5)
                d.text((W / 2, 1360), "Watch closely...", font=BOLD(52), fill=tc((255, 255, 255), a), anchor="mm")
        else:
            st = t - split_t
            sc = ease_back(st / 0.4)
            if st < 2.5:
                a = 1 if st < 1.8 else 1 - (st - 1.8) / 0.7
                d.text((W / 2, 230), "CHAOS", font=BLACK(170 * max(sc, .01)), fill=tc((255, 70, 110), a), anchor="mm")
            else:
                a = ease_out((st - 2.5) / 0.6)
                d.text((W / 2, 180), "Same start.", font=BLACK(80), fill=tc((255, 255, 255), a), anchor="mm")
                d.text((W / 2, 280), "150 different futures.", font=BLACK(64), fill=tc((255, 212, 59), a), anchor="mm")
            a2 = ease_out((st - 4) / 0.6)
            if a2 > 0:
                d.text((W / 2, 1380), f"It took {split_t:.1f} seconds", font=BOLD(50), fill=tc((230, 230, 255), a2), anchor="mm")
                d.text((W / 2, 1450), "for 0.0000001° to change everything", font=SEMI(38), fill=tc((170, 180, 220), a2), anchor="mm")
        # HUD
        d.text((W / 2, 1560), f"{t:5.2f}s", font=mono(64), fill=(255, 255, 255), anchor="mm")
        spread_deg = spread / (2 * L) * 180
        d.text((W / 2, 1630), f"max spread: {spread:7.2f} px", font=mono(30), fill=(130, 140, 180), anchor="mm")
    else:
        ot = t - SIM_T
        ov = Image.new("RGBA", (W, H), (5, 6, 12, int(200 * min(1, ot / 0.5)))); im.paste(ov, (0, 0), ov); d = ImageDraw.Draw(im)
        k = ease_back((ot - 0.2) / 0.5)
        if k > 0:
            d.text((W / 2, 620), "THIS IS", font=BLACK(80 * max(k, .01)), fill=(255, 255, 255), anchor="mm")
            d.text((W / 2, 740), "CHAOS THEORY", font=BLACK(110 * max(k, .01)), fill=(255, 70, 110), anchor="mm")
        k2 = ease_out((ot - 0.7) / 0.5)
        if k2 > 0:
            d.text((W / 2, 860), "Tiny change. Totally different future.", font=SEMI(44), fill=tc((220, 220, 240), k2), anchor="mm")
        k3 = ease_back((ot - 1.3) / 0.5)
        if k3 > 0:
            d.text((W / 2, 1060), "PYTHONIC", font=BLACK(120 * max(k3, .01)), fill=(55, 160, 235), anchor="mm")
            d.text((W / 2, 1190), "LIFE", font=BLACK(120 * max(k3, .01)), fill=(255, 212, 59), anchor="mm")
            d.text((W / 2, 1300), "100% Python  •  Follow for more", font=BOLD(40), fill=tc((255, 92, 122), min(1, k3)), anchor="mm")
        if ot < 1 / FPS:
            for i, fr in enumerate([261.6, 329.6, 392.0, 523.3]): chime(t + 0.1 * i, fr, 0.2, 2.2)

    out = glow(np.asarray(im), 0.8)
    if t < 0.15: out = (out * (t / 0.15)).astype(np.uint8)
    if t > TOTAL - 0.4: out = (out * max(0, (TOTAL - t) / 0.4)).astype(np.uint8)
    proc.stdin.write(out.tobytes())
proc.stdin.close(); proc.wait()

# soft pad underneath
tt = np.arange(len(audio)) / SR
audio += ((np.sin(2 * np.pi * 65.4 * tt) + 0.6 * np.sin(2 * np.pi * 98.0 * tt)) * 0.035).astype(np.float32)
a = audio[: int(TOTAL * SR)]
a = np.tanh(a / (np.percentile(np.abs(a), 99.9) + 1e-6) * 1.1) * 0.92
with wave.open("chaos_audio.wav", "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR); wf.writeframes((a * 32767).astype(np.int16).tobytes())
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "chaos_noaudio.mp4", "-i", "chaos_audio.wav", "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k", "-shortest", "pythonic_life_chaos_pendulums.mp4"], check=True)
print("done", TOTAL)
