"""Pythonic Life — Zooming 1,000,000,000,000x into one equation (YouTube Shorts 1080x1920)
Mandelbrot set z = z² + c, computed with Numba (parallel, float64), smooth-coloured with NumPy,
HUD with Pillow, ambient soundtrack synthesized in NumPy, encoded with ffmpeg.
Usage: python3 infinite_zoom.py          (takes ~20-30 min on 2 cores)
"""
import math, subprocess, wave
from functools import lru_cache
import numpy as np, cv2
from numba import njit, prange
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SR = 1080, 1920, 30, 44100
RW, RH = 540, 960                       # render resolution (upscaled 2x)
CX, CY = -0.7436438870371587, 0.13182590420531197     # "seahorse valley" deep-zoom point
ZMAX_EXP = 12                           # final zoom = 10^12
T_HOOK = 1.2
T_ZOOM = 26.0
T_HOLD = 2.6
T_BACK = 1.2
T_OUTRO = 4.0
Z_END = T_HOOK + T_ZOOM
TOTAL = Z_END + T_HOLD + T_BACK + T_OUTRO

@njit(parallel=True, fastmath=True)
def mandel(cx, cy, scale, w, h, maxit, out):
    for j in prange(h):
        for i in range(w):
            x0 = cx + (i - w / 2) * scale
            y0 = cy + (j - h / 2) * scale
            x = 0.0; y = 0.0; x2 = 0.0; y2 = 0.0; k = 0
            while x2 + y2 <= 256.0 and k < maxit:
                y = 2 * x * y + y0
                x = x2 - y2 + x0
                x2 = x * x; y2 = y * y; k += 1
            if k == maxit:
                out[j, i] = -1.0
            else:
                out[j, i] = k + 1 - math.log(math.log(math.sqrt(x2 + y2))) / math.log(2)

def colorize(n, shift):
    v = np.log(np.maximum(n, 1.0)) * 0.9 + shift
    r = 0.5 + 0.5 * np.cos(2 * np.pi * (v + 0.00))
    g = 0.5 + 0.5 * np.cos(2 * np.pi * (v + 0.15))
    b = 0.5 + 0.5 * np.cos(2 * np.pi * (v + 0.30))
    img = np.stack([r, g, b], -1) ** 1.2
    img[n < 0] = 0
    return (img * 255).astype(np.uint8)

def zoom_at(t):
    u = min(max((t - T_HOOK) / T_ZOOM, 0.0), 1.0)
    # gentle ease at both ends, constant exponential speed in the middle
    e = u - (math.sin(2 * math.pi * u) / (2 * math.pi)) * 0.12
    return 10 ** (ZMAX_EXP * e)

buf = np.zeros((RH, RW))
def fractal(zoom, t):
    it = int(150 + 120 * math.log10(zoom + 1) ** 1.25)
    mandel(CX, CY, 3.0 / RW / zoom, RW, RH, it, buf)
    img = colorize(buf, t * 0.04)
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)

# ---------------- fonts / helpers ----------------
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
STROKE = dict(stroke_width=7, stroke_fill=(5, 5, 12))

# zoom milestones: if your 7 cm phone screen were scaled up by the zoom...
MILESTONES = [(3, "Your screen would be", "70 METERS wide"),
              (6, "Your screen would be", "70 KILOMETERS wide"),
              (9, "Your screen would be", "5x WIDER THAN EARTH"),
              (11, "Your screen would reach", "18x FARTHER THAN THE MOON")]
def zoom_time(exp):
    lo, hi = T_HOOK, Z_END
    for _ in range(50):
        mid = (lo + hi) / 2
        if math.log10(zoom_at(mid)) < exp: lo = mid
        else: hi = mid
    return lo
MS_T = [zoom_time(e) for e, _, _ in MILESTONES]

def fmt_zoom(z):
    return f"{int(round(z)):,}"

# ---------------- audio ----------------
audio = np.zeros(int((TOTAL + 2) * SR), np.float32)
def put(t0, sig):
    s = int(t0 * SR); n = min(len(sig), len(audio) - s)
    if n > 0: audio[s:s + n] += sig[:n].astype(np.float32)
def pad(t0, freqs, dur, amp):
    tt = np.arange(int(dur * SR)) / SR
    sig = sum(np.sin(2 * np.pi * f * tt + 0.3 * np.sin(2 * np.pi * 0.2 * tt)) + 0.3 * np.sin(2 * np.pi * 2 * f * tt) for f in freqs)
    env = np.minimum(1, tt / 1.5) * np.minimum(1, (dur - tt) / 1.5)
    put(t0, sig / len(freqs) * env * amp)
def chime(t0, f, amp=0.25, dur=2.5):
    tt = np.arange(int(dur * SR)) / SR
    w = np.sin(2 * np.pi * f * tt) + 0.4 * np.sin(2 * np.pi * 2.76 * f * tt) * np.exp(-tt * 4)
    put(t0, w * np.exp(-tt * 2.2) * np.minimum(1, tt * 600) * amp)
def boom(t0, amp=0.9):
    tt = np.arange(int(2.0 * SR)) / SR
    put(t0, np.sin(2 * np.pi * np.cumsum(70 * np.exp(-tt * 2) + 32) / SR) * np.exp(-tt * 1.6) * amp)
def whoosh(t0, dur, amp, up=False):
    tt = np.arange(int(dur * SR)) / SR
    nz = np.random.default_rng(int(t0 * 31)).standard_normal(len(tt)).astype(np.float32)
    k = 30
    nz = np.convolve(nz, np.ones(k) / k, mode="same")
    env = (tt / dur) ** 2 if up else np.sin(np.pi * tt / dur) ** 2
    put(t0, nz * env * amp * 3)

PROG = [[110, 164.8, 220, 277.2], [98, 146.8, 196, 246.9], [87.3, 130.8, 174.6, 220], [98, 146.8, 196, 293.7]]
pt, pi_ = 0.0, 0
while pt < Z_END + T_HOLD:
    pad(pt, PROG[pi_ % 4], 4.6, 0.32); pt += 4.0; pi_ += 1
# slow shimmering arpeggio that speeds up as we go deeper
at, k = T_HOOK, 0
ARP = [440, 523.3, 659.3, 784, 880, 1046.5]
while at < Z_END:
    chime(at, ARP[k % len(ARP)] * (0.5 if k % 7 == 3 else 1), 0.07, 1.8)
    u = (at - T_HOOK) / T_ZOOM
    at += 0.5 - 0.28 * u; k += 1
for mt in MS_T: chime(mt, 1318.5, 0.25, 3.0); chime(mt + 0.08, 987.8, 0.18, 3.0)
whoosh(Z_END - 3.0, 3.0, 0.25, up=True)
boom(Z_END)
chime(Z_END + 0.2, 659.3, 0.3, 4); chime(Z_END + 0.25, 830.6, 0.25, 4); chime(Z_END + 0.3, 987.8, 0.25, 4)
whoosh(Z_END + T_HOLD, T_BACK + 0.3, 0.5)
boom(Z_END + T_HOLD + T_BACK, 0.6)
for i, f in enumerate([220, 277.2, 329.6, 440]): chime(Z_END + T_HOLD + T_BACK + 0.1 * i, f, 0.25, 3.5)

# ---------------- render ----------------
proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "21",
                         "-maxrate", "9M", "-bufsize", "18M", "-pix_fmt", "yuv420p", "zoom_noaudio.mp4"], stdin=subprocess.PIPE)
NF = int(TOTAL * FPS)
keyframes = {}           # zoom exponent -> image, for the fast zoom-out at the end
minibrot = None
for f in range(NF):
    t = f / FPS
    if t < Z_END + T_HOLD:
        z = zoom_at(t)
        frame = fractal(z, t)
        if f % 15 == 0: keyframes[round(math.log10(z), 3)] = cv2.resize(frame, (W // 2, H // 2))
        if minibrot is None and t >= Z_END:
            # locate the mini copy near the centre (largest interior blob)
            mask = (buf < 0).astype(np.uint8)
            n, lab, stats, cents = cv2.connectedComponentsWithStats(mask)
            if n > 1:
                best = 1 + int(np.argmin([np.hypot(c[0] - RW / 2, c[1] - RH / 2) - s[4] * 0.02
                                          for c, s in zip(cents[1:], stats[1:])]))
                cx_, cy_ = cents[best]; sz = max(stats[best][2], stats[best][3])
                minibrot = (cx_ * 2, cy_ * 2, sz * 2)
    elif t < Z_END + T_HOLD + T_BACK:
        u = (t - Z_END - T_HOLD) / T_BACK
        e = ZMAX_EXP * (1 - ease_out(u) ** 0.8)
        key = min(keyframes, key=lambda k: abs(k - e))
        frame = cv2.resize(keyframes[key], (W, H), interpolation=cv2.INTER_LINEAR)
        frame = cv2.GaussianBlur(frame, (0, 0), 1 + 6 * math.sin(math.pi * u))
    else:
        frame = (cv2.GaussianBlur(cv2.resize(keyframes[min(keyframes)], (W, H)), (0, 0), 6) * 0.28).astype(np.uint8)

    img = Image.fromarray(frame); d = ImageDraw.Draw(img)
    # dark gradient bands behind text for legibility
    if t < Z_END + T_HOLD + T_BACK:
        top = Image.new("RGBA", (W, 420), (0, 0, 0, 0)); g = np.linspace(170, 0, 420).astype(np.uint8)
        top.putalpha(Image.fromarray(np.repeat(g[:, None], W, 1)))
        img.paste(Image.new("RGB", (W, 420), (4, 4, 10)), (0, 0), top)
        g2 = np.concatenate([np.linspace(0, 190, 420), np.full(H - 1180 - 420, 190)]).astype(np.uint8)
        bot = Image.fromarray(np.repeat(g2[:, None], W, 1))
        img.paste(Image.new("RGB", (W, H - 1180), (4, 4, 10)), (0, 1180), bot)
        d = ImageDraw.Draw(img)

    if t < Z_END:
        if t < 3.2:
            k = ease_back(t / 0.45)
            d.text((W / 2, 150), "THIS IMAGE", font=BLACK(110 * max(k, .01)), fill=(255, 255, 255), anchor="mm", **STROKE)
            d.text((W / 2, 270), "HAS NO END", font=BLACK(110 * max(k, .01)), fill=(255, 212, 59), anchor="mm", **STROKE)
        else:
            a = ease_out((t - 3.2) / 0.5)
            d.text((W / 2, 150), "ONE EQUATION:", font=BLACK(64), fill=tc((255, 255, 255), a), anchor="mm", **STROKE)
            d.text((W / 2, 250), "z = z² + c", font=mono(96), fill=tc((255, 212, 59), a), anchor="mm", **STROKE)
        z = zoom_at(t)
        d.text((W / 2, 1520), "ZOOM", font=BOLD(40), fill=(200, 205, 230), anchor="mm", **STROKE)
        d.text((W / 2, 1600), "×" + fmt_zoom(z), font=mono(72 if z < 1e10 else 62), fill=(255, 255, 255), anchor="mm", **STROKE)
        for (e, l1, l2), mt in zip(MILESTONES, MS_T):
            age = t - mt
            if 0 <= age < 2.6:
                a = min(1, age / 0.25) * (1 if age < 2.1 else 1 - (age - 2.1) / 0.5)
                sc = ease_back(age / 0.35)
                d.text((W / 2, 1360), l1, font=SEMI(44), fill=tc((230, 230, 245), a), anchor="mm", **STROKE)
                d.text((W / 2, 1430), l2, font=BLACK(min(70, 1900 / max(1, len(l2))) * max(sc, .01)), fill=tc((120, 230, 255), a), anchor="mm", **STROKE)
    elif t < Z_END + T_HOLD:
        ht = t - Z_END
        sc = ease_back(ht / 0.4)
        d.text((W / 2, 150), "×1,000,000,000,000", font=mono(70 * max(sc, .01)), fill=(255, 255, 255), anchor="mm", **STROKE)
        d.text((W / 2, 250), "A TRILLION TIMES DEEPER", font=BLACK(62), fill=(255, 212, 59), anchor="mm", **STROKE)
        if minibrot and ht > 0.5:
            mx, my, ms = minibrot; r = int(ms * 1.3 + 24 + 8 * math.sin(ht * 6))
            d.ellipse([mx - r, my - r, mx + r, my + r], outline=(255, 255, 255), width=6)
            a = ease_out((ht - 0.6) / 0.4)
            d.text((W / 2, 1400), "...and a perfect copy of the", font=SEMI(46), fill=tc((235, 235, 250), a), anchor="mm", **STROKE)
            d.text((W / 2, 1480), "WHOLE SHAPE IS HIDING HERE", font=BLACK(60), fill=tc((120, 230, 255), a), anchor="mm", **STROKE)
    elif t < Z_END + T_HOLD + T_BACK:
        pass
    else:
        ot = t - Z_END - T_HOLD - T_BACK
        k = ease_back(ot / 0.45)
        d.text((W / 2, 560), "IT NEVER ENDS.", font=BLACK(104 * max(k, .01)), fill=(255, 255, 255), anchor="mm", **STROKE)
        k2 = ease_out((ot - 0.5) / 0.5)
        if k2 > 0:
            d.text((W / 2, 690), "All of it from one line of math:", font=SEMI(46), fill=tc((220, 220, 240), k2), anchor="mm", **STROKE)
            d.text((W / 2, 800), "z = z² + c", font=mono(110), fill=tc((255, 212, 59), k2), anchor="mm", **STROKE)
        k3 = ease_back((ot - 1.2) / 0.5)
        if k3 > 0:
            d.text((W / 2, 1060), "PYTHONIC", font=BLACK(120 * max(k3, .01)), fill=(55, 160, 235), anchor="mm", **STROKE)
            d.text((W / 2, 1190), "LIFE", font=BLACK(120 * max(k3, .01)), fill=(255, 212, 59), anchor="mm", **STROKE)
            d.text((W / 2, 1300), "Rendered 100% in Python  •  Follow", font=BOLD(40), fill=tc((255, 92, 122), min(1, k3)), anchor="mm", **STROKE)

    out = np.asarray(img)
    if t < 0.15: out = (out * (t / 0.15)).astype(np.uint8)
    if t > TOTAL - 0.4: out = (out * max(0, (TOTAL - t) / 0.4)).astype(np.uint8)
    proc.stdin.write(out.tobytes())
    if f % 60 == 0: print(f"frame {f}/{NF}  zoom 1e{math.log10(zoom_at(t)):.1f}", flush=True)
proc.stdin.close(); proc.wait()

tt = np.arange(len(audio)) / SR
audio += (np.sin(2 * np.pi * 55 * tt) * 0.05 * (0.6 + 0.4 * np.sin(2 * np.pi * 0.1 * tt))).astype(np.float32)
a = audio[: int(TOTAL * SR)]
a = np.tanh(a / (np.percentile(np.abs(a), 99.9) + 1e-6) * 1.1) * 0.92
with wave.open("zoom_audio.wav", "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR); wf.writeframes((a * 32767).astype(np.int16).tobytes())
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "zoom_noaudio.mp4", "-i", "zoom_audio.wav", "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k", "-shortest", "pythonic_life_infinite_zoom.mp4"], check=True)
print("done", TOTAL)
