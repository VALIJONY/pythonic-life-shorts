"""Pythonic Life — Can the Ball Escape? (YouTube Shorts 1080x1920)
Pure Python: NumPy physics + Pillow/OpenCV rendering + NumPy-synthesized audio, encoded with ffmpeg.
"""
import math, random, colorsys, subprocess, wave, sys
from functools import lru_cache
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SR = 1080, 1920, 30, 44100
SUB = 8                       # physics substeps per frame
DT = 1 / (FPS * SUB)
CX, CY = W / 2, 1010
NR = 12
R_IN, R_OUT = 105, 480
RADII = [R_IN + (R_OUT - R_IN) * i / (NR - 1) for i in range(NR)]
GAP = math.radians(38)
BR = 16                       # ball radius
G = 1500.0
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 11
MAX_SIM = 55.0

F = "/usr/share/fonts/opentype/inter/"
@lru_cache(None)
def font(n, s): return ImageFont.truetype(F + n, max(1, int(s)))
@lru_cache(None)
def mono(s): return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", max(1, int(s)))

def ring_color(i, light=1.0):
    r, g, b = colorsys.hsv_to_rgb((0.58 - i / NR * 0.75) % 1, 0.75, light)
    return (int(r * 255), int(g * 255), int(b * 255))

def ease_out(t): t = max(0, min(1, t)); return 1 - (1 - t) ** 3
def ease_back(t):
    t = max(0, min(1, t)); return 1 + 2.70158 * (t - 1) ** 3 + 1.70158 * (t - 1) ** 2

# ---------------- simulation ----------------
def simulate(seed):
    rnd = random.Random(seed)
    speeds = [(0.55 + 0.09 * i) * (1 if i % 2 == 0 else -1) * rnd.uniform(0.85, 1.15) for i in range(NR)]
    phase0 = [rnd.uniform(0, 2 * math.pi) for _ in range(NR)]
    p = np.array([CX + rnd.uniform(-20, 20), CY - 40.0]); v = np.array([rnd.uniform(-260, 260), 0.0])
    cur = 0; t = 0.0; frames = []; last_bounce = -1
    while t < MAX_SIM:
        ev = []
        for _ in range(SUB):
            v[1] += G * DT
            p += v * DT; t += DT
            if cur >= NR: continue
            R = RADII[cur]
            d = p - (CX, CY); dist = math.hypot(*d)
            ang = math.atan2(d[1], d[0])
            rot = phase0[cur] + speeds[cur] * t
            diff = (ang - rot + math.pi) % (2 * math.pi) - math.pi
            in_gap = abs(diff) < GAP / 2 - BR / R
            if dist > R + BR:                           # fully escaped this ring
                ev.append(("break", cur, t)); cur += 1; continue
            if dist + BR >= R and not in_gap:
                n = d / dist
                vn = v @ n
                if vn > 0:
                    v = v - 2 * vn * n
                    # tangential kick from rotating ring + tiny randomness keeps it lively
                    tang = np.array([-n[1], n[0]])
                    v += tang * speeds[cur] * R * 0.06
                    v *= 1.0
                    sp = np.linalg.norm(v)
                    target = 1150 + 40 * cur
                    v *= (0.85 * sp + 0.15 * target) / sp
                    v += np.array([rnd.uniform(-25, 25), rnd.uniform(-25, 25)])
                    if t - last_bounce > 0.045: ev.append(("bounce", cur, t)); last_bounce = t
                p = np.array([CX, CY]) + n * (R - BR - 0.5)
        frames.append((p.copy(), cur, ev))
        if cur >= NR and len(frames) > 0:
            # keep running a bit after escape
            if not hasattr(simulate, "_"): pass
        if cur >= NR and t > esc_time(frames) + 1.6: break
    return frames, speeds, phase0

def esc_time(frames):
    for i, (_, _, ev) in enumerate(frames):
        for e in ev:
            if e[0] == "break" and e[1] == NR - 1: return e[2]
    return 1e9

frames, SPEEDS, PHASE0 = simulate(SEED)
ESC = esc_time(frames)
print("seed", SEED, "escape at", ESC, "sim frames", len(frames))
if ESC > 50: sys.exit("too slow — try another seed")

OUTRO = 3.4
TOTAL = len(frames) / FPS + OUTRO
NF = int(TOTAL * FPS)

# ---------------- background ----------------
yy, xx = np.mgrid[0:H, 0:W]
rr = np.hypot(xx - CX, yy - CY)
BG = np.zeros((H, W, 3), np.float32)
BG[...] = (7, 8, 18)
BG += (np.exp(-(rr / 700) ** 2) * 26)[..., None] * np.array([0.5, 0.6, 1.2])
BG = np.clip(BG, 0, 255).astype(np.uint8)

def glow(a, s=0.9):
    sm = cv2.resize(a, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    sm = cv2.GaussianBlur(sm, (0, 0), 5)
    sm2 = cv2.GaussianBlur(sm, (0, 0), 14)
    big = cv2.resize(sm, (W, H)); big2 = cv2.resize(sm2, (W, H))
    return cv2.add(cv2.addWeighted(a, 1, big, s, 0), (big2 * 0.6).astype(np.uint8))

# ---------------- audio ----------------
audio = np.zeros(int((TOTAL + 2) * SR), np.float32)
SCALE = [0, 2, 4, 7, 9]   # major pentatonic
def note_freq(k):
    octv, deg = divmod(k, 5); return 261.63 * 2 ** ((SCALE[deg] + 12 * octv) / 12)
def bell(t0, f, amp=0.3, dur=0.9):
    s = int(t0 * SR); n = int(dur * SR); tt = np.arange(n) / SR
    w = (np.sin(2 * np.pi * f * tt) + 0.35 * np.sin(2 * np.pi * 2 * f * tt) * np.exp(-tt * 6)
         + 0.15 * np.sin(2 * np.pi * 3.01 * f * tt) * np.exp(-tt * 10))
    env = np.exp(-tt * 4.5) * np.minimum(1, tt * 600)
    seg = (w * env * amp).astype(np.float32); audio[s:s + n] += seg[: len(audio) - s]
def burst(t0, amp=0.35, dur=0.5):
    s = int(t0 * SR); n = int(dur * SR); tt = np.arange(n) / SR
    nz = np.random.default_rng(int(t0 * 1000)).standard_normal(n)
    nz = np.diff(nz, prepend=0)          # crude high-pass -> crisp shatter
    env = np.exp(-tt * 9)
    audio[s:s + n] += (nz * env * amp * 0.5).astype(np.float32)[: len(audio) - s]
def sweep(t0, f0, f1, dur, amp):
    s = int(t0 * SR); n = int(dur * SR); tt = np.arange(n) / SR
    fr = f0 + (f1 - f0) * (tt / dur) ** 2; ph = 2 * np.pi * np.cumsum(fr) / SR
    audio[s:s + n] += (np.sin(ph) * np.sin(np.pi * tt / dur) ** 2 * amp).astype(np.float32)

MELODY = [5, 7, 9, 7, 8, 10, 9, 7, 6, 8, 10, 12, 11, 9, 7, 5]
mel_i = 0

# ---------------- render ----------------
proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                         "-crf", "18", "-pix_fmt", "yuv420p", "ring_noaudio.mp4"], stdin=subprocess.PIPE)

trail = []
particles = []        # [x, y, vx, vy, life, max_life, color, size]
broken_at = {}
pulse = [0.0] * NR
shake = 0.0
prng = np.random.default_rng(5)
bounces = 0

for f in range(NF):
    t = f / FPS
    sim = f < len(frames)
    if sim:
        p, cur, evs = frames[f]
        for kind, ri, et in evs:
            if kind == "bounce":
                bounces += 1; pulse[ri] = 1.0
                bell(et, note_freq(MELODY[mel_i % len(MELODY)] + ri // 4), 0.26); mel_i += 1
            else:
                broken_at[ri] = et; shake = 14 + ri
                bell(et, note_freq(10 + ri % 5), 0.32, 1.3); bell(et, note_freq(12 + ri % 5), 0.22, 1.3)
                burst(et, 0.45)
                R = RADII[ri]; rot = PHASE0[ri] + SPEEDS[ri] * et
                col = ring_color(ri)
                for _ in range(130):
                    a = rot + GAP / 2 + prng.uniform(0, 2 * math.pi - GAP)
                    sp = prng.uniform(80, 520)
                    particles.append([CX + R * math.cos(a), CY + R * math.sin(a),
                                      math.cos(a) * sp + prng.uniform(-80, 80), math.sin(a) * sp + prng.uniform(-200, 60),
                                      0.0, prng.uniform(0.9, 2.0), col, prng.uniform(3, 7)])
        if cur >= NR and ESC <= t < ESC + 1 / FPS: sweep(t, 300, 1600, 1.2, 0.3)
        trail.append(tuple(p)); trail = trail[-26:]
    ox = oy = 0
    if shake > 0.3:
        ox, oy = prng.uniform(-shake, shake), prng.uniform(-shake, shake); shake *= 0.82

    img = Image.fromarray(BG.copy()); d = ImageDraw.Draw(img)

    if sim:
        # rings
        for i, R in enumerate(RADII):
            if i in broken_at: continue
            rot = PHASE0[i] + SPEEDS[i] * t
            pulse[i] *= 0.82
            col = ring_color(i, 0.75 + 0.25 * pulse[i])
            if pulse[i] > 0.05: col = tuple(int(c + (255 - c) * pulse[i] * 0.7) for c in col)
            wdt = 7 + int(6 * pulse[i])
            start = math.degrees(rot + GAP / 2); end = math.degrees(rot - GAP / 2 + 2 * math.pi)
            box = [CX - R + ox, CY - R + oy, CX + R + ox, CY + R + oy]
            d.arc(box, start, end, fill=col, width=wdt)
        # trail
        hue_col = ring_color(min(cur, NR - 1))
        for k, (tx, ty) in enumerate(trail):
            a = (k + 1) / len(trail); r = BR * (0.3 + 0.7 * a)
            c = tuple(int(cc * a) for cc in hue_col)
            d.ellipse([tx - r + ox, ty - r + oy, tx + r + ox, ty + r + oy], fill=c)
        px, py = trail[-1]
        d.ellipse([px - BR + ox, py - BR + oy, px + BR + ox, py + BR + oy], fill=(255, 255, 255), outline=hue_col, width=4)

    # particles
    alive = []
    for q in particles:
        q[4] += 1 / FPS
        if q[4] > q[5]: continue
        q[3] += 900 / FPS; q[0] += q[2] / FPS; q[1] += q[3] / FPS; q[2] *= 0.985
        a = 1 - q[4] / q[5]; s = q[7] * (0.5 + 0.5 * a)
        c = tuple(int(cc * a) for cc in q[6])
        d.rectangle([q[0] - s + ox, q[1] - s + oy, q[0] + s + ox, q[1] + s + oy], fill=c)
        alive.append(q)
    particles = alive

    # HUD
    left = NR - len(broken_at)
    if sim and t < ESC:
        k = ease_back(t / 0.6)
        d.text((W / 2, 200), "CAN IT ESCAPE?", font=font("Inter-Black.otf", 96 * max(k, .01)), fill=(255, 255, 255), anchor="mm")
        d.text((W / 2, 300), "12 rotating rings. One tiny gap each.", font=font("Inter-SemiBold.otf", 40),
               fill=(160, 170, 210), anchor="mm")
        d.text((W / 2 - 200, 1590), f"{left:02d}", font=mono(96), fill=ring_color(min(cur, NR - 1)), anchor="mm")
        d.text((W / 2 - 200, 1665), "RINGS LEFT", font=font("Inter-Bold.otf", 30), fill=(160, 170, 210), anchor="mm")
        d.text((W / 2 + 200, 1590), f"{t:4.1f}", font=mono(96), fill=(255, 255, 255), anchor="mm")
        d.text((W / 2 + 200, 1665), "SECONDS", font=font("Inter-Bold.otf", 30), fill=(160, 170, 210), anchor="mm")
        # popup when a ring breaks
        latest = sorted(((bt, ri) for ri, bt in broken_at.items() if bt <= t))[-1:]
        for bt, ri in latest:
            age = t - bt
            if 0 <= age < 0.9:
                a = 1 - ease_out((age - 0.4) / 0.5) if age > 0.4 else 1
                sc = ease_back(age / 0.3)
                d.text((W / 2, 430), f"RING {ri + 1} DOWN!", font=font("Inter-Black.otf", 64 * max(sc, .01)),
                       fill=tuple(int(c * a) for c in ring_color(ri)), anchor="mm")
    elif sim:
        et = t - ESC
        sc = ease_back(et / 0.5)
        d.text((W / 2, 900), "ESCAPED!", font=font("Inter-Black.otf", 150 * max(sc, .01)), fill=(140, 255, 160), anchor="mm")
        k2 = ease_out((et - 0.3) / 0.5)
        if k2 > 0:
            d.text((W / 2, 1060), f"in {ESC:.1f} seconds  •  {bounces} bounces", font=font("Inter-Bold.otf", 50),
                   fill=(int(255 * k2),) * 3, anchor="mm")
    else:
        ot = t - len(frames) / FPS
        if ot < 1 / FPS: bell(t, 261.6, 0.25, 2); bell(t, 329.6, 0.22, 2); bell(t, 392, 0.22, 2); bell(t, 523.3, 0.18, 2)
        k = ease_back(ot / 0.6)
        # orbiting dots logo
        for i in range(NR):
            a = ot * (0.8 + 0.15 * i) * (1 if i % 2 == 0 else -1) + i
            r = 60 + i * 14
            x, y = CX + r * math.cos(a), 620 + r * math.sin(a) * 0.5
            d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=ring_color(i))
        d.text((W / 2, 950), "PYTHONIC", font=font("Inter-Black.otf", 150 * max(k, .01)), fill=(55, 160, 235), anchor="mm")
        d.text((W / 2, 1110), "LIFE", font=font("Inter-Black.otf", 150 * max(k, .01)), fill=(255, 212, 59), anchor="mm")
        k2 = ease_out((ot - 0.6) / 0.6)
        if k2 > 0:
            d.text((W / 2, 1270), "Pure Python physics", font=font("Inter-SemiBold.otf", 48), fill=(int(220 * k2),) * 3, anchor="mm")
            d.text((W / 2, 1360), "Follow for more", font=font("Inter-Bold.otf", 52),
                   fill=tuple(int(c * k2) for c in (255, 92, 122)), anchor="mm")

    out = glow(np.asarray(img), 0.85)
    if t < 0.2: out = (out * (t / 0.2)).astype(np.uint8)
    if t > TOTAL - 0.4: out = (out * max(0, (TOTAL - t) / 0.4)).astype(np.uint8)
    proc.stdin.write(out.tobytes())
proc.stdin.close(); proc.wait()

# soft pad underneath
tt = np.arange(len(audio)) / SR
pad = (np.sin(2 * np.pi * 65.4 * tt) + 0.5 * np.sin(2 * np.pi * 98 * tt)) * 0.04 * (0.6 + 0.4 * np.sin(2 * np.pi * 0.25 * tt))
audio += pad.astype(np.float32)
a = audio[: int(TOTAL * SR)]
a = np.tanh(a / (np.max(np.abs(a)) + 1e-6) * 1.6) * 0.9
with wave.open("ring_audio.wav", "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR); wf.writeframes((a * 32767).astype(np.int16).tobytes())
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "ring_noaudio.mp4", "-i", "ring_audio.wav", "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k", "-shortest", "pythonic_life_ring_escape.mp4"], check=True)
print("done", TOTAL, "s, bounces", bounces)
