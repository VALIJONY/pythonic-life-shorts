"""Pythonic Life — Sorting Algorithm Race (YouTube Shorts, 1080x1920).
100% Python: PIL + NumPy + OpenCV for frames, NumPy for synthesized audio, ffmpeg for encoding.
"""
import random, math, subprocess, wave
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1080, 1920, 30
SR = 44100
N = 48
random.seed(7)

F = "/usr/share/fonts/opentype/inter/"
from functools import lru_cache
@lru_cache(maxsize=None)
def font(name, size): return ImageFont.truetype(F + name, max(1, int(size)))
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
f_black = lambda s: font("Inter-Black.otf", s)
f_bold = lambda s: font("Inter-Bold.otf", s)
f_semi = lambda s: font("Inter-SemiBold.otf", s)
f_mono = lambda s: _mono(max(1, int(s)))
@lru_cache(maxsize=None)
def _mono(s): return ImageFont.truetype(MONO, s)

BLUE = np.array([55, 118, 171]); YELLOW = np.array([255, 212, 59])
BG_TOP = (8, 11, 22); BG_BOT = (16, 22, 44)
ACCENTS = [(255, 92, 122), (120, 220, 255), (255, 212, 59), (140, 255, 160)]

def lerp(a, b, t): return a + (b - a) * t
def ease_out(t): t = max(0, min(1, t)); return 1 - (1 - t) ** 3
def ease_back(t):
    t = max(0, min(1, t)); c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2

def val_color(v):
    t = v / (N - 1)
    c = BLUE * (1 - t) + YELLOW * t
    return tuple(int(x) for x in c)

# ---------------- algorithms -> recorded steps ----------------
def record(fn, data):
    a = data[:]; steps = []
    def cmp(i, j): steps.append([(i, j), []])
    def write(i, v):
        a[i] = v
        if steps: steps[-1][1].append((i, v))
    fn(a, cmp, write)
    assert a == sorted(data)
    return steps

def bubble(a, cmp, write):
    n = len(a)
    for i in range(n):
        sw = False
        for j in range(n - 1 - i):
            cmp(j, j + 1)
            if a[j] > a[j + 1]:
                x, y = a[j + 1], a[j]; write(j, x); write(j + 1, y); sw = True
        if not sw: break

def insertion(a, cmp, write):
    for i in range(1, len(a)):
        key = a[i]; j = i - 1
        while j >= 0:
            cmp(j, j + 1)
            if a[j] > key: write(j + 1, a[j]); j -= 1
            else: break
        write(j + 1, key)

def merge_sort(a, cmp, write):
    def ms(lo, hi):
        if hi - lo < 2: return
        mid = (lo + hi) // 2; ms(lo, mid); ms(mid, hi)
        L, R = a[lo:mid], a[mid:hi]; i = j = 0; k = lo
        while i < len(L) and j < len(R):
            cmp(lo + i, mid + j)
            if L[i] <= R[j]: write(k, L[i]); i += 1
            else: write(k, R[j]); j += 1
            k += 1
        while i < len(L): write(k, L[i]); i += 1; k += 1
        while j < len(R): write(k, R[j]); j += 1; k += 1
    ms(0, len(a))

def quick(a, cmp, write):
    def qs(lo, hi):
        if lo >= hi: return
        p = a[hi]; i = lo
        for j in range(lo, hi):
            cmp(j, hi)
            if a[j] < p:
                x, y = a[j], a[i]; write(i, x); write(j, y); i += 1
        x, y = a[hi], a[i]; write(i, x); write(hi, y)
        qs(lo, i - 1); qs(i + 1, hi)
    qs(0, len(a) - 1)

DATA = list(range(N)); random.shuffle(DATA)
ALGOS = [("BUBBLE SORT", "O(n²)", bubble), ("INSERTION SORT", "O(n²)", insertion),
         ("QUICK SORT", "O(n log n)", quick), ("MERGE SORT", "O(n log n)", merge_sort)]
STEPS = [record(fn, DATA) for _, _, fn in ALGOS]

# ---------------- timeline ----------------
T_INTRO, T_COUNT = 2.6, 1.8
RACE_START = T_INTRO + T_COUNT
RACE_LEN = 18.5
SPEED = max(len(s) for s in STEPS) / (RACE_LEN * FPS - 10)   # comparisons per frame
RES_START = RACE_START + RACE_LEN + 0.8
RES_LEN = 6.0
END_START = RES_START + RES_LEN
TOTAL = END_START + 3.6
NF = int(TOTAL * FPS)

# ---------------- background ----------------
def make_bg():
    g = np.linspace(0, 1, H)[:, None, None]
    bg = np.array(BG_TOP) * (1 - g) + np.array(BG_BOT) * g
    bg = np.repeat(bg, W, axis=1).astype(np.float32)
    for x in range(0, W, 60): bg[:, x] += 7
    for y in range(0, H, 60): bg[y, :] += 7
    yy, xx = np.mgrid[0:H, 0:W]
    vign = 1 - 0.45 * (((xx - W / 2) / W) ** 2 + ((yy - H / 2) / H) ** 2) * 2.2
    return np.clip(bg * vign[..., None], 0, 255).astype(np.uint8)
BG = make_bg()

def glow(img_np, strength=0.85):
    small = cv2.resize(img_np, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 6)
    big = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(img_np, 1.0, big, strength, 0)

def text_c(d, xy, s, fnt, fill, anchor="mm"):
    d.text(xy, s, font=fnt, fill=fill, anchor=anchor)

# ---------------- drawing helpers ----------------
PANEL_TOP, PANEL_H, PANEL_GAP = 300, 300, 24
PX0, PX1 = 60, W - 60

def draw_bars(d, arr, x0, x1, ybase, hmax, hl=(), wr=(), done_flash=0.0, alpha=1.0):
    bw = (x1 - x0) / N
    for i, v in enumerate(arr):
        h = 8 + (hmax - 8) * (v + 1) / N
        c = np.array(val_color(v), float)
        if i in hl: c = np.array([255, 255, 255.])
        elif i in wr: c = np.array([255, 92, 122.])
        if done_flash > 0: c = c + (np.array([140, 255, 160.]) - c) * done_flash
        c = tuple(int(x * alpha) for x in c)
        xa = x0 + i * bw + 1.5; xb = x0 + (i + 1) * bw - 1.5
        d.rounded_rectangle([xa, ybase - h, xb, ybase], radius=3, fill=c)

def fmt_t(sec): return f"{sec:4.1f}s"

# ---------------- audio buffer ----------------
audio = np.zeros(int(TOTAL * SR) + SR, np.float32)
def add_tone(t0, freq, dur, amp, kind="sine"):
    s = int(t0 * SR); n = int(dur * SR)
    t = np.arange(n) / SR
    env = np.exp(-t * (6 / dur)) * np.minimum(1, t * 400)
    if kind == "sine": w = np.sin(2 * np.pi * freq * t)
    else: w = np.sign(np.sin(2 * np.pi * freq * t)) * 0.5 + np.sin(2 * np.pi * freq * t) * 0.5
    audio[s:s + n] += (w * env * amp).astype(np.float32)[: len(audio) - s]
def add_sweep(t0, f0, f1, dur, amp):
    s = int(t0 * SR); n = int(dur * SR); t = np.arange(n) / SR
    fr = f0 + (f1 - f0) * (t / dur) ** 2
    ph = 2 * np.pi * np.cumsum(fr) / SR
    env = np.sin(np.pi * t / dur) ** 2
    audio[s:s + n] += (np.sin(ph) * env * amp).astype(np.float32)
def chord(t0, freqs, dur, amp):
    for fr in freqs: add_tone(t0, fr, dur, amp / len(freqs))

# ---------------- render ----------------
proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                         "-crf", "18", "-pix_fmt", "yuv420p", "video_noaudio.mp4"], stdin=subprocess.PIPE)

state = [DATA[:] for _ in ALGOS]
pos = [0] * len(ALGOS)
finish = [None] * len(ALGOS)
order = []
shuffle_arr = DATA[:]
rng = random.Random(3)

# intro audio
add_sweep(0.0, 120, 900, 1.2, 0.25)
chord(0.9, [261.6, 329.6, 392.0], 1.0, 0.35)
for k in range(3): add_tone(T_INTRO + k * 0.5, 660, 0.25, 0.4)
add_tone(T_INTRO + 1.5, 1320, 0.5, 0.45)

for f in range(NF):
    t = f / FPS
    frame = Image.fromarray(BG.copy())
    d = ImageDraw.Draw(frame)

    if t < RACE_START:
        # --- intro: shuffling bars + title ---
        if f % 2 == 0:
            i, j = rng.randrange(N), rng.randrange(N); shuffle_arr[i], shuffle_arr[j] = shuffle_arr[j], shuffle_arr[i]
        draw_bars(d, shuffle_arr, 60, W - 60, 1500, 520, alpha=0.55)
        k = ease_back(t / 0.7)
        s1 = f_black(int(118 * max(k, 0.01)))
        text_c(d, (W / 2, 470), "4 SORTING", s1, (255, 255, 255))
        k2 = ease_back((t - 0.35) / 0.7)
        if k2 > 0: text_c(d, (W / 2, 610), "ALGORITHMS", f_black(int(118 * k2)), (255, 212, 59))
        k3 = ease_out((t - 0.9) / 0.5)
        if k3 > 0:
            text_c(d, (W / 2, 760), "ONE RACE.", f_black(int(96 * k3)), (255, 92, 122))
        k4 = ease_out((t - 1.4) / 0.6)
        if k4 > 0:
            text_c(d, (W / 2, 880), "Which one finishes first?", f_semi(46), (int(200 * k4),) * 3)
        if t >= T_INTRO:
            ct = t - T_INTRO; idx = int(ct / 0.5)
            lab = ["3", "2", "1", "GO!"][min(idx, 3)]
            sub = (ct % 0.5) / 0.5 if idx < 3 else (ct - 1.5) / 0.3
            sc = 1.6 - 0.6 * ease_out(sub)
            ov = Image.new("RGBA", (W, H), (0, 0, 0, 150)); frame.paste(ov, (0, 0), ov); d = ImageDraw.Draw(frame)
            text_c(d, (W / 2, H / 2 - 80), lab, f_black(int(300 * sc)),
                   (140, 255, 160) if lab == "GO!" else (255, 255, 255))

    elif t < RES_START:
        # --- race ---
        rf = f - int(RACE_START * FPS)
        target = rf * SPEED
        text_c(d, (W / 2, 120), "SORTING RACE", f_black(84), (255, 255, 255))
        text_c(d, (W / 2, 200), f"{N} random numbers  •  same speed", f_semi(36), (150, 165, 200))
        tones = []
        for a in range(len(ALGOS)):
            steps = STEPS[a]; hl = (); wr = set()
            while pos[a] < len(steps) and pos[a] < target:
                (i, j), writes = steps[pos[a]]
                for (k, v) in writes: state[a][k] = v; wr.add(k)
                hl = (i, j); pos[a] += 1
            if hl: tones.append(state[a][hl[0]])
            if pos[a] >= len(steps) and finish[a] is None:
                finish[a] = rf / FPS; order.append(a)
                chord(t, [523.3 * (1 + 0.25 * (3 - len(order))), 659.3, 784.0], 0.7, 0.35)
            y0 = PANEL_TOP + a * (PANEL_H + PANEL_GAP)
            acc = ACCENTS[a]
            d.rounded_rectangle([PX0 - 20, y0, PX1 + 20, y0 + PANEL_H], radius=26,
                                fill=(18, 24, 46), outline=acc if finish[a] is None else (140, 255, 160), width=3)
            name, big_o, _ = ALGOS[a]
            d.text((PX0 + 6, y0 + 22), name, font=f_black(42), fill=acc)
            d.text((PX0 + 6, y0 + 74), big_o, font=f_mono(28), fill=(150, 165, 200))
            d.text((PX1 - 6, y0 + 26), f"{pos[a]:>5} cmp", font=f_mono(34), fill=(230, 235, 255), anchor="ra")
            flash = 0.0
            if finish[a] is not None:
                since = rf / FPS - finish[a]
                flash = max(0.0, 1 - since / 0.6) * 0.9 + 0.35
            draw_bars(d, state[a], PX0, PX1, y0 + PANEL_H - 20, PANEL_H - 130,
                      hl=hl if finish[a] is None else (), wr=wr if finish[a] is None else set(),
                      done_flash=min(1.0, flash) if finish[a] is not None else 0.0)
            # progress bar
            prog = pos[a] / len(steps)
            d.rounded_rectangle([PX0, y0 + 112, PX0 + (PX1 - PX0) * prog, y0 + 118], radius=3, fill=acc)
            if finish[a] is not None:
                rank = order.index(a) + 1
                pop = ease_back((rf / FPS - finish[a]) / 0.4)
                badge = f"#{rank}  {fmt_t(finish[a])}"
                fb = f_black(int(54 * max(pop, 0.01)))
                bx, by = W / 2, y0 + PANEL_H / 2 + 20
                tw = d.textlength(badge, font=fb)
                d.rounded_rectangle([bx - tw / 2 - 28, by - 46 * pop, bx + tw / 2 + 28, by + 46 * pop],
                                    radius=24, fill=(10, 14, 28), outline=(140, 255, 160), width=4)
                text_c(d, (bx, by), badge, fb, (140, 255, 160))
        # sonify comparisons (one blended tone per frame)
        for k, v in enumerate(tones):
            add_tone(t + k * 0.004, 220 + v * 18, 0.07, 0.10 / max(1, len(tones)) * 1.8, "square")
        elapsed = rf / FPS
        text_c(d, (W / 2, 1640), f"TIME {elapsed:5.1f}s", f_mono(60), (255, 255, 255))

    elif t < END_START:
        # --- results chart ---
        rt = t - RES_START
        if abs(rt - 0.0) < 1 / FPS: add_sweep(t, 200, 1200, 0.9, 0.22)
        text_c(d, (W / 2, 250), "RESULTS", f_black(110), (255, 255, 255))
        text_c(d, (W / 2, 350), "comparisons needed to sort 48 numbers", f_semi(36), (150, 165, 200))
        counts = [len(s) for s in STEPS]
        mx = max(counts)
        ranked = sorted(range(len(ALGOS)), key=lambda a: counts[a])
        for r, a in enumerate(ranked):
            y = 520 + r * 230
            kk = ease_out((rt - 0.3 - r * 0.35) / 1.2)
            if kk <= 0: continue
            acc = ACCENTS[a]
            d.text((PX0, y), f"#{r + 1}  {ALGOS[a][0]}", font=f_black(46), fill=acc)
            bar_w = (PX1 - PX0) * counts[a] / mx * kk
            d.rounded_rectangle([PX0, y + 70, PX0 + max(bar_w, 20), y + 150], radius=18, fill=acc)
            num = int(counts[a] * kk)
            lbl_x = PX0 + bar_w + 20
            if lbl_x + 200 > PX1:
                d.text((PX0 + bar_w - 20, y + 110), f"{num}", font=f_mono(46), fill=(10, 14, 28), anchor="rm")
            else:
                d.text((lbl_x, y + 110), f"{num}", font=f_mono(46), fill=(255, 255, 255), anchor="lm")
            if abs(rt - (0.3 + r * 0.35)) < 1 / FPS: add_tone(t, 440 * 2 ** (r / 4), 0.3, 0.3)
        k5 = ease_out((rt - 2.6) / 0.6)
        if k5 > 0:
            ratio = counts[0] / min(counts)
            text_c(d, (W / 2, 1460), f"Bubble Sort did {ratio:.1f}x more work", f_bold(50), (int(255 * k5),) * 3)
            text_c(d, (W / 2, 1540), "Same Python. Different algorithm.", f_semi(40),
                   tuple(int(c * k5) for c in (255, 212, 59)))

    else:
        # --- end card ---
        et = t - END_START
        if abs(et) < 1 / FPS: chord(t, [261.6, 329.6, 392.0, 523.3], 2.2, 0.4)
        draw_bars(d, sorted(DATA), 60, W - 60, 1500, 520, alpha=0.35 + 0.2 * math.sin(et * 3))
        k = ease_back(et / 0.6)
        text_c(d, (W / 2, 700), "PYTHONIC", f_black(int(150 * max(k, .01))), (55, 160, 235))
        text_c(d, (W / 2, 860), "LIFE", f_black(int(150 * max(k, .01))), (255, 212, 59))
        k2 = ease_out((et - 0.6) / 0.6)
        if k2 > 0:
            text_c(d, (W / 2, 1020), "Built 100% in Python", f_semi(48), (int(220 * k2),) * 3)
            text_c(d, (W / 2, 1110), "Follow for more", f_bold(52), tuple(int(c * k2) for c in (255, 92, 122)))

    out = glow(np.asarray(frame), 0.7)
    # fade in/out
    if t < 0.25: out = (out * (t / 0.25)).astype(np.uint8)
    if t > TOTAL - 0.4: out = (out * max(0, (TOTAL - t) / 0.4)).astype(np.uint8)
    proc.stdin.write(out.tobytes())

proc.stdin.close(); proc.wait()

# ---------------- audio: light bass pulse under race + write ----------------
beat = 60 / 120
tt = RACE_START
while tt < RES_START - 0.2:
    add_tone(tt, 55, 0.35, 0.35); tt += beat
a = audio[: int(TOTAL * SR)]
a = a / (np.max(np.abs(a)) + 1e-6) * 0.85
pcm = (a * 32767).astype(np.int16)
with wave.open("audio.wav", "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR); wf.writeframes(pcm.tobytes())

subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "video_noaudio.mp4", "-i", "audio.wav",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "pythonic_life_sorting_race.mp4"], check=True)
print("frames", NF, "dur", TOTAL, "steps", [len(s) for s in STEPS], "finish", finish)
