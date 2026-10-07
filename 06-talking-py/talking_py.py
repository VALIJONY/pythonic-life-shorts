"""Pythonic Life — "Py" the talking snake: 3 Python tricks that feel illegal (YouTube Shorts 1080x1920)
Voice: Piper neural TTS (offline). Character, lip-sync, captions, code cards and music: Pillow + NumPy + OpenCV.
Usage: python3 talking_py.py /path/to/en-us-lessac-medium.onnx
"""
import math, re, subprocess, sys, wave, os
from functools import lru_cache
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SR = 1080, 1920, 30, 44100
VOICE = sys.argv[1] if len(sys.argv) > 1 else "en-us-lessac-medium.onnx"

# ---------------- script ----------------
# (text, card index shown from this line, code typing starts on this line?)
LINES = [
    ("Stop scrolling.", 0, False),
    ("Here are three Python tricks that feel illegal to know.", 0, False),
    ("Number one.", 1, False),
    ("Swap two variables, without a temp variable.", 1, True),
    ("a comma b, equals b comma a. That's it.", 1, False),
    ("Number two.", 2, False),
    ("Find the most common item in any list, in one line.", 2, True),
    ("Number three. The walrus operator.", 3, False),
    ("Assign and check, in the same line.", 3, True),
    ("Which one did you already know?", 4, False),
    ("Comment below, and follow for part two.", 4, False),
]
CARDS = {
    1: ["a, b = 1, 2", "a, b = b, a", "print(a, b)  # 2 1"],
    2: ["from collections import Counter", "votes = ['py', 'js', 'py', 'go', 'py']",
        "Counter(votes).most_common(1)", "# [('py', 3)]"],
    3: ["data = list(range(42))", "if (n := len(data)) > 10:", "    print(f'{n} items!')", "# 42 items!"],
}
CARD_TITLES = {1: "SWAP WITHOUT TEMP", 2: "MOST COMMON IN 1 LINE", 3: "THE WALRUS :="}

# ---------------- TTS ----------------
def tts(text, path):
    subprocess.run([sys.executable, "-m", "piper", "-m", VOICE, "-f", path, "--length-scale", "0.92"],
                   input=text.encode(), check=True, capture_output=True)
    with wave.open(path) as wf:
        sr = wf.getframerate(); x = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float32) / 32768
    env = np.abs(x) > 0.01
    idx = np.where(env)[0]
    x = x[max(0, idx[0] - int(0.02 * sr)): idx[-1] + int(0.05 * sr)]
    t_new = np.arange(int(len(x) * SR / sr)) / SR
    return np.interp(t_new, np.arange(len(x)) / sr, x).astype(np.float32)

os.makedirs("tts_tmp", exist_ok=True)
segments = []
t = 0.25
GAP = 0.14
for i, (txt, card, typing) in enumerate(LINES):
    wav = tts(txt, f"tts_tmp/l{i}.wav")
    dur = len(wav) / SR
    words = txt.split()
    weights = np.array([len(re.sub(r"[^\w]", "", w)) + 1.5 for w in words], float)
    starts = t + np.concatenate([[0], np.cumsum(weights)[:-1]]) / weights.sum() * dur
    segments.append(dict(text=txt, card=card, typing=typing, start=t, end=t + dur, wav=wav,
                         words=list(zip(words, starts))))
    t += dur + (GAP + 0.18 if txt.endswith(".") and i in (1, 4, 6, 8) else GAP)
TOTAL = t + 2.6
print("duration", TOTAL)

voice = np.zeros(int((TOTAL + 1) * SR), np.float32)
for s in segments:
    a = int(s["start"] * SR); voice[a:a + len(s["wav"])] += s["wav"]
voice /= np.abs(voice).max() + 1e-6

# mouth amplitude per frame
NF = int(TOTAL * FPS)
hop = SR // FPS
amp = np.array([np.sqrt(np.mean(voice[i * hop:(i + 1) * hop] ** 2)) for i in range(NF)])
amp = np.clip(amp / (np.percentile(amp, 95) + 1e-6), 0, 1.2)
amp = np.convolve(amp, [0.25, 0.5, 0.25], mode="same")

# ---------------- fonts / helpers ----------------
F = "/usr/share/fonts/opentype/inter/"
@lru_cache(None)
def font(n, s): return ImageFont.truetype(F + n, max(1, int(s)))
@lru_cache(None)
def mono(s): return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", max(1, int(s)))
BLACK = lambda s: font("Inter-Black.otf", s)
BOLD = lambda s: font("Inter-Bold.otf", s)
def ease_out(x): x = max(0, min(1, x)); return 1 - (1 - x) ** 3
def ease_back(x): x = max(0, min(1, x)); return 1 + 2.70158 * (x - 1) ** 3 + 1.70158 * (x - 1) ** 2

PY_BLUE = (55, 118, 171); PY_BLUE_L = (85, 155, 215); PY_YEL = (255, 212, 59); NAVY = (14, 22, 48)

# ---------------- background ----------------
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
def background(t):
    g = (np.sin(xx / 240 + t * 0.6) + np.sin(yy / 310 - t * 0.4) + np.sin((xx + yy) / 420 + t * 0.3)) / 3
    r = 18 + 10 * g; gg = 16 + 8 * g; b = 40 + 22 * g
    return np.stack([r, gg, b], -1)
BG_CACHE = {}
def bg(t):
    k = int(t * 10)                      # update background 10x per second (cheap)
    if k not in BG_CACHE:
        BG_CACHE.clear(); BG_CACHE[k] = np.clip(background(k / 10), 0, 255).astype(np.uint8)
    return BG_CACHE[k]

# ---------------- mascot ----------------
MS = 2                                   # supersampling
def draw_py(t, a, blink, brow, bounce):
    w, h = 760, 720
    im = Image.new("RGBA", (w * MS, h * MS), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    S = lambda v: v * MS
    # coiled body: chain of circles from tail (left-bottom) to neck
    pts = []
    for i in range(70):
        u = i / 69
        ang = math.pi * 1.15 + u * math.pi * 2.0
        rad = 250 - 110 * u
        x = 380 + rad * math.cos(ang) * 1.15
        y = 600 + rad * math.sin(ang) * 0.42 - u * 120
        pts.append((x, y, 26 + 46 * min(1, u * 1.6)))
    pts.append((380, 420 + bounce * 0.5, 66))
    for (x, y, r) in pts:
        d.ellipse([S(x - r - 7), S(y - r - 7), S(x + r + 7), S(y + r + 7)], fill=NAVY)
    for (x, y, r) in pts:
        d.ellipse([S(x - r), S(y - r), S(x + r), S(y + r)], fill=PY_BLUE)
    for (x, y, r) in pts[::3]:
        d.ellipse([S(x - r * 0.35), S(y - r * 0.55), S(x + r * 0.35), S(y - r * 0.15)], fill=PY_YEL)
    # head
    hx, hy = 380, 290 + bounce
    d.ellipse([S(hx - 205), S(hy - 165), S(hx + 205), S(hy + 165)], fill=NAVY)
    d.ellipse([S(hx - 195), S(hy - 155), S(hx + 195), S(hy + 155)], fill=PY_BLUE)
    d.ellipse([S(hx - 150), S(hy - 140), S(hx + 60), S(hy - 40)], fill=PY_BLUE_L)          # highlight
    # cheeks
    for sx in (-1, 1):
        d.ellipse([S(hx + sx * 130 - 34), S(hy + 40), S(hx + sx * 130 + 34), S(hy + 76)], fill=(240, 120, 150))
    # eyes + glasses
    for sx in (-1, 1):
        ex, ey = hx + sx * 78, hy - 30
        eh = 46 * (1 - blink) + 3
        d.ellipse([S(ex - 42), S(ey - eh), S(ex + 42), S(ey + eh)], fill=(255, 255, 255))
        if blink < 0.8:
            d.ellipse([S(ex - 18), S(ey - min(22, eh) + 6), S(ex + 18), S(ey + min(22, eh) + 6)], fill=(12, 14, 24))
            d.ellipse([S(ex - 6), S(ey - 6), S(ex + 4), S(ey + 4)], fill=(255, 255, 255))
        d.ellipse([S(ex - 62), S(ey - 62), S(ex + 62), S(ey + 62)], outline=NAVY, width=S(11))
        # eyebrows
        by = ey - 82 - brow * 14
        d.line([S(ex - 34), S(by + sx * 0), S(ex + 34), S(by - 6)] if sx > 0 else [S(ex - 34), S(by - 6), S(ex + 34), S(by)],
               fill=NAVY, width=S(12))
    d.line([S(hx - 18), S(hy - 34), S(hx + 18), S(hy - 34)], fill=NAVY, width=S(10))
    # mouth
    mo = max(0.0, min(1.0, a))
    my = hy + 78
    if mo > 0.08:
        mh = 10 + 52 * mo; mw = 64 + 18 * mo
        d.ellipse([S(hx - mw), S(my - mh * 0.45), S(hx + mw), S(my + mh)], fill=(40, 10, 30), outline=NAVY, width=S(7))
        d.ellipse([S(hx - mw * 0.55), S(my + mh * 0.35), S(hx + mw * 0.55), S(my + mh * 0.95)], fill=(235, 90, 120))
    else:
        d.arc([S(hx - 60), S(my - 40), S(hx + 60), S(my + 30)], 20, 160, fill=NAVY, width=S(9))
    # little forked tongue flick when idle
    return im.resize((w, h), Image.LANCZOS)

# ---------------- code card ----------------
KW = {"from", "import", "if", "for", "in", "def", "return", "print", "list", "range", "len"}
def tokens(line):
    out = []
    for m in re.finditer(r"#.*|f?'[^']*'|\d+|\w+|:=|\S|\s+", line):
        s = m.group()
        if s.startswith("#"): c = (120, 135, 170)
        elif s.startswith("'") or s.startswith("f'"): c = (140, 230, 140)
        elif s.isdigit(): c = (255, 170, 90)
        elif s in KW: c = (255, 110, 160)
        elif s == ":=": c = (255, 212, 59)
        elif re.match(r"\w", s) and s[0].isupper(): c = (120, 200, 255)
        elif re.match(r"\w", s): c = (235, 238, 250)
        else: c = (130, 210, 230)
        out.append((s, c))
    return out

def draw_card(d, k, appear, typed_chars, cy=700):
    x0, x1 = 50, W - 50
    lines = CARDS[k]
    hgt = 130 + 62 * len(lines)
    y0 = cy - hgt / 2 + (1 - appear) * 80
    alpha = appear
    d.rounded_rectangle([x0, y0, x1, y0 + hgt], radius=28, fill=(16, 19, 34), outline=(70, 90, 150), width=3)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([x0 + 30 + i * 38, y0 + 28, x0 + 52 + i * 38, y0 + 50], fill=c)
    d.text((x1 - 30, y0 + 39), "trick.py", font=BOLD(28), fill=(120, 135, 170), anchor="rm")
    left = typed_chars
    fnt = mono(34)
    for li, line in enumerate(lines):
        x = x0 + 40; y = y0 + 100 + li * 62
        for s, c in tokens(line):
            if left <= 0: break
            part = s[:left]; left -= len(part)
            d.text((x, y), part, font=fnt, fill=c)
            x += d.textlength(part, font=fnt)
        if left <= 0:
            if (int(t_global * 3) % 2 == 0): d.rectangle([x + 2, y + 2, x + 18, y + 40], fill=(255, 212, 59))
            break
        left -= 0   # newline costs nothing
    return y0, hgt

# ---------------- audio: music bed + sfx ----------------
music = np.zeros_like(voice)
def put(buf, t0, sig):
    s = int(t0 * SR); n = min(len(sig), len(buf) - s)
    if n > 0: buf[s:s + n] += sig[:n]
def kick(t0):
    tt = np.arange(int(0.25 * SR)) / SR
    put(music, t0, (np.sin(2 * np.pi * np.cumsum(120 * np.exp(-tt * 28) + 48) / SR) * np.exp(-tt * 11) * 0.9).astype(np.float32))
def hat(t0, a=0.12):
    tt = np.arange(int(0.05 * SR)) / SR
    nz = np.diff(np.random.default_rng(int(t0 * 1e3)).standard_normal(len(tt) + 1)).astype(np.float32)
    put(music, t0, nz * np.exp(-tt * 70) * a)
def keys(t0, freqs, dur=1.2, a=0.07):
    tt = np.arange(int(dur * SR)) / SR
    sig = sum(np.sin(2 * np.pi * f * tt) for f in freqs) / len(freqs)
    put(music, t0, (sig * np.exp(-tt * 1.8) * np.minimum(1, tt * 40) * a).astype(np.float32))
def pop(t0, f=900, a=0.35):
    tt = np.arange(int(0.12 * SR)) / SR
    put(music, t0, (np.sin(2 * np.pi * (f + 900 * np.exp(-tt * 40)) * tt) * np.exp(-tt * 35) * a).astype(np.float32))
def whoosh(t0, a=0.25, dur=0.35):
    tt = np.arange(int(dur * SR)) / SR
    nz = np.random.default_rng(int(t0 * 77)).standard_normal(len(tt)).astype(np.float32)
    nz = np.convolve(nz, np.ones(12) / 12, mode="same")
    put(music, t0, nz * np.sin(np.pi * tt / dur) ** 2 * a)

bpm = 96; beat = 60 / bpm
CHORDS = [[220, 261.6, 329.6], [174.6, 220, 261.6], [196, 246.9, 293.7], [164.8, 207.7, 246.9]]
bt, bi = 0.0, 0
while bt < TOTAL - 1.5:
    if bi % 2 == 0: kick(bt)
    hat(bt + beat / 2)
    if bi % 4 == 0: keys(bt, CHORDS[(bi // 4) % 4], beat * 4)
    bt += beat; bi += 1

card_on = {}
for s in segments:
    if s["card"] not in card_on: card_on[s["card"]] = s["start"]
    if s["text"].startswith("Number"): pop(s["start"], 1200, 0.45)
for k, t0 in card_on.items():
    if k in (1, 2, 3): whoosh(t0 - 0.1)
type_start = {s["card"]: s["start"] for s in segments if s["typing"]}
for k, t0 in type_start.items():
    total = sum(len(l) for l in CARDS[k])
    for c in range(0, total, 3): hat(t0 + c / total * 1.4, 0.06)

# ---------------- render ----------------
proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                         "-maxrate", "8M", "-bufsize", "16M", "-pix_fmt", "yuv420p", "talk_noaudio.mp4"], stdin=subprocess.PIPE)
blink_times = [1.6, 4.9, 7.2, 10.8, 13.1, 16.5, 19.4, 22.0, 25.3]
def blink_at(t):
    for b in blink_times:
        if 0 <= t - b < 0.16: return math.sin((t - b) / 0.16 * math.pi)
    return 0.0

def seg_at(t):
    for s in segments:
        if s["start"] - 0.05 <= t < s["end"] + 0.25: return s
    return None

for f in range(NF):
    t = f / FPS; t_global = t
    img = Image.fromarray(bg(t)); d = ImageDraw.Draw(img)
    cur = None
    for s in segments:
        if s["start"] <= t: cur = s
    card = cur["card"] if cur else 0

    # ---- header ----
    d.text((W / 2, 120), "PYTHON TRICKS THAT", font=BLACK(64), fill=(255, 255, 255), anchor="mm")
    d.text((W / 2, 200), "FEEL ILLEGAL", font=BLACK(84), fill=(255, 92, 122), anchor="mm")
    for i in range(3):
        cx = W / 2 + (i - 1) * 110; active = card == i + 1; done = card > i + 1
        col = PY_YEL if active else ((120, 220, 140) if done else (60, 66, 96))
        d.ellipse([cx - 36, 260, cx + 36, 332], fill=col)
        d.text((cx, 296), "✓" if done and False else str(i + 1), font=BLACK(40), fill=NAVY, anchor="mm")

    # ---- centre: hook stamp / code cards / outro poll ----
    if card == 0:
        a = ease_back((t - 0.1) / 0.5)
        d.text((W / 2, 560), "3", font=BLACK(300 * max(a, .01)), fill=PY_YEL, anchor="mm")
        d.text((W / 2, 760), "TRICKS", font=BLACK(110 * max(a, .01)), fill=(255, 255, 255), anchor="mm")
        if t > 1.2:
            st = ease_back((t - 1.2) / 0.35)
            stamp = Image.new("RGBA", (620, 180), (0, 0, 0, 0)); sd = ImageDraw.Draw(stamp)
            sd.rounded_rectangle([8, 8, 612, 172], radius=20, outline=(255, 60, 80), width=12)
            sd.text((310, 92), "ILLEGAL?!", font=BLACK(110), fill=(255, 60, 80), anchor="mm")
            stamp = stamp.rotate(-12, expand=True, resample=Image.BICUBIC)
            sw, sh = stamp.size; sc = max(0.01, 1.6 - 0.6 * st)
            stamp = stamp.resize((int(sw * sc), int(sh * sc)))
            img.paste(stamp, (int(W / 2 - stamp.size[0] / 2), int(900 - stamp.size[1] / 2)), stamp)
            d = ImageDraw.Draw(img)
            if 1.2 <= t < 1.2 + 1 / FPS: pop(t, 300, 0.6)
    elif card in (1, 2, 3):
        appear = ease_out((t - card_on[card]) / 0.35)
        d.text((W / 2, 440), f"#{card}  {CARD_TITLES[card]}", font=BLACK(52), fill=PY_YEL, anchor="mm")
        total = sum(len(l) for l in CARDS[card])
        if card in type_start and t >= type_start[card]:
            typed = int(total * min(1, (t - type_start[card]) / 1.4))
        else:
            typed = 0
        draw_card(d, card, appear, typed, cy=720)
    else:
        ot = t - card_on[4]
        d.text((W / 2, 470), "WHICH ONE DID", font=BLACK(76), fill=(255, 255, 255), anchor="mm")
        d.text((W / 2, 560), "YOU KNOW?", font=BLACK(76), fill=PY_YEL, anchor="mm")
        for i, lab in enumerate(["1  a, b = b, a", "2  Counter().most_common", "3  walrus  :="]):
            a = ease_back((ot - 0.2 - i * 0.18) / 0.4)
            if a <= 0: continue
            y = 690 + i * 120
            d.rounded_rectangle([110, y - 46 * a, W - 110, y + 46 * a], radius=24, fill=(24, 30, 56), outline=(90, 110, 180), width=3)
            d.text((150, y), lab, font=mono(38), fill=(235, 238, 250), anchor="lm")
            if 0.2 + i * 0.18 <= ot < 0.2 + i * 0.18 + 1 / FPS: pop(t, 800 + i * 200, 0.3)

    # ---- captions (3-word chunks, current word highlighted) ----
    s = seg_at(t)
    if s:
        ws = s["words"]
        ci = max([i for i, (_, st) in enumerate(ws) if st <= t] or [0])
        c0 = (ci // 3) * 3
        chunk = ws[c0:c0 + 3]
        fnt = BLACK(76)
        widths = [d.textlength(w.upper(), font=fnt) for w, _ in chunk]
        x = W / 2 - (sum(widths) + 24 * (len(chunk) - 1)) / 2
        for j, ((w_, st), wd) in enumerate(zip(chunk, widths)):
            active = c0 + j == ci
            pop_s = 1 + 0.15 * (1 - ease_out((t - st) / 0.15)) if active else 1
            fz = BLACK(76 * pop_s)
            col = PY_YEL if active else (255, 255, 255)
            d.text((x + wd / 2, 1110), w_.upper(), font=fz, fill=col, anchor="mm", stroke_width=8, stroke_fill=(10, 10, 20))
            x += wd + 24

    # ---- mascot ----
    a = amp[f] if f < len(amp) else 0
    talking = s is not None
    bounce = math.sin(t * 2.4) * 8 - a * 14
    brow = a * 0.8 + (0.6 if s and s["text"].startswith(("Number", "Stop")) else 0)
    py = draw_py(t, a if talking else 0, blink_at(t), brow, bounce)
    img.paste(py, (int(W / 2 - 380), 1180), py)

    out = np.asarray(img)
    if t < 0.12: out = (out * (t / 0.12)).astype(np.uint8)
    if t > TOTAL - 0.4: out = (out * max(0, (TOTAL - t) / 0.4)).astype(np.uint8)
    proc.stdin.write(out.tobytes())
proc.stdin.close(); proc.wait()

# duck the music under the voice
venv = np.convolve(np.abs(voice), np.ones(4410) / 4410, mode="same")
duck = 1 - 0.55 * np.clip(venv / (venv.max() * 0.3 + 1e-6), 0, 1)
mix = voice * 0.95 + music * duck * 0.55
mix = mix[: int(TOTAL * SR)]
mix = np.tanh(mix / (np.percentile(np.abs(mix), 99.9) + 1e-6) * 1.05) * 0.92
with wave.open("talk_audio.wav", "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR); wf.writeframes((mix * 32767).astype(np.int16).tobytes())
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "talk_noaudio.mp4", "-i", "talk_audio.wav", "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k", "-shortest", "pythonic_life_talking_py.mp4"], check=True)
print("done", TOTAL)
