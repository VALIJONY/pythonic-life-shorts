"""Shared helpers for the Pythonic Life "pick and watch" race Shorts: themes, fonts, chips, banner, end card,
synthesized audio and the ffmpeg encoder. Each video script imports this file."""
import math, os, random, re, subprocess, wave
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SR = 1080, 1920, 30, 44100
YEL, BLUE, GREEN, RED = (255, 196, 0), (60, 140, 255), (46, 170, 80), (235, 70, 70)

DARK = dict(bg=(13, 17, 23), panel=(22, 27, 34), panel2=(30, 36, 44), border=(48, 54, 61), text=(230, 237, 243),
            muted=(139, 148, 158), stroke=(8, 10, 16), sw=5, arena=(18, 23, 30), dot=(26, 33, 44), title2=(255, 212, 59))
LIGHT = dict(bg=(246, 247, 250), panel=(255, 255, 255), panel2=(236, 239, 244), border=(206, 212, 222), text=(22, 26, 34),
             muted=(100, 108, 122), stroke=(255, 255, 255), sw=0, arena=(255, 255, 255), dot=(226, 230, 237), title2=(240, 150, 0))

F = "/usr/share/fonts/opentype/inter/"
@lru_cache(None)
def font(n, s): return ImageFont.truetype(F + n, max(1, int(s)))
BLACK = lambda s: font("Inter-Black.otf", s)
BOLD = lambda s: font("Inter-Bold.otf", s)
SEMI = lambda s: font("Inter-SemiBold.otf", s)
@lru_cache(None)
def mono(s): return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", max(1, int(s)))
def ease_out(x): x = max(0.0, min(1.0, x)); return 1 - (1 - x) ** 3
def ease_back(x): x = max(0.0, min(1.0, x)); return 1 + 2.70158 * (x - 1) ** 3 + 1.70158 * (x - 1) ** 2
def mix(a, b, k): return tuple(int(a[i] + (b[i] - a[i]) * k) for i in range(3))

def background(th):
    im = Image.new("RGB", (W, H), th["bg"]); d = ImageDraw.Draw(im)
    for y in range(0, H, 54):
        for x in range(0, W, 54): d.ellipse([x - 2.5, y - 2.5, x + 2.5, y + 2.5], fill=th["dot"])
    return im

def text2(d, y, parts, fnt, th, cx=W / 2):
    tot = sum(d.textlength(s, font=fnt) for s, _ in parts); x = cx - tot / 2
    for s, c in parts:
        d.text((x, y), s, font=fnt, fill=c, anchor="lm", stroke_width=th["sw"], stroke_fill=th["stroke"]); x += d.textlength(s, font=fnt)

def title(d, th, word, rest=""):
    text2(d, 196, [("PICK A ", th["text"]), (word, th["title2"])] + ([(rest, th["text"])] if rest else []), BLACK(92), th)

def chips(d, th, items, t, t_win, win, y=286, per_row=4, w=232, gap=14):
    """items: list of (label, colour, value_text, alive)"""
    n = len(items); rows = (n + per_row - 1) // per_row
    for k, (lab, col, val, alive) in enumerate(items):
        r, c = divmod(k, per_row); m = min(per_row, n - r * per_row)
        cx = W / 2 + (c - (m - 1) / 2) * (w + gap); cy = y + r * 76
        iswin = t >= t_win and k == win; kk = ease_back((t - t_win) / 0.4) if iswin else 0
        dead = (not alive and not iswin) or (t >= t_win and not iswin)
        cc = mix(col, th["bg"], 0.6) if dead else col
        wd, hh = w * (1 + 0.1 * kk), 62 * (1 + 0.1 * kk)
        d.rounded_rectangle([cx - wd / 2, cy - hh / 2, cx + wd / 2, cy + hh / 2], radius=hh / 2, fill=mix(th["bg"], cc, 0.2),
                            outline=th["title2"] if iswin else cc, width=5 if iswin else 3)
        d.ellipse([cx - wd / 2 + 10, cy - 21, cx - wd / 2 + 52, cy + 21], fill=cc)
        d.text((cx - wd / 2 + 31, cy + 1), lab, font=BLACK(26), fill=(20, 22, 28), anchor="mm")
        d.text((cx + wd / 2 - 18, cy), val, font=mono(28) if len(val) < 6 else BOLD(24), fill=th["muted"] if dead else th["text"], anchor="rm")
        if not alive and not iswin: d.line([(cx - wd / 2 + 60, cy), (cx + wd / 2 - 12, cy)], fill=RED, width=4)

def burst(d, t, t0, cx, cy, seed, cols, n=40, life=1.0):
    dt = t - t0
    if not 0 <= dt < life: return
    rnd = random.Random(seed)
    for _ in range(n):
        a = rnd.uniform(0, 2 * math.pi); v = rnd.uniform(300, 900); s = rnd.uniform(5, 12)
        x = cx + math.cos(a) * v * dt; y = cy + math.sin(a) * v * dt + 900 * dt * dt; k = 1 - dt / life
        d.ellipse([x - s * k, y - s * k, x + s * k, y + s * k], fill=rnd.choice(cols))

def pop_text(d, th, t, t0, x, y, txt, col, life=0.8, size=42):
    if 0 <= t - t0 < life:
        a = 1 - (t - t0) / life
        d.text((min(max(x, 120), 960), y - 60 * (1 - a)), txt, font=BLACK(size), fill=mix(th["bg"], col, a), anchor="mm",
               stroke_width=4 if th["sw"] else 0, stroke_fill=th["stroke"])

def banner(d, th, t, t_win, cy, parts, sub, col, until):
    if not t_win <= t < until: return
    kb = ease_back((t - t_win) / 0.35)
    if kb <= 0.02: return
    d.rounded_rectangle([W / 2 - 360 * kb, cy - 100 * kb, W / 2 + 360 * kb, cy + 100 * kb], radius=30 * kb, fill=th["panel"], outline=col, width=6)
    if kb > 0.5:
        text2(d, cy - 30, parts, BLACK(88 * kb), th)
        d.text((W / 2, cy + 52), sub, font=BLACK(52 * kb), fill=th["title2"], anchor="mm", stroke_width=4 if th["sw"] else 0, stroke_fill=th["stroke"])

def end_card(img, th, t, et, pick_parts, ask, code_title, code):
    k = min(1.0, et / 0.25)
    img.paste(Image.blend(img, Image.new("RGB", img.size, th["bg"]), 0.85 * k))
    d = ImageDraw.Draw(img)
    kp = ease_out(et / 0.3); y0 = 360 + (1 - kp) * 120
    d.rounded_rectangle([40, y0, 940, y0 + 1180], radius=34, fill=th["panel"], outline=th["border"], width=3)
    if et < 0.1: return
    k1 = ease_back((et - 0.05) / 0.35)
    d.text((490, y0 + 100), "DID YOU PICK", font=BLACK(84 * k1), fill=th["text"], anchor="mm")
    text2(d, y0 + 210, pick_parts, BLACK(110 * k1), th, cx=490)
    k3 = ease_out((et - 0.45) / 0.25)
    if k3 > 0:
        d.text((490, y0 + 355), ask, font=BLACK(58), fill=mix(th["panel"], th["title2"], k3), anchor="mm")
        ay = y0 + 418 + 10 * math.sin(t * 8)
        d.polygon([(470, ay), (510, ay), (490, ay + 26)], fill=mix(th["panel"], th["title2"], k3))
    k4 = ease_out((et - 0.8) / 0.25)
    if k4 > 0:
        bx0, by0 = 70, y0 + 490; hh = 90 + 48 * len(code)
        d.rounded_rectangle([bx0, by0, 910, by0 + hh], radius=18, fill=(10, 13, 18), outline=th["border"], width=2)
        d.text((bx0 + 24, by0 + 28), code_title, font=SEMI(26), fill=(139, 148, 158), anchor="lm")
        for li, ln in enumerate(code):
            x = bx0 + 24; y = by0 + 80 + li * 48
            for m in re.finditer(r'(\b(?:if|for|in|and|or|not|else|return|while)\b)|(\w+(?=\())|(\d+\.?\d*)|(.)', ln):
                s = m.group(0); col = (255, 123, 114) if m.group(1) else (210, 168, 255) if m.group(2) else (121, 192, 255) if m.group(3) else (230, 237, 243)
                d.text((x, y), s, font=mono(25), fill=mix((10, 13, 18), col, k4), anchor="lm"); x += d.textlength(s, font=mono(25))
    k5 = ease_out((et - 1.2) / 0.25)
    if k5 > 0:
        d.text((490, y0 + 960), "Full code: link in profile", font=BOLD(40), fill=mix(th["panel"], th["text"], k5), anchor="mm")
        text2(d, y0 + 1070, [("PYTHONIC ", mix(th["panel"], BLUE, k5)), ("LIFE", mix(th["panel"], th["title2"], k5))], BLACK(68), th, cx=490)


# ------------------------------------------------------------------ audio
class Mixer:
    def __init__(self, total):
        self.total = total; self.ns = int((total + 1.5) * SR)
        self.mus = np.zeros(self.ns, np.float32); self.sfx = np.zeros(self.ns, np.float32)
    def put(self, buf, t0, sig):
        s = int(t0 * SR); n = min(len(sig), self.ns - s)
        if n > 0 and s >= 0: buf[s:s + n] += sig[:n].astype(np.float32)
    @staticmethod
    def tt(dur): return np.arange(int(dur * SR)) / SR
    @staticmethod
    def noise(n, seed): return np.random.default_rng(seed).standard_normal(n)
    @staticmethod
    def lp(x, a): return np.convolve(x, np.ones(a) / a, mode="same")
    def ping(self, t0, f, amp):
        x = self.tt(0.35); ph = 2 * np.pi * f * x
        self.put(self.sfx, t0, (np.sin(ph) + 0.4 * np.sin(2 * ph) + 0.15 * np.sin(3 * ph)) * np.exp(-x * 14) * np.minimum(1, x * 1500) * amp)
    def pop(self, t0, amp=0.5):
        x = self.tt(0.25)
        self.put(self.sfx, t0, self.noise(len(x), int(t0 * 1e4)) * np.exp(-x * 45) * 0.5 * amp)
        self.put(self.sfx, t0, np.sin(2 * np.pi * np.cumsum(900 * np.exp(-x * 30) + 120) / SR) * np.exp(-x * 20) * 0.5 * amp)
    def crack(self, t0, amp=0.25):
        x = self.tt(0.08); self.put(self.sfx, t0, (self.noise(len(x), int(t0 * 7e3)) - self.lp(self.noise(len(x), int(t0 * 7e3)), 4)) * np.exp(-x * 60) * amp)
    def thud(self, t0, amp=0.3):
        x = self.tt(0.25); self.put(self.sfx, t0, np.sin(2 * np.pi * np.cumsum(160 * np.exp(-x * 25) + 60) / SR) * np.exp(-x * 14) * amp)
    def impact(self, t0, amp=1.0):
        x = self.tt(1.0)
        self.put(self.sfx, t0, np.sin(2 * np.pi * np.cumsum(90 * np.exp(-x * 12) + 40) / SR) * np.exp(-x * 4) * 0.6 * amp)
        self.put(self.sfx, t0, self.lp(self.noise(len(x), 5), 5) * 2.2 * np.exp(-x * 10) * 0.3 * amp)
    def chord(self, t0, notes, amp, dec=2.5):
        x = self.tt(1.6)
        self.put(self.sfx, t0, sum(np.sin(2 * np.pi * f * x) + 0.3 * np.sin(4 * np.pi * f * x) for f in notes) * np.exp(-x * dec) * np.minimum(1, x * 500) * amp)
    def whoosh(self, t0, dur, amp, seed=1):
        x = self.tt(dur); n = self.noise(len(x), seed); k = x / dur
        self.put(self.sfx, t0, (self.lp(n, 16) * 3.5 * (1 - k) + (n - self.lp(n, 5)) * k) * np.sin(np.pi * k) ** 1.5 * amp)
    def beat(self, bpm=124, hot_from=1e9, roots=(110.0, 87.31, 130.81, 98.0)):
        B = 60 / bpm; t0 = 0.0; bar = 0
        while t0 < self.total + 0.5:
            hot = t0 >= hot_from
            for b in range(4):
                x = self.tt(0.3); self.put(self.mus, t0 + b * B, np.sin(2 * np.pi * np.cumsum(140 * np.exp(-x * 30) + 46) / SR) * np.exp(-x * 9) * (0.34 if hot else 0.28))
                if b in (1, 3):
                    x = self.tt(0.2); self.put(self.mus, t0 + b * B, (0.7 * self.noise(len(x), int((t0 + b) * 1000)) * np.exp(-x * 22) + 0.5 * np.sin(2 * np.pi * 200 * x) * np.exp(-x * 30)) * (0.16 if hot else 0.11))
                for h in range(4 if hot else 2):
                    x = self.tt(0.04); self.put(self.mus, t0 + b * B + h * B / (4 if hot else 2), np.diff(self.noise(len(x) + 1, int((t0 + b + h) * 991))) * np.exp(-x * 90) * (0.045 if h % 2 else 0.028))
            for e in range(8):
                x = self.tt(B / 2); ph = 2 * np.pi * roots[bar % 4] * x
                self.put(self.mus, t0 + e * B / 2, (np.sin(ph) + 0.2 * np.sign(np.sin(ph))) * np.minimum(1, x * 300) * np.exp(-x * 4) * 0.12)
            t0 += 4 * B; bar += 1
    def win(self, t_win, t_end):
        self.whoosh(t_win - 1.0, 1.0, 0.12, 8)
        self.impact(t_win, 1.3); self.chord(t_win + 0.05, [440.0, 554.4, 659.3, 880.0], 0.06)
        self.impact(t_end, 0.6); self.chord(t_end + 0.5, [1318.5, 1975.5], 0.05, 3)
    def final(self):
        a = (self.mus + self.sfx)[: int(self.total * SR)]
        a = np.tanh(a / (np.percentile(np.abs(a), 99.7) + 1e-6) * 1.1) * 0.9
        nf = int(0.08 * SR); a[-nf:] *= np.linspace(1, 0.3, nf)
        return a


def encode(render, total, audio, out, loop=0.5):
    """render(frame_index) -> HxWx3 uint8; the last `loop` seconds cross-fade into frame 0"""
    tmp = out.replace(".mp4", "_noaudio.mp4"); wav = out.replace(".mp4", ".wav")
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                             "-c:v", "libx264", "-preset", "slow", "-crf", "15", "-maxrate", "10M", "-bufsize", "20M", "-pix_fmt", "yuv420p",
                             "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", tmp], stdin=subprocess.PIPE)
    nf = int(total * FPS); F0 = render(0).astype(np.float32); tl = total - loop
    for f in range(nf):
        t = f / FPS; fr = render(f)
        if t > tl:
            a = min(1.0, (t - tl) / (total - tl - 1 / FPS)); fr = (fr * (1 - a) + F0 * a).astype(np.uint8)
        proc.stdin.write(np.ascontiguousarray(fr).tobytes())
    proc.stdin.close(); proc.wait()
    with wave.open(wav, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR); wf.writeframes((audio * 32767).astype(np.int16).tobytes())
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-i", wav, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", out], check=True)
    os.remove(tmp); os.remove(wav)

def stills(render, ts):
    os.makedirs("chk", exist_ok=True)
    for x in ts: Image.fromarray(render(int(float(x) * FPS))).save(f"chk/still_{x}.png")
