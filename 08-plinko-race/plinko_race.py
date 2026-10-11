"""Pythonic Life — PICK A NUMBER 1-9: Plinko Race (YouTube Shorts 1080x1920)
Galton-board physics in NumPy (ball-peg collisions), rendering with OpenCV + Pillow,
synthesized audio in NumPy, encoded with ffmpeg.
Usage: python3 plinko_race.py [seed]        (python3 plinko_race.py probe -> scan seeds)
"""
import math, subprocess, sys, wave, colorsys
from functools import lru_cache
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SR = 1080, 1920, 30, 44100
NB = 9                         # buckets
BX0, BW = 90, 100              # board left edge, bucket width  (9 * 100 = 900 -> 90..990)
BX1 = BX0 + NB * BW
TARGET = 60                    # first bucket to reach this many balls wins
RATE = 14.0                    # balls released per second
PEG_R, BALL_R = 11, 14
G = 1800.0
DRAG = 3.0                    # horizontal air drag (1/s): keeps balls from skipping pegs
REST = 0.30
SUB = 8
DT = 1 / (FPS * SUB)
Y_SPAWN = 400
ROWS, ROW_DY, Y0 = 11, 70, 480
Y_LAND = Y0 + (ROWS - 1) * ROW_DY + 80       # balls are counted below this line
Y_BASE, BAR_H = 1700, 440                   # bottom of buckets, bar height at TARGET
T_START = 0.35

def build_pegs():
    pts = []
    for r in range(ROWS):
        off = 0.5 if r % 2 else 0.0
        for c in range(NB + 1):
            x = BX0 + (c + off) * BW
            if BX0 + 8 < x < BX1 - 8 or (r % 2 == 0 and c in (0, NB)):
                pts.append((x, Y0 + r * ROW_DY))
    return np.array(pts, float)
PEGS = build_pegs()

# ---------------- simulation ----------------
def simulate(seed, max_t=40.0):
    rnd = np.random.default_rng(seed)
    cap = 500
    x = np.zeros(cap); y = np.zeros(cap); vx = np.zeros(cap); vy = np.zeros(cap)
    alive = np.zeros(cap, bool); n_spawned = 0
    counts = np.zeros(NB, int)
    frames = []          # per frame: (positions of alive balls, events list)
    winner = None; t_win = None
    spawn_acc = 0.0
    nframes = int(max_t * FPS)
    for f in range(nframes):
        t = f / FPS
        events = {"land": [], "peg": 0}
        for _ in range(SUB):
            if t >= T_START and winner is None:
                spawn_acc += RATE * DT
                while spawn_acc >= 1 and n_spawned < cap:
                    spawn_acc -= 1
                    x[n_spawned] = BX0 + BW * NB / 2 + rnd.uniform(-5, 5); y[n_spawned] = Y_SPAWN
                    vx[n_spawned] = rnd.uniform(-25, 25); vy[n_spawned] = 0
                    alive[n_spawned] = True; n_spawned += 1
            idx = np.where(alive)[0]
            if len(idx) == 0: continue
            vy[idx] += G * DT; vx[idx] *= (1 - DRAG * DT)
            x[idx] += vx[idx] * DT; y[idx] += vy[idx] * DT
            # walls
            lo = x[idx] < BX0 + BALL_R; hi = x[idx] > BX1 - BALL_R
            x[idx[lo]] = BX0 + BALL_R; vx[idx[lo]] = abs(vx[idx[lo]]) * 0.5
            x[idx[hi]] = BX1 - BALL_R; vx[idx[hi]] = -abs(vx[idx[hi]]) * 0.5
            # ball-peg collisions
            dx = x[idx][:, None] - PEGS[None, :, 0]; dy = y[idx][:, None] - PEGS[None, :, 1]
            d2 = dx * dx + dy * dy; R = PEG_R + BALL_R
            d2m = np.where(d2 < R * R, d2, np.inf)
            j = np.argmin(d2m, axis=1)
            hit = np.isfinite(d2m[np.arange(len(idx)), j])
            if hit.any():
                ii = idx[hit]; jj = j[hit]
                ddx = x[ii] - PEGS[jj, 0]; ddy = y[ii] - PEGS[jj, 1]
                dist = np.sqrt(ddx * ddx + ddy * ddy) + 1e-9
                nx, ny = ddx / dist, ddy / dist
                x[ii] = PEGS[jj, 0] + nx * R; y[ii] = PEGS[jj, 1] + ny * R
                vn = vx[ii] * nx + vy[ii] * ny
                m = vn < 0
                vx[ii[m]] -= (1 + REST) * vn[m] * nx[m]; vy[ii[m]] -= (1 + REST) * vn[m] * ny[m]
                vx[ii] += rnd.normal(0, 4, len(ii))                      # microscopic jitter
                events["peg"] += int(m.sum())
            # landing
            low = idx[y[idx] > Y_LAND]
            for b in low:
                alive[b] = False
                k = int(np.clip((x[b] - BX0) // BW, 0, NB - 1))
                if winner is None:
                    counts[k] += 1; events["land"].append((k, x[b]))
                    if counts[k] >= TARGET: winner = k; t_win = t
        idx = np.where(alive)[0]
        frames.append((np.stack([x[idx], y[idx]], 1).copy(), events, counts.copy()))
        if winner is not None and t > t_win + 1.2: break
    return frames, counts, winner, t_win

if len(sys.argv) > 1 and sys.argv[1] == "probe":
    for s in range(1, 41):
        fr, c, w, tw = simulate(s)
        srt = np.sort(c)[::-1]
        print(s, "winner", w + 1, "t", round(tw, 1) if tw else None, "margin", srt[0] - srt[1], "counts", list(c))
    sys.exit()

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
FR, COUNTS, WINNER, T_WIN = simulate(SEED)
assert WINNER is not None
print("seed", SEED, "winner", WINNER + 1, "at", round(T_WIN, 1), "counts", list(COUNTS))
RANK = np.argsort(-COUNTS)
RUNNER = int(RANK[1])
T_RES = T_WIN + 0.4
T_OUT = T_RES + 5.2
TOTAL = T_OUT + 3.6
NF = int(TOTAL * FPS)

# ---------------- helpers ----------------
F = "/usr/share/fonts/opentype/inter/"
@lru_cache(None)
def font(n, s): return ImageFont.truetype(F + n, max(1, int(s)))
@lru_cache(None)
def mono(s): return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", max(1, int(s)))
BLACK = lambda s: font("Inter-Black.otf", s)
BOLD = lambda s: font("Inter-Bold.otf", s)
SEMI = lambda s: font("Inter-SemiBold.otf", s)
def ease_out(v): v = max(0, min(1, v)); return 1 - (1 - v) ** 3
def ease_back(v): v = max(0, min(1, v)); return 1 + 2.70158 * (v - 1) ** 3 + 1.70158 * (v - 1) ** 2
def tc(c, a=1.0): return tuple(int(min(255, max(0, v * a))) for v in c)
BCOL = [tuple(int(255 * v) for v in colorsys.hsv_to_rgb(0.62 - i * 0.075, 0.75, 1.0)) for i in range(NB)]
STROKE = dict(stroke_width=6, stroke_fill=(5, 6, 14))

def glow(a, s=0.55):
    sm = cv2.resize(a, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    b1 = cv2.GaussianBlur(sm, (0, 0), 3); b2 = cv2.GaussianBlur(sm, (0, 0), 10)
    return cv2.add(cv2.addWeighted(a, 1, cv2.resize(b1, (W, H)), s, 0), (cv2.resize(b2, (W, H)) * 0.4).astype(np.uint8))

# static background: gradient, pegs, bucket walls
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
base = np.zeros((H, W, 3), np.float32); base[...] = (6, 7, 16)
base += (np.exp(-(((xx - W / 2) / 700) ** 2 + ((yy - 1000) / 800) ** 2)) * 20)[..., None] * np.array([0.6, 0.7, 1.3])
base = np.clip(base, 0, 255).astype(np.uint8)
for (px, py) in PEGS:
    cv2.circle(base, (int(px), int(py)), PEG_R, (170, 180, 215), -1, cv2.LINE_AA)
    cv2.circle(base, (int(px - 3), int(py - 3)), 3, (255, 255, 255), -1, cv2.LINE_AA)
for k in range(NB + 1):
    xk = BX0 + k * BW
    cv2.line(base, (xk, Y_BASE - BAR_H - 30), (xk, Y_BASE), (80, 92, 150), 3, cv2.LINE_AA)
cv2.line(base, (BX0, Y_BASE), (BX1, Y_BASE), (80, 92, 150), 3, cv2.LINE_AA)
# funnel
cv2.fillPoly(base, [np.array([[BX0 + BW * NB // 2 - 90, 300], [BX0 + BW * NB // 2 + 90, 300],
                              [BX0 + BW * NB // 2 + 14, Y_SPAWN - 14], [BX0 + BW * NB // 2 - 14, Y_SPAWN - 14]])],
             (60, 70, 120), cv2.LINE_AA)

# ---------------- audio ----------------
audio = np.zeros(int((TOTAL + 2) * SR), np.float32)
def put(t0, sig):
    s = int(t0 * SR); n = min(len(sig), len(audio) - s)
    if n > 0: audio[s:s + n] += sig[:n].astype(np.float32)
def tone(t0, f, dur, amp, decay=8.0):
    tt = np.arange(int(dur * SR)) / SR
    w = np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt) * np.exp(-tt * 10)
    put(t0, w * np.exp(-tt * decay) * np.minimum(1, tt * 900) * amp)
def kick(t0, amp=0.4):
    tt = np.arange(int(0.25 * SR)) / SR
    put(t0, np.sin(2 * np.pi * np.cumsum(130 * np.exp(-tt * 26) + 46) / SR) * np.exp(-tt * 10) * amp)
def boom(t0, amp=0.8):
    tt = np.arange(int(1.6 * SR)) / SR
    put(t0, np.sin(2 * np.pi * np.cumsum(90 * np.exp(-tt * 2.5) + 36) / SR) * np.exp(-tt * 2) * amp)
SC = [0, 2, 4, 7, 9, 12, 14, 16, 19]
def nfreq(k): return 261.63 * 2 ** (SC[k] / 12)
bt = 0.0
while bt < T_RES + 4:
    kick(bt, 0.32); bt += 0.5
for f, (pos, ev, cnt) in enumerate(FR):
    t = f / FPS
    for (k, _) in ev["land"]: tone(t, nfreq(k), 0.5, 0.17, 7)
    for i in range(min(ev["peg"], 3)):
        tone(t + 0.01 * i, 1500 + 900 * ((f * 7 + i * 13) % 10) / 10, 0.05, 0.018, 60)

# ---------------- render ----------------
proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                         "-maxrate", "8M", "-bufsize", "16M", "-pix_fmt", "yuv420p", "plinko_noaudio.mp4"], stdin=subprocess.PIPE)
sparks = []
srng = np.random.default_rng(3)
shown = np.zeros(NB)
final_counts = COUNTS.copy()
for f in range(NF):
    t = f / FPS
    sim = f < len(FR)
    pos, ev, cnt = FR[min(f, len(FR) - 1)]
    img = base.copy()
    if sim:
        for (k, bx) in ev["land"]:
            for _ in range(10):
                a = srng.uniform(math.pi, 2 * math.pi); sp = srng.uniform(80, 360)
                sparks.append([bx, Y_BASE - BAR_H * cnt[k] / TARGET, math.cos(a) * sp, math.sin(a) * sp, 0.0, BCOL[k]])
    shown += (cnt - shown) * 0.35
    # bars
    for k in range(NB):
        h = int(BAR_H * min(1.0, shown[k] / TARGET))
        if h > 1:
            x0 = BX0 + k * BW + 6; x1 = BX0 + (k + 1) * BW - 6
            bar = img[Y_BASE - h:Y_BASE, x0:x1].astype(np.float32)
            img[Y_BASE - h:Y_BASE, x0:x1] = np.clip(bar * 0.25 + np.array(BCOL[k]) * 0.75, 0, 255).astype(np.uint8)
            cv2.rectangle(img, (x0, Y_BASE - h), (x1, Y_BASE - h + 5), (255, 255, 255), -1)
    # balls
    for (bx, by) in (pos if sim else []):
        cv2.circle(img, (int(bx), int(by)), BALL_R + 4, (90, 60, 20), -1, cv2.LINE_AA)
        cv2.circle(img, (int(bx), int(by)), BALL_R, (255, 212, 59), -1, cv2.LINE_AA)
        cv2.circle(img, (int(bx - 4), int(by - 4)), 4, (255, 255, 255), -1, cv2.LINE_AA)
    keep = []
    for s_ in sparks:
        s_[4] += 1 / FPS
        if s_[4] > 0.6: continue
        s_[3] += 700 / FPS; s_[0] += s_[2] / FPS; s_[1] += s_[3] / FPS
        a = 1 - s_[4] / 0.6
        cv2.circle(img, (int(s_[0]), int(s_[1])), 4, tc(s_[5], a), -1, cv2.LINE_AA)
        keep.append(s_)
    sparks = keep
    # winner highlight
    if t >= T_WIN:
        pulse = 0.5 + 0.5 * math.sin((t - T_WIN) * 8)
        x0 = BX0 + WINNER * BW; cv2.rectangle(img, (x0 + 2, Y_BASE - BAR_H - 30), (x0 + BW - 2, Y_BASE), (255, 255, 255), 3 + int(2 * pulse))

    im = Image.fromarray(img); d = ImageDraw.Draw(im)
    # bucket numbers + counts
    for k in range(NB):
        cx = BX0 + k * BW + BW / 2
        win = t >= T_WIN and k == WINNER
        d.text((cx, Y_BASE + 62), str(k + 1), font=BLACK(70 if not win else 84), fill=BCOL[k], anchor="mm", **STROKE)
        d.text((cx, Y_BASE + 135), str(int(round(shown[k]))), font=mono(34), fill=(230, 232, 250), anchor="mm")

    # titles
    leader = int(np.argmax(cnt)) if cnt.max() > 0 else None
    if t < T_WIN:
        if t < 3.2:
            k = ease_back(t / 0.45)
            d.text((W / 2, 140), "PICK A NUMBER", font=BLACK(110 * max(k, .01)), fill=(255, 255, 255), anchor="mm", **STROKE)
            d.text((W / 2, 250), "1 TO 9", font=BLACK(96 * max(k, .01)), fill=(255, 212, 59), anchor="mm", **STROKE)
        else:
            a = ease_out((t - 3.2) / 0.4)
            d.text((W / 2, 140), "FIRST BUCKET TO", font=BLACK(64), fill=tc((255, 255, 255), a), anchor="mm", **STROKE)
            d.text((W / 2, 235), f"{TARGET} BALLS WINS", font=BLACK(92), fill=tc((255, 212, 59), a), anchor="mm", **STROKE)
        if leader is not None and t > 3.6:
            d.text((W / 2, 340), f"LEADING: #{leader + 1}  ({cnt[leader]})", font=mono(40), fill=tc(BCOL[leader]), anchor="mm", **STROKE)
    elif t < T_OUT:
        rt = t - T_WIN
        sc = ease_back(rt / 0.4)
        d.text((W / 2, 150), f"#{WINNER + 1} WINS!", font=BLACK(150 * max(sc, .01)), fill=tc(BCOL[WINNER]), anchor="mm", **STROKE)
        k2 = ease_out((rt - 0.5) / 0.5)
        if k2 > 0:
            d.text((W / 2, 280), f"{final_counts[WINNER]} balls vs {final_counts[RUNNER]} for #{RUNNER + 1}", font=BOLD(46),
                   fill=tc((235, 235, 250), k2), anchor="mm", **STROKE)
        k3 = ease_out((rt - 1.6) / 0.5)
        if k3 > 0:
            mid = int(final_counts[3] + final_counts[4] + final_counts[5]); pct = round(100 * mid / final_counts.sum())
            d.text((W / 2, 400), f"The middle 3 buckets caught {pct}% of all balls", font=SEMI(40), fill=tc((255, 120, 150), k3), anchor="mm", **STROKE)
            d.text((W / 2, 465), "Did you pick right?", font=BOLD(40), fill=tc((200, 205, 230), k3), anchor="mm", **STROKE)
    else:
        ot = t - T_OUT
        ov = Image.new("RGBA", (W, H), (5, 6, 14, int(210 * min(1, ot / 0.4)))); im.paste(ov, (0, 0), ov); d = ImageDraw.Draw(im)
        k = ease_back((ot - 0.1) / 0.5)
        if k > 0:
            d.text((W / 2, 620), "THIS IS THE", font=BLACK(76 * max(k, .01)), fill=(255, 255, 255), anchor="mm")
            d.text((W / 2, 735), "BELL CURVE", font=BLACK(120 * max(k, .01)), fill=(255, 212, 59), anchor="mm")
        k2 = ease_out((ot - 0.6) / 0.5)
        if k2 > 0:
            d.text((W / 2, 860), "The middle wins. The edges almost never do.", font=SEMI(40), fill=tc((220, 220, 240), k2), anchor="mm")
        k3 = ease_back((ot - 1.2) / 0.5)
        if k3 > 0:
            d.text((W / 2, 1060), "PYTHONIC", font=BLACK(120 * max(k3, .01)), fill=(55, 160, 235), anchor="mm")
            d.text((W / 2, 1190), "LIFE", font=BLACK(120 * max(k3, .01)), fill=(255, 212, 59), anchor="mm")
            d.text((W / 2, 1300), "Physics simulated 100% in Python  •  Follow", font=BOLD(36), fill=tc((255, 92, 122), min(1, k3)), anchor="mm")

    out = glow(np.asarray(im))
    if t < 0.15: out = (out * (t / 0.15)).astype(np.uint8)
    if t > TOTAL - 0.4: out = (out * max(0, (TOTAL - t) / 0.4)).astype(np.uint8)
    proc.stdin.write(out.tobytes())
proc.stdin.close(); proc.wait()

boom(T_WIN, 0.7)
for i, k in enumerate([0, 2, 4, 6, 8]): tone(T_WIN + 0.06 * i, nfreq(k) * 2, 1.4, 0.2, 2.5)
for i, fr in enumerate([261.6, 329.6, 392.0, 523.3]): tone(T_OUT + 0.12 * i, fr, 2.2, 0.22, 2.5)
a = audio[: int(TOTAL * SR)]
a = np.tanh(a / (np.percentile(np.abs(a), 99.9) + 1e-6) * 1.1) * 0.92
with wave.open("plinko_audio.wav", "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR); wf.writeframes((a * 32767).astype(np.int16).tobytes())
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "plinko_noaudio.mp4", "-i", "plinko_audio.wav", "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k", "-shortest", "pythonic_life_plinko_race.mp4"], check=True)
print("done", TOTAL)
