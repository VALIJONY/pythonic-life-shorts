"""Pythonic Life — COLOR WAR: Pick a color, only one survives. (YouTube Shorts 1080x1920)
Pure Python: NumPy simulation + Pillow/OpenCV rendering + NumPy-synthesized audio, encoded with ffmpeg.
Usage: python3 color_war.py [seed]          (python3 color_war.py probe  -> scan seeds)
"""
import math, random, subprocess, wave, sys
from functools import lru_cache
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SR = 1080, 1920, 30, 44100
N, CS = 40, 24                      # grid cells, cell size px
BX, BY = 60, 400                    # board origin
BOARD = N * CS
SUB = 6
DT = 1 / (FPS * SUB)
BR = 11

TEAMS = ["RED", "BLUE", "GREEN", "YELLOW"]
COLS = [np.array(c, float) for c in [(255, 60, 90), (110, 90, 255), (70, 235, 90), (255, 205, 40)]]
NEUTRAL = np.array([38, 42, 60.])
ELIMS = [9.0, 16.0]                 # elimination times (4->3, 3->2)
FINAL_END = 28.0                    # final duel ends, winner decided
WIN_LEN, OUTRO_LEN = 3.6, 3.6
TOTAL = FINAL_END + WIN_LEN + OUTRO_LEN
SIM_FRAMES = int(FINAL_END * FPS)

# ---------------- simulation ----------------
def simulate(seed):
    rnd = random.Random(seed)
    own = np.zeros((N, N), np.int8)
    h = N // 2
    own[:h, :h] = 0; own[:h, h:] = 1; own[h:, :h] = 2; own[h:, h:] = 3   # [row, col]
    centers = {0: (h * CS / 2, h * CS / 2), 1: (h * CS * 1.5, h * CS / 2),
               2: (h * CS / 2, h * CS * 1.5), 3: (h * CS * 1.5, h * CS * 1.5)}
    balls = []
    def spawn(team, x, y, speed):
        a = rnd.uniform(0, 2 * math.pi)
        a = a if abs(math.cos(a)) > 0.25 and abs(math.sin(a)) > 0.25 else a + 0.6
        balls.append([x, y, math.cos(a) * speed, math.sin(a) * speed, team])
    speed = 560.0
    for t_ in range(4): spawn(t_, *centers[t_], speed)
    alive = [True] * 4
    log = []
    elim_i = 0
    for f in range(SIM_FRAMES):
        t = f / FPS
        ev = {"flips": [], "elim": None, "spawn": [], "final": False}
        if elim_i < len(ELIMS) and t >= ELIMS[elim_i]:
            counts = [(np.sum(own == k) if alive[k] else 1e9) for k in range(4)]
            loser = int(np.argmin(counts))
            alive[loser] = False
            dead_balls = [(b[0], b[1]) for b in balls if b[4] == loser]
            balls = [b for b in balls if b[4] != loser]
            own[own == loser] = -1
            ev["elim"] = (loser, dead_balls)
            speed *= 1.12
            for b in balls: s = math.hypot(b[2], b[3]); b[2] *= speed / s; b[3] *= speed / s
            for k in range(4):
                if alive[k]:
                    src = next(b for b in balls if b[4] == k)
                    spawn(k, src[0], src[1], speed); ev["spawn"].append(k)
            elim_i += 1
            if elim_i == len(ELIMS):
                ev["final"] = True
                speed *= 1.25
                for b in balls: s = math.hypot(b[2], b[3]); b[2] *= speed / s; b[3] *= speed / s
        for _ in range(SUB):
            for b in balls:
                x, y, vx, vy, tm = b
                nx, ny = x + vx * DT, y + vy * DT
                if nx - BR < 0 or nx + BR > BOARD: vx = -vx
                if ny - BR < 0 or ny + BR > BOARD: vy = -vy
                px = int((nx + math.copysign(BR, vx)) // CS); py = int(ny // CS)
                if 0 <= px < N and 0 <= py < N and own[py, px] != tm:
                    own[py, px] = tm; vx = -vx; ev["flips"].append((py, px, tm))
                px = int(nx // CS); py = int((ny + math.copysign(BR, vy)) // CS)
                if 0 <= px < N and 0 <= py < N and own[py, px] != tm:
                    own[py, px] = tm; vy = -vy; ev["flips"].append((py, px, tm))
                if ev["flips"] and ev["flips"][-1][2] == tm and rnd.random() < 0.3:
                    a = rnd.uniform(-0.06, 0.06); c, s = math.cos(a), math.sin(a)
                    vx, vy = vx * c - vy * s, vx * s + vy * c
                x = min(max(x + vx * DT, BR), BOARD - BR); y = min(max(y + vy * DT, BR), BOARD - BR)
                b[:] = [x, y, vx, vy, tm]
        log.append((own.copy(), [(b[0], b[1], b[4]) for b in balls], ev))
    counts = [int(np.sum(own == k)) for k in range(4)]
    winner = int(np.argmax(counts))
    return log, counts, winner, alive

if len(sys.argv) > 1 and sys.argv[1] == "probe":
    for s in range(1, 40):
        log, c, w, al = simulate(s)
        order = [log[int(e * FPS)][2]["elim"][0] for e in ELIMS]
        fin = [c[k] for k in range(4) if al[k]]
        share = max(fin) / sum(fin)
        print(s, "out:", [TEAMS[o] for o in order], "win:", TEAMS[w], f"{share:.2f}", fin)
    sys.exit()

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
LOG, FINAL_COUNTS, WINNER, ALIVE_END = simulate(SEED)
print("winner", TEAMS[WINNER], FINAL_COUNTS)

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

yy, xx = np.mgrid[0:H, 0:W]
BG = np.zeros((H, W, 3), np.float32); BG[...] = (6, 7, 14)
BG += (np.exp(-(((xx - W / 2) / 650) ** 2 + ((yy - 880) / 800) ** 2)) * 22)[..., None] * np.array([0.6, 0.7, 1.3])
BG = np.clip(BG, 0, 255).astype(np.uint8)

PALETTE = np.stack(COLS + [NEUTRAL])          # index -1 -> NEUTRAL (last)
# cell mask with 2px gaps (rounded-ish tiles)
tile = np.ones((CS, CS), np.float32); tile[:2, :] = 0.55; tile[:, :2] = 0.55; tile[-1:, :] = 0.75; tile[:, -1:] = 0.75
TILE = np.tile(tile, (N, N))[..., None]

def glow(a, s=0.8):
    sm = cv2.resize(a, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    b1 = cv2.GaussianBlur(sm, (0, 0), 4); b2 = cv2.GaussianBlur(sm, (0, 0), 12)
    return cv2.add(cv2.addWeighted(a, 1, cv2.resize(b1, (W, H)), s, 0), (cv2.resize(b2, (W, H)) * 0.5).astype(np.uint8))

# ---------------- audio ----------------
audio = np.zeros(int((TOTAL + 2) * SR), np.float32)
def put(t0, sig):
    s = int(t0 * SR); n = min(len(sig), len(audio) - s)
    if n > 0: audio[s:s + n] += sig[:n].astype(np.float32)
def tone(t0, f, dur, amp, decay=6.0, harm=0.3):
    tt = np.arange(int(dur * SR)) / SR
    w = np.sin(2 * np.pi * f * tt) + harm * np.sin(4 * np.pi * f * tt)
    put(t0, w * np.exp(-tt * decay) * np.minimum(1, tt * 800) * amp)
def boom(t0, amp=0.9):
    tt = np.arange(int(1.4 * SR)) / SR
    fr = 120 * np.exp(-tt * 3) + 35; ph = 2 * np.pi * np.cumsum(fr) / SR
    nz = np.random.default_rng(int(t0 * 99)).standard_normal(len(tt))
    put(t0, (np.sin(ph) * np.exp(-tt * 2.5) + nz * np.exp(-tt * 7) * 0.35) * amp)
def sweep(t0, f0, f1, dur, amp):
    tt = np.arange(int(dur * SR)) / SR
    fr = f0 + (f1 - f0) * (tt / dur) ** 2
    put(t0, np.sin(2 * np.pi * np.cumsum(fr) / SR) * (tt / dur) ** 1.5 * amp)
def kick(t0, amp=0.5):
    tt = np.arange(int(0.3 * SR)) / SR
    fr = 150 * np.exp(-tt * 25) + 45
    put(t0, np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-tt * 9) * amp)
TEAM_F = [523.3, 392.0, 659.3, 440.0]

# beat: speeds up each round
bt = 0.0
while bt < FINAL_END:
    kick(bt, 0.45)
    bpm = 110 if bt < ELIMS[0] else 124 if bt < ELIMS[1] else 145
    bt += 60 / bpm

# ---------------- render ----------------
proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                         "-pix_fmt", "yuv420p", "war_noaudio.mp4"], stdin=subprocess.PIPE)

flash = np.zeros((N, N), np.float32)
particles = []
prng = np.random.default_rng(1)
alive = [True] * 4
elim_events = []          # (time, team)
spawn_events = []
shake = 0.0
disp_counts = [N * N / 4] * 4
trails = {}
final_flag_t = None
NF = int(TOTAL * FPS)
last_own = LOG[-1][0]

def burst(x, y, col, n=160, spd=(150, 750)):
    for _ in range(n):
        a = prng.uniform(0, 2 * math.pi); s = prng.uniform(*spd)
        particles.append([x, y, math.cos(a) * s, math.sin(a) * s - 150, 0.0, prng.uniform(0.7, 1.8), col, prng.uniform(3, 8)])

for f in range(NF):
    t = f / FPS
    img_np = BG.copy()
    in_sim = f < len(LOG)
    if in_sim:
        own, balls, ev = LOG[f]
        flash *= 0.80
        team_flips = [0] * 4
        for (r, c, tm) in ev["flips"]: flash[r, c] = 1.0; team_flips[tm] += 1
        for k in range(4):
            if team_flips[k]: tone(t + k * 0.003, TEAM_F[k] * (1 + 0.5 * (team_flips[k] > 2)), 0.06, 0.05, 40, 0.1)
        if ev["elim"]:
            loser, dead = ev["elim"]; alive[loser] = False; elim_events.append((t, loser)); shake = 26
            flash[own == -1] = 1.0
            for (x, y) in dead: burst(BX + x, BY + y, tc(COLS[loser]), 220)
            boom(t)
            for k in ev["spawn"]: spawn_events.append((t, k))
            if ev["spawn"]: tone(t + 0.45, 880, 0.35, 0.25, 8); tone(t + 0.55, 1174.7, 0.4, 0.25, 8)
        if ev["final"]:
            final_flag_t = t; sweep(t + 0.2, 200, 1400, 1.0, 0.25)
    else:
        own = last_own; balls = []
        flash *= 0.85

    # winner takeover wave after FINAL_END
    wt = t - FINAL_END
    grid_idx = own.copy()
    if wt >= 0:
        rr, cc = np.mgrid[0:N, 0:N]
        wx, wy = N / 2, N / 2
        wave_r = wt * 22
        dist = np.hypot(rr - wy, cc - wx)
        mask = dist < wave_r
        newly = mask & (grid_idx != WINNER)
        grid_idx = np.where(mask, WINNER, grid_idx)
        ring = (np.abs(dist - wave_r) < 1.5)
        flash = np.maximum(flash, ring.astype(np.float32))
        if 0 <= wt < 1 / FPS:
            boom(t, 0.6)
            for i, fr in enumerate([523.3, 659.3, 784.0, 1046.5]): tone(t + 0.12 * i, fr, 1.2, 0.3, 3)
            tone(t + 0.6, 1318.5, 1.6, 0.3, 2.5)

    # draw board via numpy
    cols = PALETTE[np.where(grid_idx < 0, 4, grid_idx)]                   # (N,N,3)
    cols = cols + (255 - cols) * (flash[..., None] * 0.75)
    board = np.repeat(np.repeat(cols, CS, 0), CS, 1) * TILE
    ox = oy = 0
    if shake > 0.4: ox, oy = int(prng.uniform(-shake, shake)), int(prng.uniform(-shake, shake)); shake *= 0.8
    y0, x0 = BY + oy, BX + ox
    img_np[y0:y0 + BOARD, x0:x0 + BOARD] = np.clip(board, 0, 255).astype(np.uint8)

    img = Image.fromarray(img_np); d = ImageDraw.Draw(img)
    d.rounded_rectangle([BX - 8 + ox, BY - 8 + oy, BX + BOARD + 8 + ox, BY + BOARD + 8 + oy], radius=14,
                        outline=(90, 100, 150), width=3)

    # balls with trails
    for (x, y, tm) in balls:
        key = (tm, round(x / 200), round(y / 200))
        col = COLS[tm]
        cx, cy = BX + x + ox, BY + y + oy
        d.ellipse([cx - BR - 4, cy - BR - 4, cx + BR + 4, cy + BR + 4], fill=tc(col * 0.35))
        d.ellipse([cx - BR, cy - BR, cx + BR, cy + BR], fill=(255, 255, 255), outline=tc(col * 0.6), width=3)

    # particles
    keep = []
    for q in particles:
        q[4] += 1 / FPS
        if q[4] > q[5]: continue
        q[3] += 1100 / FPS; q[0] += q[2] / FPS; q[1] += q[3] / FPS; q[2] *= 0.98
        a = 1 - q[4] / q[5]; s = q[7] * (0.4 + 0.6 * a)
        d.rectangle([q[0] - s, q[1] - s, q[0] + s, q[1] + s], fill=tc(q[6], a))
        keep.append(q)
    particles = keep

    # ---------- HUD ----------
    counts = [int(np.sum(grid_idx == k)) for k in range(4)]
    for k in range(4): disp_counts[k] += (counts[k] - disp_counts[k]) * 0.35
    if t < 2.4:
        k1 = ease_back(t / 0.5)
        d.text((W / 2, 150), "PICK A COLOR", font=BLACK(110 * max(k1, .01)), fill=(255, 255, 255), anchor="mm")
        k2 = ease_out((t - 0.4) / 0.5)
        for i in range(4):
            cx = W / 2 + (i - 1.5) * 190; r = 52 * ease_back((t - 0.25 - i * 0.12) / 0.4)
            if r > 1: d.ellipse([cx - r, 300 - r, cx + r, 300 + r], fill=tc(COLS[i]), outline=(255, 255, 255), width=4)
        if t < 1 / FPS:
            for i in range(4): tone(0.25 + i * 0.12, TEAM_F[i], 0.4, 0.25, 7)
    elif in_sim:
        # live leaderboard: 4 rows
        for i in range(4):
            yb = 140 + i * 60
            col = COLS[i]
            out = not alive[i]
            share = disp_counts[i] / (N * N)
            top = max(disp_counts[j] for j in range(4) if alive[j]) / (N * N)
            rel = 0.35 + 0.65 * (share / top) ** 4 if top > 0 else 0
            d.text((60, yb), TEAMS[i], font=BLACK(38), fill=tc(col, 0.35 if out else 1), anchor="lm")
            bx0, bx1 = 270, 870
            d.rounded_rectangle([bx0, yb - 16, bx1, yb + 16], radius=16, fill=(24, 28, 44))
            if not out:
                d.rounded_rectangle([bx0, yb - 16, bx0 + max(32, (bx1 - bx0) * rel), yb + 16], radius=16, fill=tc(col))
                d.text((W - 60, yb), f"{share * 100:4.1f}%", font=mono(36), fill=(240, 240, 255), anchor="rm")
            else:
                d.text((W - 60, yb), "OUT", font=BLACK(38), fill=(255, 64, 96), anchor="rm")
                d.line([60, yb, 235, yb], fill=(255, 255, 255), width=4)
    # countdown / round info under board
    if in_sim and t >= 2.4:
        nxt = next((e for e in ELIMS if e > t), None)
        target = nxt if nxt is not None else FINAL_END
        rem = target - t
        label = "NEXT ELIMINATION" if nxt is not None else "FINAL DUEL ENDS"
        lab_col = (255, 64, 96) if nxt is not None else (255, 212, 59)
        d.text((W / 2, 1420), label, font=BOLD(38), fill=lab_col, anchor="mm")
        if rem <= 3.0:
            ph = rem % 1.0; sc = 1 + 0.35 * ease_out(ph)
            d.text((W / 2, 1530), f"{math.ceil(rem)}", font=BLACK(120 * sc), fill=(255, 255, 255), anchor="mm")
            if abs(ph - 0.999) < 1 / FPS or (rem % 1.0) > 1 - 1 / FPS:
                tone(t, 1046.5 if nxt is not None else 1318.5, 0.18, 0.35, 12)
        else:
            d.text((W / 2, 1520), f"{rem:4.1f}s", font=mono(80), fill=(255, 255, 255), anchor="mm")
        # progress bar
        prev = max([0.0] + [e for e in ELIMS if e <= t])
        prog = (t - prev) / (target - prev)
        d.rounded_rectangle([160, 1600, 920, 1614], radius=7, fill=(30, 34, 52))
        d.rounded_rectangle([160, 1600, 160 + 760 * prog, 1614], radius=7, fill=lab_col)

    # elimination banner
    for (et, k) in elim_events:
        age = t - et
        if 0 <= age < 1.8:
            a = 1 if age < 1.3 else 1 - (age - 1.3) / 0.5
            sc = ease_back(age / 0.35)
            d.rounded_rectangle([60, 790 - 95 * sc, W - 60, 790 + 95 * sc], radius=28, fill=tc((10, 10, 18), a),
                                outline=tc(COLS[k], a), width=6)
            d.text((W / 2, 765), f"{TEAMS[k]}", font=BLACK(86 * max(sc, .01)), fill=tc(COLS[k], a), anchor="mm")
            d.text((W / 2, 840), "ELIMINATED", font=BLACK(50 * max(sc, .01)), fill=tc((255, 255, 255), a), anchor="mm")
        if 1.2 <= age < 2.6:
            a = min(1, (age - 1.2) / 0.2) * (1 if age < 2.2 else 1 - (age - 2.2) / 0.4)
            msg = "FINAL DUEL!  SPEED x1.4" if final_flag_t and abs(et - final_flag_t) < 0.1 else "SURVIVORS GET +1 BALL"
            d.text((W / 2, 1000), msg, font=BLACK(58), fill=tc((255, 212, 59), a), anchor="mm")

    # winner screen
    if wt >= 0 and t < FINAL_END + WIN_LEN:
        sc = ease_back((wt - 0.4) / 0.5)
        if sc > 0:
            ov = Image.new("RGBA", (W, H), (0, 0, 0, int(190 * min(1, wt)))); img.paste(ov, (0, 0), ov); d = ImageDraw.Draw(img)
            d.text((W / 2, 780), TEAMS[WINNER], font=BLACK(190 * max(sc, .01)), fill=tc(COLS[WINNER]), anchor="mm")
            d.text((W / 2, 930), "WINS!", font=BLACK(150 * max(sc, .01)), fill=(255, 255, 255), anchor="mm")
            fin = [FINAL_COUNTS[k] for k in range(4) if ALIVE_END[k]]
            k2 = ease_out((wt - 1.0) / 0.5)
            if k2 > 0:
                diff = max(fin) - min(fin)
                d.text((W / 2, 1060), f"WON BY {diff} CELL{'S' if diff != 1 else ''}!", font=BLACK(64),
                       fill=tc((255, 212, 59), k2), anchor="mm")
                d.text((W / 2, 1140), f"{max(fin)} vs {min(fin)}", font=mono(44), fill=tc((230, 230, 255), k2), anchor="mm")
        if int(wt * FPS) % 3 == 0 and wt < 2.5:
            for _ in range(2):
                burst(prng.uniform(100, W - 100), prng.uniform(500, 900), tc(COLS[prng.integers(4)]), 25, (200, 600))
        # leaderboard frozen on top
        for i in range(4):
            yb = 140 + i * 60
            d.text((60, yb), TEAMS[i], font=BLACK(38), fill=tc(COLS[i], 1 if i == WINNER else 0.3), anchor="lm")

    # outro
    if t >= FINAL_END + WIN_LEN:
        ot = t - FINAL_END - WIN_LEN
        ov = Image.new("RGBA", (W, H), (6, 7, 14, int(235 * min(1, ot / 0.4)))); img.paste(ov, (0, 0), ov); d = ImageDraw.Draw(img)
        k = ease_back((ot - 0.2) / 0.5)
        if k > 0:
            d.text((W / 2, 640), "DID YOUR COLOR", font=BLACK(84 * max(k, .01)), fill=(255, 255, 255), anchor="mm")
            d.text((W / 2, 750), "SURVIVE?", font=BLACK(110 * max(k, .01)), fill=tc(COLS[WINNER]), anchor="mm")
        k2 = ease_out((ot - 0.7) / 0.5)
        if k2 > 0:
            d.text((W / 2, 870), "Comment your color below", font=SEMI(46), fill=tc((220, 220, 240), k2), anchor="mm")
            for i in range(4):
                cx = W / 2 + (i - 1.5) * 120; r = 30 * k2
                d.ellipse([cx - r, 960 - r, cx + r, 960 + r], fill=tc(COLS[i]))
        k3 = ease_back((ot - 1.2) / 0.5)
        if k3 > 0:
            d.text((W / 2, 1180), "PYTHONIC", font=BLACK(120 * max(k3, .01)), fill=(55, 160, 235), anchor="mm")
            d.text((W / 2, 1310), "LIFE", font=BLACK(120 * max(k3, .01)), fill=(255, 212, 59), anchor="mm")
            d.text((W / 2, 1420), "100% Python  •  Follow for Round 2", font=BOLD(40), fill=tc((255, 92, 122), min(1, k3)), anchor="mm")
        if ot < 1 / FPS:
            for i, fr in enumerate([392, 523.3, 659.3]): tone(t + 0.25 + 0.1 * i, fr, 1.0, 0.25, 4)

    out = glow(np.asarray(img), 0.65)
    if t < 0.15: out = (out * (t / 0.15)).astype(np.uint8)
    if t > TOTAL - 0.4: out = (out * max(0, (TOTAL - t) / 0.4)).astype(np.uint8)
    proc.stdin.write(out.tobytes())
proc.stdin.close(); proc.wait()

a = audio[: int(TOTAL * SR)]
a = np.tanh(a / (np.percentile(np.abs(a), 99.9) + 1e-6) * 1.1) * 0.92
with wave.open("war_audio.wav", "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR); wf.writeframes((a * 32767).astype(np.int16).tobytes())
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "war_noaudio.mp4", "-i", "war_audio.wav", "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k", "-shortest", "pythonic_life_color_war.mp4"], check=True)
print("done", TOTAL)
