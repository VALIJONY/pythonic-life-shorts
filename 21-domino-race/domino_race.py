"""Pythonic Life — "PICK A TRACK: which dominoes ring the bell first?" (YouTube Shorts 1080x1920, ~14 s, loops, light theme)
Four tracks of dominoes with different heights and gaps. Each domino is a thin rod tipping over its front edge
(theta'' = 3g / 2h * sin(theta)); when its top touches the next one, the next starts to fall. First bell wins.
Usage: python3 domino_race.py  |  python3 domino_race.py still 0 5
"""
import math, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
from race_kit import *

TH = LIGHT
TRACK_Y = [650, 910, 1170, 1430]            # floor of each track
X0, XEND = 70, 880                          # first domino, bell
WD = 14                                     # domino thickness
GRAV = 210.0
KT = 0.55                                   # share of angular speed passed on at contact
LCOL = [(240, 70, 80), (255, 150, 30), (40, 175, 90), (40, 130, 240)]
NAMES = ["RED", "ORANGE", "GREEN", "BLUE"]
DT = 1 / 240

def build(rnd):
    """a track = segments with their own height/gap style"""
    xs, hs = [], []; x = X0
    while x < XEND - 40:
        style = rnd.choice(["short", "tall", "grow", "shrink", "mixed"]); n = rnd.randint(3, 6)
        for k in range(n):
            if style == "short": h = rnd.uniform(44, 56)
            elif style == "tall": h = rnd.uniform(100, 125)
            elif style == "grow": h = 46 + 76 * k / max(1, n - 1)
            elif style == "shrink": h = 122 - 76 * k / max(1, n - 1)
            else: h = rnd.uniform(46, 122)
            if x >= XEND - 40: break
            xs.append(x); hs.append(h)
            x += WD + h * rnd.uniform(0.30, 0.48)
    return xs, hs

def simulate(seed, T=16.0):
    rnd = random.Random(seed)
    tracks = []
    for l in range(4):
        xs, hs = build(rnd)
        n = len(xs)
        tracks.append(dict(xs=xs, hs=hs, th=[0.0] * n, om=[0.0] * n, state=[0] * n, lean=[0.0] * n, done=None))   # state 0 stand,1 fall,2 leaning
        tr = tracks[-1]; tr["state"][0] = 1; tr["th"][0] = 0.25; tr["om"][0] = 1.6
    frames, hits = [], []; t = 0.0; winner = t_win = None
    steps_per_frame = int(round(1 / (FPS * DT)))
    for f in range(int(T * FPS)):
        for s in range(steps_per_frame):
            t += DT
            for l, tr in enumerate(tracks):
                xs, hs, th, om, st = tr["xs"], tr["hs"], tr["th"], tr["om"], tr["state"]
                n = len(xs)
                for i in range(n):
                    if st[i] != 1: continue
                    om[i] += 1.5 * GRAV / hs[i] * math.sin(th[i]) * DT; th[i] = min(math.pi / 2, th[i] + om[i] * DT)
                    tip = xs[i] + WD + hs[i] * math.sin(th[i])
                    nxt = xs[i + 1] if i + 1 < n else XEND
                    if tip >= nxt:                                              # top touches the next domino (or the bell)
                        st[i] = 2; tr["lean"][i] = th[i]
                        if i + 1 < n:
                            st[i + 1] = 1; th[i + 1] = 0.02; om[i + 1] = KT * om[i]; hits.append((t, l, i + 1))
                        elif tr["done"] is None:
                            tr["done"] = t; hits.append((t, l, -1))
                            if winner is None: winner, t_win = l, t
                    elif th[i] >= math.pi / 2: st[i] = 3                       # fell flat without reaching (gap too big)
                # leaning dominoes follow the one in front of them down
                for i in range(n - 1):
                    if st[i] == 2:
                        th[i] = tr["lean"][i] + (math.pi / 2 - 0.06 - tr["lean"][i]) * (th[i + 1] / (math.pi / 2)) ** 1.6
        frames.append([[list(tr["th"]), tr["done"]] for tr in tracks])
        if t_win is not None and t > t_win + 5: break
    return tracks, frames, hits, winner, t_win

def progress(tracks, fr, l, t):
    th = fr[min(int(t * FPS), len(fr) - 1)][l][0]
    return sum(1 for a in th if a > 0.6) / len(th)

def pick_seed():
    for seed in range(1, 3000):
        tr, fr, hits, w, tw = simulate(seed, 12.0)
        if w is None or not 8.8 <= tw <= 10.2: continue
        if any(t_["done"] is None for t_ in tr): continue
        ends = sorted(t_["done"] for t_ in tr)
        if ends[1] - ends[0] > 0.7: continue
        p3 = [progress(tr, fr, l, tw - 3.0) for l in range(4)]
        if max(range(4), key=lambda l: p3[l]) != w: return seed
    raise RuntimeError("no seed")

SEED = pick_seed()
TRACKS, FR, HITS, WIN, T_WIN = simulate(SEED, 16.0)
T_END = T_WIN + 1.1
TOTAL = T_END + 3.3
print(f"seed {SEED}: {NAMES[WIN]} rings first at {T_WIN:.2f} s, ends {[round(t['done'], 2) for t in TRACKS]}, total {TOTAL:.1f} s", flush=True)
BGI = background(TH)

def rot(px, py, th):
    return px * math.cos(th) - py * math.sin(th), px * math.sin(th) + py * math.cos(th)

def render(fi):
    t = fi / FPS; k_ = min(fi, len(FR) - 1)
    S = 2; box = (30, 430, 1050, 1500)
    im = BGI.crop(box).resize(((box[2] - box[0]) * S, (box[3] - box[1]) * S), Image.NEAREST); d = ImageDraw.Draw(im)
    P = lambda x, y: ((x - box[0]) * S, (y - box[1]) * S)
    for l in range(4):
        fy = TRACK_Y[l]; tr = TRACKS[l]; ths, done = FR[k_][l]
        a, b = P(40, fy - 205); c, e = P(1040, fy + 22)
        d.rounded_rectangle([a, b, c, e], radius=18 * S, fill=mix((255, 255, 255), LCOL[l], 0.06), outline=mix((255, 255, 255), LCOL[l], 0.35), width=2 * S)
        a, b = P(50, fy); c, e = P(1030, fy + 10); d.rectangle([a, b, c, e], fill=(120, 126, 140))
        # bell
        rang = done is not None and t >= done
        sw = 0.35 * math.sin((t - done) * 22) * math.exp(-(t - done) * 2.5) if rang else 0.0
        bx, by = XEND + 34, fy - 120
        pts = [(-26, 0), (26, 0), (20, -40), (8, -52), (-8, -52), (-20, -40)]
        bp = [P(bx + x * math.cos(sw) - y * math.sin(sw), by + 40 + x * math.sin(sw) + y * math.cos(sw)) for x, y in [(px, py + 40) for px, py in pts]]
        a, b = P(bx, by - 16); c, e = P(bx, fy); d.line([(a, b), (c, e)], fill=(150, 156, 168), width=4 * S)
        d.polygon(bp, fill=(240, 180, 20) if rang else (225, 190, 90), outline=(170, 120, 10))
        if rang and t - done < 0.8:
            for rr in (1, 2):
                q = P(bx, by + 10); r_ = (30 + 70 * (t - done) * rr) * S
                d.arc([q[0] - r_, q[1] - r_, q[0] + r_, q[1] + r_], -60, 60, fill=(240, 160, 0), width=4 * S)
        # dominoes
        for i in range(len(tr["xs"])):
            th = ths[i]; h = tr["hs"][i]; px0, py0 = tr["xs"][i] + WD, fy
            corners = [(-WD, 0), (-WD, -h), (0, -h), (0, 0)]
            poly = [P(px0 + rx, py0 + ry) for rx, ry in (rot(cx, cy, th) for cx, cy in corners)]
            col = LCOL[l] if th < 0.6 else mix(LCOL[l], (40, 40, 50), 0.25)
            d.polygon(poly, fill=col, outline=mix(col, (0, 0, 0), 0.4))
            for k in (0.3, 0.7):
                dx, dy = rot(-WD / 2, -h * k, th); q = P(px0 + dx, py0 + dy)
                d.ellipse([q[0] - 2.4 * S, q[1] - 2.4 * S, q[0] + 2.4 * S, q[1] + 2.4 * S], fill=(255, 255, 255))
        d.text(P(70, fy - 180), NAMES[l], font=BLACK(26 * S), fill=LCOL[l], anchor="lm")
    img = BGI.copy(); img.paste(im.reduce(S), (box[0], box[1])); d = ImageDraw.Draw(img)
    title(d, TH, "TRACK", "")
    items = [(NAMES[l][0], LCOL[l], f"{int(100 * progress(TRACKS, FR, l, min(t, T_WIN)))}%", True) for l in range(4)]
    chips(d, TH, items, t, T_WIN, WIN, y=300, per_row=4, w=232, gap=14)
    if t < T_WIN: d.text((W / 2, 1560), "First bell to ring wins", font=SEMI(34), fill=TH["muted"], anchor="mm")
    banner(d, TH, t, T_WIN, 1040, [(NAMES[WIN] + " ", LCOL[WIN]), ("WINS", TH["text"])], "RANG THE BELL FIRST!", LCOL[WIN], T_END + 0.3)
    if T_WIN <= t: burst(d, t, T_WIN, XEND + 34, TRACK_Y[WIN] - 110, 9, [LCOL[WIN], (255, 196, 0), (60, 140, 255), (40, 175, 90)], 50)
    if t >= T_END:
        end_card(img, TH, t, t - T_END, [(NAMES[WIN] + "?", LCOL[WIN])], "Comment your track", "one domino, in Python:",
                 ["omega += 1.5 * g / h * sin(theta) * dt", "theta += omega * dt", "if x + h * sin(theta) >= next_x:", "    next.omega = 0.55 * omega"])
    return np.asarray(img)

def audio():
    m = Mixer(TOTAL); m.beat(120, hot_from=T_WIN - 4, roots=(130.81, 98.0, 110.0, 87.31))
    base = [523.3, 587.3, 659.3, 784.0]; last = -1
    for (th, l, i) in HITS:
        if i == -1:
            x = m.tt(1.5)
            m.put(m.sfx, th, sum(np.sin(2 * np.pi * f * x) * a for f, a in ((1318.5, 1), (1975.5, 0.5), (2637, 0.25))) * np.exp(-x * 2.5) * (0.12 if l == WIN else 0.05))
            continue
        if th > T_WIN or th - last < 0.03: continue
        m.crack(th, 0.12); m.ping(th, base[l] * (1 + 0.5 * i / len(TRACKS[l]["xs"])), 0.035); last = th
    m.impact(T_WIN, 1.1); m.chord(T_WIN + 0.05, [440.0, 554.4, 659.3, 880.0], 0.05)
    m.impact(T_END, 0.5); m.chord(T_END + 0.5, [1318.5, 1975.5], 0.05, 3)
    return m.final()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "still": stills(render, sys.argv[2:]); sys.exit()
    encode(render, TOTAL, audio(), "domino_race.mp4"); print("done", round(TOTAL, 1))
