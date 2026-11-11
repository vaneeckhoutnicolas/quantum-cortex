"""An original piece of music for the presentation video, synthesised here from scratch (no sample, no existing work):
a slow electronic bed in D minor, 84 beats per minute, built scene by scene on the video's own timeline (read from the
render's scenes.json), kept under the picture. Usage: python music.py <out.wav> <scenes.json>"""
import json
import sys
import wave
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve

SR = 48000
SCENES = {s['id']: (s['t0'], s['t1']) for s in json.load(open(sys.argv[2]))}
TOTAL = max(t1 for _, t1 in SCENES.values())
S = lambda sid: SCENES[sid]
N = int(TOTAL * SR)
t = np.arange(N) / SR
BPM = 84.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
rng = np.random.default_rng(20261111)


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


# ---------- the scenes of the video (seconds), and the music's sections mapped on them ----------
# the scene times come from scenes.json, written by render.js from page.html (the comment that stood here gave an earlier cut's times)
def env_between(a, b, fade_in=2.0, fade_out=2.0):
    """A smooth gain that is 1 between a and b, with raised cosine edges."""
    e = np.zeros(N)
    i0, i1 = int(a * SR), int(min(b, TOTAL) * SR)
    e[i0:i1] = 1.0
    fi, fo = int(fade_in * SR), int(fade_out * SR)
    if fi and i0 + fi < N:
        r = np.arange(fi) / fi; e[i0:i0 + fi] = 0.5 - 0.5 * np.cos(np.pi * r)
    if fo and i1 - fo >= 0:
        r = np.arange(fo) / fo; e[i1 - fo:i1] = 0.5 + 0.5 * np.cos(np.pi * r)
    return e


# chord progression in D minor (MIDI numbers), one chord per two bars; the list repeats
# Dm9, Bbmaj7, Fmaj7(9), Cadd9, Gm7, Bbmaj7, Dm9, Am7
PROG = [
    [50, 57, 60, 64, 65],   # D2 A2 C3 E3 F3  (Dm9, open)
    [46, 53, 57, 60, 65],   # Bb1 F2 A2 C3 F3 (Bbmaj7)
    [41, 48, 53, 57, 60],   # F1 C2 F2 A2 C3  (Fmaj7)
    [48, 55, 59, 62, 64],   # C2 G2 B2 D3 E3  (Cadd9)
    [43, 50, 53, 58, 60],   # G1 D2 F2 Bb2 C3 (Gm7)
    [46, 53, 57, 60, 65],   # Bbmaj7
    [50, 57, 60, 64, 65],   # Dm9
    [45, 52, 55, 60, 62],   # A1 E2 G2 C3 D3  (Am7)
]
CHORD_LEN = 2 * BAR


def chord_at(time):
    return PROG[int(time // CHORD_LEN) % len(PROG)]


# ---------- the pad: detuned saws through a slowly moving low pass, stereo by detune ----------
def saw(freq, n, phase=0.0):
    x = (phase + np.arange(n) * freq / SR) % 1.0
    return 2.0 * x - 1.0


def render_pad():
    L = np.zeros(N); R = np.zeros(N)
    n_chords = int(np.ceil(TOTAL / CHORD_LEN))
    for c in range(n_chords):
        a = c * CHORD_LEN; b = min(TOTAL, (c + 1) * CHORD_LEN + 1.2)   # chords overlap by 1.2 s and cross fade
        i0, i1 = int(a * SR), int(b * SR); n = i1 - i0
        if n <= 0:
            continue
        notes = PROG[c % len(PROG)]
        segL = np.zeros(n); segR = np.zeros(n)
        for k, m in enumerate(notes):
            f = midi(m)
            for d, side in ((-7, 0), (0, 0.5), (+7, 1)):
                fr = f * 2 ** (d / 1200)
                s = saw(fr, n, rng.random()) * (0.5 if k == 0 else 0.35)
                segL += s * (1 - side) ; segR += s * side
        # the edges of the segment
        ramp = np.ones(n); fi = int(1.2 * SR)
        ramp[:fi] = np.linspace(0, 1, fi); ramp[-fi:] = np.minimum(ramp[-fi:], np.linspace(1, 0, fi))
        L[i0:i1] += segL * ramp; R[i0:i1] += segR * ramp
    # a slow low pass whose cutoff breathes (0.045 Hz) and opens a little in the later sections
    out = []
    for ch in (L, R):
        y = np.zeros(N)
        block = SR // 2
        for s in range(0, N, block):
            tt = s / SR
            lift = 0.0 if tt < S('s8')[0] else min(1.0, (tt - S('s8')[0]) / 12)          # opens from the integrators' scene
            cutoff = 520 + 220 * np.sin(2 * np.pi * 0.045 * tt) + 380 * lift
            sos = butter(2, cutoff / (SR / 2), btype='low', output='sos')
            y[s:s + block] = sosfilt(sos, ch[s:s + block])
        out.append(y)
    return np.stack(out, axis=1) * 0.11


# ---------- the sub bass: a sine on the root, one note per chord, soft attack ----------
def render_bass():
    y = np.zeros(N)
    n_chords = int(np.ceil(TOTAL / CHORD_LEN))
    for c in range(n_chords):
        a = c * CHORD_LEN; b = min(TOTAL, (c + 1) * CHORD_LEN)
        i0, i1 = int(a * SR), int(b * SR); n = i1 - i0
        if n <= 0:
            continue
        f = midi(PROG[c % len(PROG)][0])
        ph = np.cumsum(np.full(n, f / SR))
        env = np.ones(n); fi = int(0.8 * SR); env[:fi] = np.linspace(0, 1, fi); env[-fi:] = np.minimum(env[-fi:], np.linspace(1, 0, fi))
        y[i0:i1] += (np.sin(2 * np.pi * ph) + 0.15 * np.sin(4 * np.pi * ph)) * env
    return np.stack([y, y], axis=1) * 0.16


# ---------- the pulse: a soft kick on beats one and three, a tick on two and four, from the second scene ----------
def kick(n):
    tt = np.arange(n) / SR
    f = 42 + 70 * np.exp(-tt * 28)
    ph = np.cumsum(f / SR)
    return np.sin(2 * np.pi * ph) * np.exp(-tt * 9.0)


def tick(n):
    tt = np.arange(n) / SR
    noise = rng.standard_normal(n)
    sos = butter(2, [3000 / (SR / 2), 9000 / (SR / 2)], btype='band', output='sos')
    return sosfilt(sos, noise) * np.exp(-tt * 90.0) * 0.5


def render_pulse():
    y = np.zeros((N, 2))
    k = kick(int(0.45 * SR)); tk = tick(int(0.08 * SR))
    beat = 0
    while beat * BEAT < TOTAL:
        time = beat * BEAT; i = int(time * SR)
        if beat % 2 == 0:
            seg = k[:min(len(k), N - i)]; y[i:i + len(seg), 0] += seg; y[i:i + len(seg), 1] += seg
        else:
            seg = tk[:min(len(tk), N - i)]; y[i:i + len(seg), 0] += seg * 0.8; y[i:i + len(seg), 1] += seg * 1.0
        beat += 1
    # present from the "how it is built" scene, softer during "what fails" and "security", gone before the end card
    g = env_between(S('s2')[0], S('s6')[0], 3.0, 2.0) * 1.0 + env_between(S('s6')[0], S('s8')[0], 2.0, 2.0) * 0.55 + env_between(S('s8')[0], S('s12')[0] - 1.0, 2.0, 4.0) * 1.0
    return y * g[:, None] * 0.22


# ---------- the arpeggio: a plucked tone on eighth notes over the chord, alternating sides (the travelling dots) ----------
def pluck(freq, n):
    tt = np.arange(n) / SR
    ph = 2 * np.pi * freq * tt
    return (np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.12 * np.sin(3 * ph)) * np.exp(-tt * 7.5) * (1 - np.exp(-tt * 600))


def render_arp():
    y = np.zeros((N, 2))
    step = BEAT / 2
    k = 0
    while k * step < TOTAL:
        time = k * step; i = int(time * SR)
        notes = chord_at(time)
        order = [1, 2, 3, 4, 3, 2]                                     # up and down over the upper four notes
        m = notes[order[k % len(order)]] + 12 + (12 if (k // 16) % 4 == 3 else 0)
        n = int(0.9 * SR); seg = pluck(midi(m), min(n, N - i))
        side = 0.3 + 0.4 * ((k % 2) == 0)                             # alternating sides
        y[i:i + len(seg), 0] += seg * (1 - side); y[i:i + len(seg), 1] += seg * side
        k += 1
    # the arpeggio enters with the flow scene, rests in the console scene, returns for the measured scene,
    # lifts for the integrators and the roadmap, and leaves with the pillars
    g = (env_between(S('s2')[0], S('s2')[1], 2.5, 1.5) * 0.9 + env_between(S('s4')[0], S('s6')[0], 2.0, 2.0) * 0.75
         + env_between(S('s8')[0], S('s11')[0], 2.0, 2.0) * 1.0 + env_between(S('s11')[0], S('s11')[1] + 2.0, 1.0, 5.0) * 0.5)
    return y * g[:, None] * 0.09


# ---------- a high, slow shimmer: two sines a fifth apart, very quiet, for the title and the end ----------
def render_shimmer():
    y = np.zeros(N)
    for f, w in ((midi(81), 1.0), (midi(88), 0.6), (midi(93), 0.35)):
        y += np.sin(2 * np.pi * f * t + 0.8 * np.sin(2 * np.pi * 0.09 * t)) * w
    g = env_between(S('s1')[0] - 2.0, S('s1')[1] + 1.0, 2.5, 3.0) * 0.8 + env_between(S('s11')[0] - 2.0, TOTAL, 4.0, 5.0) * 1.0 + env_between(0.0, TOTAL, 1.0, 6.0) * 0.25
    return np.stack([y, y * 0.9], axis=1) * g[:, None] * 0.03


# ---------- reverb: a synthetic room (exponentially decaying noise, 2.4 s), mixed under ----------
def reverb(x, wet=0.28, seconds=2.4):
    n = int(seconds * SR); tt = np.arange(n) / SR
    ir = rng.standard_normal((n, 2)) * np.exp(-tt / 0.55)[:, None]
    sos = butter(2, 4200 / (SR / 2), btype='low', output='sos')
    ir = sosfilt(sos, ir, axis=0); ir /= np.sqrt((ir ** 2).sum(axis=0))
    w = np.stack([fftconvolve(x[:, 0], ir[:, 0])[:N], fftconvolve(x[:, 1], ir[:, 1])[:N]], axis=1)
    return x + wet * w


def main():
    out = sys.argv[1]
    mix = render_pad() + render_bass() + render_pulse() + render_arp() + render_shimmer()
    mix = reverb(mix)
    # the whole piece: in over the hook, out over the end card
    g = env_between(0.0, TOTAL, 3.5, 6.0)
    mix *= g[:, None]
    # a gentle limiter
    peak = np.abs(mix).max(); mix = mix / peak * 0.6
    mix = np.tanh(mix * 1.4) / np.tanh(1.4) * 0.25      # 2026-11-11: 6 dB lower than the first mix (which sat at minus 18.2 LUFS integrated), at the founder's request
    rms = np.sqrt((mix ** 2).mean())
    pcm = (np.clip(mix, -1, 1) * 32767).astype('<i2')
    with wave.open(out, 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
    print('wrote', out, 'seconds', TOTAL, 'peak', round(float(np.abs(mix).max()), 3), 'rms', round(float(rms), 4), 'rms dBFS', round(20 * np.log10(rms), 1))


main()
