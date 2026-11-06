"""Audio pipeline: preprocess -> strum onsets -> per-strum pitch evidence.

Audio answers, per strum: when did it happen, which pitches sounded, was it the right chord.
Because the target chord is known we *verify* expected pitches instead of transcribing, and
onset detection is score-informed: a strict global pass, then a gentler search inside each
expected strum's window for strums the global pass missed (quiet up-strums).
"""
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

from ..theory import CHORDS, OPEN_MIDI, chord_chroma
from .params import P

log = logging.getLogger(__name__)

SR = 44100
CLICK_HZ = 3000.0          # metronome click frequency; above the guitar's fundamental range
BP_MIDI_OFFSET = 21        # basic-pitch pitch bin 0 = MIDI 21 (A0)
HOP = 256
# Thresholds live in params.py (P.*). Note P.strum_time_bias_s: the flux peak lands mid-strum
# (a strum spreads its string attacks over 10-40 ms) and the analysis frame adds latency, so a
# constant is subtracted to approximate the first attack.

UPSTRUM_STRINGS = (2, 3, 4, 5)  # array idx: D, G, B, high E -- up-strums rarely reach the bass


@dataclass
class Strum:
    t: float                     # seconds, attack time
    strength: float              # normalized onset strength
    recovered: bool = False      # found by the score-informed pass
    end: float = 0.0             # window end used for pitch evidence
    presence: dict[int, float] = field(default_factory=dict)   # midi -> activation 0..1
    cqt_db: dict[int, float] = field(default_factory=dict)     # midi -> dB rel. frame max
    chroma: list[float] = field(default_factory=list)
    energy: float = 0.0          # RMS of the first 120 ms, relative to the take's typical strum
    rise: float = 1.0            # RMS(after 60 ms) / RMS(before 60 ms): a real attack is > ~1.5
    gap: float = 0.0             # seconds to the next strum
    sustain: float = 1.0         # RMS of the last 80 ms before the next strum / attack RMS
    attack: dict[int, float] = field(default_factory=dict)     # midi -> basic-pitch onset activation
    cqt_rise: dict[int, float] = field(default_factory=dict)   # midi -> dB rise vs just before the strum
    direction: tuple[str | None, float] = (None, 0.0)


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------

def load_wav(path: str) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        sr = SR
    return y, sr


def preprocess(y: np.ndarray, sr: int, click_hz: float = CLICK_HZ) -> np.ndarray:
    """High-pass at 70 Hz (low E = 82.4 Hz) and notch out the metronome click band."""
    sos_hp = signal.butter(4, 70, btype="highpass", fs=sr, output="sos")
    y = signal.sosfiltfilt(sos_hp, y)
    lo, hi = click_hz * 0.88, min(click_hz * 1.12, sr / 2 - 100)
    sos_bs = signal.butter(4, [lo, hi], btype="bandstop", fs=sr, output="sos")
    y = signal.sosfiltfilt(sos_bs, y)
    peak = np.max(np.abs(y)) or 1.0
    return (y / peak * 0.9).astype(np.float32)


# ---------------------------------------------------------------------------
# click bleed: measure the true click offset from the recording (if audible)
# ---------------------------------------------------------------------------

def measure_click_offset(y: np.ndarray, sr: int, clicks_s: list[float],
                         click_hz: float = CLICK_HZ, count_in: int | None = None) -> float | None:
    """If the click is audible in the mic, find each logged click in the audio and return the
    median (audio - logged) offset in ms. Returns None when the click isn't reliably found
    (e.g. headphones). This calibrates away browser output/input latency.

    With `count_in`, only the count-in clicks are used: nothing but the click sounds there, so
    they can't be confused with strums, and most of them must be found."""
    if not clicks_s:
        return None
    if count_in:
        clicks_s = clicks_s[:count_in]
    need = max(2, int(np.ceil(0.75 * len(clicks_s)))) if count_in else 3
    hop = 64

    def band_env(lo: float, hi: float) -> np.ndarray:
        sos = signal.butter(4, [lo, min(hi, sr / 2 - 100)], btype="bandpass", fs=sr, output="sos")
        b = np.abs(signal.sosfiltfilt(sos, y))
        return b[: len(b) // hop * hop].reshape(-1, hop).max(axis=1)

    env = band_env(click_hz * 0.92, click_hz * 1.08)
    if env.size == 0:
        return None
    # The click is a pure tone; a strum attack is broadband. Requiring the click band to stand
    # well above both neighbouring bands stops guitar attacks that land on the beat from being
    # mistaken for click bleed (which shifted the whole beat grid on headphone takes).
    flank = np.maximum(band_env(click_hz * 0.70, click_hz * 0.82), band_env(click_hz * 1.20, click_hz * 1.35))
    floor = np.median(env) + 1e-9
    offsets = []
    for c in clicks_s:
        a, b = int((c - 0.05) * sr / hop), int((c + 0.20) * sr / hop)
        a, b = max(a, 0), min(b, len(env))
        if b - a < 3:
            continue
        seg = env[a:b]
        k = int(np.argmax(seg))
        if seg[k] < 8 * floor or seg[k] < 3 * flank[a + k]:
            continue
        k0 = int(np.argmax(seg >= seg[k] * 0.3))  # attack = first frame at 30% of the peak (robust to reverb build-up)
        offsets.append(((a + k0) * hop / sr - c) * 1000.0)
    if len(offsets) < need:
        return None
    offsets = np.array(offsets)
    med = float(np.median(offsets))
    if float(np.median(np.abs(offsets - med))) > 10.0:
        return None
    return med


# ---------------------------------------------------------------------------
# basic-pitch + spectral helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _bp_model():
    from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
    from basic_pitch.inference import Model
    return Model(build_icassp_2022_model_path(FilenameSuffix.onnx))


def run_basic_pitch(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Returns (frame_times, note_activation[frames, 88], onset_activation[frames, 88], note events)."""
    from basic_pitch.inference import predict
    from basic_pitch.note_creation import model_frames_to_time

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(path, y, sr)
        out, _midi, events = predict(path, _bp_model(), onset_threshold=0.5,
                                     frame_threshold=0.3, minimum_note_length=58,
                                     minimum_frequency=70, maximum_frequency=1400)
    finally:
        os.unlink(path)
    note, onset = out["note"], out["onset"]
    times = model_frames_to_time(note.shape[0])
    notes = [{"onset": round(float(s), 3), "offset": round(float(e), 3), "pitch": int(p),
              "confidence": round(float(a), 3)} for s, e, p, a, _ in events]
    return times, note, onset, notes


def _window_max(times: np.ndarray, mat: np.ndarray, a: float, b: float) -> np.ndarray:
    i0, i1 = np.searchsorted(times, a), np.searchsorted(times, b)
    if i1 <= i0:
        i1 = min(i0 + 1, len(times))
    return mat[i0:i1].max(axis=0) if i1 > i0 else np.zeros(mat.shape[1])


def _harmonic_partner(m: int, others: list[int]) -> bool:
    """True if pitch m could be a harmonic (2x, 3x, 4x) of another sounding pitch, which
    makes a 'missing' call on m less certain."""
    return any(m - o in (12, 19, 24) for o in others)


# ---------------------------------------------------------------------------
# the analysis object
# ---------------------------------------------------------------------------

class AudioAnalysis:
    def __init__(self, y: np.ndarray, sr: int, click_offset_ms: float | None, use_basic_pitch: bool = True):
        self.y, self.sr = y, sr
        self.duration = len(y) / sr
        self.click_offset_ms = click_offset_ms
        # onset envelope: spectral flux below 2 kHz (the 3 kHz click stays out of it)
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP, lag=1, max_size=3,
                                           fmax=2000, n_mels=96)
        peaks = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0, wait=1)
        ref = np.percentile(env[peaks], 95) if len(peaks) else (env.max() or 1.0)
        self.env = env / (ref or 1.0)
        self.env_t = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=HOP)

        self.note_events: list[dict] = []
        self.used_basic_pitch = False
        self._bp = None
        if use_basic_pitch:
            try:
                ft, note_act, onset_act, self.note_events = run_basic_pitch(y, sr)
                self._bp = (ft, note_act, onset_act)
                self.used_basic_pitch = True
            except Exception:  # pragma: no cover - model/runtime issues
                log.exception("basic-pitch failed; falling back to CQT-only evidence")

        C = np.abs(librosa.cqt(y, sr=sr, hop_length=512, fmin=librosa.midi_to_hz(40),
                               n_bins=48, bins_per_octave=12))
        db = librosa.amplitude_to_db(C, ref=1.0)
        self.cqt_abs = db.T                       # absolute dB (for per-pitch attack rise)
        self.cqt_db = (db - db.max(axis=0, keepdims=True)).T
        self.cqt_t = librosa.frames_to_time(np.arange(C.shape[1]), sr=sr, hop_length=512)
        self.chroma = librosa.feature.chroma_cqt(C=C, sr=sr, hop_length=512, n_chroma=12,
                                                 bins_per_octave=12, fmin=librosa.midi_to_hz(40))
        self.detect()

    def detect(self) -> None:
        """Strum detection + per-strum evidence. Cheap compared with the feature extraction in
        __init__, so tuning re-runs only this when onset parameters change."""
        frames = librosa.onset.onset_detect(onset_envelope=self.env, sr=self.sr, hop_length=HOP,
                                            delta=P.global_delta, wait=4, normalize=False)
        self.strums: list[Strum] = []
        for fr in frames:
            self._add(fr, recovered=False)
        self._energy_ref = None
        self._finalize()
        # spectral flux also peaks while a chord decays (partials beating); a real strum raises
        # the level, so drop onsets whose energy doesn't rise and re-derive the windows
        kept = [s for s in self.strums if s.rise >= P.min_rise]
        if len(kept) != len(self.strums):
            self.strums = kept
            self._energy_ref = None
            self._finalize()

    # -- building strums --------------------------------------------------------
    def _add(self, frame: int, recovered: bool) -> Strum | None:
        t = float(self.env_t[frame] - P.strum_time_bias_s)
        for s in self.strums:
            if abs(s.t - t) < P.merge_window_s:
                s.strength = max(s.strength, float(self.env[frame]))
                return None
        st = Strum(t=t, strength=float(self.env[frame]), recovered=recovered)
        self.strums.append(st)
        self.strums.sort(key=lambda s: s.t)
        return st

    def _finalize(self) -> None:
        y, sr = self.y, self.sr
        for st in self.strums:
            a, b = int(max(st.t, 0) * sr), int(min(st.t + 0.12, self.duration) * sr)
            st.energy = float(np.sqrt(np.mean(y[a:b] ** 2))) if b > a else 0.0
            p0, p1 = int(max(st.t - 0.06, 0) * sr), a
            a1 = int(min(st.t + 0.06, self.duration) * sr)
            pre = float(np.sqrt(np.mean(y[p0:p1] ** 2))) if p1 > p0 else 1e-6
            post = float(np.sqrt(np.mean(y[a:a1] ** 2))) if a1 > a else 0.0
            st.rise = post / (pre + 1e-6)
        raw = [st.energy for st in self.strums]
        if self._energy_ref is None:
            e = np.array([s.energy for s in self.strums if not s.recovered])
            self._energy_ref = float(np.percentile(e, 75)) if len(e) else 1.0
        for st in self.strums:
            st.energy = st.energy / (self._energy_ref or 1.0)
        real_t = [s.t for s in self.strums if s.energy >= P.ghost_energy]
        for st, e_raw in zip(self.strums, raw):
            # sustain: does the chord still ring just before the next real strum (or 0.8 s
            # later)? Ghost-level onsets (noise in the silence after a choked chord) don't count.
            later = [t for t in real_t if t > st.t + 0.01]
            nxt = later[0] if later else min(st.t + 0.8, self.duration)
            st.gap = float(nxt - st.t)
            e0, e1 = int(max(nxt - 0.08, st.t + 0.12) * sr), int(nxt * sr)
            tail = float(np.sqrt(np.mean(y[e0:e1] ** 2))) if e1 > e0 else e_raw
            st.sustain = tail / (e_raw + 1e-9)
        for i, st in enumerate(self.strums):
            nxt = self.strums[i + 1].t if i + 1 < len(self.strums) else self.duration
            st.end = float(min(nxt, st.t + 0.40))
            self._evidence(st)

    def _evidence(self, st: Strum) -> None:
        a = st.t + 0.025
        b = max(min(st.end - 0.01, st.t + 0.25), a + 0.03)
        cq = _window_max(self.cqt_t, self.cqt_db, a, b)
        st.cqt_db = {40 + k: float(cq[k]) for k in range(48)}
        n = self.chroma.shape[1]
        i0 = min(int(np.searchsorted(self.cqt_t, a)), n - 1)
        i1 = min(max(int(np.searchsorted(self.cqt_t, b)), i0 + 1), n)
        st.chroma = self.chroma[:, i0:i1].mean(axis=1).tolist()
        # attack evidence: did this pitch get a *new* attack at this strum? Separates a string
        # that was struck from one that is only implied by another string's harmonics or by
        # the previous strum still ringing.
        pre = self.cqt_abs[max(0, int(np.searchsorted(self.cqt_t, st.t - 0.07))):
                           max(1, int(np.searchsorted(self.cqt_t, st.t - 0.01)))]
        post = _window_max(self.cqt_t, self.cqt_abs, st.t + 0.01, st.t + 0.12)
        pre_db = pre.mean(axis=0) if len(pre) else post - 30
        st.cqt_rise = {40 + k: float(post[k] - pre_db[k]) for k in range(48)}
        if self._bp is not None:
            ft, note_act, onset_act = self._bp
            pres = _window_max(ft, note_act, a, b)
            st.presence = {m: float(pres[m - BP_MIDI_OFFSET]) for m in range(36, 100)}
            ons = _window_max(ft, onset_act, st.t - 0.03, st.t + 0.09)
            st.attack = {m: float(ons[m - BP_MIDI_OFFSET]) for m in range(36, 100)}
        else:  # CQT-only fallback: map dB to a pseudo-activation
            st.presence = {m: float(np.clip((st.cqt_db.get(m, -80) + 30) / 25, 0, 1)) for m in range(36, 100)}

    def recover(self, t_expected_s: float, window_s: float) -> Strum | None:
        """Score-informed pass: look for a strum near an expected time that the strict global
        pass missed. Accepts the strongest flux peak in the window if it stands clear of the
        local background."""
        i0 = np.searchsorted(self.env_t, t_expected_s - window_s + P.strum_time_bias_s)
        i1 = np.searchsorted(self.env_t, t_expected_s + window_s + P.strum_time_bias_s)
        if i1 - i0 < 3:
            return None
        seg = self.env[i0:i1]
        k = int(np.argmax(seg))
        bg = float(np.median(self.env[max(0, i0 - 40):i1 + 40]))
        if seg[k] < P.recover_min or seg[k] < 4 * bg:
            return None
        st = self._add(i0 + k, recovered=True)
        if st is not None:
            self._finalize()
        return st

    # -- per-strum verdicts ---------------------------------------------------
    def direction(self, st: Strum, chord: str) -> tuple[str | None, float]:
        return attack_order_direction(self.y, self.sr, st.t, chord)


def analyze_audio(wav_path: str, clicks_s: list[float] | None = None,
                  use_basic_pitch: bool = True, count_in: int | None = None) -> AudioAnalysis:
    y_raw, sr = load_wav(wav_path)
    click_offset = measure_click_offset(y_raw, sr, clicks_s or [], count_in=count_in)
    y = preprocess(y_raw, sr)
    return AudioAnalysis(y, sr, click_offset, use_basic_pitch)


# ---------------------------------------------------------------------------
# per-strum verification against the expected chord
# ---------------------------------------------------------------------------

def verify_chord(st: Strum, chord: str, direction: str = "D") -> dict:
    """Compare a strum's pitch evidence with the expected voicing.

    Returns per-string status dict keyed by array idx (0 = low E):
      ok | not_required | missing | wrong_fret | muted_sounding | muted_ok
    plus a pitch score and a list of unexpected pitches.
    """
    shape = CHORDS[chord]
    expected = [m for m in shape.midi if m >= 0]
    strings: dict[int, dict] = {}
    claimed: set[int] = set(expected)

    for i, m in enumerate(shape.midi):
        if m < 0:
            continue
        p = st.presence.get(m, 0.0)
        if direction == "U" and i not in UPSTRUM_STRINGS:
            # bass strings on an up-strum: not required, still credited if they ring
            strings[i] = {"status": "ok" if p >= P.present_thr else "not_required", "midi": m,
                          "presence": round(p, 3), "confidence": 0.5}
            continue
        if p >= P.present_thr:
            strings[i] = {"status": "ok", "midi": m, "presence": round(p, 3), "confidence": round(min(1.0, p + 0.2), 2)}
            continue
        # a clearly stronger neighbouring semitone (not explained by the chord) => wrong fret
        best = None
        for d in (-1, 1, -2, 2):
            q = m + d
            if q in expected or q < OPEN_MIDI[i]:
                continue
            pq = st.presence.get(q, 0.0)
            if pq >= P.present_thr and pq > p + P.wrong_fret_margin and (best is None or pq > best[1]):
                best = (d, pq)
        if best:
            claimed.add(m + best[0])
            strings[i] = {"status": "wrong_fret", "midi": m, "observed_midi": m + best[0],
                          "delta": best[0], "presence": round(p, 3),
                          "confidence": round(min(0.95, 0.5 + best[1] / 2), 2)}
            continue
        # missing: require basic-pitch miss AND weak CQT energy unless basic-pitch is very sure
        cqt_weak = st.cqt_db.get(m, -80.0) < P.missing_cqt_db
        if p < P.absent_thr or cqt_weak:
            conf = 0.85 if (p < P.absent_thr and cqt_weak) else 0.6
            if _harmonic_partner(m, [o for o in expected if o != m]):
                conf -= 0.2
            strings[i] = {"status": "missing", "midi": m, "presence": round(p, 3), "confidence": round(conf, 2)}
        else:
            strings[i] = {"status": "ok", "midi": m, "presence": round(p, 3), "confidence": 0.5}

    for i in shape.muted_idx:
        m = OPEN_MIDI[i]
        p = st.presence.get(m, 0.0)
        # an open muted string that sounds and isn't explained by the chord
        # (only judged on down-strums; on up-strums it's usually leftover ring)
        if direction == "D" and p >= P.muted_sounding_presence and m not in expected and st.cqt_db.get(m, -80) > P.muted_sounding_cqt_db:
            strings[i] = {"status": "muted_sounding", "midi": m, "presence": round(p, 3),
                          "confidence": round(min(0.95, p), 2)}
            claimed.add(m)
        else:
            strings[i] = {"status": "muted_ok", "midi": -1, "presence": round(p, 3), "confidence": 0.7}

    unexpected = [m for m, p in st.presence.items()
                  if p >= P.unexpected_presence and m not in claimed and 40 <= m <= 88 and not _harmonic_partner(m, expected)]

    required = [s for s in strings.values() if s["status"] not in ("muted_ok", "muted_sounding", "not_required")]
    n_ok = sum(1 for s in required if s["status"] == "ok")
    n_extra = sum(1 for s in strings.values() if s["status"] == "muted_sounding") + len(unexpected)
    pitch_score = max(0.0, n_ok / len(required) - 0.15 * n_extra) if required else 1.0

    # chroma backup check: robust "right chord / wrong chord" even if basic-pitch is noisy
    ch = np.array(st.chroma) if st.chroma else np.zeros(12)
    sims = {}
    for name in CHORDS:
        t = np.array(chord_chroma(name))
        sims[name] = round(float(ch @ t / (np.linalg.norm(ch) * np.linalg.norm(t) + 1e-9)), 3)

    return {"strings": strings, "pitch_score": round(pitch_score, 3), "unexpected": unexpected,
            "chroma_sim": sims[chord], "chroma_all": sims}


# ---------------------------------------------------------------------------
# strum direction from audio: per-string attack order
# ---------------------------------------------------------------------------

def _direction_probe_pitches(chord: str) -> list[tuple[int, int]]:
    """(array idx, midi) of strings whose fundamental isn't a harmonic of a lower chord tone,
    so its energy rise is caused by that string alone."""
    shape = CHORDS[chord]
    sounding = [(i, m) for i, m in enumerate(shape.midi) if m >= 0]
    out = []
    for i, m in sounding:
        lower = [o for _, o in sounding if o < m]
        if not _harmonic_partner(m, lower):
            out.append((i, m))
    return out


def attack_order_direction(y: np.ndarray, sr: int, t: float, chord: str) -> tuple[str | None, float]:
    """Down-strums hit low strings first. Measure when each probe string's fundamental starts
    rising (fine hop, same window for all bins so relative timing survives) and correlate
    with string order. Returns (D|U|None, confidence)."""
    probes = _direction_probe_pitches(chord)
    if len(probes) < 2:
        return None, 0.0
    a, b = int(max(0, t - 0.04) * sr), int(min(len(y) / sr, t + 0.10) * sr)
    seg = y[a:b]
    if len(seg) < 2048:
        return None, 0.0
    S = np.abs(librosa.stft(seg, n_fft=2048, hop_length=64, center=True))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    times = np.arange(S.shape[1]) * 64 / sr
    pts = []
    for i, m in probes:
        f0 = librosa.midi_to_hz(m)
        k = int(np.argmin(np.abs(freqs - f0)))
        e = S[max(0, k - 1):k + 2].sum(axis=0)
        e = np.convolve(e, np.ones(3) / 3, "same")
        pre = e[: max(1, int(0.03 * sr / 64))].mean()
        if e.max() <= 0 or e.max() < 2.5 * pre:  # this string wasn't (re-)struck
            continue
        # rise time: first frame reaching 50% of the window max
        pts.append((i, times[int(np.argmax(e >= 0.5 * e.max()))]))
    if len(pts) < 3:
        return None, 0.0
    idx = np.array([p[0] for p in pts], float)
    ts = np.array([p[1] for p in pts])
    slope = np.polyfit(idx, ts, 1)[0] * 1000  # ms per string
    if abs(slope) < 1.0:
        return None, 0.0
    r = np.corrcoef(idx, ts)[0, 1]
    conf = float(min(1.0, abs(slope) / 6.0)) * (float(abs(r)) if np.isfinite(r) else 0.0)
    return ("D" if slope > 0 else "U"), round(conf, 2)
