"""
segmentation.py — Onset/offset detection for signal events
===========================================================
AI 1 (DSP / Signal Processing Lead)

WHY
---
"Signal segmentation" is an AI 1 charter deliverable (§2) and was the one item
with no implementation at all. It is also a hard prerequisite for the
AI -> Backend contract (charter §14), which asks for:

    "duration": 3.2

A single analysis frame is 2048 samples at 48 kHz = ~42.7 ms. Duration cannot
come from one frame. It needs a start and an end, which is what this module
finds.

WHAT IT DOES
------------
Tracks band energy over time and reports discrete events:

    energy ---------________-----------________
                   ^      ^            ^      ^
                 onset  offset       onset  offset
                 |<-- event 1 -->|   |<-- event 2 -->|

Hysteresis (separate enter/exit thresholds) stops a signal hovering near the
threshold from producing dozens of spurious micro-events.

USAGE
-----
Offline, on a complete recording:

    from dsp.segmentation import segment_signal
    events = segment_signal(audio, sample_rate=48000)
    for e in events:
        print(e["start_sec"], e["end_sec"], e["duration_sec"])

Streaming, one chunk at a time:

    from dsp.segmentation import StreamingSegmenter
    seg = StreamingSegmenter(sample_rate=48000)
    for chunk in stream:
        ev = seg.push(chunk)          # returns an event dict when one closes
        if ev:
            print("event ended:", ev["duration_sec"], "sec")
"""

from __future__ import annotations

import numpy as np


DEFAULT_BAND = (18000.0, 21000.0)     # matches dsp.ULTRASONIC_LOW / _HIGH


def band_energy(chunk: np.ndarray, sample_rate: int, band=DEFAULT_BAND) -> float:
    """Energy inside `band` for one chunk, via Hann-windowed rFFT."""
    x = np.asarray(chunk, dtype=np.float64)
    if x.size == 0:
        return 0.0
    x = x - x.mean()
    w = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * w)) / (np.sum(w) + 1e-20)
    freqs = np.fft.rfftfreq(len(x), 1.0 / sample_rate)
    m = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(m):
        return 0.0
    return float(np.sum(spec[m] ** 2))


def _frame_energies(audio, sample_rate, frame_size, hop, band):
    x = np.asarray(audio, dtype=np.float64)
    energies, times = [], []
    for start in range(0, max(1, len(x) - frame_size + 1), hop):
        energies.append(band_energy(x[start:start + frame_size], sample_rate, band))
        times.append(start / sample_rate)
    return np.array(energies), np.array(times)


def segment_signal(
    audio,
    sample_rate: int = 48000,
    frame_size: int = 2048,
    hop: int = 1024,
    band=DEFAULT_BAND,
    threshold_db: float = 10.0,
    release_db: float = 6.0,
    min_duration_sec: float = 0.05,
    merge_gap_sec: float = 0.10,
) -> list:
    """
    Find signal events in a complete recording.

    Thresholds are RELATIVE to the recording's own noise floor (median frame
    energy), so this adapts to quiet and loud rooms without recalibration.

    Parameters
    ----------
    threshold_db : float
        Rise above the noise floor required to OPEN an event.
    release_db : float
        Level below which an open event CLOSES. Lower than threshold_db —
        this hysteresis prevents flicker at the boundary.
    min_duration_sec : float
        Events shorter than this are discarded as transients.
    merge_gap_sec : float
        Events separated by less than this are merged into one.

    Returns
    -------
    list of dict, each with start_sec, end_sec, duration_sec, peak_energy_db,
    mean_energy_db, start_sample, end_sample.
    """
    energies, times = _frame_energies(audio, sample_rate, frame_size, hop, band)
    if energies.size == 0:
        return []

    eps = 1e-20
    e_db = 10 * np.log10(energies + eps)

    # Noise floor estimation.
    #
    # A fixed percentile is fragile: the median only works when signal covers
    # less than half the file (a 3 s tone in a 4.5 s recording puts the median
    # INSIDE the tone, so the threshold lands above the signal and nothing is
    # ever detected — segment_signal() silently returned [] for that case).
    # A low fixed percentile then fails in the opposite direction on recordings
    # that are almost entirely signal.
    #
    # Instead: if the dynamic range is wide, the recording clearly contains
    # both quiet and loud regions, so split them at the midpoint between the
    # low and high percentiles and take the floor from the quiet side only.
    p10 = float(np.percentile(e_db, 10))
    p90 = float(np.percentile(e_db, 90))

    if (p90 - p10) > threshold_db:
        midpoint = (p10 + p90) / 2.0
        quiet = e_db[e_db < midpoint]
        floor_db = float(np.median(quiet)) if quiet.size else p10
    else:
        # Narrow range: no obvious signal, or uniformly loud. Use a low
        # percentile and let the threshold decide.
        floor_db = float(np.percentile(e_db, 20))

    open_at = floor_db + threshold_db
    close_at = floor_db + release_db

    events = []
    active = False
    start_i = 0

    for i, v in enumerate(e_db):
        if not active and v >= open_at:
            active = True
            start_i = i
        elif active and v < close_at:
            active = False
            events.append((start_i, i))

    if active:
        events.append((start_i, len(e_db) - 1))

    # Merge events separated by a short gap
    frame_dur = hop / sample_rate
    merged = []
    for ev in events:
        if merged and (ev[0] - merged[-1][1]) * frame_dur <= merge_gap_sec:
            merged[-1] = (merged[-1][0], ev[1])
        else:
            merged.append(ev)

    out = []
    for a, b in merged:
        start_sec = float(times[a])
        end_sec = float(times[b] + frame_size / sample_rate)
        dur = end_sec - start_sec
        if dur < min_duration_sec:
            continue
        seg = e_db[a:b + 1]
        out.append({
            "start_sec": round(start_sec, 4),
            "end_sec": round(end_sec, 4),
            "duration_sec": round(dur, 4),
            "start_sample": int(a * hop),
            "end_sample": int(b * hop + frame_size),
            "peak_energy_db": round(float(np.max(seg)), 2),
            "mean_energy_db": round(float(np.mean(seg)), 2),
            "noise_floor_db": round(floor_db, 2),
            "snr_db": round(float(np.max(seg) - floor_db), 2),
        })
    return out


class StreamingSegmenter:
    """
    Real-time onset/offset tracking, one chunk at a time.

    Unlike segment_signal(), the noise floor cannot be known in advance, so it
    is estimated adaptively from frames that are not part of an active event.

        seg = StreamingSegmenter(sample_rate=48000)
        for chunk in stream:
            event = seg.push(chunk)
            if event:
                ...   # an event just closed
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        band=DEFAULT_BAND,
        threshold_db: float = 10.0,
        release_db: float = 6.0,
        min_duration_sec: float = 0.05,
        floor_adapt: float = 0.05,
    ):
        self.sample_rate = sample_rate
        self.band = band
        self.threshold_db = threshold_db
        self.release_db = release_db
        self.min_duration_sec = min_duration_sec
        self.floor_adapt = floor_adapt

        self._floor_db = None
        self._active = False
        self._start_time = None
        self._elapsed = 0.0
        self._peak_db = -np.inf
        self.events = []

    @property
    def is_active(self) -> bool:
        """True while an event is currently open."""
        return self._active

    @property
    def noise_floor_db(self):
        return self._floor_db

    def push(self, chunk: np.ndarray):
        """
        Feed one chunk. Returns an event dict if an event just CLOSED,
        otherwise None.
        """
        chunk = np.asarray(chunk, dtype=np.float64)
        dur = len(chunk) / self.sample_rate
        e_db = 10 * np.log10(band_energy(chunk, self.sample_rate, self.band) + 1e-20)

        if self._floor_db is None:
            self._floor_db = e_db

        open_at = self._floor_db + self.threshold_db
        close_at = self._floor_db + self.release_db

        closed = None

        if not self._active:
            if e_db >= open_at:
                self._active = True
                self._start_time = self._elapsed
                self._peak_db = e_db
            else:
                # Only adapt the floor while idle, so a long tone cannot
                # slowly drag the threshold up and hide itself.
                self._floor_db = ((1 - self.floor_adapt) * self._floor_db
                                  + self.floor_adapt * e_db)
        else:
            self._peak_db = max(self._peak_db, e_db)
            if e_db < close_at:
                self._active = False
                length = self._elapsed - self._start_time + dur
                if length >= self.min_duration_sec:
                    closed = {
                        "start_sec": round(self._start_time, 4),
                        "end_sec": round(self._elapsed + dur, 4),
                        "duration_sec": round(length, 4),
                        "peak_energy_db": round(float(self._peak_db), 2),
                        "noise_floor_db": round(float(self._floor_db), 2),
                        "snr_db": round(float(self._peak_db - self._floor_db), 2),
                    }
                    self.events.append(closed)

        self._elapsed += dur
        return closed

    def flush(self):
        """Close any open event at end of stream. Returns it, or None."""
        if not self._active:
            return None
        self._active = False
        length = self._elapsed - self._start_time
        if length < self.min_duration_sec:
            return None
        ev = {
            "start_sec": round(self._start_time, 4),
            "end_sec": round(self._elapsed, 4),
            "duration_sec": round(length, 4),
            "peak_energy_db": round(float(self._peak_db), 2),
            "noise_floor_db": round(float(self._floor_db or 0.0), 2),
            "snr_db": round(float(self._peak_db - (self._floor_db or 0.0)), 2),
            "truncated": True,
        }
        self.events.append(ev)
        return ev

    def total_active_sec(self) -> float:
        """Total duration of all completed events."""
        return round(sum(e["duration_sec"] for e in self.events), 4)

    def reset(self):
        self._floor_db = None
        self._active = False
        self._start_time = None
        self._elapsed = 0.0
        self._peak_db = -np.inf
        self.events = []
