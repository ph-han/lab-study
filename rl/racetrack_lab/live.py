"""Live streaming between a training script and the viewer, through files.

Your code publishes; the viewer tails. Nothing here knows what an update *is*
-- a sweep, an episode, a single environment step -- you call
:meth:`LiveRun.publish` whenever something worth seeing changed, with whatever
you have at that moment. Everything is optional.

    from racetrack_lab import get_track
    from racetrack_lab.live import LiveRun

    track = get_track("Easy")
    run = LiveRun("runs/my_first_try", track, scalar_label="V(s)", vector_label="pi(s)")

    for k in range(...):
        ...                                   # your update
        run.publish(scalar_field=V, vector_field=pi, metrics={"delta": d}, label=f"sweep {k}")

    for t in range(...):
        ...                                   # your rollout
        run.publish(state=(x, y, vx, vy))     # the viewer draws the growing trail
    run.new_episode()                         # start a fresh trail
    run.close()

Files live in the run directory: ``meta.json`` (track + labels), ``latest.pkl``
(the newest update, replaced atomically), ``snap_000012.pkl`` (kept updates,
for the iteration slider), ``metrics.jsonl`` (one line per update, for charts)
and ``done`` once :meth:`LiveRun.close` is called. Only those files are ever
touched, so the directory is safe to share with your own outputs.
"""

from __future__ import annotations

import json
import os
import pickle
import time
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .schema import (
    Iteration,
    LabSession,
    ScalarField,
    TrackMap,
    Trajectory,
    VectorField,
    VehicleState,
)

# The folder that holds ``racetrack_lab/`` (and ``app.py``). Relative run paths
# are resolved against it on both sides, so a script started from any working
# directory and the viewer agree on where ``runs/`` is.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_run_path(path: str | os.PathLike) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p)


META = "meta.json"
LATEST = "latest.pkl"
METRICS = "metrics.jsonl"
DONE = "done"
SNAP_PREFIX = "snap_"
SNAP_SUFFIX = ".pkl"
_OWN_FILES = {META, LATEST, METRICS, DONE}


# ---------------------------------------------------------------------------
# Normalisation helpers (plain Python only, so the viewer never needs your classes)
# ---------------------------------------------------------------------------
def _plain_state(state: Any) -> dict[str, Any]:
    s = VehicleState.from_any(state)
    d: dict[str, Any] = {"x": float(s.x), "y": float(s.y), "vx": float(s.vx), "vy": float(s.vy)}
    if s.step is not None:
        d["step"] = int(s.step)
    if s.action is not None and len(s.action) >= 2:
        d["action"] = [float(s.action[0]), float(s.action[1])]
    if s.reward is not None:
        d["reward"] = float(s.reward)
    for k, v in s.extras.items():
        if isinstance(v, (bool, int, float, str)):
            d[k] = v
        elif isinstance(v, (tuple, list)) and len(v) == 2:
            d[k] = [v[0], v[1]]
    return d


def _plain_trajectory(states: Iterable[Any] | None) -> list[dict[str, Any]] | None:
    if states is None:
        return None
    return [_plain_state(s) for s in states]


def _plain_field(data: Any) -> Any:
    """Mappings and arrays pass through; anything exotic is rejected early."""
    if data is None:
        return None
    if isinstance(data, Mapping):
        return {tuple(float(c) for c in k): _plain_value(v) for k, v in data.items()}
    arr = np.asarray(data)
    if arr.ndim == 2:
        return arr.astype(float)
    raise TypeError("fields must be a mapping keyed by (x, y[, vx, vy]) or a 2D array [y, x]")


def _plain_value(v: Any) -> Any:
    if isinstance(v, (tuple, list, np.ndarray)):
        return [float(c) for c in list(v)[:2]]
    return float(v)


def _plain_metrics(metrics: Mapping[str, Any] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in (metrics or {}).items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _atomic_write(path: Path, payload: bytes, attempts: int = 25) -> bool:
    """Write ``payload`` to ``path`` via a temp file and an atomic replace.

    On Windows the replace fails while another process (the viewer) has the
    target open, so retry briefly; report ``False`` instead of raising when it
    still cannot land -- the caller keeps the update pending for next time.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(payload)
    delay = 0.01
    for _ in range(attempts):
        try:
            os.replace(tmp, path)
            return True
        except PermissionError:
            time.sleep(delay)
            delay = min(delay * 1.5, 0.08)
    return False


def _unlink_quietly(path: Path) -> None:
    for _ in range(10):
        try:
            path.unlink()
            return
        except FileNotFoundError:
            return
        except PermissionError:
            time.sleep(0.02)


# ---------------------------------------------------------------------------
# Publisher
# ---------------------------------------------------------------------------
class LiveRun:
    """Publish updates from your training loop to a run directory.

    ``keep_every`` decides which updates are stored as separate snapshots for
    the iteration slider (every one by default; raise it for long runs -- each
    kept snapshot is a full copy of the fields). ``min_interval`` throttles disk
    writes when you publish very often; the latest update is never lost because
    :meth:`flush` / :meth:`close` write whatever is pending.
    """

    def __init__(
        self,
        path: str | os.PathLike = "runs/live",
        track: TrackMap | str | None = None,
        *,
        scalar_label: str = "Value",
        vector_label: str = "Vector",
        trajectory_label: str = "Trajectory",
        note: str = "",
        reset: bool = True,
        keep_every: int = 1,
        min_interval: float = 0.05,
    ) -> None:
        self.path = resolve_run_path(path)
        self.keep_every = max(1, int(keep_every))
        self.min_interval = max(0.0, float(min_interval))
        self.seq = 0
        self._episode: list[dict[str, Any]] = []
        self._pending: tuple[dict[str, Any], bool] | None = None
        self._last_write = 0.0
        self._trace: Any = None

        self.path.mkdir(parents=True, exist_ok=True)
        if reset:
            for f in list(self.path.iterdir()):
                if f.name in _OWN_FILES or (
                    f.name.startswith(SNAP_PREFIX) and f.name.endswith((SNAP_SUFFIX, ".tmp"))
                ):
                    _unlink_quietly(f)

        meta: dict[str, Any] = {
            "created": time.time(),
            "pid": os.getpid(),
            "scalar_label": scalar_label,
            "vector_label": vector_label,
            "trajectory_label": trajectory_label,
            "note": note,
        }
        if track is not None:
            if hasattr(track, "grid"):
                meta["track"] = {
                    "name": getattr(track, "name", "Track"),
                    "difficulty": getattr(track, "difficulty", ""),
                    "grid": np.asarray(track.grid).astype(int).tolist(),
                }
            else:
                meta["track"] = {"name": str(track)}
        (self.path / META).write_text(json.dumps(meta), encoding="utf-8")

    # -- what to publish -----------------------------------------------------
    def new_episode(self) -> None:
        """Forget the states appended so far; the next ``state=`` starts a new trail."""
        self._episode = []

    def trace(self, *, coords: str = "yx", gamma: float | None = None, actions: Any = None) -> Any:
        """Open the calculation channel for this run (see :mod:`racetrack_lab.trace`).

        Call it once, before the loop, so the viewer knows how you write states::

            run.trace(coords="yx", gamma=agent.gamma)
        """
        from .trace import TraceWriter

        self._trace = TraceWriter(self.path, coords=coords, gamma=gamma, actions=actions)
        return self._trace

    def backup(self, **kw: Any) -> Any:
        """Record one Bellman backup. Opens the channel with defaults if needed."""
        if self._trace is None:
            self.trace()
        return self._trace.backup(**kw)

    def publish(
        self,
        *,
        scalar_field: Any = None,
        vector_field: Any = None,
        trajectory: Iterable[Any] | None = None,
        state: Any = None,
        metrics: Mapping[str, Any] | None = None,
        label: str | None = None,
        keep: bool | None = None,
    ) -> bool:
        """Record one update. Returns ``True`` if it reached disk right away.

        ``state`` appends one state to the current trail; ``trajectory``
        replaces the trail wholesale. Pass only what changed -- the viewer keeps
        showing the last field it received.
        """
        if state is not None:
            self._episode.append(_plain_state(state))
        traj = (
            _plain_trajectory(trajectory)
            if trajectory is not None
            else (list(self._episode) if self._episode else None)
        )

        self.seq += 1
        snap = {
            "seq": self.seq,
            "time": time.time(),
            "label": label if label is not None else f"update {self.seq}",
            "scalar": _plain_field(scalar_field),
            "vector": _plain_field(vector_field),
            "trajectory": traj,
            "metrics": _plain_metrics(metrics),
        }
        keep_it = (self.seq % self.keep_every == 0) if keep is None else bool(keep)
        self._pending = (snap, keep_it)

        if keep_it or (time.time() - self._last_write) >= self.min_interval:
            self.flush()
            return True
        return False

    def flush(self) -> None:
        """Write the pending update, if any."""
        if self._pending is None:
            return
        snap, keep_it = self._pending
        blob = pickle.dumps(snap, protocol=pickle.HIGHEST_PROTOCOL)
        if not _atomic_write(self.path / LATEST, blob):
            # The viewer had the file open at that instant (Windows locks it).
            # Nothing is lost: the update stays pending and goes out next time.
            self._skipped = getattr(self, "_skipped", 0) + 1
            if self._skipped == 20 and not getattr(self, "_warned", False):
                self._warned = True
                warnings.warn(
                    "LiveRun: latest.pkl was busy 20 times in a row -- is another process "
                    "holding it open?",
                    RuntimeWarning,
                    stacklevel=3,
                )
            return
        self._skipped = 0
        if keep_it:
            _atomic_write(self.path / f"{SNAP_PREFIX}{snap['seq']:07d}{SNAP_SUFFIX}", blob)
        with open(self.path / METRICS, "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {"seq": snap["seq"], "time": snap["time"], "label": snap["label"],
                     "metrics": snap["metrics"]}
                )
                + "\n"
            )
        self._last_write = time.time()
        self._pending = None

    def close(self) -> None:
        """Flush and mark the run finished (the viewer stops polling)."""
        self.flush()
        (self.path / DONE).write_text(str(time.time()), encoding="utf-8")

    def __enter__(self) -> "LiveRun":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Reader (viewer side)
# ---------------------------------------------------------------------------
def list_runs(root: str | os.PathLike) -> list[Path]:
    """Run directories under ``root`` (relative to the project root), newest first."""
    base = resolve_run_path(root)
    if not base.is_dir():
        return []
    runs = [p for p in base.iterdir() if p.is_dir() and (p / META).exists()]
    runs.sort(key=lambda p: (p / LATEST).stat().st_mtime if (p / LATEST).exists() else 0, reverse=True)
    return runs


class LiveReader:
    """Tail a run directory. Instances cache what they have already parsed."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = resolve_run_path(path)
        self._meta: dict[str, Any] | None = None
        self._latest: tuple[tuple[int, int], dict[str, Any]] | None = None
        self._snaps: dict[str, dict[str, Any]] = {}
        self._parsed: dict[str, tuple[ScalarField | None, VectorField | None]] = {}
        self._metrics: tuple[int, list[dict[str, Any]]] = (0, [])

    def exists(self) -> bool:
        return (self.path / META).exists()

    def meta(self) -> dict[str, Any]:
        file = self.path / META
        try:
            stamp = file.stat().st_mtime_ns
        except OSError:
            return self._meta or {}
        if self._meta is None or stamp != getattr(self, "_meta_stamp", None):
            try:
                meta = json.loads(file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                meta = self._meta or {}
            if self._meta is not None and meta.get("created") != self._meta.get("created"):
                # the publisher started over: nothing cached is valid any more
                self._snaps.clear()
                self._parsed.clear()
                self._latest = None
                self._metrics = (0, [])
            self._meta = meta
            self._meta_stamp = stamp
        return self._meta

    def done(self) -> bool:
        return (self.path / DONE).exists()

    def age(self) -> float | None:
        """Seconds since the last update landed, or None if nothing yet."""
        try:
            return max(0.0, time.time() - (self.path / LATEST).stat().st_mtime)
        except OSError:
            return None

    def _read_pickle(self, file: Path) -> dict[str, Any] | None:
        for _ in range(3):  # the writer replaces atomically, but be forgiving
            try:
                blob = file.read_bytes()  # open only for the copy, not the parse
                obj = pickle.loads(blob)
                return obj if isinstance(obj, dict) else None
            except (OSError, EOFError, pickle.UnpicklingError):
                time.sleep(0.02)
        return None

    def latest(self) -> dict[str, Any] | None:
        self.meta()
        file = self.path / LATEST
        try:
            st = file.stat()
        except OSError:
            return None
        stamp = (st.st_mtime_ns, st.st_size)
        if self._latest and self._latest[0] == stamp:
            return self._latest[1]
        snap = self._read_pickle(file)
        if snap is not None:
            self._latest = (stamp, snap)
        return snap

    def kept_files(self) -> list[Path]:
        return sorted(
            p for p in self.path.glob(f"{SNAP_PREFIX}*{SNAP_SUFFIX}") if p.suffix == SNAP_SUFFIX
        )

    def snapshot(self, file: Path) -> dict[str, Any] | None:
        key = file.name
        if key not in self._snaps:
            snap = self._read_pickle(file)
            if snap is None:
                return None
            self._snaps[key] = snap
        return self._snaps[key]

    def parsed(
        self, key: str, snap: dict[str, Any], s_label: str, v_label: str
    ) -> tuple[ScalarField | None, VectorField | None]:
        """Fields of a snapshot, normalised once and cached under ``key``."""
        if key not in self._parsed:
            self._parsed[key] = (
                ScalarField.from_any(snap.get("scalar"), label=s_label),
                VectorField.from_any(snap.get("vector"), label=v_label),
            )
        return self._parsed[key]

    def metrics(self) -> list[dict[str, Any]]:
        file = self.path / METRICS
        try:
            size = file.stat().st_size
        except OSError:
            return []
        if size == self._metrics[0]:
            return self._metrics[1]
        rows: list[dict[str, Any]] = []
        try:
            with open(file, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        continue
        except OSError:
            return self._metrics[1]
        self._metrics = (size, rows)
        return rows

    def count(self) -> int:
        latest = self.latest()
        return int(latest["seq"]) if latest else 0


def _pick_frames(files: list[Path], max_frames: int) -> list[Path]:
    """Evenly spaced subset that always keeps the first and the last file."""
    if len(files) <= max_frames:
        return files
    idx = np.linspace(0, len(files) - 1, num=max_frames).round().astype(int)
    return [files[i] for i in sorted(set(idx.tolist()))]


def track_from_meta(meta: Mapping[str, Any], fallback: TrackMap) -> TrackMap:
    info = meta.get("track") or {}
    grid = info.get("grid")
    if grid:
        return TrackMap(
            grid=np.asarray(grid, dtype=int),
            name=str(info.get("name", "Track")),
            difficulty=str(info.get("difficulty", "")),
        )
    name = info.get("name")
    if name:
        from .tracks import TRACK_SPECS, get_track

        if name in TRACK_SPECS:
            return get_track(name)
    return fallback


def build_live_session(
    reader: LiveReader,
    fallback_track: TrackMap,
    max_frames: int = 40,
) -> LabSession:
    """Assemble what the run directory holds right now into a :class:`LabSession`."""
    meta = reader.meta()
    track = track_from_meta(meta, fallback_track)
    s_label = meta.get("scalar_label", "Value")
    v_label = meta.get("vector_label", "Vector")
    t_label = meta.get("trajectory_label", "Trajectory")

    latest = reader.latest()
    iterations: list[Iteration] = []
    last_scalar: ScalarField | None = None
    last_vector: VectorField | None = None
    seen_seq = -1

    files = _pick_frames(reader.kept_files(), max_frames)
    for file in files:
        snap = reader.snapshot(file)
        if snap is None:
            continue
        scalar, vector = reader.parsed(file.name, snap, s_label, v_label)
        last_scalar = scalar or last_scalar
        last_vector = vector or last_vector
        seen_seq = int(snap.get("seq", seen_seq))
        iterations.append(
            Iteration(
                index=seen_seq,
                scalar=scalar or last_scalar,
                vector=vector or last_vector,
                metrics=dict(snap.get("metrics") or {}),
                label=str(snap.get("label") or f"update {seen_seq}"),
            )
        )

    if latest is not None and int(latest.get("seq", -1)) != seen_seq:
        scalar, vector = reader.parsed(f"latest:{latest.get('seq')}", latest, s_label, v_label)
        last_scalar = scalar or last_scalar
        last_vector = vector or last_vector
        iterations.append(
            Iteration(
                index=int(latest.get("seq", 0)),
                scalar=scalar or last_scalar,
                vector=vector or last_vector,
                metrics=dict(latest.get("metrics") or {}),
                label=str(latest.get("label") or "latest"),
            )
        )

    trajectory = None
    if latest is not None and latest.get("trajectory"):
        trajectory = Trajectory.from_any(latest["trajectory"], label=t_label)

    metrics_log = [
        {"index": int(r.get("seq", i)), "label": r.get("label"), "metrics": r.get("metrics") or {}}
        for i, r in enumerate(reader.metrics())
    ]

    return LabSession(
        track=track,
        trajectory=trajectory,
        scalar=last_scalar,
        vector=last_vector,
        iterations=iterations,
        source="live",
        title=str(reader.path.name),
        note=str(meta.get("note", "")),
        metrics_log=metrics_log,
    )
