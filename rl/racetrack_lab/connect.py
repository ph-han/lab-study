"""Where your experiment plugs into the viewer.

The viewer calls :func:`load_session` once per rerun with the track the user
selected, and draws whatever comes back. Returning ``None`` leaves the app in
its preview/empty state, so this file is safe to leave untouched.

You own everything upstream of here: file names, function names, data
structures, how a run is produced and when. This function only has to hand the
viewer a :class:`~racetrack_lab.schema.LabSession`, and
:func:`~racetrack_lab.schema.build_session` accepts loose shapes:

    from racetrack_lab.schema import build_session

    def load_session(track):
        my_output = ...                      # however you produced it

        return build_session(
            track,
            # any iterable of (x, y), (x, y, vx, vy), dicts or your own objects
            trajectory=my_output["path"],
            # {(x, y): number} or {(x, y, vx, vy): number} or a 2D array [y, x]
            scalar_field=my_output["numbers"],
            # {(x, y): (u, v)} or {(x, y, vx, vy): (u, v)}
            vector_field=my_output["directions"],
            # optional per-iteration snapshots for the timeline
            iterations=[
                {"scalar": snap, "metrics": {"delta": d}}
                for snap, d in my_output["history"]
            ],
            # labels are yours; they are what the UI prints
            scalar_label="V(s)",
            vector_label="a(s)",
        )

Only ``track`` is required -- pass whichever of the rest exist and the UI shows
an empty state for the others.

To fill the Bellman panel as well, record the calculation while you compute it
and hand back ``(session, trace)``. No files, no ``runs/`` directory::

    from racetrack_lab.rooms import parse
    from racetrack_lab.schema import build_session
    from racetrack_lab.trace import TraceRecorder

    def load_session(track):
        rec = TraceRecorder(coords="yx", gamma=0.9, actions=ACTION_VEC)

        for s in env.states():                      # your sweep
            terms = []
            for a, p in pi[s].items():
                ns = env.next_state(s, a)
                terms.append({"action": a, "prob": p, "next_state": ns,
                              "reward": env.reward(s, a, ns), "next_value": V[ns]})
            rec.backup(state=s, value_before=V[s], terms=terms, value_after=new, sweep=k)

        return build_session(track, scalar_field=V_xy), rec.trace()

``coords`` says whether you write states as ``(y, x)`` or ``(x, y)``;
``actions`` maps each action to its ``(dx, dy)`` so a blocked move still shows
which way it was tried.
"""

from __future__ import annotations

from .schema import LabSession, TrackMap
from .trace import Trace


def load_session(track: TrackMap) -> LabSession | tuple[LabSession, Trace] | None:
    """Return your data for ``track``, or ``None`` when nothing is connected."""
    return None
