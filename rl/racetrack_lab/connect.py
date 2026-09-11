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
"""

from __future__ import annotations

from .schema import LabSession, TrackMap


def load_session(track: TrackMap) -> LabSession | None:
    """Return your data for ``track``, or ``None`` when nothing is connected."""
    return None
