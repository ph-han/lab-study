"""DEMO stream -- publishes placeholder geometry as if a run were in progress.

Nothing here learns anything. It replays the same closed-form demo shapes
(``demo.py``) through :class:`~racetrack_lab.live.LiveRun` so the live path of
the viewer -- polling, gliding car, recoloring field, iteration slider,
metrics chart -- can be watched before any experiment exists.

    python -m racetrack_lab.demo_stream            # Easy, ~45 s
    python -m racetrack_lab.demo_stream --track Hard --seconds 90
"""

from __future__ import annotations

import argparse
import math
import time

from .demo import demo_scalar_values, demo_trajectory, demo_vector_values
from .live import LiveRun
from .tracks import get_track


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--track", default="Easy", choices=["Easy", "Medium", "Hard"])
    parser.add_argument("--path", default="runs/demo_stream")
    parser.add_argument("--seconds", type=float, default=45.0)
    parser.add_argument("--hz", type=float, default=3.0, help="updates per second")
    args = parser.parse_args()

    track = get_track(args.track)
    line = demo_trajectory(track)
    directions = demo_vector_values(track)
    total = max(2, int(args.seconds * args.hz))

    run = LiveRun(
        args.path,
        track,
        scalar_label="Demo field",
        vector_label="Demo directions",
        trajectory_label="Demo line",
        note="DEMO stream -- geometric placeholder, not learning",
        keep_every=max(1, total // 40),
    )
    print(f"publishing {total} demo updates to {run.path} ... (Ctrl+C to stop)")
    try:
        for k in range(total):
            progress = k / (total - 1)
            step = k % len(line)
            if step == 0:
                run.new_episode()
            s = line.states[step]
            run.publish(
                scalar_field=demo_scalar_values(track, sharpness=progress),
                vector_field=directions if progress > 0.3 else None,
                state=(s.x, s.y, s.vx, s.vy),
                metrics={"blend": progress, "demo delta": 10.0 * math.exp(-4.0 * progress)},
                label=f"demo update {k}",
            )
            time.sleep(1.0 / args.hz)
    except KeyboardInterrupt:
        pass
    finally:
        run.close()
        print("done")


if __name__ == "__main__":
    main()
