"""Stacked tracking-progress picture, shared by the trainer and the offline script.

One panel per evaluation step on the same fixed trajectory, so a run can be read as a
sequence of pictures instead of only as metric tables.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np


def _ragged(values: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    return [values[offsets[k] : offsets[k + 1]] for k in range(len(offsets) - 1)]


def _truth_tracks(archive) -> tuple[dict[int, np.ndarray], np.ndarray]:
    offsets = archive["state_offsets"]
    states, identifiers = archive["states"], archive["target_ids"]
    tracks: dict[int, list[np.ndarray]] = {}
    for index in range(len(offsets) - 1):
        span = slice(offsets[index], offsets[index + 1])
        for state, label in zip(states[span], identifiers[span]):
            tracks.setdefault(int(label), []).append(state[:2])
    return (
        {label: np.asarray(points) for label, points in tracks.items()},
        np.asarray(archive["measurements"]),
    )


def draw_panel(
    axis,
    tracks: dict[int, np.ndarray],
    measurements: np.ndarray,
    positions: Sequence[np.ndarray],
    identifiers: Sequence[np.ndarray],
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    axis.scatter(measurements[:, 0], measurements[:, 1], s=2, marker="x", color="0.82", alpha=0.25)
    for points in tracks.values():
        axis.plot(points[:, 0], points[:, 1], "k-", lw=1.0, alpha=0.8)
        axis.plot(points[0, 0], points[0, 1], "ko", ms=3)
        axis.plot(points[-1, 0], points[-1, 1], "k^", ms=3)
    grouped: dict[int, list[np.ndarray]] = {}
    for frame_positions, frame_ids in zip(positions, identifiers):
        for position, label in zip(frame_positions, frame_ids):
            grouped.setdefault(int(label), []).append(position)
    palette = plt.colormaps["tab20"]
    for index, (_, points) in enumerate(sorted(grouped.items())):
        stacked = np.asarray(points)
        axis.plot(stacked[:, 0], stacked[:, 1], ".", ms=3.0, color=palette(index % 20))
    estimates = sum(len(item) for item in positions)
    axis.set_title(f"{title}  ({len(grouped)} tracks, {estimates} estimates)", fontsize=9)
    axis.set_xlim(-11, 11)
    axis.set_ylim(-11, 11)
    axis.grid(alpha=0.2)
    axis.tick_params(labelsize=7)


def plot_demo_progress(
    demo_dir: Path,
    dataset_dir: Path,
    output: Path,
    *,
    max_panels: int = 8,
    columns: int = 2,
    extra_panels: Sequence[tuple[str, list[np.ndarray], list[np.ndarray]]] = (),
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = sorted(Path(demo_dir).glob("step_*.npz"))
    if not files:
        raise FileNotFoundError(f"no demo dumps in {demo_dir}")
    if max_panels and len(files) > max_panels:
        # Keep the endpoints and spread the rest, so early behaviour stays visible.
        keep = np.linspace(0, len(files) - 1, max_panels).round().astype(int)
        files = [files[index] for index in sorted(set(keep.tolist()))]
    archives = [dict(np.load(path)) for path in files]
    scene = str(archives[0]["scene"])
    run = int(archives[0]["run"])
    with np.load(Path(dataset_dir) / scene / f"run_{run:03d}.npz") as source:
        tracks, measurements = _truth_tracks(source)

    panels = [
        (
            f"step {int(archive['step'])}",
            _ragged(archive["positions"], archive["offsets"]),
            _ragged(archive["track_ids"], archive["offsets"]),
        )
        for archive in archives
    ]
    panels.extend(extra_panels)
    rows = int(np.ceil(len(panels) / max(columns, 1)))
    figure, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 5.0 * rows), squeeze=False)
    for index, (title, positions, identifiers) in enumerate(panels):
        draw_panel(
            axes[index // columns][index % columns],
            tracks,
            measurements,
            positions,
            identifiers,
            title,
        )
    for index in range(len(panels), rows * columns):
        axes[index // columns][index % columns].axis("off")
    figure.suptitle(
        f"Tracking progress on {scene} run {run:03d} "
        "(black: ground truth, grey: measurements, colours: estimated tracks)",
        fontsize=11,
    )
    figure.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def demo_steps(demo_dir: Path) -> list[int]:
    return sorted(
        int(re.findall(r"\d+", path.stem)[-1]) for path in Path(demo_dir).glob("step_*.npz")
    )
