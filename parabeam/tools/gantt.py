"""Gantt charts of one problem in a level-sync run, drawn only from its logs.

    python -m parabeam.tools.gantt runs/levelsync-weak20-n256-p4-keep_local --id 17
    python -m parabeam.tools.gantt RUN_DIR            # the slowest problem

The top panel has one row per rank: for each level, the time spent receiving
moved beams, generating, scoring, and waiting in the allgather. Async runs score
while they generate, so their levels show one "generate + score" span; their
wait can hold early starts of the next level. The bottom panel
counts each rank's requests still generating, which shows how much of a level
is a few long steps. Needs matplotlib, which the CARC environment leaves out;
copy the run directory and plot locally.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

PHASES = ("move", "generate", "score", "wait")
ASYNC_PHASES = ("move", "work", "wait")  # async logs have four marks, not five
# Validated categorical slots 1-3 (dataviz palette), and gray for idle time. An
# async chart never has generate and work together, so they share a colour.
COLORS = {"move": "#1baf7a", "generate": "#2a78d6", "score": "#eb6834", "wait": "#c3c2b7",
          "work": "#2a78d6"}
LABELS = {"work": "generate + score"}
# Ranks use other validated slots, so a rank never shares a phase's color.
RANK_COLORS = ("#6250d6", "#eda100", "#e87ba4", "#008300")
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5df"


@dataclass(frozen=True)
class Span:
    rank: int
    level: int
    phase: str
    start: float
    stop: float


def read_logs(run: Path, problem_id: int, job: str | None = None) -> list[dict]:
    """Every rank's lines for one problem, from one job (the newest if not given)."""
    lines = [json.loads(line) for path in sorted((run / "ranks").glob("rank-*.jsonl"))
             for line in path.read_text().splitlines() if line.strip()]
    lines = [line for line in lines if line["id"] == problem_id]
    if not lines:
        raise SystemExit(f"no log lines for problem {problem_id} in {run / 'ranks'}")
    jobs = sorted({str(line.get("job")) for line in lines})
    wanted = job if job is not None else jobs[-1]
    return [line for line in lines if str(line.get("job")) == wanted]


def spans(lines: list[dict]) -> list[Span]:
    out = []
    for line in lines:
        marks = line["marks"]
        phases = ASYNC_PHASES if len(marks) == len(ASYNC_PHASES) + 1 else PHASES
        for phase, (start, stop) in zip(phases, pairwise(marks), strict=True):
            out.append(Span(line["rank"], line["level"], phase, start, stop))
    return out


def running(lines: list[dict], rank: int) -> list[tuple[float, int]]:
    """(time, requests still generating) steps for one rank; requests start at their level's
    generate mark. An async step started early really began before that mark, so for
    async runs the count is approximate and can exceed the cap."""
    events = []
    for line in lines:
        if line["rank"] != rank:
            continue
        begin = line["marks"][1]
        for _, _, seconds in line["steps"]:
            events += [(begin, 1), (begin + seconds, -1)]
    events.sort()
    count, curve = 0, [(0.0, 0)]
    for t, delta in events:
        count += delta
        curve.append((t, count))
    return curve


def level_tails(lines: list[dict]) -> list[dict]:
    """Per level: the longest step over all ranks, its finish, and the median step time."""
    by_level: dict[int, list[list]] = {}
    for line in lines:
        by_level.setdefault(line["level"], []).extend(line["steps"])
    rows = []
    for level, steps in sorted(by_level.items()):
        times = sorted(s[2] for s in steps)
        longest = max(steps, key=lambda s: s[2])
        rows.append({"level": level, "requests": len(steps), "median_s": times[len(times) // 2],
                     "longest_s": longest[2], "longest_tokens": longest[0],
                     "longest_finish": longest[1],
                     "length_finishes": sum(s[1] == "length" for s in steps)})
    return rows


def draw(lines: list[dict], title: str, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ranks = sorted({line["rank"] for line in lines})
    fig, (top, bottom) = plt.subplots(
        2, 1, sharex=True, figsize=(11, 1.2 + 0.55 * len(ranks) + 2.4),
        gridspec_kw={"height_ratios": [max(1.5, 0.55 * len(ranks)), 2.2]})
    drawn = spans(lines)
    for span in drawn:
        if span.stop > span.start:
            # A 2 px gap between neighbouring bars comes from the white edge.
            top.barh(span.rank, span.stop - span.start, left=span.start, height=0.6,
                     color=COLORS[span.phase], edgecolor="white", linewidth=0.6)
    top.set_yticks(ranks, [f"rank {r}" for r in ranks])
    top.invert_yaxis()
    top.set_title(title, loc="left", color=INK, fontsize=11)
    shown = list(dict.fromkeys(span.phase for span in drawn))
    top.legend(handles=[Patch(color=COLORS[p], label=LABELS.get(p, p)) for p in shown],
               ncols=len(shown),
               loc="lower left", bbox_to_anchor=(0, 1.08), frameon=False, fontsize=9)

    for rank in ranks:
        curve = running(lines, rank)
        bottom.step([t for t, _ in curve], [n for _, n in curve], where="post", linewidth=1.5,
                    color=RANK_COLORS[rank % len(RANK_COLORS)], label=f"rank {rank}")
    bottom.set_ylabel("requests generating", color=MUTED)
    bottom.set_xlabel("seconds since the problem started", color=MUTED)
    peak = max(max(n for _, n in running(lines, rank)) for rank in ranks)
    bottom.set_ylim(0, max(1, peak) * 1.3)  # headroom keeps the legend off the lines
    if len(ranks) > 1:
        bottom.legend(frameon=False, fontsize=9, ncols=len(ranks), loc="upper right")
    for ax in (top, bottom):
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--id", type=int, help="problem to draw (default: the slowest)")
    parser.add_argument("--job", help="Slurm job whose lines to use (default: the newest)")
    parser.add_argument("--out", type=Path, help="PNG path (default: RUN/gantt-<id>.png)")
    args = parser.parse_args()

    records = [json.loads(line) for line in (args.run / "results.jsonl").read_text().splitlines()
               if line.strip()]
    record = (max(records, key=lambda r: r["seconds"]) if args.id is None
              else next(r for r in records if r["id"] == args.id))
    lines = read_logs(args.run, record["id"], args.job)
    manifest = json.loads((args.run / "manifest.json").read_text())["config"]
    how = (f"async, cap {manifest.get('cap')}, early {manifest.get('early')}"
           if manifest.get("mode") == "async" else manifest["placement"])
    title = (f"problem {record['id']}, P={manifest['P']}, N={manifest['N']}, "
             f"{how}: {record['seconds']:.1f} s, {record['levels']} levels")
    out = args.out or args.run / f"gantt-{record['id']}.png"
    draw(lines, title, out)
    print(f"wrote {out}")
    print("level  requests  median step s  longest step s  its tokens  its finish  length")
    for row in level_tails(lines):
        print(f"{row['level']:>5}  {row['requests']:>8}  {row['median_s']:>13.2f}  "
              f"{row['longest_s']:>14.2f}  {row['longest_tokens']:>10}  "
              f"{row['longest_finish']:>10}  {row['length_finishes']:>6}")


if __name__ == "__main__":
    main()
