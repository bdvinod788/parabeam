"""Asynchronous search over P ranks, the scheduler level-sync is the baseline for.

Selection stays global and exact: each level still ends in one allgather of
(path, score, finished), and every rank picks the same survivors, so this keeps
exactly the beams the serial search keeps. What changes is how a rank spends a
level, in three parts that can be switched off one at a time for the ablation:

- Scoring overlaps generation (score="each"). Each child is scored as soon as it
  is generated, in one PRM call with whatever else finished while the PRM was
  busy, instead of all together after the level's longest step. In its one GPU
  run (job 14398026: P = 1, cap 64) this took 4.02 s per level against level-sync's
  2.73. The two models slowing each other on one GPU is the likely cause, but it
  was not isolated from the cap or from the many small PRM calls. With
  score="level" a rank scores its children of a level in one call once they are
  all generated, as level-sync
  does, and keeps the other two parts.
- Stealing. Work waits in a rank-local queue with at most `cap` requests
  generating at once. A rank with nothing queued and free slots asks the rank
  that started the level with the longest queue, then the next, and takes as
  many units as it can start, worst-ranked first, parents included. With
  placement="stay" survivors are not rebalanced before a level, so stealing is
  what evens out the load; a cap of N / P then queues only the overflow.
- Early starts. Child i of a parent is the same request however many children
  the parent ends up with, and every survivor gets at least M. So a rank with
  free slots starts the first M children of its best-scored beams before
  selection. It reports which parents it started, placement keeps their first
  units where they are, and the children of parents that lose are aborted.
- Early stealing (early_steal=True). Ranks tell each other their scores as soon as
  they have them. A rank drops early starts that can no longer be among the top
  N / M, and uses the slots for likely survivors of ranks that hold more than
  their share, which hand over the parent. So the likely survivors are spread
  while ranks wait, with no cap needed. A rank still waiting for a long step can
  score what it has once at most `tail` requests are left, and join in.

    python -m parabeam.dist.async_sched --local 4 --subset pilot
    mpiexec -n 4 python -m parabeam.dist.async_sched --subset pilot
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

from parabeam.backends.base import Backend
from parabeam.core.beam import (
    Beam,
    Expansion,
    Problem,
    SearchConfig,
    expand,
    is_finished,
)
from parabeam.core.seeding import Path
from parabeam.core.select import Scored, child_counts, select_top
from parabeam.core.serial import LevelStats, SearchResult, candidate_digest
from parabeam.dist.levelsync import Candidate, Step, _move
from parabeam.dist.placement import ASYNC_PLACEMENTS, Owner, check_placement, make_units
from parabeam.dist.runtime import EARLY, GIVE, HAND, SCORES, STEAL, Comm


@dataclass(frozen=True, slots=True)
class AsyncOptions:
    cap: int | None = None  # requests generating at once on a rank; None for no limit
    steal: bool = True
    early: float = 1.0  # parents a rank may start early, in shares of N / (M * P); 0 for none
    early_after: float = 0.5  # share of its own children scored before a rank starts early
    score: str = "each"  # "each": as children finish; "level": once a rank's level is generated
    placement: str = "keep_local"  # or "stay": leave survivors where they are, and let stealing act
    early_steal: bool = False  # start early the likely survivors of other ranks; needs score="level"
    tail: int = 0  # with score="level", score early once this few requests are left; 0 to wait

    def __post_init__(self) -> None:
        if self.score not in ("each", "level"):
            raise ValueError(f"score must be 'each' or 'level', got {self.score!r}")
        if self.cap is not None and self.cap < 1:
            raise ValueError(f"cap must be positive, got {self.cap}")
        if self.early < 0 or not 0 <= self.early_after <= 1:
            raise ValueError("early must be at least 0 and early_after in [0, 1]")
        if self.placement not in ASYNC_PLACEMENTS:
            raise ValueError(f"placement must be one of {sorted(ASYNC_PLACEMENTS)}")
        if (self.early_steal or self.tail) and self.score != "level":
            raise ValueError("early_steal and tail need score='level'")
        if self.early_steal and self.early <= 0:
            raise ValueError("early_steal needs early starts")


@dataclass(frozen=True, slots=True)
class AsyncRankLevel:
    """What one rank did in one level."""

    rank: int
    level: int
    requests: int  # children of this level generated and scored here
    gen_tokens: int
    moved_in: int  # survivors received from other ranks before the level
    stolen: int  # units this rank took from others
    given: int  # units others took from this rank
    early_started: int  # children of the next level started before selection
    early_kept: int  # of those, children whose parent survived
    early_taken: int  # parents started early here that another rank scored
    sent_bytes: int
    # When the rank began moving beams, began the level's work, finished it, and
    # left the allgather. Work for the next level can run between the last two.
    marks: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    steps: tuple[Step, ...] = ()

    @property
    def wait_s(self) -> float:
        return self.marks[3] - self.marks[2]


@dataclass(frozen=True, slots=True)
class AsyncResult:
    result: SearchResult  # the same on every rank
    ranks: tuple[AsyncRankLevel, ...]  # by level, then rank


class _Work(NamedTuple):
    """Children start to stop of a parent: a unit, with the parent itself, so it can move."""

    parent: Beam
    start: int
    stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start


class _Child(NamedTuple):
    beam: Beam  # scored, unless the rank scores whole levels
    step: Step


@dataclass(eq=False, slots=True)
class _Slot:
    """A child's place among the requests generating; given back exactly once."""

    held: bool = True


class _Scorer:
    """Scores children as they come, batching whatever arrives while the PRM is busy."""

    def __init__(self, backend: Backend) -> None:
        self._backend = backend
        self._waiting: list[tuple[list[int], asyncio.Future[float]]] = []
        self._worker: asyncio.Task[None] | None = None

    async def score(self, prm_ids: list[int]) -> float:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._waiting.append((prm_ids, future))
        if self._worker is None or self._worker.done():
            self._worker = loop.create_task(self._run())
        return await future

    async def _run(self) -> None:
        while self._waiting:
            batch = [(ids, f) for ids, f in self._waiting if not f.done()]
            self._waiting = []
            if not batch:
                continue
            try:
                scores = await self._backend.score([ids for ids, _ in batch])
            except Exception as error:  # noqa: BLE001 - handed to every waiter, not swallowed
                for _, future in batch:
                    if not future.done():
                        future.set_exception(error)
                continue
            for (_, future), score in zip(batch, scores, strict=True):
                if not future.done():
                    future.set_result(score)


def _quiet(task: asyncio.Task) -> None:
    """Retrieve an aborted task's outcome, so asyncio does not warn about it."""
    if not task.cancelled():
        task.exception()


@dataclass(eq=False)
class _Rank:
    problem: Problem
    cfg: SearchConfig
    opts: AsyncOptions
    backend: Backend
    comm: Comm
    clock: Callable[[], float]
    scorer: _Scorer = field(init=False)
    level: int = 0  # the level whose queue is open; others see it to answer steals
    queue: deque[_Work] = field(default_factory=deque)
    generating: int = 0
    given: int = 0
    early: dict[Path, list[asyncio.Task[_Child]]] = field(default_factory=dict)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    opened: dict[int, asyncio.Event] = field(default_factory=dict)  # by level
    # Early stealing, for the current level:
    scored: dict[Path, Beam] = field(default_factory=dict)  # this rank's, as they are scored
    handed: set[Path] = field(default_factory=set)  # given to another rank to start
    foreign: dict[Path, Beam] = field(default_factory=dict)  # taken from another rank
    known: dict[int, dict[int, tuple[int, list[Scored]]]] = field(default_factory=dict)
    news: asyncio.Event = field(default_factory=asyncio.Event)
    sending: set[asyncio.Task] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.scorer = _Scorer(self.backend)

    # Children -------------------------------------------------------------

    def _release(self, slot: _Slot) -> None:
        if slot.held:
            slot.held = False
            self.generating -= 1
            self.changed.set()

    async def _child(self, e: Expansion, slot: _Slot) -> _Child:
        try:
            out = await self.backend.generate(e.req)
        finally:
            self._release(slot)
        child = e.parent.child(e.index, out, self.backend.prm_step_ids(out))
        step = Step(len(out.token_ids), out.finish, out.t_end - out.t_start)
        if self.opts.score == "level":
            return _Child(child, step)  # scored with the rest of its level
        return _Child(child.scored(await self.scorer.score(child.prm_ids)), step)

    def _start(self, work: _Work) -> list[asyncio.Task[_Child]]:
        """Start a unit's children. Each holds a slot from now until its step is generated,
        or until it is cancelled, even before it first runs."""
        loop = asyncio.get_running_loop()
        tasks = []
        for e in expand(work.parent, work.stop, self.cfg)[work.start:]:
            slot = _Slot()
            self.generating += 1
            task = loop.create_task(self._child(e, slot))
            task.add_done_callback(lambda _, slot=slot: self._release(slot))
            tasks.append(task)
        return tasks

    def _free(self) -> int:
        cap = self.opts.cap
        return sys.maxsize if cap is None else cap - self.generating

    def _fill(self, tasks: set[asyncio.Task[_Child]]) -> None:
        """Start queued units while there are slots; a unit always starts on an idle rank."""
        while self.queue and (self.queue[0].size <= self._free() or self.generating == 0):
            tasks.update(self._start(self.queue.popleft()))

    # Stealing ---------------------------------------------------------------

    def _opened(self, level: int) -> asyncio.Event:
        return self.opened.setdefault(level, asyncio.Event())

    async def serve(self) -> None:
        """Answer steal requests for the whole problem, whatever this rank is doing."""
        answering: set[asyncio.Task[None]] = set()
        try:
            while True:
                thief, (level, want) = await self.comm.recv(STEAL)
                task = asyncio.ensure_future(self._answer(thief, level, want))
                answering.add(task)
                task.add_done_callback(answering.discard)
        finally:
            for task in answering:
                task.cancel()

    async def _answer(self, thief: int, level: int, want: int) -> None:
        """Give up to `want` units, worst first. A thief can reach a level before this
        rank has opened it, so the answer waits for that rather than having it ask again."""
        await self._opened(level).wait()
        if self.level > level or not self.queue:
            reply = (level, "empty", [])
        else:
            works = [self.queue.pop() for _ in range(min(want, len(self.queue)))][::-1]
            self.given += len(works)
            reply = (level, "gave", works)
        await self.comm.send(reply, thief, GIVE)

    async def _steal(self, level: int, victims: Sequence[int], empty: set[int]) -> int:
        """Ask each rank not known to be empty once; return how many units arrived."""
        for victim in victims:
            if victim in empty:
                continue
            want = max(1, min(self._free(), self.cfg.N) // self.cfg.M)
            await self.comm.send((level, want), victim, STEAL)
            source, (for_level, status, works) = await self.comm.recv(GIVE)
            if source != victim or for_level != level:
                raise RuntimeError(f"rank {self.comm.rank} asked {victim} for level {level} "
                                   f"and heard from {source} about {for_level}")
            if status == "gave":
                self.queue.extend(works)
                return len(works)
            empty.add(victim)
        return 0

    # Early starts -----------------------------------------------------------

    def _start_early(self, children: dict[Path, Beam], budget: int) -> None:
        """Start the first M children of the best local beams not yet started, up to budget."""
        if len(self.early) >= budget or self._free() < self.cfg.M:
            return
        ranked = sorted((b for b in children.values()
                         if not is_finished(b, self.cfg) and b.path not in self.early),
                        key=lambda b: (-b.score, b.path))
        for beam in ranked:
            if len(self.early) >= budget or self._free() < self.cfg.M:
                return
            self.early[beam.path] = self._start(_Work(beam, 0, self.cfg.M))

    def _abort_early(self, keep: set[Path]) -> int:
        """Abort early children of parents not kept here; return how many are kept."""
        kept = 0
        for parent, tasks in list(self.early.items()):
            if parent in keep:
                kept += len(tasks)
                continue
            for task in tasks:
                # Cancelling is the abort: the mock drops the request, and the vLLM
                # client closes its connection, which makes vLLM abort it.
                task.cancel()
                task.add_done_callback(_quiet)
            del self.early[parent]
        return kept

    # Early stealing ---------------------------------------------------------

    def _budget(self) -> int:
        return math.ceil(self.opts.early * self.cfg.survivors / self.comm.size)

    def _publish(self, level: int, children: dict[Path, Beam]) -> None:
        """Tell every rank this rank's scores so far for a level."""
        self.scored = children
        mine = (len(children), [Scored(b.score, b.path) for b in children.values()
                                if not is_finished(b, self.cfg)])
        self.known.setdefault(level, {})[self.comm.rank] = mine
        self.news.set()
        for dest in range(self.comm.size):
            if dest != self.comm.rank:
                task = asyncio.ensure_future(
                    self.comm.send((self.problem.id, level, mine), dest, SCORES))
                self.sending.add(task)
                task.add_done_callback(self.sending.discard)

    async def listen(self) -> None:
        """Collect the other ranks' scores. Another problem's are dropped."""
        while True:
            source, (problem, level, scores) = await self.comm.recv(SCORES)
            if problem == self.problem.id and level >= self.level:
                self.known.setdefault(level, {})[source] = scores
                self.news.set()

    def _likely(self, level: int) -> set[Path]:
        """Which scored beams would survive, judging by every score known so far."""
        known = self.known.get(level, {}).values()
        share = sum(count for count, _ in known) / self.cfg.N
        return {s.path for s in select_top((s for _, scores in known for s in scores),
                                           math.ceil(self.cfg.survivors * share))}

    def _spare(self, level: int, rank: int) -> list[Scored]:
        """A rank's likely survivors beyond its own budget, best first."""
        likely = self._likely(level)
        scores = self.known.get(level, {}).get(rank, (0, []))[1]
        mine = sorted((s for s in scores if s.path in likely), key=lambda s: (-s.score, s.path))
        return mine[self._budget():]

    async def hand(self) -> None:
        """Answer requests for likely survivors, at once, for the whole problem."""
        while True:
            thief, (level, want) = await self.comm.recv(EARLY)
            beams = []
            if level == self.level and level in self.known:
                spare = [s.path for s in self._spare(level, self.comm.rank)]
                free = [p for p in spare if p in self.scored and p not in self.early
                        and p not in self.handed]
                beams = [self.scored[p] for p in free[:want]]
                self.handed.update(b.path for b in beams)
            await self.comm.send((level, beams), thief, HAND)

    async def speculate(self, level: int, stop: asyncio.Event) -> None:
        """Keep the early starts to likely survivors, taking other ranks' spare ones, until
        the level is decided. One request is out at a time, and it is always answered."""
        budget, dry = self._budget(), set()
        while not stop.is_set():
            self.news.clear()
            if level in self.known and self.comm.rank in self.known[level]:
                likely = self._likely(level)
                for parent in [p for p in self.early if p not in likely]:
                    self._cancel(parent)
                own = sorted((b for p, b in self.scored.items() if p in likely
                              and p not in self.early and p not in self.handed),
                             key=lambda b: (-b.score, b.path))
                for beam in own:
                    if len(self.early) >= budget or self._free() < self.cfg.M:
                        break
                    self.early[beam.path] = self._start(_Work(beam, 0, self.cfg.M))
                room = min(budget - len(self.early), self._free() // self.cfg.M)
                spare = {r: len(self._spare(level, r)) for r in self.known[level]
                         if r != self.comm.rank and (r, len(self.known[level])) not in dry}
                victim = max(spare, key=lambda r: (spare[r], -r), default=None)
                if room > 0 and victim is not None and spare[victim] > 0:
                    asked = (victim, len(self.known[level]))
                    await self.comm.send((level, min(room, spare[victim])), victim, EARLY)
                    _, (for_level, beams) = await self.comm.recv(HAND)
                    if for_level != level:
                        raise RuntimeError(f"asked for level {level}, heard about {for_level}")
                    if not beams:
                        dry.add(asked)
                    for beam in beams:
                        self.foreign[beam.path] = beam
                        self.early[beam.path] = self._start(_Work(beam, 0, self.cfg.M))
                    continue
            waits = [asyncio.ensure_future(self.news.wait()), asyncio.ensure_future(stop.wait())]
            await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
            for wait in waits:
                wait.cancel()

    def _cancel(self, parent: Path) -> None:
        for task in self.early.pop(parent):
            task.cancel()
            task.add_done_callback(_quiet)

    async def _score(self, beams: list[Beam]) -> dict[Path, Beam]:
        scores = await self.backend.score([b.prm_ids for b in beams])
        return {b.path: b.scored(x) for b, x in zip(beams, scores, strict=True)}

    # One level --------------------------------------------------------------

    async def work(self, level: int, mine: list[_Work], victims: Sequence[int]) -> tuple[
            dict[Path, Beam], list[Step], int, int]:
        """Generate and score this rank's children of a level, stealing when idle.

        Returns the scored children, their steps, units stolen, and children
        started early for the next level.
        """
        tasks: set[asyncio.Task[_Child]] = set()
        for work in mine:
            started = self.early.pop(work.parent.path, None) if work.start == 0 else None
            if started is not None:
                tasks.update(started)
            else:
                self.queue.append(work)
        self.level = level  # with the queue in place, steal requests see both together
        self.scored, self.handed, self.foreign = {}, set(), {}
        self.known = {k: v for k, v in self.known.items() if k >= level}
        self._opened(level).set()
        own = sum(work.size for work in mine)
        budget = self._budget()
        done: dict[Path, Beam] = {}  # scored before the rest, by a tail call
        tail_call: asyncio.Task | None = None
        early_ok = self.opts.early > 0 and level < self.cfg.max_depth
        children: dict[Path, Beam] = {}
        steps: list[Step] = []
        empty: set[int] = set()
        stolen = 0
        while True:
            self._fill(tasks)
            for task in [t for t in tasks if t.done()]:
                tasks.discard(task)
                beam, step = task.result()
                children[beam.path] = beam
                steps.append(step)
            can_start = self._free() >= self.cfg.M or self.generating == 0
            if not self.queue and self.opts.steal and can_start and len(empty) < len(victims):
                got = await self._steal(level, victims, empty)
                if got:
                    stolen += got
                    continue
            if early_ok and self.opts.score == "each" and not self.queue \
                    and len(children) >= self.opts.early_after * own:
                self._start_early(children, budget)
            if not tasks:
                break
            if self.opts.tail and tail_call is None and children and not self.queue \
                    and len(tasks) <= self.opts.tail:
                tail_call = asyncio.ensure_future(self._score(list(children.values())))
            if tail_call is not None and tail_call.done() and not done:
                done = tail_call.result()
                if early_ok and self.opts.early_steal:
                    self._publish(level, dict(done))
                elif early_ok:
                    self._start_early(done, budget)
            self.changed.clear()
            waiter = asyncio.ensure_future(self.changed.wait())
            scoring = {tail_call} if tail_call is not None and not tail_call.done() else set()
            await asyncio.wait({*tasks, waiter, *scoring}, return_when=asyncio.FIRST_COMPLETED)
            waiter.cancel()
        if tail_call is not None:
            done = await tail_call
        if self.opts.score == "level" and children:
            # Unless early starts are already running here, the PRM has the GPU to itself.
            rest = [b for path, b in children.items() if path not in done]
            children = {**done, **(await self._score(rest) if rest else {})}
        if early_ok and self.opts.early_steal:
            self._publish(level, children)
        elif early_ok:
            self._start_early(children, budget)  # every child is scored now: top up
        return children, steps, stolen, sum(len(t) for t in self.early.values())


async def search(
    problem: Problem,
    cfg: SearchConfig,
    backend: Backend,
    comm: Comm,
    opts: AsyncOptions = AsyncOptions(),
    clock: Callable[[], float] = time.perf_counter,
    sync_clock: Callable[[float], None] | None = None,
) -> AsyncResult:
    """Run one problem on every rank at once. Every rank returns the same result."""

    async def gather(payload):
        reports = await comm.allgather((payload, clock()))
        if sync_clock is not None:
            sync_clock(max(at for _, at in reports))
        return [payload for payload, _ in reports]

    rank = _Rank(problem, cfg, opts, backend, comm, clock)
    servers = [asyncio.ensure_future(rank.serve())]
    if opts.early_steal:
        servers += [asyncio.ensure_future(rank.listen()), asyncio.ensure_future(rank.hand())]
    speculator: asyncio.Task | None = None
    try:
        start = clock()
        survivors: list[Path] = [()]
        owners: list[Owner] = [None]
        pinned: set[int] = set()
        held: dict[Path, Beam] = {(): Beam(problem)}
        finished: list[Beam] = []
        kept: list[tuple[Path, ...]] = []
        digests: list[str] = []
        mine: list[AsyncRankLevel] = []
        early_kept = 0
        while survivors:
            level = len(survivors[0]) + 1
            units = make_units(child_counts(len(survivors), cfg.N), cfg.M)
            where = ASYNC_PLACEMENTS[opts.placement](units, owners, comm.size, pinned)
            check_placement(units, where, comm.size)
            sent_before, given_before, moving = comm.sent_bytes, rank.given, clock()
            moved_in = await _move(comm, level, units, where, owners, survivors, held)
            begin = clock()
            work = [_Work(held[survivors[u.survivor]], u.start, u.stop)
                    for u, dest in zip(units, where, strict=True) if dest == comm.rank]
            load = [0] * comm.size
            for unit, dest in zip(units, where, strict=True):
                load[dest] += unit.size
            victims = sorted((r for r in range(comm.size) if r != comm.rank),
                             key=lambda r: (-load[r], (r - comm.rank) % comm.size))
            stop = asyncio.Event()
            if opts.early_steal and level < cfg.max_depth:
                rank.level = level  # its speculator and servers act on this level from here
                speculator = asyncio.ensure_future(rank.speculate(level, stop))
            children, steps, stolen, early_started = await rank.work(level, work, victims)
            done_at = clock()

            report = ([Candidate(c.path, c.score, is_finished(c, cfg)) for c in children.values()],
                      sorted(rank.early))
            reports = await gather(report)
            candidates = [c for cands, _ in reports for c in cands]
            if len(candidates) != len({c.path for c in candidates}) or len(candidates) != sum(
                    unit.size for unit in units):
                raise RuntimeError(f"level {level} lost or repeated beams")
            owner_of = {c.path: r for r, (cands, _) in enumerate(reports) for c in cands}
            starters = {path: r for r, (_, early) in enumerate(reports) for path in early}
            if opts.early_steal:
                # Early starts went on until every score was in; now settle who started what.
                stop.set()
                if speculator is not None:
                    await speculator
                    speculator = None
                early_started = sum(len(t) for t in rank.early.values())
                lists = await gather(sorted(rank.early))
                starters = {path: r for r, early in enumerate(lists) for path in early}
                if len(starters) != sum(len(early) for early in lists):
                    raise RuntimeError(f"level {level}: two ranks started the same parent early")
            taken = len(rank.foreign)
            top = select_top((Scored(c.score, c.path) for c in candidates if not c.finished),
                             cfg.survivors)
            survivors = [c.path for c in top]
            owners = [starters.get(path, owner_of[path]) for path in survivors]
            pinned = {i for i, path in enumerate(survivors) if path in starters}
            keep = set(survivors)
            early_kept = rank._abort_early(keep)
            held = {path: beam for path, beam in {**children, **rank.foreign}.items()
                    if path in keep}
            finished.extend(b for b in children.values() if is_finished(b, cfg))
            kept.append(tuple(survivors))
            digests.append(candidate_digest(Scored(c.score, c.path) for c in candidates))
            gathered_at = clock()
            mine.append(AsyncRankLevel(
                comm.rank, level, len(children), sum(s.tokens for s in steps), moved_in,
                stolen, rank.given - given_before, early_started, early_kept, taken,
                comm.sent_bytes - sent_before, (moving, begin, done_at, gathered_at),
                tuple(steps)))

        # Every steal of this problem was answered before its thief reported, so the
        # server can go now. Stopping it before the final gather matters: once that
        # returns, a fast rank starts the next problem and may ask this one for work,
        # and a server still running would take the request and never answer it.
        if rank.sending:
            await asyncio.gather(*rank.sending)
        for server in servers:
            await _stop(server)
        gathered = await gather((finished, mine))
    finally:
        for task in [*servers, *([speculator] if speculator is not None else [])]:
            await _stop(task)
        rank._abort_early(set())
    every = sorted((s for _, stats in gathered for s in stats), key=lambda s: (s.level, s.rank))
    return AsyncResult(
        SearchResult(
            problem.id,
            tuple(sorted((b for beams, _ in gathered for b in beams), key=lambda b: b.path)),
            tuple(kept),
            tuple(digests),
            _per_level(every),
            clock() - start,
        ),
        tuple(every),
    )


async def _stop(task: asyncio.Task) -> None:
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def _per_level(stats: Sequence[AsyncRankLevel]) -> tuple[LevelStats, ...]:
    """Requests and tokens summed over ranks; a level's time is its slowest rank's work."""
    by_level: dict[int, list[AsyncRankLevel]] = {}
    for s in stats:
        by_level.setdefault(s.level, []).append(s)
    return tuple(
        LevelStats(level, sum(s.requests for s in group),
                   max(s.marks[2] - s.marks[1] for s in group), 0.0,
                   sum(s.gen_tokens for s in group))
        for level, group in sorted(by_level.items())
    )


async def _mock_rank(comm: Comm, problem_id: int, cfg: SearchConfig, opts: AsyncOptions,
                     chaos: int | None, clock=None) -> AsyncResult:
    from parabeam.backends.mock import Chaos, MockBackend, mock_problem

    backend = MockBackend(chaos=None if chaos is None else Chaos(chaos + 1000 * comm.rank),
                          clock=clock)
    return await search(mock_problem(problem_id), cfg, backend, comm, opts, backend.now,
                        backend.advance_to)


def main() -> None:
    from parabeam.dist.levelsync import _serial
    from parabeam.eval.datasets import SUBSETS_PATH

    parser = argparse.ArgumentParser(description="Async search on the mock backend.")
    parser.add_argument("--local", type=int, metavar="P",
                        help="run P ranks in this process instead of over MPI")
    parser.add_argument("--subset", default="pilot", choices=["pilot", "weak", "timing"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--M", type=int, default=4)
    parser.add_argument("--max-depth", type=int, default=40)
    parser.add_argument("--max-step-tokens", type=int, default=2048)
    parser.add_argument("--cap", type=int, help="requests generating at once per rank")
    parser.add_argument("--no-steal", action="store_true")
    parser.add_argument("--early", type=float, default=1.0)
    parser.add_argument("--early-after", type=float, default=0.5)
    parser.add_argument("--score", default="each", choices=["each", "level"])
    parser.add_argument("--placement", default="keep_local", choices=sorted(ASYNC_PLACEMENTS))
    parser.add_argument("--early-steal", action="store_true")
    parser.add_argument("--tail", type=int, default=0)
    parser.add_argument("--chaos", type=int, help="seed for chaos timing and delivery")
    args = parser.parse_args()

    cfg = SearchConfig(args.N, args.M, args.max_depth, args.max_step_tokens)
    opts = AsyncOptions(args.cap, not args.no_steal, args.early, args.early_after, args.score,
                        args.placement, args.early_steal, args.tail)
    ids = json.loads(SUBSETS_PATH.read_text())["math500"]["subsets"][args.subset][: args.limit]

    def line(problem_id: int, run: AsyncResult, serial: SearchResult) -> tuple[bool, str]:
        from parabeam.core.serial import fingerprint

        same = fingerprint(run.result) == fingerprint(serial)
        r = run.ranks
        return same, (
            f"problem {problem_id:>3}: {len(run.result.levels):>2} levels, "
            f"{run.result.seconds:7.1f} s simulated vs {serial.seconds:7.1f} s serial "
            f"({serial.seconds / run.result.seconds:4.2f}x), stolen {sum(s.stolen for s in r):>3}, "
            f"taken {sum(s.early_taken for s in r):>3}, "
            f"early {sum(s.early_kept for s in r):>4}/{sum(s.early_started for s in r):>4} kept, "
            f"{'match' if same else 'MISMATCH'}")

    def rank_main(problem_id: int, clock=None):
        return lambda comm: _mock_rank(comm, problem_id, cfg, opts, args.chaos, clock)

    matches = 0
    if args.local:
        from parabeam.backends.mock import SimClock
        from parabeam.dist.runtime import run_local

        size = args.local
        for problem_id in ids:
            run = asyncio.run(run_local(size, rank_main(problem_id, SimClock()), args.chaos))[0]
            same, text = line(problem_id, run, asyncio.run(_serial(problem_id, cfg)))
            matches += same
            print(text, flush=True)
    else:
        from parabeam.dist.runtime import MpiComm, abort_on_error

        async def over_mpi() -> tuple[int, int, int]:
            comm = MpiComm()
            found = 0
            for problem_id in ids:
                run = await rank_main(problem_id)(comm)
                if comm.rank == 0:
                    same, text = line(problem_id, run, await _serial(problem_id, cfg))
                    found += same
                    print(text, flush=True)
            await comm.close()
            return comm.rank, comm.size, found

        with abort_on_error():
            rank, size, matches = asyncio.run(over_mpi())
        if rank != 0:
            return
    print(f"{matches} of {len(ids)} match the serial search, P={size}, {opts}")
    sys.exit(0 if matches == len(ids) else 1)


if __name__ == "__main__":
    main()
