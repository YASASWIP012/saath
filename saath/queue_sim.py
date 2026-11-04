"""Discrete-event replay of a text helpline queue.

People arrive (Poisson), wait for one of ``counsellors``, and leave unanswered
if their patience runs out. The question we care about is not accuracy but
how many people at levels 4-6 leave before anyone answers, and how long the
ones who stay have to wait.

Every policy sees exactly the same arrivals, service times and patience
draws in a given replay, so differences between policies come from the order
of the queue alone.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from .data import HIGH_RISK


@dataclass(frozen=True)
class Scenario:
    counsellors: int = 4
    load: float = 1.0  # offered load: arrival rate x mean service / counsellors
    mean_service: float = 25.0  # minutes per conversation
    mean_patience: float = 20.0  # minutes before an unanswered person gives up
    hours: float = 12.0
    aging: float = 45.0  # minutes of waiting worth one full unit of risk


def _draws(scn: Scenario, n_pool: int, rng):
    rate = scn.load * scn.counsellors / scn.mean_service
    horizon = scn.hours * 60
    n = rng.poisson(rate * horizon)
    arrive = np.sort(rng.uniform(0, horizon, n))
    who = rng.integers(0, n_pool, n)
    # Lognormal service with the requested mean and a long right tail.
    sigma = 0.6
    service = rng.lognormal(np.log(scn.mean_service) - sigma**2 / 2, sigma, n)
    patience = rng.exponential(scn.mean_patience, n)
    return arrive, who, service, patience


def replay(priority, level, scn: Scenario, seed: int):
    """Run one replay.

    ``priority`` is a per-person urgency in [0, 1] (or None for first come,
    first served). Waiting time is added through ``scn.aging`` so nobody
    starves. Returns a dict of per-arrival arrays.
    """
    rng = np.random.default_rng(seed)
    arrive, who, service, patience = _draws(scn, len(level), rng)
    lv = np.asarray(level)[who]
    prio = np.zeros(len(who)) if priority is None else np.asarray(priority)[who]

    n = len(arrive)
    wait = np.full(n, np.nan)
    abandoned = np.zeros(n, dtype=bool)
    free_at = [0.0] * scn.counsellors
    heapq.heapify(free_at)
    waiting: list[int] = []
    nxt = 0
    t = 0.0

    def pick(now):
        # Drop anyone whose patience ran out before now.
        alive = []
        for i in waiting:
            if arrive[i] + patience[i] < now:
                abandoned[i] = True
            else:
                alive.append(i)
        waiting[:] = alive
        if not waiting:
            return None
        if priority is None:
            best = min(waiting, key=lambda i: arrive[i])
        else:
            best = max(waiting, key=lambda i: (prio[i] + (now - arrive[i]) / scn.aging, -arrive[i]))
        waiting.remove(best)
        return best

    while nxt < n or waiting:
        t_free = free_at[0]
        if nxt < n and arrive[nxt] <= t_free:
            waiting.append(nxt)
            t = arrive[nxt]
            nxt += 1
            continue
        if not waiting:
            # Counsellor idle until the next person arrives.
            heapq.heapreplace(free_at, arrive[nxt])
            continue
        t = max(t, t_free)
        i = pick(t)
        if i is None:
            continue
        heapq.heapreplace(free_at, t + service[i])
        wait[i] = t - arrive[i]

    return {"level": lv, "wait": wait, "abandoned": abandoned}


def outcomes(run) -> dict:
    hr = run["level"] >= HIGH_RISK
    low = ~hr
    served_hr = hr & ~run["abandoned"]
    return {
        "hr_abandon": float(run["abandoned"][hr].mean()) if hr.any() else np.nan,
        "hr_wait_mean": float(np.nanmean(run["wait"][served_hr])) if served_hr.any() else np.nan,
        "hr_wait_p90": float(np.nanpercentile(run["wait"][served_hr], 90)) if served_hr.any() else np.nan,
        "low_abandon": float(run["abandoned"][low].mean()) if low.any() else np.nan,
        "all_abandon": float(run["abandoned"].mean()),
        "n": len(hr),
        "n_hr": int(hr.sum()),
    }


def compare(policies: dict, level, scn: Scenario, seeds=range(200)) -> dict:
    """Run every policy over the same seeds; returns {policy: {metric: array}}."""
    res = {name: [] for name in policies}
    for s in seeds:
        for name, prio in policies.items():
            res[name].append(outcomes(replay(prio, level, scn, seed=s)))
    return {name: {k: np.array([r[k] for r in rows]) for k in rows[0]} for name, rows in res.items()}
