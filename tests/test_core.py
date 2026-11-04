import numpy as np
import pytest

from saath import conformal, models, queue_sim
from saath.data import HIGH_RISK, N_LEVELS
from saath.models import cumulative_targets


def test_level_proba_is_a_distribution_even_when_heads_cross():
    exceed = np.array([[0.9, 0.95, 0.2, 0.3, 0.1, 0.0]])  # head 1 > head 0: crossed
    p = models.to_level_proba(exceed)
    assert p.shape == (1, N_LEVELS)
    assert np.all(p >= 0)
    assert p.sum() == pytest.approx(1)


def test_predict_level_is_median_of_distribution():
    exceed = np.array([[0.99, 0.9, 0.8, 0.6, 0.4, 0.1]])
    assert models.predict_level(exceed)[0] == 4


def test_keyword_rule_picks_highest_matching_level():
    kw = models.KeywordRule()
    assert kw.levels(["I took all the pills"])[0] == 6
    assert kw.levels(["lovely day at the beach"])[0] == 0
    assert kw.levels(["sometimes I wish I were dead"])[0] == 1


def test_cumulative_targets_blend_jury_votes():
    y = np.array([4])
    jury = np.array([[4, 4, 2, np.nan, 6]])
    t = cumulative_targets(y, jury, jury_weight=0.5)
    # Threshold "level > 3": human yes (1.0), jury 3 of 4 yes (0.75).
    assert t[0, 3] == pytest.approx(0.5 * 1 + 0.5 * 0.75)
    # Threshold "level > 4": human no, jury 1 of 4 yes.
    assert t[0, 4] == pytest.approx(0.5 * 0.25)
    assert np.array_equal(cumulative_targets(y), (4 > np.arange(6))[None].astype(np.float32))


def test_urgent_threshold_keeps_its_promise_on_exchangeable_data():
    rng = np.random.default_rng(0)
    misses = []
    for _ in range(300):
        level = rng.integers(0, N_LEVELS, 400)
        score = np.clip((level >= HIGH_RISK) * 0.3 + rng.normal(0.4, 0.2, 400), 0, 1)
        cal, test = slice(0, 300), slice(300, 400)
        u = conformal.UrgentThreshold(0.1).fit(score[cal], level[cal])
        hr = level[test] >= HIGH_RISK
        misses.append((~u.urgent(score[test]))[hr].mean())
    assert np.mean(misses) <= 0.1 + 0.01


def test_urgent_threshold_flags_everyone_when_calibration_is_too_small():
    u = conformal.UrgentThreshold(0.05).fit(np.array([0.9, 0.8]), np.array([5, 6]))
    assert u.urgent(np.array([0.0])).all()


def test_level_sets_cover_and_are_contiguous():
    rng = np.random.default_rng(1)
    n = 3000
    level = rng.integers(0, N_LEVELS, n)
    logits = rng.normal(0, 1, (n, N_LEVELS))
    logits[np.arange(n), level] += 2
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    ls = conformal.LevelSets(0.1).fit(p[:2000], level[:2000])
    sets = ls.predict(p[2000:])
    assert np.all(sets[:, 0] <= sets[:, 1])
    assert conformal.covered(sets, level[2000:]).mean() >= 0.88


def test_queue_every_arrival_is_served_or_abandons():
    level = np.array([0, 2, 4, 6] * 50)
    scn = queue_sim.Scenario(load=1.2, hours=4)
    run = queue_sim.replay(level / 6, level, scn, seed=3)
    served = ~np.isnan(run["wait"])
    assert np.all(served ^ run["abandoned"])
    assert np.all(run["wait"][served] >= 0)


def test_queue_policies_see_identical_arrivals():
    level = np.array([0, 5] * 100)
    scn = queue_sim.Scenario(hours=3)
    a = queue_sim.replay(None, level, scn, seed=7)
    b = queue_sim.replay(level / 6, level, scn, seed=7)
    assert np.array_equal(a["level"], b["level"])


def test_oracle_order_does_not_lose_more_high_risk_people_than_fifo():
    level = np.random.default_rng(0).integers(0, N_LEVELS, 500)
    res = queue_sim.compare(
        {"fifo": None, "oracle": (level >= HIGH_RISK).astype(float)},
        level,
        queue_sim.Scenario(load=1.2),
        seeds=range(40),
    )
    assert res["oracle"]["hr_abandon"].mean() < res["fifo"]["hr_abandon"].mean()


def test_level_sets_by_level_cover_every_level():
    rng = np.random.default_rng(2)
    n = 6000
    level = rng.integers(0, N_LEVELS, n)
    logits = rng.normal(0, 1, (n, N_LEVELS))
    logits[np.arange(n), level] += 1.5 + level * 0.3  # some levels easier than others
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    ls = conformal.LevelSets(0.1, by_level=True).fit(p[:4000], level[:4000])
    cov = conformal.covered(ls.predict(p[4000:]), level[4000:])
    for v in range(N_LEVELS):
        assert cov[level[4000:] == v].mean() >= 0.85


def test_risk_dependent_patience_changes_only_high_risk_texters():
    level = np.array([0, 6] * 100)
    base = queue_sim.replay(None, level, queue_sim.Scenario(hours=4), seed=5)
    impatient = queue_sim.replay(None, level, queue_sim.Scenario(hours=4, hr_patience=0.25), seed=5)
    hr = base["level"] >= HIGH_RISK
    assert impatient["abandoned"][hr].mean() >= base["abandoned"][hr].mean()
