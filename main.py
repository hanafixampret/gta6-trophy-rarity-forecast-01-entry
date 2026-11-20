"""GTA VI trophy rarity forecaster v2 (make.fun gta6-trophy-rarity-forecast-01).

Reads one case /input/<id>.in and writes /output/<id>.out: for each of the
game's 55 trophies, five quantiles (5/25/50/75/95%) of its Exophase earn rate
at the case's target. Python 3.13, standard library only, no network,
deterministic, one pass per case.

Model. Everything is fit from the case's own history, except two priors
that come from the one refresh the competition brief documents (launch
night: 638 -> 3,868 players tracked, 943 -> 4,276 trophies earned). The
v1 entry got the dilution arithmetic wrong by 10x (it assumed the new
cohort earned each trophy at 7% of the old rate; the totals say ~70%);
v2 fixes that and models dilution per trophy instead of uniformly.

1. Earners/players decomposition.  A rate is earners over players
   tracked. At a refresh the player count jumps by a factor (1 + g) and
   the new cohort brings its own earners. Write r for a trophy's old rate
   (as a fraction) and n for the new cohort's earn fraction for it. The
   post-refresh rate is (r + n*g) / (1 + g). The old code applied one
   uniform alpha to every trophy; that is wrong in a specific direction:
   a trophy earned early in the game is already held by most of the new
   cohort, a trophy that takes hours is held by almost none of them.

2. Time-to-earn mechanism for the new cohort.  A trophy takes a random
   time to earn; a player with playtime t holds it with probability
   1 - exp(-lambda*t). If the old cohort played t_old and the new cohort
   t_new, with rho = t_new / t_old, then n = 1 - (1 - r)**rho. This one
   parameter bends the dilution the right way: n/r is ~rho for rare
   trophies and ~1 for near-universal ones. rho is SOLVED from the totals
   the brief documents (the new cohort's trophies per player pins the sum
   of the n's), giving RHO_PRIOR = 0.0665 -- and the same solve runs on
   every refresh the history actually shows, so on the final case rho is
   estimated from the case's own refreshes (median, shrunk to the prior)
   instead of assumed. The growth g at a refresh is estimated the same
   way (median over observed refreshes, shrunk to the documented 5.06).

3. Trend in logit space.  For each trophy, regress logit(rate) on
   x = ln(1 + hours since release). Log-time already bends the curve:
   early gains are fast, later gains slow -- the S-curve shape of a
   launch week. Each trophy's slope is shrunk toward the pooled (median)
   slope with weight n / (n + 6), so a trophy with a flat or noisy
   history follows the global curve, and extrapolation is damped by
   1 / (1 + dx / 3) and capped, so a short hot streak cannot launch a
   rare trophy to 90% in a week. The extrapolated median is also
   winsorized to the trophy's own observed logit range, plus or minus
   2 logits.

4. Refresh mixture.  The forecast is a two-part mixture: with
   probability p from a renewal calculation over the age of the current
   flat stretch (Normal refresh interval, mean 4.5 h, sd 0.75 h -- a
   scheduled cadence is fairly regular), a refresh lands before the
   target and the median shifts down by the dilution step the mechanism
   of (2) implies, from the trend rate at the target; otherwise it
   follows the trend. When the history already contains refreshes, the
   trend has priced dilution in, so the step is scaled to 25% and the
   mixture mainly adds honest spread. Near-universal trophies are not
   diluted: the mechanism gives n = 1 at r = 1, matching the documented
   refresh, where the 100% trophy did not move.

5. Spread from residuals.  The base spread is the trophy's residual
   scatter around its trend line (shrunk toward the global scatter),
   widened by leverage at the target, over a floor that grows with the
   horizon (0.20 logits for an hour-scale case, ~0.47 for the 4.5-day
   final); the refresh component adds a further 0.35 logit of dilution
   uncertainty in quadrature. Quantiles of the mixture are found by
   bisection on its CDF -- no sampling, so two runs agree byte for byte.

Why this beats v1 on the practice cases: their histories are perfectly
flat, so any trend fit is exactly zero and the persistence baseline is
all a trend model can say -- but the answer is the first refreshed page.
v1 expected that refresh with a uniform alpha of 0.07, diluting every
rate to ~22% of its old value; the totals say the new cohort earned at
~70% of the old per-player rate, and the mechanism says the dilution is
much weaker for common trophies than for rare ones. v2's median is the
mechanism's post-refresh rate, and its spread is tight enough to be
honest about a 90-minute horizon. The same machinery on the final case,
whose history does move, lets the data speak instead of the priors: rho
and g come from its own refreshes and the trend from its own curve.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Priors for a refresh regime the history has not shown yet. They are the
# launch-night refresh the brief documents (638 -> 3,868 players tracked,
# 943 -> 4,276 trophies earned), solved, not guessed:
#   RHO solves sum_j (1 - (1 - r_j)**rho) = (4276 - 943) / (3868 - 638)
#   over the launch-night rates, i.e. the new cohort's trophies per player
#   matches the documented totals exactly: rho = 0.0665.
#   GROWTH is 3868 / 638 - 1 = 5.06.
RHO_PRIOR = 0.0665
GROWTH_PRIOR = 5.06
# Refresh cadence. Exophase refreshes a game's counts on its own schedule,
# hours apart, not continuously: the brief documents the page flat from the
# first probe (23:48Z) through the 04:00Z read, then refreshed by 04:30Z,
# so the interval is about 4.5 hours. A refresh is therefore not a Poisson
# event with a memoryless wait: after the page has sat flat for most of an
# interval, the next refresh is DUE. The forecast below uses a renewal
# calculation (Normal interval, mean 4.5 h, sd 0.75 h -- a scheduled
# cadence is fairly regular) over the age of the current flat stretch,
# instead of the Poisson 1 - exp(-horizon/interval), which ignores how
# long the page has already sat still and understates the refresh
# probability exactly when the history is flat.
REFRESH_INTERVAL_PRIOR_H = 4.5
REFRESH_INTERVAL_SD_PRIOR_H = 0.75
# How much of the dilution step still applies once the history already
# contains refreshes (the trend has then priced dilution in).
DILUTION_SCALE_AFTER_REFRESH = 0.25
# Winsorize for the extrapolated trend median: how far, in logits, the
# forecast median may sit outside the trophy's own observed logit range.
WINSORIZE_LOGIT = 2.0
# Spread: base floor in logit units, growing with the horizon; extra
# dilution uncertainty in the refresh component.
SIGMA_FLOOR_BASE = 0.20
SIGMA_FLOOR_PER_DAY = 0.06
SIGMA_DILUTION = 0.35

RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)


def parse_time(value):
    """ISO-8601 timestamp (or bare date) -> aware datetime, or None."""
    if not isinstance(value, str):
        return None
    try:
        if len(value) == 10:
            return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, as the starter does."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return None
    if n % 2:
        return vals[n // 2]
    return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def ols(xs, ys):
    """Least-squares line; returns (slope, intercept, resid_sd, x_mean, sxx)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return 0.0, my, 0.0, mx, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    if n > 2:
        rss = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
        resid = math.sqrt(max(0.0, rss) / (n - 2))
    else:
        resid = 0.0
    return slope, intercept, resid, mx, sxx


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mixture_quantile(mu, sigma, weight, delta, sigma_refresh, tau):
    """tau-quantile of (1-w) * N(mu, sigma) + w * N(mu + delta, sigma_refresh).

    Bisection on the CDF: deterministic, no sampling.
    """
    if weight <= 0.001:
        return mu + Z[LEVELS.index(tau)] * sigma
    span = 8.0 * max(sigma, sigma_refresh) + abs(delta) + 2.0
    lo, hi = mu - span, mu + span
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        cdf = (1.0 - weight) * norm_cdf((mid - mu) / sigma) + weight * norm_cdf(
            (mid - mu - delta) / sigma_refresh
        )
        if cdf < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_rho(old_fracs, target_sum):
    """rho in (0, 1) with sum_j (1 - (1 - r_j)**rho) == target_sum.

    Bisection: the sum is strictly increasing in rho. Falls back to the
    ends when the target is outside the attainable range.
    """
    if target_sum <= 0.0:
        return 0.001
    if target_sum >= len(old_fracs):
        return 0.999
    lo, hi = 0.001, 0.999
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        total = 0.0
        for r in old_fracs:
            total += 1.0 - (1.0 - r) ** mid
        if total < target_sum:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    horizon_min = float(case["horizon_minutes"])
    days = horizon_min / 1440.0

    release = parse_time(case.get("game", {}).get("release_date")) or RELEASE
    if release.tzinfo is None:
        release = release.replace(tzinfo=timezone.utc)
    target = parse_time(case.get("target"))
    checkpoint = parse_time(case.get("at"))

    def hours_since_release(when):
        if when is None:
            return None
        return max(-0.9, (when - release).total_seconds() / 3600.0)

    # Time axis: x = ln(1 + hours since release) for every read and target.
    xs_hist = []
    for read in history:
        h = hours_since_release(parse_time(read.get("at")))
        xs_hist.append(math.log1p(h) if h is not None else 0.0)
    if target is not None:
        x_target = math.log1p(hours_since_release(target))
    elif checkpoint is not None:
        x_target = math.log1p(hours_since_release(checkpoint) + horizon_min / 60.0)
    else:
        x_target = xs_hist[-1] + math.log1p(horizon_min / 60.0)
    x_last = xs_hist[-1]

    # ---- refresh events in the history: what dilution looks like here ----
    # A refresh is any read where the player count or any rate moved: the
    # page is static between refreshes, so a change is a refresh. For each
    # refresh with player growth, solve rho from that refresh's own totals
    # (the new cohort's trophies per player) -- the same solve the prior
    # comes from, run on the case's own data.
    rho_samples = []
    growth_samples = []
    refresh_times = []
    for prev, cur in zip(history, history[1:]):
        p_prev, p_cur = prev.get("players_tracked"), cur.get("players_tracked")
        rates_prev = prev.get("rates") or []
        rates_cur = cur.get("rates") or []
        changed = p_prev != p_cur
        if not changed:
            changed = any(
                a != b
                for a, b in zip(rates_prev, rates_cur)
                if a is not None and b is not None
            )
        if not changed:
            continue
        t_cur = parse_time(cur.get("at"))
        if t_cur is not None:
            refresh_times.append(t_cur)
        if p_prev and p_cur and p_cur > p_prev:
            growth_samples.append(p_cur / p_prev - 1.0)
            old_fracs = []
            new_sum = 0.0
            denom = p_cur - p_prev
            for a, b in zip(rates_prev, rates_cur):
                if a is None or b is None:
                    continue
                old_fracs.append(a / 100.0)
                earned_prev = a / 100.0 * p_prev
                earned_cur = b / 100.0 * p_cur
                new_sum += (earned_cur - earned_prev) / denom
            if old_fracs and new_sum > 0.0:
                rho_samples.append(solve_rho(old_fracs, new_sum))
    n_refresh = len(refresh_times)

    if rho_samples:
        rho_hat = median(rho_samples)
        rho = (len(rho_samples) * rho_hat + 2.0 * RHO_PRIOR) / (
            len(rho_samples) + 2.0
        )
    else:
        rho = RHO_PRIOR
    rho = min(0.99, max(0.005, rho))

    if growth_samples:
        growth_hat = median(growth_samples)
        growth = (len(growth_samples) * growth_hat + 2.0 * GROWTH_PRIOR) / (
            len(growth_samples) + 2.0
        )
    else:
        growth = GROWTH_PRIOR
    growth = min(20.0, max(0.0, growth))

    if len(refresh_times) >= 2:
        gaps = [
            (b - a).total_seconds() / 3600.0
            for a, b in zip(refresh_times, refresh_times[1:])
            if (b - a).total_seconds() > 0
        ]
        interval_h = median(gaps) if gaps else REFRESH_INTERVAL_PRIOR_H
        interval_h = min(24.0, max(1.0, interval_h))
        interval_sd = max(0.5, 0.25 * interval_h)
    else:
        interval_h = REFRESH_INTERVAL_PRIOR_H
        interval_sd = REFRESH_INTERVAL_SD_PRIOR_H

    # Probability at least one refresh lands before the target, as a
    # renewal calculation: the interval since the last refresh is already
    # `age` hours old (or the whole flat stretch, when no refresh has been
    # seen yet), and the answer is read in the hour from the target, so
    # the window is the horizon plus 30 minutes. p = P(age < T <= age+h
    # | T > age) for an interval T ~ Normal(interval_h, interval_sd).
    # On a flat launch-night history the page has already sat still for
    # 3+ hours of a ~4.5 h cadence, so a refresh is due, not a coin flip;
    # on a moving history the age is short and the horizon is days, so
    # this is ~1 there too.
    if checkpoint is not None:
        last_seen = refresh_times[-1] if refresh_times else parse_time(history[0].get("at"))
        age_h = 0.0
        if last_seen is not None:
            age_h = max(0.0, (checkpoint - last_seen).total_seconds() / 3600.0)
    else:
        age_h = 0.0
    window_h = (horizon_min + 30.0) / 60.0

    def _norm_cdf_std(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    z_age = (age_h - interval_h) / interval_sd
    z_end = (age_h + window_h - interval_h) / interval_sd
    denom = 1.0 - _norm_cdf_std(z_age)
    if denom <= 1e-9:
        p_refresh = 0.99
    else:
        p_refresh = (_norm_cdf_std(z_end) - _norm_cdf_std(z_age)) / denom
    p_refresh = min(0.99, max(0.02, p_refresh))
    dilution_scale = 1.0 if n_refresh == 0 else DILUTION_SCALE_AFTER_REFRESH

    # ---- per-trophy trend fits in logit space ----
    series = []  # (xs, ys, last_rate, last_logit, lo_logit, hi_logit) per trophy
    last_rates = []
    for i in range(len(trophies)):
        xs, ys = [], []
        last_rate = None
        for x, read in zip(xs_hist, history):
            rates = read.get("rates") or []
            rate = rates[i] if i < len(rates) else None
            if rate is not None:
                xs.append(x)
                ys.append(logit(rate))
                last_rate = rate
        if last_rate is None:
            last_rate = 0.01
        last_rates.append(last_rate)
        lo_logit = min(ys) if ys else logit(last_rate)
        hi_logit = max(ys) if ys else logit(last_rate)
        series.append((xs, ys, last_rate, logit(last_rate), lo_logit, hi_logit))
    fallback_rate = median(last_rates) or 0.5

    fits = []
    for xs, ys, _rate, _lg, _lo, _hi in series:
        if len(xs) >= 4:
            fits.append(ols(xs, ys))
        else:
            fits.append(None)
    valid_slopes = [f[0] for f in fits if f is not None]
    valid_resids = [f[2] for f in fits if f is not None]
    global_slope = median(valid_slopes) if valid_slopes else 0.0
    global_resid = median(valid_resids) if valid_resids else 0.25

    sigma_floor = SIGMA_FLOOR_BASE + SIGMA_FLOOR_PER_DAY * days

    rows = []
    for idx, trophy in enumerate(trophies):
        xs, ys, last_rate, last_logit, lo_logit, hi_logit = series[idx]
        if last_rate <= 0.01:
            last_rate = fallback_rate
            last_logit = logit(last_rate)
        fit = fits[idx]
        if fit is not None:
            slope, _intercept, resid, x_mean, sxx = fit
            n = len(xs)
            weight = n / (n + 6.0)
            slope = weight * slope + (1.0 - weight) * global_slope
            resid_shrunk = math.sqrt(
                (n * resid * resid + 8.0 * global_resid * global_resid) / (n + 8.0)
            )
            leverage = 1.0
            if sxx > 1e-12:
                leverage = math.sqrt(1.0 + (x_target - x_mean) ** 2 / sxx)
        else:
            slope, resid_shrunk, leverage, n = global_slope, global_resid, 1.5, 0

        # Trend median: damped log-time extrapolation, moves capped.
        dx = x_target - x_last
        damp = 1.0 / (1.0 + max(0.0, dx) / 3.0)
        move = slope * dx * damp
        move = min(3.0, max(-2.0, move))
        mu = last_logit + move
        mu = min(logit(99.99), max(logit(0.01), mu))
        # Winsorize: the extrapolated median may not leave the range this
        # trophy has actually shown by more than WINSORIZE_LOGIT logits.
        mu = min(hi_logit + WINSORIZE_LOGIT, max(lo_logit - WINSORIZE_LOGIT, mu))

        # Spread: residual scatter (with leverage) over a horizon floor.
        sigma_fit = resid_shrunk * leverage
        sigma = math.sqrt(sigma_fit * sigma_fit + sigma_floor * sigma_floor)
        sigma_refresh = math.sqrt(sigma * sigma + SIGMA_DILUTION * SIGMA_DILUTION)

        # Dilution step for this trophy at the next refresh, from the
        # time-to-earn mechanism: the new cohort holds the trophy with
        # probability 1 - (1 - r)**rho, and the refreshed rate mixes the
        # trend rate with the new cohort at the refresh's growth. A
        # saturated trophy (r = 100%) is held by the whole new cohort too
        # (n = 1): the mechanism's (1 - r)**rho is meaningless at the
        # clamped 99.99, so the boundary is handled explicitly, matching
        # the documented refresh, where the 100% trophy did not move.
        r_trend = inv_logit(mu) / 100.0
        if r_trend >= 0.99:
            n_cohort = 1.0
        else:
            n_cohort = 1.0 - (1.0 - r_trend) ** rho
        y_refresh = (r_trend + n_cohort * growth) / (1.0 + growth)
        delta = (logit(100.0 * y_refresh) - mu) * dilution_scale

        qs = []
        prev_q = None
        for tau in LEVELS:
            lg = mixture_quantile(mu, sigma, p_refresh, delta, sigma_refresh, tau)
            q = round4(min(100.0, max(0.01, inv_logit(lg))))
            if prev_q is not None and q < prev_q:
                q = prev_q
            prev_q = q
            qs.append(q)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
