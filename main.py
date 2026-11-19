"""GTA VI trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads one case /input/<id>.in and writes /output/<id>.out: for each of the
game's 55 trophies, five quantiles (5/25/50/75/95%) of its Exophase earn rate
at the case's target. Python 3.13, standard library only, no network,
deterministic, one pass per case.

Model, in three parts. Everything is fit from the case's own history; the
only numbers fixed in advance are priors for a regime the history cannot
show yet (a page that has never refreshed), and those priors come from the
one refresh the brief documents: on launch night Exophase's page sat
unchanged (638 players tracked) from 23:48Z to 04:00Z, then refreshed to
3,868 players with every rate but the 100% one lower.

1. Trend in logit space.  For each trophy, regress logit(rate) on
   x = ln(1 + hours since release).  Log-time already bends the curve:
   early gains are fast, later gains slow -- the S-curve shape of a launch
   week.  Each trophy's slope is shrunk toward the pooled (median) slope
   with weight n / (n + 6), so a trophy with a flat or noisy history
   follows the global curve, and extrapolation is damped by
   1 / (1 + dx / 3) and capped, so a short hot streak cannot launch a
   rare trophy to 90% in a week.

2. Refresh dilution.  Exophase does not refresh continuously: the page
   freezes, then jumps.  At a refresh the tracked-player count jumps and
   the new cohort has earned each trophy at only a fraction alpha of the
   old rate, so rates fall by the factor (1 + g*alpha) / (1 + g), where g
   is the fractional player growth at the refresh.  Refresh events are
   detected in the history (any change in players tracked or in any rate)
   and alpha, g and the refresh interval are estimated from them
   (median over events, shrunk toward the launch-night priors
   alpha = 0.07, g = 5.0, interval = 3 h).  The forecast is a two-part
   mixture: with probability p = 1 - exp(-(horizon + 30 min) / interval)
   a refresh lands before the target and the median shifts down by the
   dilution step; otherwise it follows the trend.  When the history
   already contains refreshes, the trend has priced dilution in, so the
   step is scaled to 25% and the mixture mainly adds honest spread.
   Near-universal trophies (rate ~100%) are not diluted: alpha grows
   with the rate itself, alpha_eff = alpha + (1 - alpha) * (r/100)^2,
   which is 1 at r = 100, matching the documented refresh.

3. Spread from residuals.  The base spread is the trophy's residual
   scatter around its trend line (shrunk toward the global scatter),
   widened by leverage at the target and by a horizon term, with a
   0.55-logit floor; the refresh component adds a further 0.35 logit of
   dilution uncertainty in quadrature.  Quantiles of the mixture are
   found by bisection on its CDF -- no sampling, so two runs agree
   byte for byte.

A note on the practice cases: their histories are perfectly flat (every
read identical), so any trend fit is exactly zero and the persistence
baseline is all a trend model can say.  The mixture above is what beats
it there: it expects the refresh the flat page is hiding, shifts the
median down toward the diluted rates, and spreads wide enough to cover
them.  The same machinery on the final case, whose history does move,
   lets the data speak instead of the priors.
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

# Priors for a refresh regime the history has not shown yet.  They are the
# launch-night refresh the brief documents (638 -> 3,868 players tracked;
# the new cohort earned the non-trivial trophies at ~7% of the old rate),
# rounded: alpha 0.07, player growth x5, one refresh about every 3 hours.
ALPHA_PRIOR = 0.07
GROWTH_PRIOR = 5.0
REFRESH_INTERVAL_PRIOR_H = 3.0
# How much of the dilution step still applies once the history already
# contains refreshes (the trend has then priced dilution in).
DILUTION_SCALE_AFTER_REFRESH = 0.25
# Spread: floor in logit units, extra dilution uncertainty, horizon term.
SIGMA_FLOOR = 0.55
SIGMA_DILUTION = 0.35
SIGMA_HORIZON_PER_SQRT_DAY = 0.18

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
    alpha_samples = []
    growth_samples = []
    refresh_times = []
    for prev, cur in zip(history, history[1:]):
        p_prev, p_cur = prev.get("players_tracked"), cur.get("players_tracked")
        changed = p_prev != p_cur
        if not changed:
            changed = any(
                a != b
                for a, b in zip(prev.get("rates") or [], cur.get("rates") or [])
                if a is not None and b is not None
            )
        if not changed:
            continue
        t_cur = parse_time(cur.get("at"))
        if t_cur is not None:
            refresh_times.append(t_cur)
        if p_prev and p_cur and p_cur > p_prev:
            growth_samples.append(p_cur / p_prev - 1.0)
            denom = p_cur - p_prev
            for a, b in zip(prev.get("rates") or [], cur.get("rates") or []):
                if a is None or b is None or not (0.2 <= a <= 50.0):
                    continue
                earned_prev = a / 100.0 * p_prev
                earned_cur = b / 100.0 * p_cur
                new_rate = (earned_cur - earned_prev) / denom
                alpha_samples.append(min(1.5, max(0.0, new_rate / (a / 100.0))))
    n_refresh = len(refresh_times)

    if alpha_samples:
        alpha_hat = median(alpha_samples)
        alpha = (len(alpha_samples) * alpha_hat + 25.0 * ALPHA_PRIOR) / (
            len(alpha_samples) + 25.0
        )
    else:
        alpha = ALPHA_PRIOR
    alpha = min(1.0, max(0.01, alpha))

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
    else:
        interval_h = REFRESH_INTERVAL_PRIOR_H

    # Probability at least one refresh lands before the target (the answer
    # is read in the hour from the target, so add 30 minutes of window).
    p_refresh = 1.0 - math.exp(-(horizon_min + 30.0) / (interval_h * 60.0))
    dilution_scale = 1.0 if n_refresh == 0 else DILUTION_SCALE_AFTER_REFRESH

    # ---- per-trophy trend fits in logit space ----
    series = []  # (xs, ys, last_rate, last_logit) per trophy
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
        series.append((xs, ys, last_rate, logit(last_rate)))
    fallback_rate = median(last_rates) or 0.5

    fits = []
    for xs, ys, _rate, _lg in series:
        if len(xs) >= 4:
            fits.append(ols(xs, ys))
        else:
            fits.append(None)
    valid_slopes = [f[0] for f in fits if f is not None]
    valid_resids = [f[2] for f in fits if f is not None]
    global_slope = median(valid_slopes) if valid_slopes else 0.0
    global_resid = median(valid_resids) if valid_resids else 0.25

    rows = []
    for idx, trophy in enumerate(trophies):
        xs, ys, last_rate, last_logit = series[idx]
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

        # Spread: residual scatter (with leverage) + horizon term, floored.
        sigma_fit = resid_shrunk * leverage
        sigma = max(
            SIGMA_FLOOR,
            math.sqrt(sigma_fit * sigma_fit + (SIGMA_HORIZON_PER_SQRT_DAY ** 2) * days),
        )
        sigma_refresh = math.sqrt(sigma * sigma + SIGMA_DILUTION * SIGMA_DILUTION)

        # Dilution step for this trophy at the next refresh.
        alpha_eff = alpha + (1.0 - alpha) * (last_rate / 100.0) ** 2
        factor = (1.0 + growth * alpha_eff) / (1.0 + growth)
        delta = (logit(last_rate * factor) - last_logit) * dilution_scale

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
