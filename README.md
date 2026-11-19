[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hanafixampret/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hanafixampret/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast

Entry for the make.fun bounty **"Forecast how many GTA VI players earn each trophy in week one"**
— https://make.fun/bounty/gta6-trophy-rarity-forecast-01

<sub>Entered on [make.fun](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>

`main.py` (Python 3.13, standard library only, no network) reads one case
`/input/<id>.in` and writes `/output/<id>.out`: for each of the game's 55
trophies, five quantiles (5%, 25%, 50%, 75%, 95%) of its Exophase earn rate
at the case's target. Deterministic, well under the 8-second limit.

## How it forecasts

Three parts, all fit from the case's own history (see the docstring at the
top of `main.py` for the full description):

1. **Trend in logit space.** Per trophy, least-squares fit of
   `logit(rate)` against `ln(1 + hours since release)` — log-time bends the
   curve into the S-shape of a launch week (fast early gains, then
   saturation). Each trophy's slope is shrunk toward the pooled median
   slope (`n / (n + 6)` weighting), and extrapolation is damped and capped.
2. **Refresh dilution.** Exophase refreshes discretely: the page freezes,
   then jumps as newly tracked players (who have earned fewer trophies)
   dilute every rate. Refresh events are detected in the history and the
   new cohort's earn fraction, the player-growth jump, and the refresh
   interval are estimated from them (with launch-night priors when the
   history shows no refresh yet). The forecast is a two-component mixture:
   with probability `1 - exp(-(horizon + 30 min) / interval)` a refresh
   lands before the target and the median shifts down by the dilution step.
3. **Spread from residuals.** Quantile spread comes from each trophy's
   residual scatter around its trend line (shrunk toward the global
   scatter), widened by leverage and horizon, plus dilution uncertainty
   in the refresh component — not a fixed constant.

## Why this beats the persistence baseline

The starter says every rate stays where it is. On the practice cases the
histories are perfectly flat, so a trend fit is exactly zero — but the
answer is the first refreshed page, where players tracked jumped
638 → 3,868 and every rate but the 100% one fell. The mixture above
expects that refresh: it shifts medians down toward the diluted rates and
spreads wide enough to cover them. On the reconstructed practice answers
(rebuilt from the documented refresh totals: 638 → 3,868 players,
943 → 4,276 trophies earned) it scores a mean pinball loss of ~0.40 on
the logit scale, against ~1.40–1.51 for the starter baseline — on both
practice cases.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

From the kit directory, with the practice cases in `input/`:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT — see [LICENSE](LICENSE).
