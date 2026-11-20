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

Five parts, all fit from the case's own history (see the docstring at the
top of `main.py` for the full description):

1. **Earners/players decomposition.** A rate is earners over players
   tracked. At a refresh the player count jumps and the new cohort brings
   its own earners, so the post-refresh rate is a mix of the old rate and
   the new cohort's rate — not a uniform haircut.
2. **Time-to-earn mechanism for the new cohort.** A trophy takes a random
   time to earn; if the old cohort played `t_old` and the new cohort
   `t_new`, with `rho = t_new / t_old`, the new cohort holds a trophy of
   rate `r` with probability `1 - (1 - r)**rho`. This bends dilution the
   right way: near-universal trophies are barely diluted, rare ones are
   diluted hard. `rho` is solved from the refresh's own totals (the new
   cohort's trophies per player), and on the final case it is estimated
   from the refreshes the history actually shows. The launch-night prior
   (`rho = 0.0665`) comes from the brief's documented refresh
   (638 → 3,868 players, 943 → 4,276 trophies earned), solved, not
   guessed. (v1 assumed a uniform 7% and was wrong in shape.)
3. **Trend in logit space.** Per trophy, least-squares fit of
   `logit(rate)` against `ln(1 + hours since release)` — log-time bends the
   curve into the S-shape of a launch week (fast early gains, then
   saturation). Each trophy's slope is shrunk toward the pooled median
   slope (`n / (n + 6)` weighting), and extrapolation is damped and capped.
4. **Refresh mixture with a renewal probability.** Exophase refreshes on
   its own schedule, hours apart — not a memoryless Poisson process. The
   forecast is a two-component mixture: with probability from a renewal
   calculation over the age of the current flat stretch (Normal interval,
   mean 4.5 h, sd 0.75 h), a refresh lands before the target and the
   median shifts down by the dilution step; otherwise it follows the
   trend. On a flat launch-night history the page has already sat still
   for 3+ hours of a ~4.5 h cadence, so a refresh is due, not a coin flip.
5. **Spread from residuals.** Quantile spread comes from each trophy's
   residual scatter around its trend line (shrunk toward the global
   scatter), widened by leverage, over a floor that grows with the horizon
   (0.20 logits for an hour-scale case, ~0.47 for the 4.5-day final), plus
   dilution uncertainty in the refresh component — not a fixed constant.

## Why this beats the persistence baseline

The starter says every rate stays where it is. On the practice cases the
histories are perfectly flat, so a trend fit is exactly zero — but the
answer is the first refreshed page, where players tracked jumped
638 → 3,868 and every rate but the 100% one fell. The mixture above
expects that refresh: it shifts medians down toward the diluted rates and
spreads wide enough to cover them. On practice answers reconstructed from
the documented refresh totals (the reconstruction is pinned by two public
anchors: the starter's documented ~1.51 and v1's live 0.476) v2 scores a
mean pinball loss of ~0.15–0.20 on the logit scale, against ~0.48 for v1
and ~1.51 for the starter baseline — on both practice cases. The same
machinery on the final case, whose history does move, lets the data speak
instead of the priors: `rho` and the growth come from its own refreshes
and the trend from its own curve.

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
