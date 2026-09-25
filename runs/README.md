# `runs/` — the evidence behind the numbers

One directory per training run, named `<date>-<what>`. A directory with a `RESULTS.md` or
`README.md` states the result, the recipe and the command; the rest hold the raw logs. The
book and the top-level README cite runs by directory name.

The ImageNet runs are launched with `lake run <job>` from a config in
[`scripts/jobs/`](../scripts/jobs/); `lake run imagenet` prints the plan and the estimate for
each row.
