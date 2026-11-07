---
type: data-science
complexity: complex
status: in progress
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Rename the rate-penalty coefficient from g^2 to r in every LaTeX file

PI request (2026-11-06): the notes' rate-penalty scale $g$ clashes with the
code's hidden-layer gain `g` (`fct/model.py:92`, `NetConfig.g`); rename it so
that, after `rate-penalty-code`, `train.rate_lambda == r` exactly.

**Reading (confirmed by the PI 2026-11-07).** The code's penalty will be $\frac{\text{rate\_lambda}}{2\beta}\sum\phi^2$
in $\mathcal L$, i.e. $\frac{\text{rate\_lambda}}{2}\sum\phi^2$ in $\beta\mathcal L$,
against the notes' $\frac{g^2}{2}\sum\phi^2$. For `rate_lambda == r` the notes
must write the penalty as $\frac r2\sum\phi^2$, so the rename is $g^2 \to r$
(not $g \to r$, which would give `rate_lambda == r^2`). Consequences:
$G = g^2 I \to G = rI$; $g^2\sigma^2 \to r\sigma^2$; $1/g^2 \to 1/r$;
$g^4 \to r^2$; "$g = 0$" / "$g > 0$" / "no penalty $g$" $\to$ "$r = 0$" etc.;
$\gamma_k = g^2\lambda_0\hat s_k^2 \to r\lambda_0\hat s_k^2$.

## Scope (occurrences of `g^`, approximate)
`sections/02_renormalisation_group.tex` (10), `sections/A1_comments.tex` (7),
`notes/continuum_limit.tex` (29), `notes/fct_experiment_plan.tex` (22),
`notes/linear_solution_limits.tex` (61), `notes/linear_without_ansatz.tex` (27),
`notes/rg_diagonal_reduction.tex` (19), `notes/rg_kernel_normalisation.tex` (4),
`notes/setup_conjugate_rotation.tex` (1), `notes/input_noise_average.tex`
(written with $r$ already; remove its footnote about the pending rename),
plus bare-$g$ prose in `00_motivation.tex`, `A1_comments.tex` and the notes.
Also the scripts under `latex/notes/scripts/` and their tests under `tests/`
(`g2` arguments and variables become `r`), and the plan's dictionary row
("$g^2 = 2\beta\,$`rate_lambda`" becomes "$r = $ `rate_lambda`").
The hidden-layer gain `g` of the code (`NetConfig.g`) is not the same symbol
and stays.

Also: `fct/linear_theory.py` and `tests/test_linear_theory.py` (merged into
`sigma0`) load `latex/notes/scripts/check_linear_soln_limits.py` by path and
call its functions; rename their `g2` arguments to `r` in the script and update
every caller in `tests/` so the suite stays green. The input-noise note
`notes/input_noise_average.tex` already writes $r$; remove its footnote about
the pending rename.

## Acceptance criteria
- `grep -n 'g\^' latex` finds no rate-penalty use (the only remaining `g^`
  are unrelated symbols, listed in the report).
- `make -C latex check` passes; every note and `main.tex` build.
- The scripts' tests pass under the renamed arguments; the quoted numbers in
  every note are unchanged (the rename changes no value).
- A reviewer reading `02_renormalisation_group.tex` eq. "G = r I" and the
  code's loss sees the same coefficient without a factor.
