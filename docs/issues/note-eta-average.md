---
type: data-science
complexity: simple
status: done
---

<!-- status: todo | in progress | in review | done | blocked |
     awaiting your confirmation -->

# Write the note on the input-noise average (annealed, quenched, fresh draws) and point to it from the setup section

PI decision (2026-11-06): the notes' $\eta$ is a formal regulariser and every
analysis and experiment is stated in the limit $\sigma \to 0^+$; the `fct`
code has no $\eta$ at all for now. The reasoning behind that decision, and
the proper treatment should $\sigma > 0$ ever be needed, must be written
down once so nobody re-derives it. The plan `latex/notes/fct_experiment_plan.tex`
(Section D2 and the remark "Fresh draws are the wrong sampler") contains the
seed of the argument; this note is its full, self-contained form.

## Inputs
- `latex/sections/01_setup.tex` (`eq:ff-network`, the paragraph "Input Noise",
  `eq:single-site-reg`); `latex/sections/02_renormalisation_group.tex` (the
  first-layer covariance $\Phi^{(0)} + \sigma^2 I$).
- `latex/notes/linear_solution_limits.tex`: the implicit solution, Lemma on the
  floor ($d_q = \sigma_0^2 \equiv \sigma^2/(1+g^2\sigma^2)$ for null modes), and
  Section `sec:sigma0` ($\sigma \to 0$ is regular; floor $0$; null modes $d_q = 0$).
- `latex/notes/fct_experiment_plan.tex`, Section D2 and the remark.
- `latex/macros.tex` (`\major`, `\majornote`), another note (e.g.
  `setup_site_factorisation.tex`) as the preamble pattern, `latex/Makefile`
  (`make note NOTE=<name>`, `make check`).

## Outputs
- `latex/notes/input_noise_average.tex`, standalone article in the style of the
  other notes, building with `make -C latex note NOTE=input_noise_average`.
- In `01_setup.tex`, directly after the sentence "... and so suppress it in the
  notation for the single-site below.", one `\major{...}` sentence stating that
  the expectation taken in `eq:single-site-reg` is the annealed average (eta a
  sampled variable of the joint Gibbs measure), that a network with a fixed or a
  freshly drawn noise realises a different measure, and that all predictions
  are taken at sigma -> 0+ where the three coincide, followed by
  `\majornote{\detokenize{input_noise_average.tex}}`.

## Required content of the note
1. **The three measures.** Write the network of `eq:ff-network` with
   $h^{(1)} = Ux/\sqrt m + \eta$ and the loss $\mathcal L(\Theta, \eta)$.
   (a) *Annealed* (the notes): $\mathcal Z = \int \mathcal D\Theta\, \mathcal D\eta\,
   e^{-\beta \mathcal L(\Theta,\eta)} e^{-\|\eta\|^2/2\sigma^2}$; $\eta$ is a sampled
   variable whose posterior is tilted by the loss; `eq:single-site-reg` is this
   Gaussian integral and it is what gives $\Phi^{(0)} + \sigma^2 I$.
   (b) *Quenched* (one fixed draw per dataset, the same at every step):
   the Gibbs measure over $\Theta$ at fixed $\eta$, averaged over $\eta$ outside
   the log. (c) *Fresh draws* (a new $\eta$ at every SGLD step): the drift
   averages to $-\nabla_\Theta \mathbb E_\eta \mathcal L$, so the stationary law
   is $e^{-\beta \mathbb E_\eta \mathcal L(\Theta,\eta)}$, the averaged-loss law;
   for the linear net $\mathbb E_\eta \mathcal L = \mathcal L(\Theta,0) +
   \frac{N}{2Pn}\sum_{\mu,\alpha}\mathrm{Var}_\eta f^\mu_\alpha + \text{const}$,
   with $\mathrm{Var}_\eta f^\mu_\alpha = \sigma^2 \sum_i V_{i\alpha}^2/N^2$: a
   ridge on the readout.
2. **Where they differ, for the linear one-hidden-layer net.** Use the
   limits note's language (modes, $\lambda_q$, $s_X^2$, $s_Y^2$, the rate
   penalty). The sharpest statement is the null floor: the $P-K$ directions of
   $h^{(1)}$ outside $\mathrm{col}(Z)$ are pure $\eta$. Annealed: the rate
   penalty shrinks them to $\sigma_0^2 = \sigma^2/(1 + r\sigma^2)$ (the limits
   note's floor lemma, with $r$ the rate coefficient). Quenched: the weights
   can move $h^{(1)}$ only inside the $K$-dimensional latent span, so the null
   level stays at $\sigma^2$, untouched by $r$; inside the span the frozen
   $\eta$ acts as a random shift of the prior mean of $W^{(0)}$. Fresh draws:
   no null floor at all in the tuning map of the noise-free output, and a
   ridge on $V$. State that the three agree as $\sigma \to 0$ and that the
   difference is $O(1)$ at $\sigma^2 = 1$, $r = 1$ (floor $0.5$ against $1$).
3. **The proper treatment at $\sigma > 0$.** Two exact samplers of the
   annealed measure. (i) $\eta = \sigma\zeta$ with $\zeta \in \mathbb R^{P\times N}$
   a parameter carrying the term $\|\zeta\|^2/(2\beta)$ in $\mathcal L$, updated
   by SGLD with the weights (the plan's D2); works for any activation; costs a
   $P \times N$ parameter. (ii) For a linear net, marginalise $\eta$ in closed
   form. With $\bar h^\mu = Ux^\mu/\sqrt m$, $r^\mu = V^\top \bar h^\mu/N - y^\mu$,
   $c = \sigma^{-2} + r$: per pattern the exponent is $-\tfrac12 \eta^\top A \eta
   + b^\top \eta$ with $A = c I_N + \frac{\beta}{PnN} V V^\top$ and
   $b^\mu = -r\bar h^\mu - \frac{\beta}{Pn} V r^\mu$, so
   $\beta \mathcal L_{\mathrm{eff}}(\Theta) = \beta\mathcal L(\Theta, 0) +
   \tfrac P2 \log\det A - \tfrac12 \sum_\mu b^{\mu\top} A^{-1} b^\mu$ (+ const).
   Woodbury (rank $n$) gives $A^{-1}$ and $\log\det A$ at $O(Nn^2)$; SGLD on
   $\mathcal L_{\mathrm{eff}}$ samples the exact $\Theta$-marginal, and
   $\eta^\mu \mid \Theta \sim \mathcal N(A^{-1} b^\mu, A^{-1})$ drawn once per
   measurement gives an exact joint sample for the tuning-map statistics. Derive
   $A$ and $b$ explicitly from the three quadratic terms (data, rate, prior) so
   the reader can check them. Note the restriction to linear activations and
   that deeper linear nets keep a closed form with a full $N\times N$ precision.
4. **The policy.** $\sigma$ is a formal regulariser that makes the first-layer
   covariance invertible; all predictions are stated at $\sigma \to 0^+$
   (cite the limits note's `sec:sigma0`: $\alpha_q = \lambda_q s_X^2$, floor
   $0$, null modes $d_q = 0$, every other formula regular); the code has no
   $\eta$; predictions 4 and 5 of the limits note (the noise floor) are deferred
   and need (i) or (ii) when they are taken up.

Write the rate penalty as $\tfrac r2 \sum \phi^2$ throughout and say in one
footnote that the main text currently writes $g^2$ for this $r$ (a rename is
pending). Use the notation of the sections (`\hat h`, $\Phi$, $K$, $\beta$,
$\kappa = \beta/(Pn)$), and prefer short derivations to prose.

## Acceptance criteria
- `make -C latex note NOTE=input_noise_average` builds without errors;
  `make -C latex check` passes (every note and main still build).
- The note contains the three measures, the null-floor comparison with the
  numbers $0.5$ versus $1$, both exact samplers with $A$, $b$ and
  $\mathcal L_{\mathrm{eff}}$ derived, and the $\sigma\to0$ policy with the
  cross-reference to the limits note.
- `01_setup.tex` carries the `\major` sentence and the `\majornote` pointing
  at the file, and the main document still builds.
