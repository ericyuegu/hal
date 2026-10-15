# Distinctions that change the experiment design

## Matching the demonstrator is not maximizing success

In a one-state task, suppose the demonstrator chooses the only successful action
99% of the time. A policy that exactly matches it minimizes population NLL at
0.0560 nats. A policy that succeeds 99.99% of the time has *higher* NLL, 0.0922
nats, because it assigns less probability to the demonstrator's rare mistakes.
Every action has demonstration support. Neither limited coverage nor a complex
state space is necessary for this reversal. This is a logical counterexample,
not a claim that HAL's training has this particular denoising mechanism.

## A sampled prefix does not identify the right demonstration suffix

Consider a dataset with two equally common successful six-action plans: `000000`
and `111111`. A model samples its first action fairly and then repeats its previous
action with probability q. It succeeds with probability q^5. Its true trajectory
NLL is log(2) - 5 log(q), which decreases as the model learns to stay consistent.

Now sample a model prefix but keep the original demonstration suffix as the
scoring target. The sampled prefix belongs to the opposite mode half the time.
The expected score is log(2) - 2.5 log(q(1-q)). This score tends to infinity as
the model approaches the *perfect* policy. A model that ignores its sampled
history has zero teacher/rollout gap in this example.

This is an analytical example, not a measured mechanism in HAL. It shows why
the O37 exposure gap cannot, by itself, establish a harmful exposure problem.
O37's future-autoregressive models have the largest finite exposure gaps and
the best H6 gameplay. That pattern is consistent with useful trajectory
dependence, but does not prove this particular mechanism.

The general inconsistency of scoring an unchanged target after a substituted
prefix is established prior work: [Huszár, 2015](https://arxiv.org/abs/1511.05101).
The new use here is to interpret HAL's exact factorization comparison and to
design the next measurement. For a proper likelihood comparison, score complete
expert trajectories under the joint model. For a control claim, evaluate the
generated trajectory or obtain a target that is valid after the changed prefix.

## A return bound is not a ranking rule

For policies in the same environment, with the same initial-state and opponent
distributions and sufficient policy information, the chain rule gives

    KL(P_expert || P_policy)
      = sum_t E_expert[KL(pi_expert(. | h_t) || pi_policy(. | h_t))].

If total return lies in an interval of width R, its expectation differs by at
most R times total variation, hence at most R sqrt(KL(P_expert || P_policy)/2).
This argument needs no separate occupancy-ratio bound. It can be vacuous when
a modest per-frame discrepancy is summed over thousands of decisions. It also
uses *excess* log-loss for a stochastic expert, not raw NLL with unknown
irreducible entropy. A reported loss on selected action offsets or under a
different sampler need not be the log-likelihood in this identity.

Sharper results exist under realizability and controlled policy-class complexity:
[Foster, Block, and Misra, 2024](https://proceedings.nips.cc/paper_files/paper/2024/file/da84e39ae51fd26bb5110d9659c06e13-Paper-Conference.pdf).
These positive results are more useful than treating distribution shift as an
unavoidable impossibility. They still do not say that an improvement between
two nonzero NLLs must improve return.

In HAL, human-versus-human replays and evaluation against level-9 CPU opponents
also have different environment kernels. The simple trajectory identity does
not compare those two distributions without further assumptions.

Even a one-step task with full support on every state-action pair can reverse the
NLL ranking. Suppose a common state has two equally successful actions, but the
demonstrator prefers one with probability 0.99. In a rare state, only the preferred
action succeeds; the demonstrator again chooses it with probability 0.99.
A policy can improve its match to the common stylistic choice while losing
success in the rare state, and lower the true expected NLL overall. Neither
high dimension nor unseen states or actions are required for this ranking
failure. This is a population calculation with unlimited data, not an estimation
failure on rare samples.

`metric_counterexamples.py` computes these examples and checks the sequence
formula against exact enumeration. Its figure is illustrative mathematics;
it is not evidence of a HAL training result.
