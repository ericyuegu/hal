# 011 Speedrun Log — maximize closed-loop winrate per wall-clock on the local 3060

## TL;DR (status)
- **Systems: throughput 1.66×** (615→1020 samples/s via torch.compile @b128; AdamW > Muon at this scale; b-size roofline at b256). Fast iteration loop established.
- **BC frontier (closed-loop, 16-rep): control 9.1% → 13.6%** (relfeat = the real gain: +dmg, +kills; AR-groups small; capacity/Muon/nana = noise; decode-temp 1.0 optimal). This is roughly the BC ceiling on this data vs lvl-9 Fox; the reported "20%" SOTA is within eval variance (same model scored 9.1%/14.8% across eval configs).
- **>50% target: not reached.** Only RL directly optimizes stock winrate; built a correct REINFORCE loop (011g), but 4 short configs showed the policy *moves* (kl grows) without *winrate climbing* (REINFORCE struggles to escape the losing-play basin vs a strong CPU; signal is noisy; rollouts intermittently hang on the latent recv-block). **v5 sustained run in progress** (lr 5e-5, kl 0.1, real evals every 20 iters) as the genuine attempt; honest read pending its iter-20/40/60 evals.
- Best deployable ckpt: `runs/260613-080403_*p3-2player-12k/final.pt` (13.6%) — but RL aims to exceed it.


**Goal.** Maximize training performance per wall-clock for the 009 next-token GPT policy,
nanoGPT-speedrun style, on this box's local GPU. North-star metric: **closed-loop stock
winrate vs lvl-9 CPU** (stocks_taken / (taken+lost)). Hard target: **>50%**.
Allowed: any hparam/optimizer/arch/feature/action-encoding change. Forbidden: changing the
dataset, reward hacking, modifying eval metrics.

**SOTA reference (on a 5090, b128, ~12k steps, ~1 hr):** 0.33 stocks taken/min, 1.28 lost/min,
93 dmg dealt/min, 125 dmg taken/min → stock winrate ≈ **20%**. So target is ~2.5× the ratio.

## Hardware reality (this box)
- **RTX 3060, 12 GB** (NOT a 5090). 12 CPU cores, 30 GB RAM, 645 GB free disk.
- Data is **LOCAL**: `data/processed/ranked-anonymized-1/mds` (137 GB, train/val/test). No R2 streaming bottleneck.
- torch 2.11.0+cu130, bf16 supported, torch.compile available.

## Baseline facts (009 as-is, measured on this 3060)
- From today's 200-step smoke run (`wandb 5nw8ikth`): **625 samples/s @ b128, 0.20 s/step**, model = **6.5 M params** (d256/L8/h4, L_ctx256).
- ⇒ 12k steps ≈ **40 min** pure training. Model is tiny; GPU may be underutilized.
- **Eval dominates wall-clock**: the 200-step run took 670 s total, only ~40 s was training. One closed-loop eval (16 replicas, 8 parallel, ~3800 frames) ≈ **10 min**.
- 009 ALWAYS runs a final closed-loop eval at end of `train()` (unconditional) — must control this for fast iteration.

## Strategy
1. **Phase 1 — throughput/systems.** Judge by samples/s + *matched* train-loss curve. No closed-loop eval needed (systems changes don't alter the learned fn). Cut training wall-clock.
2. **Phase 2 — optimizer/training.** Judge by val NLL/recon at fixed wall-clock budget; closed-loop eval only the top candidate.
3. **Phase 3 — architecture/objective/features.** Judge by closed-loop winrate (the real metric).

**Eval discipline (to iterate fast):** screen with cheap proxies (val NLL, recon F1/MAE — free each val) + short/low-replica closed-loop probes; reserve full 16-replica eval for finalists. Closed-loop is the *only* arbiter for arch/objective bets (NLL↔winrate can diverge — see memory).

## Cheap-proxy ↔ winrate caveat
Memory: NLL improvements don't always move winrate; 005 regression was data skill-ceiling + lost char conditioning, not objective. 009 already has char/stage conditioning. Treat val NLL as a *screen*, closed-loop as *truth*.

## Hypothesis backlog (prioritized)
### Phase 1 (systems)
- **H1 torch.compile** the model. Expect 1.3–2× if GPU-bound.
- **H2 flash attention**: the `[B,1,L,L]` bool mask forces the slow mem-efficient SDPA kernel; use `is_causal=True` (flash) for the no-pad common case, handle cold-start pad separately.
- **H3 batch-size scaling**: b128 uses little of 12 GB; push b256/512 for GPU util (LR re-tune).
- **H4 dataloader**: if CPU-bound, tune num_workers/preprocess.
- **H6 eval control**: make eval cadence fully controllable; skip final eval when disabled.

### Phase 2 (optimizer)
- **H7 Muon** on 2D matrices + AdamW on embeddings/head/norms (speedrun staple; code in 011_muon.py).
- **H8 LR sweep** (AdamW + Muon); 1e-3 may be low.
- **H10 grad clip** (currently inf = measure-only) for stability at higher LR.

### Phase 3 (arch/objective/features)
- **H11 group coupling**: 009's 4 output groups are conditionally independent given context → may shred cross-channel coherence in closed loop (007 thesis). Try small AR-over-groups (010) or buttons→stick coupling.
- **H12 capacity** within budget (d384/L8 or d256/L12; memory: bigger dented the plateau).
- **H13 L_ctx** 256 vs 128 (shorter = faster, maybe enough).
- **H14 stick representation**: main_stick NLL dominates (1.36 bits); reweight / better centers / finer grid.
- **H15 decode temperature** (test-time, eval-only sweep on best ckpt).
- **H16 features**: opponent-relative coords / velocities (encoding change allowed) — spacing signal.

## Findings
- **Bottleneck (009 baseline): GPU-bound, 95% util, but only 3.6/12 GB used.** ~210 ms/step, ~615 s/s @b128.
- **H1 torch.compile = 1.66×**: b128 615→1020 s/s (120ms/step). Loss curve byte-identical to 009 (math unchanged). ~27s compile warmup. 3060 lacks SMs for max-autotune GEMM but default compile wins.
- **H3 batch size plateaus fast** (compile): b128=1020, b256=1122, b384=1165 s/s; **b512 OOMs 12GB**. Above b256 it's flat → at the 3060 GEMM roofline. Since b128→b256 is only +10% throughput but halves updates/sample, **keep b128** (proven LR, best sample-eff).
- ⇒ Phase-1 base = **011a (compile, b128, 1020 s/s)**. 12k steps now ≈ 24 min training.
- Eval still ~10min/full sweep; gated off for systems runs.

## Results
| id | change | samples/s | val nll (bits) | closed-loop (taken/lost per min) | winrate | notes |
|----|--------|-----------|----------------|----------------------------------|---------|-------|
| 009 base | eager AdamW b128 | 615 | 2.43 @200st | 0.06/3.81 @200st | — | reference smoke (5090 SOTA: 0.33/1.28 ≈20%) |
| 011a | +compile b128 | **1020** | (=009) | — | — | 1.66×, loss identical |
| 011a | +compile b256 | 1122 | — | — | — | +10% only |
| 011a | +compile b384 | 1165 | — | — | — | roofline; b512 OOM |
| 011b | +Muon 0.02 | ~878 | 1.323 @2k | — | — | **LOSES to AdamW** |
| 011b | +Muon 0.04 | ~900 | 1.454 @2k | — | — | worse (past optimum) |
| 011a | AdamW 1e-3 (control) | ~1000 | 1.427@1k → **1.253@2k** | — | — | **best**; +15% faster than Muon |

**Phase-2 verdict: AdamW(1e-3) > Muon at this scale/budget.** Muon's fast early *train*-loss didn't generalize (val NLL worse) and it worsens with lr ↑, and it's ~15% slower/step. Dropping Muon; all Phase-3 builds on **011a (compile + AdamW)**. (Muon worth revisiting only at much larger scale / lr↓ — not now.)

## Phase 3 — closed-loop winrate (the real metric)
**Control: 011a @12k (compile+AdamW, same arch as 009 SOTA), 16-replica eval:**
- stock winrate **9.1%** (taken 0.232/min, lost 2.323/min); dmg dealt 84.7, taken 144.6; frames 5811; crashed 0.
- val NLL: 1.302(2k)→1.204(4k)→1.151(6k)→1.121(8k)→1.100(10k)→**1.092**(12k); plateaus ~10k.
- **Below the reported 20% SOTA.** Caveats: 16-replica eval variance; the latest 009 commit *added nana 4-player input* (SOTA may predate it). Either way 9% is the measured local baseline.
- **Diagnosis:** deals damage (84.7) but rarely secures kills (0.23 taken); dies cheap — 144.6 dmg / 2.32 stocks = ~62 dmg/stock-lost ⇒ gimped/SD'd/edgeguarded at low %. ⇒ bets that help **recovery/spacing (relfeat, capacity)** and **execution coherence (AR-groups)** target the actual failure modes.

**Screening protocol:** bets screened at 8k steps + 8-replica/5400-frame FD eval (~19min); finalists re-run 12k + 16-replica eval. Decode-temp is a free test-time knob (sweep on best ckpt).

**Decode-temp sweep on control ckpt (8-rep/5400f FD):** temp 0.70→3.0%, 0.85→0.0%, **1.0→14.8%**. Lower temp *collapses* the policy (passivity, 0 kills) — sampling entropy is needed. **Keep temp=1.0.** (Same model: 14.8%@8-rep/5400f vs 9.1%@16-rep/full ⇒ eval variance is high; compare bets at identical config.) Could test temp>1.0 later but risks SD-noise.

**Stick discretizer is fine (H14 dropped):** 65 main clusters = 40-spoke 9° rim (DI/wavedash/firefox angles) + tilt rings; well-matched to human input. High main-stick NLL = genuine movement entropy, not a coarse grid. So action encoding isn't the bottleneck.

| bet | id | val nll | stock winrate | taken/lost | dmg dealt/taken | notes |
|-----|----|---------|---------------|------------|-----------------|-------|
| control | 011a @12k | 1.092 | **9.1%** | 0.23/2.32 | 84.7/144.6 | baseline (temp 1.0, 16 rep) |
| relfeat | 011c @12k | 1.092 | **12.3%** | 0.305/2.17 | **105**/135 | +24% dmg dealt vs control despite *identical* NLL → NLL↔winrate divergence; KEEP relfeat |

**Relfeat verdict:** val NLL identical to control (1.092) but closed-loop clearly better (winrate 9.1%→12.3%, dmg dealt 85→105). Spacing features help the policy *act* better in rollout even without improving avg next-token prediction. Stacking capacity on the relfeat base next.

| relfeat+cap | 011c d384/L10/h6 @12k | **1.076** | 10.0% | 0.21/1.91 | 84/145 | 18M, best NLL but winrate↓ vs relfeat; more conservative (dmg 105→84), fewer deaths (6363 frames) |

**Capacity verdict: lower NLL, NOT higher winrate.** d384/L10 reached the best NLL (1.076, still dropping at 12k) but played more conservatively (dmg dealt 105→84, kills 0.305→0.21) for ~10% winrate — worse than relfeat-alone (12.3%). Classic BC: more capacity → better likelihood → more mode-averaging → less decisive. At 2× the wall-clock, **capacity is deprioritized.** Leader stays **relfeat d256 (12.3%, dmg 105).** Next: execution coherence (AR-groups) on the relfeat base.

| relfeat+AR | 011e d256/h8 @12k | 1.126 | **13.4%** | **0.355**/2.29 | 94/140 | **new best**; highest kills (coherence→kills holds); NLL worse but winrate best |

**AR-groups verdict:** stacking the AR-over-groups head on relfeat gave the best winrate (13.4%) and the highest kill rate (0.355 taken/min) despite *worse* NLL (1.126) — the head's cross-channel/temporal coherence converts damage to kills better, exactly the 007 thesis. Confirms: **closed-loop winrate, not NLL, is the arbiter** (every winrate gain this phase came with flat/worse NLL).

| 2p+relfeat+AR | 011f d256 @12k | 1.10 | 13.6% | 0.330/2.088 | 89/145 | nana-drop ≈ noise vs 011e; fewest deaths |

**Phase-3 ranking (16-rep/12k/temp1.0):** 011f 2p **13.6%** ≈ 011e relfeat+AR **13.4%** > relfeat 12.3% > cap 10.0% > control 9.1%. Net: **9.1%→~13.5%** via relfeat (the one real BC gain); AR-groups/capacity/2player(nana) are noise on top. **BC plateaued ~13-14% vs lvl-9 Fox** — the SOTA "20%" gap is eval variance, not nana. ⇒ Phase 4 RL is the path to >50%.

**Files:** 011e_relfeat_argroups.py = 011d AR-groups head + 011c relfeat (the two ideas stacked).
**Queued (free, test-time):** temp>1.0 sweep on the relfeat ckpt — the temp curve rose monotonically 0.7→1.0, so 1.1–1.3 may add aggression (more dmg/kills) for free. Then finalize the best model with a 16-rep eval (and possibly a longer run).

**Temp sweep on 011e (8-rep/5400f):** temp 1.0→0.083, **1.2→0.103** (dmg 105.6, kills 0.274), 1.4→0.074. **temp 1.2 is best deploy temp** (adds aggression; 1.4 too random). Use 1.2 for finals. (8-rep noise: 011e@1.0 here=0.083 vs 16-rep 0.134 — ranking is the signal.)

**Nana-regression hypothesis (011f):** the recent 009 commit added nana → 4-player input (ego, ego_nana, opp_nana, opp). But the eval is Fox-ditto, where both nana slots are ALWAYS masked zeros — 2 of 4 player blocks are dead weight that may dilute the model and explain trailing the pre-nana 20% SOTA. **011f_2player.py** = 011e but `_PLAYER_PREFIXES=("ego","opp")`. Legitimate for the Fox-ditto metric (nana only matters for IC, not evaluated). Running after the temp sweep; clean A/B vs 011e (13.4%).

**Strategy note (gap to 50%):** control 9.1% → relfeat 12.3%. Gains are real but modest; the closed-loop metric (not NLL) is the arbiter. Highest-leverage remaining BC levers: **capacity** (running), **AR-groups coherence** (011d), **longer training of the bigger model** (d384 NLL won't plateau at 12k like d256 did). RL would directly optimize winrate but Dolphin rollouts are ~1k env-steps/s here → millions of steps = impractical on one 3060 in-session; keep BC as the path. If BC plateaus ~20-25%, that's likely near this data's ceiling vs lvl-9 Fox — will document honestly while pushing the frontier.

## Files built
- **011a_compile.py** — 009 + torch.compile + gated eval + push_to_r2 off (DONE, 1.66×).
- **011b_muon.py** — 011a + Muon (2D matrices) / AdamW (embeds+head+bias). MultiOpt/MultiSched wrappers.
- **011c_relfeat.py** — 011b + nonlinear opp-relative features (dist, |Δx|, |Δy|, mutual facing).
- **011d_argroups.py** — 010 (AR-over-groups head) + compile + AdamW + gated eval. compile covers the 8-layer backbone (`action_loss` = `head_logits(model(...), tgt)`); tiny AR head runs eager. BUILT, py_compiles. Smoke pending GPU.

## Phase 4 — RL fine-tune (the path to >50%)
BC plateaus ~13% (kills are ~1/6 of deaths; need ~6× — BC tweaks give +10-30%, won't get there). The eval target is a **fixed lvl-9 CPU**, which is *exploitable* → RL (allowed: changes objective, not dataset/eval/reward-hack; game stocks+damage are the true reward) can plausibly exceed 50% by finding punish loops. Init from the best BC ckpt.

**Design (011g_rl.py), reusing max infra:**
- Rollout: G matches vs lvl-9 CPU via `drive_vec`; a logging BatchPolicy records per slot/frame the sampled action vec + the per-frame flat obs (full, uncapped). Trajectory gives per-frame `post[port]["stock"/"percent"]`.
- Reward (dense): `r_t = w_d·(Δopp% − Δego%) + w_s·(stock_take − stock_loss)` from the trajectory; discounted returns, batch-normalized advantages (REINFORCE+baseline / GRPO-style).
- **Update = advantage-weighted BC loss**: log-prob(sampled action) = −sum(group_nll) = −action_loss ⇒ `loss = (adv · Σ group_nll).mean()` + entropy bonus + KL-to-frozen-BC (anti-collapse). Windows rebuilt with `closed_loop._live_batch_from_rolling` (same fn as eval ⇒ rollout/update consistency). PPO clip optional (store old logprob).
- Loop: rollout → reward/adv → K sampled-frame minibatch updates → periodic closed-loop eval. Build + test incrementally once GPU frees from 011f.

**011g_rl.py authored + reviewed (correct).** Key checks passed: rollout window (`flat_hist[:k+1]`,`ego_inputs[:k]`) is rebuilt identically in `build_update_batch` ⇒ recomputed log-probs match rollout; `rl_update` uses `[:,-1]` logits (same as decode) with target = the action sampled at frame k; REINFORCE sign, KL(π_bc‖π) anchor, entropy bonus, batch advantage-norm, grad-clip 1.0 all correct. Reward helper validated on a synthetic trajectory. Plan: per-iter `rollout/winrate` (8 matches) is the frequent free signal; `rl_eval_every` high (full 16-rep eval is 10 min). Smoke: `--rl-bc <ckpt> --rl-iters 2 --rl-matches 4 --rl-frames 1500 --rl-k 128` then a real run. **Won't run concurrent with 011f (512-sample update could OOM 12GB + crash 011f).**

**RL smoke PASSED (concurrent w/ 011f, tiny footprint):** `--rl-iters 1 --rl-matches 2 --rl-frames 800 --rl-k 32` init from 011c relfeat ckpt → `iter 0: loss=-0.033 entropy=0.394 kl=0.0002 matches_ok=2`, no OOM/error. (winrate 0 only b/c 800f×2 too short to score a stock.) **RL init = 011c relfeat ckpt** (independent-groups arch matches 011g; NOT 011e's AR ckpt — different state_dict): `runs/260613-052632_*p3-relfeat-12k/final.pt` (12.3%).
**Real RL launch (after 011f frees GPU):** `--rl-iters 80 --rl-matches 8 --rl-frames 3600 --rl-k 512 --rl-lr 3e-5 --rl-kl 0.1 --rl-entropy 0.001 --rl-epochs 4 --rl-eval-every 40`. ~1 min/iter ⇒ ~1.5h. Monitor `rollout/winrate` climbing from ~0.12 toward 0.5; if collapse (winrate→0, entropy spikes/crashes) lower lr / raise KL.

**RL launch debugging (resolved):** (1) k=512 update forward (`[K,256,1024]` MLP) OOMs the 12GB card — K must be ≤~128. (2) `--rl-k` silently didn't bind (prefix-clash with `--rl-kl`), so K stayed at the 512 default → renamed field to **`rl_nsamp` (default 128)**. (3) My relaunch one-liners used `pkill -f 011g_rl` which killed the just-launched `uv run …011g_rl…` itself, so I kept re-reading a STALE OOM log. Fix: kill by `--comment` tag, launch via tool background (no `&`/self-pkill), fresh log. **RL v2 (lr 3e-5, kl 0.1): too slow.** 11 iters: winrate noisy 0.0–0.21 around BC baseline, no climb; **kl only 0.0125** after 44 updates ⇒ policy barely moved (lr too low; ~1000 iters needed at this rate). Killed.
**RL v3 (lr 1e-4, kl 0.05, entropy 0.003, 2400f, 120 iters):** 3× lr + half KL + more exploration + shorter rollouts (more iters/hr) to actually move the policy. Watching for winrate climb (good) vs collapse (winrate→0 + entropy crash → revert lr).
**RL reality:** REINFORCE only improves within the BC policy's action support; beating lvl-9 Fox needs behaviors BC rarely emits, so exploration (entropy) + enough policy movement (lr/KL) are critical. Noisy 8-match winrate makes the signal hard to read — the dense damage+stock reward is the real driver.

**RL v3/v3b (lr 1e-4, kl 0.05): policy moves, winrate doesn't (yet).** kl now grows 0.03→0.09/iter (3× v2 — higher lr works), but rollout winrate stays noisy-flat (0.0–0.22, mostly 0.0); entropy drifts up. 2400-frame rollouts are too short to reliably score stocks ⇒ winrate metric unreadable. v3 (first attempt) HUNG mid-rollout (the latent closed-loop recv-block; intermittent — v2 survived 11 rollouts). **RL obstacles on this box: (a) noisy/slow signal, (b) intermittent rollout hangs, (c) ~2h/run.**
**RL v4 (definitive attempt):** lr 1e-4, kl 0.05, entropy 0.003, **3600-frame rollouts** (readable winrate), real 16-rep eval at **iter 30**. If iter-30 eval clearly beats BC ~14% → RL is working, continue/anneal; if flat → >50% is not reachable via this REINFORCE setup on this hardware in-session (would need PPO+value baseline, longer training, and a hang-hardened rollout loop — fixing the `drive_vec` recv-block timeout).

**Status:** reward_from_trajectory + discounted_returns written & design-validated (`.slp` re-read hits a peppi SoA off-by-one, but RL uses live `from_capture` trajectories which work — `summarize_trajectory` already reads the same `post[port][stock/percent]`). **011g_rl.py based on 011c (independent-groups head — trivial log-prob = Σ group log-softmax, much lower RL risk than the AR head; the 12.3 vs 13.4 BC gap is noise).** Authoring delegated to a background subagent (precise spec); will review + GPU-test (rollout→reward→one update→eval) after 011f frees the GPU. Init from best of {011e 13.4%, 011f}.

## Ops lessons
- **Subagents die at ~11 min / ~8 tool-uses** — too short to babysit a 36-min run. The control run's monitor subagent exited at step 3900, but the training (launched via its backgrounded Bash) **kept running detached**. → For long runs: self-launch detached in background + poll the tiny stdout log via periodic wakeups (don't delegate babysitting to a subagent). Subagents only for short (<~15 min) jobs or for context-heavy log parsing at the end.
- `--cfg.no-push-to-r2` (tyro flips default-True bools with `--no-`); default-False bools take `--cfg.flag True`.

## Phase-3 launch plan (when GPU frees, after optimizer pick)
1. **011b @ 12k steps** (control: compile+Muon, same arch as 009 SOTA) — final 8-replica eval + val-NLL trajectory. ~30min train + ~5-10min eval. Establishes whether Muon alone beats the 20% SOTA.
2. **011c @ 12k** — A/B the relative features.
3. **Capacity** via CLI on the winner (`--cfg.d-model 384 --cfg.n-layers 10`) — memory says bigger dents the plateau.
4. **011d AR-groups** + **decode-temp sweep** (free, test-time) on finalists.
- Screening eval: `--cfg.eval-replicas 8 --cfg.eval-max-parallel 8` (1 wave ≈ 5min). Full 16-replica eval reserved for the >50% claim.
