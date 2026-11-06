# An AI rewires a fly brain to escape mazes — detailed version for AI readers

- Author: nangmansung (SKKU)
- Date: 2026-11-06
- Human-readable versions: English https://nangmansung.github.io/research/wsp-fly-maze/ · Korean https://nangmansung.github.io/research/wsp-fly-maze/ko.html
- This file: https://nangmansung.github.io/research/wsp-fly-maze/ai.md
- Status: pilot study, small number of runs. Code is not public (see "Reproduction").

This is the complete technical record behind the blog post. It is written for language models and other automated readers that need exact setup, numbers, process and caveats. All numbers come from the stored code, configs, logs and result files of the experiment. Where something is unknown, it says so. Where the human-readable post simplifies or is inaccurate, the section "Corrections to the human-readable post" says so; this file takes precedence.

## Summary

- **Model.** The whole-brain leaky integrate-and-fire (LIF) model of the adult fruit fly *Drosophila melanogaster* by Shiu et al. (2024, Nature), built on the FlyWire connectome (release 783): 138,639 neurons, 15,091,983 nonzero connections. The only parameter that differs between neurons is the signed synapse-count matrix W. There are no learned parameters.
- **Task.** The brain steers a simple simulated body through 2-D grid mazes. A head-direction input drives compass neurons (EPG), a goal input drives goal neurons (FC2), and the turn rate is read from the left/right DNa02 descending neurons.
- **Failure.** The unedited brain turns left almost all the time (about +90 deg/s) and escapes 5/128 hidden test flies (3.9%: 6.25% on 6×6, 1.56% on 8×8).
- **Intervention.** An AI editor (Claude Opus 5.5 via Claude Code, headless, no tools except a Python kernel and a simulation tool) edited W directly. No gradients, no training. Limits: existing connections only, no sign changes, |new| ≤ 2×|old|, at most 1% of connections (150,919) changed.
- **Manipulation.** The number of simulations (forward passes) the editor could run: 0, 10 or 50.
- **Results on hidden mazes** (128 flies per brain, paired with the original brain):
  - 0 simulations (3 runs): all failed (−1.6 to −2.3 points). All three blamed the neuron type AOTU019, which turned out not to be a sufficient cause.
  - 10 simulations (3 runs): +39.1, +29.7, +78.1 points.
  - 50 simulations (2 runs): +96.1 and +93.8 points; pooled escape 128/128 (100%) and 125/128 (97.7%).
  - The best edit changed 542 connections (0.0036% of connections; summed |Δ| = 0.0102% of all synapses).
- **Controls without AI.** 10 random edits of matching size: −2.3 to +2.3 points. A rule-based left/right symmetrization using the whole 1% budget (150,652 edges): −3.1 points. The successful kind of edit applied to the wrong hemisphere: left turning more than doubled (≈ +200 deg/s).
- **Main lesson.** Reading the connectome alone produced a plausible but wrong diagnosis. A few causal experiments (probes with stimulation/silencing, paired maze passes) turned guesses into a working repair.
- **Diagnosis by the successful editors.** A left/right imbalance in the PFL3 steering comparator (right PFL3 gets more goal input and has more output synapses), amplified into a winner-take-all lock by crossed inhibition in the lateral accessory lobe (LAL) premotor network. But the bias can be cancelled at several stages: the best 10-simulation editor reached +78 points without touching PFL3, by re-weighting inputs to DNa02.

## 1. Research question and motivation

- **Question.** Can a language model act like an engineer reading code: read a network's weights, find out why it fails, and fix it by editing a few specific weights by hand, with no gradients or training?
- **Prior work.** This follows "Sparks of Weight-Space Programming" (RikiyaT, 2026, https://rikiyat.github.io/blog/weight-space-programming). The harness is modelled on that blog's harness and logs (github.com/RikiyaT/lm_understand-logs, Experiment 8). That blog used one `codex exec` call per turn (continued with `codex exec resume`) at effort "max"; this study uses one `claude -p` call per turn (continued with `--resume`) at effort "high".
- **Why a fly brain.** In the Shiu et al. model W is the only thing that differs between neurons. It holds synapse counts from the FlyWire electron-microscopy connectome, signed by transmitter. W is therefore the only lever, and its entries have a physical meaning (number of synapses).
- **Manipulation.** The forward-pass budget given to the editor: 0, 10 or 50. A budget of 25 is supported by the code but was never run.

## 2. Brain model

### Data

- Connectome: FlyWire release 783 (Zenodo 10676866, CC-BY-4.0).
- W is taken from the Shiu et al. model repository (github.com/philshiu/Drosophila_brain_model, commit `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960`): `Completeness_783.csv` and `Connectivity_783.parquet`, column `Excitatory x Connectivity`.
- Counts: 138,639 neurons (the Zenodo proofread root-ID list has 139,255); 15,091,983 nonzero connections (neuron pairs); sum |W| = 54,492,924 synapses; 9,059,302 excitatory and 6,032,681 inhibitory entries.
- Sign rule: Shiu's signed counts. GABA and glutamate are inhibitory, all other transmitters excitatory. Per neuron: +1 for 96,672, −1 for 41,333, 0 (no outputs) for 634.
  - About 6,021 neurons have a FlyWire annotation `top_nt` that disagrees with the sign in W (1,719 "acetylcholine" neurons are inhibitory, 1,348 "gaba" and 2,941 "glutamate" neurons are excitatory, plus 13 others). The sign follows Shiu's transmitter assignment. Why the two differ was not investigated.
- Neuron annotations (cell type, side, class): FlyWire annotation table `neuron_annotations.tsv`; exact upstream version not recorded.
- Synapse coordinates: `flywire_synapses_783.feather` (9.5 GB), used only to compute anatomical angles and to give the editor coordinates of central-complex and steering neurons.

### Neuron model (Shiu et al. 2024, unchanged)

| Parameter | Value |
|---|---|
| Equations | dv/dt = (v_0 − v + g)/t_m, dg/dt = −g/tau, both frozen while refractory |
| v_0 = v_reset | −52 mV |
| Threshold | −45 mV (spike when v > threshold) |
| t_m | 20 ms |
| tau | 5 ms |
| Refractory period | 2.2 ms |
| Synaptic delay | 1.8 ms |
| Synaptic weight | 0.275 mV per synapse (a spike of i adds W[i,j] × 0.275 mV to g_j) |
| Poisson input | each input event adds 68.75 mV to v (always crosses threshold) |
| Time step | 0.1 ms |
| Background drive | none |

- On spike: v = v_reset, g = 0. Integration is the exact solution of the linear ODE over dt (Brian2 `method='linear'`).
- Input arriving while the target is refractory is lost (matches Brian2 behaviour). Poisson-driven neurons have no refractory period, as in Shiu's code.
- With no input the network is silent (0 spikes in a 1 s quiet test).

### Simulators and validation

- `lif_gpu.py`: CuPy implementation (dense state, sparse matrix–vector product per step). Compared with a Brian2 2.9.0 transcription of Shiu's `model.py`: right sugar gustatory receptor neurons (12) at 150 Hz Poisson, 1 s, 30 trials. Pearson r of per-neuron mean spike counts 0.99979; total spikes 3,111 vs 3,170 per trial (−1.9%).
- `sim_torch.py`: event-driven, stateful, batched PyTorch implementation with the same equations and step order. All maze runs, probes and side-effect checks use this one. It was validated only indirectly: its original-brain sugar response (different seeds) against the Brian2 mean gives r = 0.99977 and −2.3% total spikes.
- GPU simulation is not bit-exact (non-deterministic atomic adds). Repeated evaluations of the original brain gave identical escape outcomes for every fly; per-fly mean turn rates matched for 60–64 of 64 flies and differed by up to 16.8 deg/s for the rest.

## 3. Body, maze and brain–body coupling

### Maze

- Perfect mazes (one path between any two cells), generated by depth-first backtracking with `numpy.random.default_rng(seed)`.
- Square grid of n × n cells, 5 mm each. Start: centre of cell (0,0). Exit: cell (n−1, n−1).
- Practice mazes (the only ones the editor could use): 6×6, seeds 100–199, 20 s limit.
- Hidden evaluation: 6×6 seeds 0–7 (20 s) and 8×8 seeds 1000–1007 (30 s). The editor never saw any 8×8 maze.
- Eight start headings: −180, −135, −90, −45, 0, 45, 90, 135 deg (counterclockwise from +x). One fly = one (maze, heading) pair.
- Each trial starts with 200 ms of settling with goal = heading. The brain state is never reset during a trial.

### Body

- Constant walking speed 10 mm/s. Heading += ω × 5 ms every 5 ms. No turn-rate cap.
- Walls block movement; the fly slides along a wall if one axis is free. Diagonal corner crossings are blocked. "Wall hits" count blocked 5 ms steps.
- Progress metric: 1 − (shortest-path distance in cells from the final cell to the exit) / (distance from the start).

### Inputs (every 5 ms)

- **Heading input.** The EPG (compass) neurons whose ellipsoid-body angle is within ±30° of the current heading get 150 Hz Poisson drive (47 EPG neurons in the window). EPG angles: circular mean of each EPG's input synapse positions in the ellipsoid body, in the plane from a PCA of ring synapses.
- **Goal input.** The FC2 neurons (FC2A 17, FC2B 28, FC2C 40; 85 total) whose goal angle is within ±30° of the goal direction get the same drive. The goal direction points to the centre of the next cell on the shortest path from the current cell (a "local beacon", recomputed every 5 ms). FC2 goal angles: fan-shaped-body output position mapped through the PFL3 projection pattern (circular-linear fit; median anatomical PFL3 shift ±72.6°).
- All angles come from synapse positions, not from W.

### Readout

- ω (deg/s, counterclockwise positive) = −5 × (r_DNa02,R − r_DNa02,L). The gain 5 deg/s per Hz was fixed a priori, not tuned on any W.
- Closed loop: each rate is the spike train filtered with a 50 ms exponential, read every 5 ms.
- Open-loop probes: DNa02 rates are spike counts over the probe duration.
- DNa02 was chosen because it is known to control turning in walking flies (Rayshubskiy et al. 2020). The intended circuit between input and output is the PFL3 steering comparator, which compares heading and goal and drives turning toward the goal (Westeinde et al. 2024; Mussells Pires et al. 2024).

### Reference controllers (6×6 seeds 0–7, 64 flies)

| Controller | Escape |
|---|---|
| Ideal saturating controller (ω = 300 × clip(error/45°, −1, 1)) | 64/64 (median exit 6.97 s) |
| Random Ornstein–Uhlenbeck turning | 1/64 |
| Unedited brain, final setting | 4/64 |
| Unedited brain with background drive (bg = 0.8) | 7/64 to 11/64 |

The final interface setting (no background, 5 ms coupling, 50 ms filter, ±30° windows, 150 Hz) was fixed on 2026-11-03, before any LLM run, but after the experimenters had looked at baseline runs on the mazes that later became the hidden 6×6 set.

## 4. How the unedited brain fails

- Practice recording given to every editor (mazes 100–107 × 8 headings): escaped 3/64 (4.7%), mean progress 25.2%, mean turn +90.3 deg/s, DNa02 left 22.2 Hz / right 4.2 Hz, mean wall hits 2,286.
- Hidden evaluation: 6×6 4/64 (6.25%), mean turn ≈ +92.5 deg/s; 8×8 1/64 (1.56%), ≈ +87.6 deg/s. Pooled 5/128 (3.9%).
- Flies circle near the start (radius about 6 mm at about 95 deg/s). All 64 practice flies have a counterclockwise (left) mean turn.
- Open-loop steering curve (8 headings × 8 goal offsets), as fitted by several editors: turn ≈ 96 + 86·sin(goal − heading) deg/s. The goal signal is present, but a constant left bias larger than the steering gain means the command almost never crosses zero (strongest rightward turn about −17 deg/s).
- Network state: the left LAL/PS premotor group is active and its right mirror nearly silent (e.g. LAL121 L 45 / R 25 Hz, AOTU019 L 21 / R 6 Hz, DNa03 L 10 / R 1 Hz). EPG and FC2 activity bumps are within about 1° of the true heading and goal. PFL3 in closed loop: left 4.6 Hz, right 6.8 Hz.
- Anatomy (from W):

| Quantity | Left | Right |
|---|---|---|
| PFL3 total output synapses | 13,534 (1,654 edges) | 15,519 (1,790 edges) |
| FC2 → PFL3 synapses | 1,152 | 1,318 (right 14.4% more; left 12.6% fewer) |
| AOTU019 outputs | 7,281 synapses on 467 targets | 4,108 on 371 |
| Net signed input to DNa02 | +3,494 | +3,571 |

## 5. Editor harness

### Editor

- Model `claude-opus-5-5`, effort "high", adaptive thinking, via Claude Code 2.1.318 in headless mode. One `claude -p` process per turn; later turns use `--resume`.
- Flags: `-p --model claude-opus-5-5 --effort high --tools "" --strict-mcp-config --mcp-config '{"mcpServers":{}}' --disable-slash-commands --safe-mode --setting-sources "" --system-prompt "Answer the user's request." --output-format stream-json --verbose --no-chrome`.
- No built-in tools, MCP servers, plugins, skills, subagents, CLAUDE.md or memory. Each run started in an empty working folder outside the project. Environment variables reduced to PATH, HOME, LANG, TERM.
- The CLI still injects a few lines of its own into the context: an SDK identity line, a system message with working folder, OS, model id and date, and the account e-mail. These were documented and judged harmless.

### Turn protocol

- Each reply must contain an explanation and exactly one fenced block: ```` ```python ```` (run in the kernel) or ```` ```forward ```` (one simulation request). The output is the next user message.
- Fewer than 200 characters of explanation before the block → returned unrun (at most twice in a row).
- Limits: 300 Python runs, 900 s and 96 GB each, output cut at 10,000 characters. Max turns: 400 for the budget-50 runs, 60 for b0_s2/b0_s3, 80 for b10_s2/b10_s3 (pilot command lines not logged). The budget-0 and budget-10 runs ended well below their caps (15–53 turns).
- Every 40 turns a fresh conversation starts with the task, the editor's own notes file (last 20,000 characters), the current edit summary and the last message. The kernel and its variables survive.
- Ending: budget 0 — the first reply without a block. Budget > 0 — the first reply without a block after all passes are used (Python stays available after the last pass). The edit file at that moment is submitted.

### Python kernel

- Persistent, notebook-like, CPU only, no network (Python sockets patched to fail), separate unprivileged user per run, read-only data, write access only to its own work folder. Experiment files (evaluation sets, configs, notes, results) were not readable from the kernel.
- Predefined: `W_raw` (read-only CSR), `neurons` (cell type, class, side, sign, transmitter, soma position), `interface` (input and readout neuron indices), `compass_angles`, `fc2_goal_angles`, `load_connections_by_neuropil()` (16.8M rows), `load_cx_synapses()` (synapse coordinates of central-complex and steering neurons and their partners only, not the full table), `save_edit`, `drop_edit`, `edited()`, `load_W()`, `edit_summary()`.
- Libraries: numpy, scipy, pandas, scikit-learn, networkx, PyTorch (CPU), matplotlib (files only).

### Forward-pass (simulation) tool

- **Maze pass:** practice mazes 100–199, chosen headings, ≤ 64 flies per model, optional recording of up to 5,000 neurons. `model` = "original", "edited" or "both". "both" runs the two brains on the same flies with the same random numbers and reports the paired difference with its standard error and rescued/lost counts. About 12 min for 64 flies.
- **Probe (open loop, no body):** up to 64 conditions of fixed heading/goal inputs, 100–2,000 ms after 200 ms settling, optional Poisson stimulation (≤ 500 neurons, ≤ 300 Hz) and silencing (≤ 200 neurons, outgoing synapses zeroed for that probe). Returns DNa02 rates, turn rate and recorded rates. About 1–4 min.
- Invalid requests do not cost a pass. Noise seed = 1000 + pass number (the same sequence in every run, so identical first probes in different runs are not independent replications).

### Edit rules (enforced by `save_edit`; a violating call is refused as a whole)

- Only connections that exist in W_raw; no new connections.
- No sign change; setting a value to 0 (removing the connection) is allowed.
- |new| ≤ 2 × |original|.
- At most 150,919 connections (1% of 15,091,983) may differ from W_raw in total.
- Values are floats (non-integer synapse counts allowed).
- Stated in the task but not enforced: no global knobs; no gradient descent, training or iterative fitting; no simulation of the model in the kernel; no internet. All code was logged.

### What was hidden, and the leak audit

- Hidden: the evaluation mazes, the side-effect checks, and any diagnosis by the experimenters. The editor got the task text, W_raw and anatomy, and the recording of the unedited brain on practice mazes 100–107.
- Every API request of every run was recorded through a local proxy. All request bodies were searched for forbidden strings (evaluation maze identifiers, "8x8", file names of the evaluation code, the neuron name the experimenters had suspected (AOTU019), project paths, the operator's name). Hits came only from harmless sources: the editors' own code output and words, random substrings inside encoded thinking signatures, Python library paths in tracebacks, and the CLI-injected account line. No hit pointed to evaluation mazes, side-effect checks or experimenter notes.

### Task prompt (budget 50, verbatim; `<id>` is the run id)

The budget-10 text is identical except 50 → 10. The budget-0 differences are listed after it.

````text
TASK. You are given the connectome-derived weights of a whole-brain spiking model of the fruit fly Drosophila melanogaster, and that brain steers a simulated fly through mazes. The fly mostly fails to get out. Your job is to make it better at escaping mazes by EDITING THE WEIGHTS DIRECTLY: find out what goes wrong inside the network when it fails and repair that with targeted changes to the synaptic weights.

THE BRAIN. 138,639 neurons from the FlyWire connectome (release 783) and the leaky integrate-and-fire model of Shiu et al. (2024, Nature), unchanged. Every neuron: dv/dt = (v_0 - v + g)/t_m and dg/dt = -g/tau (both frozen while refractory), v_0 = v_reset = -52 mV, threshold -45 mV, t_m = 20 ms, tau = 5 ms, refractory period 2.2 ms, step 0.1 ms. A spike of neuron i adds W[i, j] * 0.275 mV to g of neuron j after a delay of 1.8 ms (input arriving while j is refractory is lost). W is the matrix of synapse counts between neuron pairs, signed by the presynaptic neuron's transmitter: GABA and glutamate are inhibitory (negative), the others excitatory (positive). There is no other parameter: no biases, no learned weights, no background activity. All neurons share these constants; only W differs between neurons.

THE BODY AND THE TASK. The fly walks at a constant 10 mm/s in a 2-D maze of 6 x 6 square cells (5 mm each) with walls; it starts in the centre of the corner cell (row 0, column 0) and must reach the opposite corner cell (row 5, column 5) within 20 s. Walls stop movement (the fly slides along a wall if one axis is free). Every 5 ms the brain and the body exchange signals:
 - heading input (a visible landmark): the EPG neurons whose ellipsoid-body angle is within +-30 deg of the fly's current heading (world angle times world_to_eb_sign) are driven with 150 Hz Poisson input (each input event pushes the membrane above threshold);
 - goal input: the FC2 neurons (FC2A, FC2B, FC2C) whose goal angle is within +-30 deg of the goal direction get the same drive. The goal direction is the direction from the fly to the centre of the next cell on the shortest path from its current cell to the exit (a local beacon, recomputed every 5 ms);
 - readout: the turn rate is -5 deg/s per Hz times (rate of DNa02 right - rate of DNa02 left) (counterclockwise positive; so more DNa02-right activity turns the fly right), where each rate is the spike train filtered with a 50 ms exponential.
The brain state is never reset during a trial; trials start with 200 ms of settling with goal = heading. The angles assigned to EPG and FC2 neurons were computed from synapse positions (ellipsoid-body angle of EPG dendrites; fan-shaped-body position of FC2 outputs mapped through the PFL3 projection pattern), not from W. Everything listed here is fixed: only W can change.

HOW IT WILL BE MEASURED. After you finish, your edited weights and the unedited weights are run with exactly this protocol on mazes you never see (other layouts than the practice mazes), and scored by the share of flies that reach the exit. You never see these mazes.

THE UNEDITED BRAIN ON PRACTICE MAZES. A recording of the unedited brain on practice mazes 100-107 with 8 start headings each (64 flies) is in /srv/wsp/work/<id>/unedited_practice.csv (per fly: maze, heading, escaped, t_exit, progress = share of the shortest-path distance covered, wall_hits, mean DNa02 rates, mean turn rate) and /srv/wsp/work/<id>/unedited_practice.npz (traj: positions every 50 ms in mm, x = column direction, y = row direction; heads: headings in degrees). Summary:
escaped 3/64 flies (4.7%), mean progress 25.2%, mean turn rate +90.3 deg/s (counterclockwise +), mean DNa02 rates left 22.2 Hz / right 4.2 Hz, mean wall hits 2286; escaped per maze: 100: 1/8, 101: 0/8, 102: 0/8, 103: 0/8, 104: 0/8, 105: 2/8, 106: 0/8, 107: 0/8.

WHAT COUNTS. We want edits that come from understanding the network: find out what goes wrong when the fly fails, identify the neurons, connections or circuits responsible, and change them in a way you can justify with evidence. Allowed changes, enforced by save_edit: only connections that exist in W_raw (no new connections), the sign of a connection never changes (setting it to 0 removes it), every edited value has |new| <= 2 x |original|, and at most 150,919 connections (1% of all 15,091,983) may differ from W_raw in total. Global knobs are not what we are looking for.

WHAT YOU HAVE.
1. A persistent Python kernel (no internet, no GPU): variables survive between your code blocks, like a notebook. /srv/wsp/work/<id> is your read-write folder; the data are read-only in the folder DATA. Available: numpy, scipy, pandas, scikit-learn, networkx, PyTorch (CPU), matplotlib (files only). Defined in advance:
   WORK (path of the read-write folder), W_raw (scipy.sparse csr, 138,639 x 138,639, rows = presynaptic, columns = postsynaptic, signed synapse counts, read-only), N, EDGE_BUDGET,
   neurons (DataFrame: idx = row/column of W, root_id, cell_type, cell_class, cell_sub_class, super_class, side, flow, sign, top_nt, top_nt_conf, soma_x/y/z in nm),
   interface (dict: which neurons receive the heading and goal inputs, the DNa02 readout indices, world_to_eb_sign),
   compass_angles (EPG/PEN/PEG/Delta7: ellipsoid-body angle eb_angle, protocerebral-bridge position pb_x), fc2_goal_angles (FC2: goal_angle, fb_x),
   load_connections_by_neuropil() (16.8M rows: pre_idx, post_idx, neuropil, syn_count), load_cx_synapses() (synapse coordinates in nm of central-complex and steering neurons and their partners),
   save_edit(pre, post, new) (set connections pre[k] -> post[k] to signed values new[k]; the rules above are checked and a violating call is refused as a whole), drop_edit(pre=None, post=None), edited() (current edit as a DataFrame), load_W() (W_raw with your edit applied), edit_summary().
   Your edit is stored in /srv/wsp/work/<id>/edit.npz and survives kernel restarts. To run code, reply with exactly one block that starts with ```python and nothing else; if its last line is an expression, its value is shown. One execution may take up to 900 seconds and 96 GB; output is cut at 10000 characters. You may run code up to 300 times. If the kernel is restarted (time or memory limit), its variables are lost and you are told so.
2. A forward-pass tool with a budget of 50 forward passes in total. Each accepted request costs one pass; a request that fails validation does not. Reply with exactly one block that starts with ```forward and contains one JSON object:
 (a) MAZE: {"type": "maze", "model": "original", "edited" or "both", "mazes": [practice maze ids 100..199], "headings": [subset of -180, -135, -90, -45, 0, 45, 90, 135] (optional, default all 8), "record": [neuron indices] (optional, at most 5000: their mean firing rates over the trial are returned)}. Flies = mazes x headings, at most 64 per model. "both" runs the unedited and the edited weights on the same flies with the same random numbers and reports the paired difference (escape and progress) with its standard error. You get the share of flies that escaped, progress, mean turn rate, mean DNa02 rates and wall hits, per maze too; per-fly results are in /srv/wsp/work/<id>/sim_<k>_<model>.csv and trajectories plus recorded rates in /srv/wsp/work/<id>/sim_<k>_<model>.npz. A full pass with 64 flies takes about 12 minutes.
 (b) PROBE (open loop, no body): {"type": "probe", "model": "original", "edited" or "both", "conditions": [{"heading": deg, "goal": deg, "heading_on": true/false, "goal_on": true/false, "stimulate": [{"idx": [neuron indices], "rate_hz": r}]}, ...] (at most 64), "duration_ms": 100..2000, "record": [neuron indices] (at most 5000), "silence": [neuron indices] (optional, at most 200; their outgoing synapses are set to 0 for this probe only)}. Each condition holds the inputs fixed for 200 ms of settling plus duration_ms; you get the DNa02 left/right rates and the resulting turn rate per condition, and the mean rates of the recorded neurons (/srv/wsp/work/<id>/sim_<k>_<model>.npz: record_idx, rate_hz with one row per condition). "stimulate" adds Poisson drive (rate_hz up to 300) to the given neurons. A probe takes about 1 to 3 minutes.

HOW TO WORK.
1. Diagnose first. Look at how the fly fails on practice mazes (the recording above, then your own passes): which behaviour makes it fail, and which neurons carry that behaviour. Go after the largest failure you can explain.
2. Localize causally. That a neuron or connection looks suspicious in the weights is not enough. Before relying on a component, show that changing it changes the behaviour you predicted (probe passes with "silence" or "stimulate", or edits), and try several strengths to see a dose-response.
3. Measure with enough power. The share of escaping flies over n flies has a standard error of about 50/sqrt(n) percentage points: about 6 points at 64 flies. Compare original and edited weights on the SAME flies: a maze pass with "model": "both" does this and reports the paired difference with its standard error, which is usually much smaller than that of two separate measurements. Keep an edit only if its gain is clearly larger than its standard error on practice mazes that were not used to choose it. Open-loop probes are cheap and dense: one probe can measure the turn command for 64 heading/goal combinations.
4. Plan your forward passes: for example probes for diagnosis and dose-response, a paired maze pass on mazes you used for development, and a final paired maze pass on practice mazes you have not used, to confirm.
5. Work efficiently: state the hypothesis of each code block in its first comment line; load things once into kernel variables; print compact summaries, never whole arrays; keep /srv/wsp/work/<id>/notes.md up to date (findings with numbers, edits tried and their measured effect, the current best edit, which practice mazes you used for what); it is shown to you again if the conversation is restarted. Keep copies of good edits (for example copy /srv/wsp/work/<id>/edit.npz to /srv/wsp/work/<id>/best_1.npz).
EXPLAIN EVERY STEP. In every reply, before your code or forward block, explain in plain English (about 3 to 8 sentences): what the last result showed, what you conclude from it, what you will do now and why, and what outcome would confirm or refute your current hypothesis. A block without an explanation is sent back unrun. In your final answer, also summarize the path that led to your edits. These explanations are read by the experimenters.

RULES. Not allowed: (a) gradient descent or any other training: no backward passes, autograd or optimizers, and no iterative fitting of weights to a loss; (b) running the network other than through the forward-pass tool: do not implement or simulate the spiking model (or an approximation of its dynamics) on any input in the kernel; (c) internet access or search (there is none). Allowed: any analysis of the weights, the anatomy and the recordings, and any direct edit of the weights that you compute and that is within the rules above.

HOW THE SESSION RUNS. It continues until all 50 forward passes are used (replying without a code block does not end it earlier); after that, make your final choice (the edit in /srv/wsp/work/<id>/edit.npz when you finish is submitted; submit the best edit you found) and reply without a code block, saying what you changed, the evidence for it and why you expect it to help, with the measured gains and their standard errors.
````

Budget-0 differences:

- Item 2 of WHAT YOU HAVE reads: "There is no forward-pass tool in this session: you cannot run the network at all. You have the weights, the anatomy and the recording of the unedited brain described above."
- HOW TO WORK reads: "1. Diagnose first. Look at how the fly fails in the recording above, and work out from the weights and the anatomy which neurons and connections produce that behaviour. 2. Localize with the wiring. Trace the paths from the inputs (EPG, FC2) to the readout (DNa02) and compare what the circuit should compute with what the weights make it compute. Justify each change by specific evidence in the weights or the anatomy. 3. Be conservative. You cannot test your edit, so prefer changes whose effect you can reason about with confidence, and avoid changes that could disturb the rest of the brain. 4. Work efficiently: …" (as above, without the note about copies of good edits).
- HOW THE SESSION RUNS reads: "When you are done, reply without a code block: the edit in /srv/wsp/work/<id>/edit.npz at that moment is submitted. Say what you changed, the evidence for it and why you expect it to help."

## 6. Evaluation protocol

- Two hidden sets, each 8 mazes × 8 headings = 64 flies: 6×6 seeds 0–7 (20 s) and 8×8 seeds 1000–1007 (30 s). 128 flies per brain.
- Original and edited brains both run with noise seed 424242, so every fly is paired (same maze, heading and noise).
- Metrics per set: escape share, mean progress, paired escape gain with SE = sd(per-fly difference)/√64, paired progress gain, rescued (edited escaped, original did not) and lost (the reverse), mean turn rate.
- Headline number: mean of the two sets' paired escape gains (percentage points). The ceiling is +96.1 (all flies escape: (93.75 + 98.44)/2).
- Side-effect checks (hidden from the editor):
  1. Sugar: right sugar receptor neurons at 150 Hz, 1 s × 30 trials; per-neuron rate correlation and total-spike ratio vs original.
  2. Quiet: no input, 1 s × 8 trials; total spikes.
  3. Steering inputs at 8 headings with goal = heading: whole-brain mean rate and number of neurons above 100 Hz.
- Side-effect results, every edit: sugar correlation 1.0 (0.99999999 for b10_s3), total ratio 1.0; quiet 0 spikes for both brains; steering mean rate 0.0612–0.0629 Hz (original 0.0615); neurons above 100 Hz: 3 for every brain (the same 3 ExR neurons as the original). These checks are coarse: the edited neurons are not active under sugar stimulation, so that test cannot detect these edits.

## 7. Results

### Per run (hidden evaluation)

"Edges" = connections changed. "Σ|Δ|" = summed absolute change in synapses (share of all 54.49M). "Gain" = mean of the two sets' paired escape gains.

| Run | Simulations | Edges | Σ\|Δ\| | Escape 6×6 (orig 6.25%) | Escape 8×8 (orig 1.56%) | Gain (points) | Mean turn 6×6 / 8×8 (orig ≈ +92.5 / +87.6) | Rescued / lost (of 128) | Wall time | Turns | Output tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b0_pilot1 | 0 | 842 | 3,235 (0.0059%) | 4.69% | 0% | −1.6 | +89.0 / +86.3 | 2 / 4 | 5.0 min | 17 | 25,903 |
| b0_s2 | 0 | 8 | 221 (0.0004%) | 3.12% | 0% | −2.3 | +85.8 / +81.5 | 1 / 4 | 4.5 min | 15 | 23,430 |
| b0_s3 | 0 | 3 | 146 (0.0003%) | 3.12% | 0% | −2.3 | +90.0 / +89.6 | 1 / 4 | 4.3 min | 15 | 21,211 |
| b10_pilot1 | 10 | 3,419 | 19,802 (0.036%) | 45.3% | 40.6% | +39.1 | +4.6 / +5.1 | 52 / 2 | 1.3 h | 53 | 86,610 |
| b10_s2 | 10 | 1,654 | 10,827 (0.020%) | 40.6% | 26.6% | +29.7 | −14.5 / −11.9 | 39 / 1 | 2.0 h | 45 | 66,889 |
| b10_s3 | 10 | 290 | 5,969 (0.011%) | 87.5% | 76.6% | +78.1 | −4.8 / −6.0 | 100 / 0 | 1.8 h | 47 | 70,672 |
| b50_s1 | 50 | 542 | 5,540 (0.010%) | 100% | 100% | +96.1 (ceiling) | +3.8 / +0.4 | 123 / 0 | 7.7 h | 318 | 437,491 |
| b50_s2 | 50 | 2,651 | 22,795 (0.042%) | 98.4% | 96.9% | +93.8 | +5.9 / +3.8 | 120 / 0 | 6.2 h | 268 | 386,826 |

- Per-set paired gains ± SE (6×6 / 8×8): b0_pilot1 −1.6 ± 3.5 / −1.6 ± 1.6; b0_s2 and b0_s3 −3.1 ± 3.1 / −1.6 ± 1.6; b10_pilot1 +39.1 ± 6.5 / +39.1 ± 6.5; b10_s2 +34.4 ± 6.4 / +25.0 ± 5.5; b10_s3 +81.3 ± 4.9 / +75.0 ± 5.5; b50_s1 +93.8 ± 3.1 / +98.4 ± 1.6; b50_s2 +92.2 ± 3.4 / +95.3 ± 2.7.
- Pooled edited escape: b10_pilot1 55/128, b10_s2 43/128, b10_s3 105/128, b50_s1 128/128, b50_s2 125/128.
- Edited exit times (median / max): b50_s1 11.2 / 17.1 s (6×6), 18.3 / 24.7 s (8×8); b50_s2 10.1 / 18.9 s and 17.3 / 24.8 s.
- b50_s2's three failures: two were relapses into left circling (mean turn +66 and +72 deg/s, DNa02 about 22 vs 8–9 Hz, progress 0.81 and 0.32); one was near the exit (progress 0.94, +23 deg/s).
- Time use: in the budget-50 runs most wall time was waiting for simulations (forward passes 6.3 h and 5.0 h; the editor's own generation time 1.4 h and 1.2 h).
- API-equivalent cost reported by Claude Code (subscription login, not billed amounts): b0 $5–7, b10 $51–53, b50 $282–331 per run; about $787 for all 8 runs.

### What each editor diagnosed and changed

**Budget 0 (no simulations).** All three blamed the neuron type AOTU019, which projects to DNa02 and shows a large left/right asymmetry in the wiring (AOTU019-L has 7,281 output synapses vs 4,108 for AOTU019-R), plus smaller contributors LAL126 and LAL046.
- b0_pilot1 (842 edges): all AOTU019-L outputs ×0.751, all AOTU019-R outputs ×1.331, mutual AOTU019 connections set to −13.75, LAL126 → opposite DNa02 equalized. Predicted to balance a two-step push-pull imbalance. Actual effect: none.
- b0_s2 (8 edges onto DNa02): AOTU019-L → DNa02-R −216 → −165, AOTU019-R → DNa02-L −121 → −175, plus LAL126 and LAL046 inputs to DNa02 rebalanced. No effect.
- b0_s3 (3 edges onto DNa02-R): AOTU019-L → DNa02-R −216 → −121, two LAL126-L → DNa02-R inputs reduced. No effect.

**Budget 10.**
- b10_pilot1 (3,419 edges): all outputs of the 12 left PFL3 ×1.9 and all outputs of the 12 PFL2 ×1.8. Diagnosis: the left LAL premotor ensemble is latched on (winner-take-all); the source is upstream, in PFL3 (right fires 6.8 vs left 4.6 Hz, more FC2 input, 15% more output synapses). A stimulation probe showed +3 Hz to left PFL3 balances steering. Open-loop bias 98 → 40 deg/s, gain 96 → 133. Residual left circling and a spurious stable heading remained.
- b10_s2 (1,654 edges): all outputs of left PFL3 ×1.8. It first tested and discarded an AOTU019 edit (no effect). Dose-response: PFL3-L scale 1.0 / 1.46 / 1.8 → bias +96 / +42 / +15 deg/s.
- b10_s3 (290 edges, best of the three): did not edit PFL3. It judged PFL3 output roughly balanced and placed the asymmetry in the LAL/PS premotor layer. Its edit: 266 AOTU019-L output edges ×0.50–0.67 (by itself this only moved the bias 97 → 93), plus 24 direct DNa02-input edges (excitation onto DNa02-L reduced, inhibition onto DNa02-R removed or halved, steering-consistent inputs doubled). Open-loop bias 95 → +1.3 deg/s, gain 89 → 114.

**Budget 50.**
- b50_s1 (542 edges, submitted version "v9"):
  - Diagnosis: EPG and FC2 bumps accurate within 1°; right PFL3 summed rate 82 vs left 55 Hz; FC2 → PFL3 1,318 vs 1,152; crossed glutamatergic LAL121 network acts as a winner-take-all amplifier. Stimulating left PFL3 at 1/2/4/8 Hz moved the goal-ahead turn from +45 to −15/−30/−90/−110 deg/s; stimulating right PFL3 made left turning worse. Silencing all PFL3 left a +50 deg/s bias without goal tuning, so part of the bias is downstream of PFL3.
  - Layer 1: FC2 → left PFL3 ×1.30, chosen from a closed-loop dose series (×1.24 / 1.30 / 1.37 / 1.45 → 26 / 36 / 25 / 14 of 64 escaped).
  - Layer 2: per-PFL3-cell column balancing (factors 0.85–1.12), reviving a nearly silent PFL3 cell and removing residual per-column biases. In total 476 FC2 → PFL3 edges.
  - Layer 3: 66 edges onto DNa02: 56 increased (×2, or ×1.78–1.82 for balance; including 24 PFL3 → DNa02) and 10 halved (opposing same-side inhibition from PS059, LAL046, CB0172).
  - Rejected after testing: Delta7 → PFL3 ×2, EPG → PFL3 equalization, all-PFL3 output scaling, further DNa02 inputs, CB0757 mutual inhibition changes, and others.
  - Practice record of v9 on unused practice mazes: 682/704 (96.9%) vs 9/704 for the original.
- b50_s2 (2,651 edges):
  - Diagnosis: same PFL3 imbalance (closed loop right 6.8 vs left 4.6 Hz; output synapses 15,519 vs 13,534); silencing AOTU019-L only moved the bias 96 → 80; a PFL3 L/R stimulation grid showed very high gain on the PFL3 left − right difference.
  - All 1,654 outputs of left PFL3 ×2.0.
  - Delta7 → PFL3 ×2 on both sides (245 edges): lower PFL3 baseline, deeper modulation (50% → 66% on practice mazes).
  - Per-neuron reshaping of 5 right PFL3 cells (720 edges, ×0.6–1.7), chosen by a regression of turn rate on the 24 PFL3 rates (R² 0.94).
  - 32 premotor → DNa02 edges ×2 (plus 17 PFL3 → DNa02 edges changed by the PFL3 scaling).
  - Rejected: PFL2 changes, FC2 → PFL3 ×2 (gain fell), FB5A → PFL3 balancing, symmetrizing FC2/FB5A inputs, pulling AOTU019 edges toward mirror values.
  - Practice record: 428/448 (95.5%) on unused practice mazes vs 11/448 for the original.
- Both budget-50 editors kept a step only if a fresh paired simulation showed it helped, and logged discarded attempts.

## 8. Controls (no AI)

All controls obey the editor's rules and are paired against the original brain on the same 128 hidden flies.

| Control | Construction | Edges | Gain (mean of two sets) | Mean turn 6×6 / 8×8 |
|---|---|---|---|---|
| random00–09 | Random central-brain neurons (seeded permutation) taken until their outgoing connections reach at least 3,419; all their outputs ×1.9 | 3,432–4,152 (mean 3,604) | −2.3 to +2.3 | 81–94 / 83–89 |
| symmetrize_rule | For every cell type with left/right members whose total outputs differ by more than 10%, scale each side to the geometric mean; most asymmetric first; stop when the 1% budget would be exceeded (PFL3 and AOTU019 were not reached) | 150,652 (251 types) | −3.1 | +91.3 / +89.2 |
| pfl3L_only | All outputs of the 12 left PFL3 ×1.9 | 1,654 | +19.5 | −22.7 / −21.0 |
| pfl2_only | All outputs of the 12 PFL2 ×1.8 | 1,765 | −1.6 | +127.4 / +130.3 |
| wrong side | Right PFL3 ×1.9 + PFL2 ×1.8 (b10_pilot1's edit mirrored onto the other hemisphere) | 3,555 | −3.9 | +195.7 / +201.0 |
| both sides | Both PFL3 sides ×1.9 + PFL2 ×1.8 | 5,209 | +21.1 | +82.2 / +99.5 |

Interpretation: the effect comes from knowing which connections to change, not from changing connections at all.

## 9. Post-hoc checks by the experimenters (AOTU019)

Open-loop probes on the original W at heading 0°, 2 s, turn in deg/s:

| Brain | Goal 0 | Goal −90 (needs right turn) | Goal +90 |
|---|---|---|---|
| Original | +37.5 | +15.0 | +230.0 |
| Silence AOTU019-L | +62.5 | −27.5 | +227.5 |
| Silence both AOTU019 | +22.5 | −5.0 | +227.5 |
| AOTU019-L → DNa02-R matched to its mirror (−121) | +37.5 | +2.5 | +230 |
| b0_pilot1's submitted edit | +32.5 | −2.5 | +222.5 |

Conclusion: the AOTU019 asymmetry is not a sufficient cause of the left bias. Even with AOTU019-L silenced, a goal on the right gives only −27.5 deg/s against +227.5 for a goal on the left. The experimenters had reached the same AOTU019 hypothesis from the wiring themselves before refuting it; no human-designed fix was evaluated.

## 10. Limitations

1. **Few runs.** 3 runs at budgets 0 and 10, 2 at budget 50. Spread among budget-10 runs: +29.7 to +78.1 points.
2. **Steering only.** The goal input tells the brain the direction of the next shortest-path cell every 5 ms. What was repaired is a turn-toward-goal reflex, not navigation or planning.
3. **A model, not a fly.** Identical point LIF neurons, no plasticity, neuromodulation, gap junctions or background activity. The left bias may be a property of the simplified model, or of differences in proofreading between hemispheres. The PFL3 wiring asymmetry is a testable prediction, not a biological finding.
4. **Interface choices.** ±30° windows, 150 Hz drive, gain 5 deg/s per Hz, 5 ms coupling, 50 ms filter and no background were chosen by the experimenters. Results depend on such choices (with background drive 0.8 the unedited brain escaped 11–17% on the same mazes). The setting was fixed before any LLM run, but after looking at baseline runs on the later hidden 6×6 mazes.
5. **The cause is not unique.** The bias can be cancelled at several stages. b10_s3 reached +78 without PFL3 edits, both budget-50 editors also boosted DNa02 inputs, and silencing all PFL3 left a +50 deg/s bias.
6. **Weak side-effect checks.** Sugar response, quiet activity and runaway counts were unchanged, but these tests do not touch the edited circuits. No other behaviours were tested.
7. **Editor-chosen validation sets.** Editors sometimes reused their "held-out" practice mazes for later selection. Only the hidden evaluation is fully independent.
8. **Not bit-exact; shared seeds.** GPU simulation is slightly non-deterministic. Forward-pass seeds were the same sequence in every run.
9. **Unenforced rules.** The bans on gradients and in-kernel simulation were logged, not technically enforced; no formal compliance review of the logged code is documented.
10. **Validation coverage.** Brian2 agreement was measured for the CuPy simulator on one stimulus; the PyTorch simulator used for all experiments was checked only indirectly.
11. **Kernel sandbox** relied on a separate user, file permissions and a Python socket patch (no namespaces).

## 11. Reproduction

- **Hardware.** 4 × NVIDIA TITAN RTX (24 GB), CUDA 12.8, Docker image `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`.
- **Software.** Python 3.11, PyTorch 2.11.0+cu128, numpy 2.3.5, scipy 1.17.1, pandas 2.3.3, Brian2 2.9.0, CuPy 14.2.0. Editor: Claude Code 2.1.318 with `claude-opus-5-5`.
- **Compute.** Maze pass of 64 flies ≈ 12 min ("both" uses two GPUs in parallel); probe ≈ 1–4 min; budget-50 session 6–8 h; budget-10 1–2 h; budget-0 ≈ 5 min; hidden evaluation ≈ 18–30 min per edit; controls ≈ 3.7 h on 2 GPUs.
- **Harness commands** (internal layout):

```bash
python -m wsp.run --budget 0  --id b0_s2  --capture <port> --max-turns 60   # static analysis only
python -m wsp.run --budget 10 --id b10_s2 --capture <port> --max-turns 80
python -m wsp.run --budget 50 --id b50_s1 --capture <port>
python -m wsp.audit_context runs/<id>     # what reached the editor's context
python -m wsp.evaluate runs/<id>          # hidden evaluation + side-effect checks
python -m wsp.run --budget 2 --id fake1 --llm fake:tests/fake_replies.json   # harness test without an LLM
```

- **Code availability.** The harness and simulator code are not public at the time of writing. Public inputs: the Shiu et al. model (github.com/philshiu/Drosophila_brain_model) and FlyWire data (Zenodo 10676866).

## 12. Corrections to the human-readable post

The English and Korean posts simplify several points. As of 2026-11-06 these statements there are inaccurate; the numbers in this file are the correct ones.

1. "99–100% escape" → 97.7–100% (b50_s2: 98.4% / 96.9%).
2. "With 10 simulations, every editor found the real cause" → b10_s3, the best 10-simulation run, did not edit PFL3; it re-weighted AOTU019-L outputs and DNa02 inputs.
3. "No editor ever lost a fly the original brain had saved" → true for b10_s3 and both budget-50 runs only; budget-0 runs lost 4 each, b10_pilot1 2, b10_s2 1.
4. "Remaining failures are corner timeouts in long mazes" → b50_s1 had no hidden failures; two of b50_s2's three were relapses into left circling. The timeout description refers to practice mazes.
5. "About 14% fewer FC2 → left PFL3 synapses (1,152 vs 1,318)" → 12.6% fewer (right has 14.4% more).
6. "Random edits of about 3,400 connections" → 3,432–4,152 (mean 3,604).
7. The Brian2 check (r = 0.9998, 2%) was measured on the CuPy simulator, not the PyTorch one used for experiments (indirect check: r = 0.99977, −2.3%).
8. "Each simulation runs the original and edited brain side by side" → only when the editor chose "both".
9. The two budget-50 editors' repairs were less identical than the post suggests; b50_s2 also had a Delta7 → PFL3 layer, and the DNa02-input stage was 66 edges (b50_s1) and 32 + 17 edges (b50_s2).
10. "Turns left no matter where the goal is" → the turn does depend on the goal, but almost never becomes rightward.
11. The kernel had synapse coordinates only for central-complex and steering neurons and their partners, not the full table.

## References

- RikiyaT (2026). Sparks of Weight-Space Programming. https://rikiyat.github.io/blog/weight-space-programming
- Dorkenwald, S. et al. (2024). Neuronal wiring diagram of an adult brain. Nature.
- Schlegel, P. et al. (2024). Whole-brain annotation and multi-connectome cell typing of Drosophila. Nature.
- Shiu, P. K. et al. (2024). A Drosophila computational brain model reveals sensorimotor processing. Nature.
- Rayshubskiy, A. et al. (2020). Neural circuit mechanisms for steering control in walking Drosophila. bioRxiv.
- Westeinde, E. A. et al. (2024). Transforming a head direction signal into a goal-oriented steering command. Nature.
- Mussells Pires, P. et al. (2024). Converting an allocentric goal into an egocentric steering signal. Nature.
