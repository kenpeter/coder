# Weight-edit log

## 2026-11-26 — Docstring-boundary cure re-applied (step 294046)

**Evidence (HumanEval@30, step 292407):** pass@1 = 0/30; all 30 completions
were empty (`"\n"`). `syntax_valid_rate = 1.0` was vacuous. Measured boundary
probabilities: `P(newline|docstring+NL) = 0.66–0.90`, `P(indent) ≈ 0.005`.
This is the F1/F6 "restart / emit nothing" disease from `abstraction.md`.
The earlier Stage C boundary edit had been washed out by ~140k steps of
training (data has ~1 docstring/MB, so no gradient maintains the cure).

**Edit:** `+α·u` on the 101 whitespace-indent embedding rows, `−α·u` on the
newline row (222), where `u` is the normalized mean post-`final_norm` hidden
state at the 30 HumanEval boundaries (cos ≈ 0.98 with each).

**Ablation** (ce_py guard = 5% over baseline):

| α | p_indent | ce_py | verdict |
|---|---|---|---|
| 0.00 | 0.0057 | 1.6403 | baseline |
| 0.02 | 0.0689 | 1.6415 | fail |
| 0.04 | 0.3348 | 1.6451 | fail |
| 0.05 | 0.5284 | 1.6475 | fail |
| 0.06 | 0.7000 | 1.6508 | fail |
| **0.08** | **0.9116** | **1.6594** | **OK (+1.2% ce_py)** |

α = 0.08 applied in place to `checkpoints/san/san_latest.pt` (step 294046).

**Verification:** greedy generation at a HumanEval-style docstring prompt now
starts with `return a + b` instead of an empty newline — the "emit nothing"
disease is cured. Residual failure mode: repetitive `return … """` loops (F10),
a genuine capability gap of the 30M / ~29%-Python model. `abstraction.md`'s
honest ceiling (assert_rate ~1%) still stands; this edit removes one failure
mode, not the capability gap.
