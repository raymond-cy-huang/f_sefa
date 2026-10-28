# R005 Tracing Log

## Round metadata

- Round: R005
- Date: 2026-10-25 (Asia/Taipei)
- Status: CLOSED — mathematical construction established; semantic pilot gated
- Track: Generator Semantic-Response Operator
- Prior report: `../round_R004_beta_calibration/experimental_report_20261025_R04_E01.md`

## Trace

| Time | Action | Evidence | Decision |
|---|---|---|---|
| 13:04 | Closed R004 re-read gate | 16 sections, 12 artifacts, unchanged 100-file source manifest | R005 opened |
| 13:08 | Read supplied GDSA-E01 designs | English pre-registration and Chinese contribution note | Treat full B2/B3 as null control; test basis value only through approximation |
| 13:12 | Rescanned literature corpus | 100 files, 903,726,158 bytes; stage02 note rechecked | InterFaceGAN primary local; LowRankGAN/Shao primary papers absent locally |
| 13:15 | Used primary-source web fallback | NeurIPS LowRankGAN; CVF Shao and InterFaceGAN; Boumal official book page | Jacobian regional control and generator pull-back geometry are prior art |
| 13:20 | Audited repository interfaces | StyleGAN checkpoints, InterfaceGAN boundaries, LPIPS, generator APIs | Model-matched StyleGAN1 W-Smile boundary is available; independent evaluator stack is not frozen |
| 13:26 | Implemented operator abstraction and exact controls | `operator_response.py`, `run.py`, unit suite | Matrix-free API and exact solve ready |
| 13:28 | Ran unit and 2D/16D controls | 3 tests pass; 15 diagnostic rows; summary PASS | H0 supported; diagonal basis dependence demonstrated only mathematically |
| 13:31 | Ran model-matched generator smoke | deterministic StyleGAN1; finite-difference epsilon sweep; adjoint test | FAIL frozen derivative gate; stop before N=20 semantic pilot |
| 13:35 | Wrote immutable R005 report and registries | pre-registration, math checks, report, raw JSON/CSV | No validated distinct contribution |
| 13:38 | Re-read and validate final report | 16 sections; 15 CSV rows; unit suite rerun; JSON statuses checked; source manifest checked | Report internally consistent; toy PASS and generator smoke FAIL are both preserved; R005 CLOSED |

## Gate checklist

- [x] Previous report re-read before starting.
- [x] Full local literature corpus rescanned.
- [x] Missing primary sources retrieved from authoritative venues.
- [x] Exact coordinate-invariance and diagonal controls implemented and tested.
- [x] Model/direction compatibility audited.
- [x] Generator derivative smoke executed without hiding failure.
- [x] N=20/N=100 stopped because mandatory derivative gate failed and user
  explicitly removed the 100-run requirement.

