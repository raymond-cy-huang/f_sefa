# R004 Tracing Log

## Round metadata

- Round: R004
- Started: 2026-10-25T12:48:00+08:00
- Status: CLOSED
- Track: Track A nonlinearity calibration
- Prior report: `../round_R003_basis_validation/experimental_report_20261025_R03_E01.md`

## Frozen hypothesis

**H-BETA-004:** equal raw alpha produces unequal effective nonlinearity across
bases on actual shared-W samples; choosing
\(\alpha_B=\beta/Q_{0.99}(|L_Bw|)\) equalizes the 99th-percentile argument
\(|\alpha_Bc|\) by construction and improves cross-basis activation
comparability.

## Provisional protocol

- 100 paired shared-W samples from the same e4e decoder mapping, seed 42.
- Identity, accepted R003 Legendre, accepted R003 Chebyshev-I.
- Record coordinate mean, SD, median, Q90/Q95/Q99/max of absolute coordinates.
- Raw-alpha controls: 0.1, 0.3, 0.5, 0.7, 1.0; alpha 2.0 stress only.
- Matched beta targets: 0.1, 0.3, 0.5, 0.7, 1.0.
- Record resulting alpha, `sinh` activation ratio, derivative distribution,
  non-finite counts, and conditioning proxies.

## Trace

| Timestamp (Asia/Taipei) | Action | Inputs | Outputs | Validation / decision |
|---|---|---|---|---|
| 2026-10-25T12:48:00+08:00 | Open R004 after R003 close gate | Re-read R003 report/trace; regenerated source manifest | R004 trace and frozen hypothesis | R003 closed; manifest unchanged |
| 2026-10-25T12:49:00+08:00 | Search complete local corpus for scaling evidence | 101-source index; dimensionless and quantile queries | Nocedal–Wright p.139 and CLRS §9.3 quantile material located | Local textbook evidence sufficient; no web fallback required |
| 2026-10-25T12:50:00+08:00 | Implement R004 calibration runner | Accepted R003 transforms; deterministic e4e mapping; frozen R004 protocol | Paired W sampling plus distribution/raw-alpha/matched-beta metrics | Outputs isolated per run; sample SHA-256 recorded |
| 2026-10-25T12:51:00+08:00 | Execute duplicate calibration runs | 100 paired W samples, three bases | Both PASS; sample hash identical | Q99 relative spread 37.6%; max beta error `1.11e-16`; no nonfinite values |
| 2026-10-25T12:51:30+08:00 | Compare duplicate artifacts and tails | Four artifact pairs; raw alpha=1; matched beta=1 | All artifact pairs byte-identical | Matched beta aligns Q99 but not maximum condition; hypothesis supported with stated limit |
| 2026-10-25T12:52:00+08:00 | Write immutable R004 report | All R004 evidence and sources | `experimental_report_20261025_R04_E01.md` | Re-read gate pending |
| 2026-10-25T13:04:00+08:00 | Re-read and validate immutable report | 16 report sections; 12 declared artifacts; regenerated source manifest | All sections present; all artifacts exist; source corpus unchanged at 100 non-git files and 903,726,158 bytes | R004 CLOSED; R005 gate opened |

## Open checks

- [x] Search full local corpus for dimensionless/scale calibration evidence.
- [x] Inspect source support; web fallback not required.
- [x] Implement paired W sampling and distribution/calibration analysis.
- [x] Validate matched-beta identities and numerical stability.
- [x] Write and re-read immutable R004 report.
