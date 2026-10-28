# Feasibility Matrix

| Variant | Track | Numerical validity | Semantic mechanism | Independent evaluator | Compute status | Decision | Last round |
|---|---|---|---|---|---|---|---|
| Existing Exp01 Legendre+sinh | A | Reproduced; one basis only | Not tested | N/A | 100-sample pilot reproduced | RETAIN AS PILOT | R001 |
| Existing Exp02 B1/B2/B3 | B | Constraint/solver reproduced | Proxy-only | Missing | 2-sample pilot reproduced; default CUDA decode nondeterministic | REVISE MEASUREMENT | R001 |
| Identity linear | A | PASS | Not tested | N/A | 512D, float32/64 | VALID CONTROL | R003 |
| Legendre linear | A | PASS | Not tested | N/A | 512D, float32/64 | VALID | R003 |
| Chebyshev-I linear | A | PASS | Not tested | N/A | 512D, float32/64 | VALID | R003 |
| SRO full W / orthogonal H | SRO | PASS on 2D/16D exact controls | Not tested | Missing | 15 analytic controls | NULL CONTROL VALIDATED | R005 |
| SRO diagonal basis comparison | SRO | Basis dependence demonstrated on toys | Not tested | Missing | Exact 2D/16D only | RETAIN FOR FUTURE PILOT | R005 |
| StyleGAN1 SRO gradient chain | SRO | FAIL frozen derivative gate | Boundary is model-matched but not independently revalidated | Mask/evaluators not frozen | One sample, 32px response | REPAIR BEFORE PILOT | R005 |
