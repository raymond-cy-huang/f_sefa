# Next Experiment

- Current round: R005 CLOSED
- Next round: none scheduled
- Status: STOPPED BY USER SCOPE
- Track: semantic-response operator (SRO)
- R005 objective: convert the supplied GDSA-E01 pre-registration into an
  auditable experiment, verify its exact coordinate-invariance and
  basis-dependent diagonal-approximation claims on analytic controls, and audit
  the repository for the generator, semantic direction, mask, evaluator, and
  prior-art dependencies required before StyleGAN execution.
- Deferred Track-A objective: full nonlinear forward/inverse/Jacobian/metric
  validation remains required and moves to a later round; R005 intentionally
  uses alpha=0 so SRO effects are not confounded with sinh reparameterization.
- R005 outcome: exact 2D/16D controls pass, but the model-matched StyleGAN JVP
  smoke fails the frozen finite-difference gate. No N=20/N=100 semantic run is
  authorized from the current numerical evidence.
- Smallest optional continuation: repair and revalidate one-sample derivative
  consistency, then run N=5 before deciding whether N=20 is warranted.
