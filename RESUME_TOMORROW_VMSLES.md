# Resume Prompt (VMS-LES Work)

Use this note to resume work in a new chat.

## Current implementation status
- VMS-LES is integrated for WCSPH in src/schemes/fluid/vms_les.jl.
- Resolved coarse operator F(Ubar) now uses pressure + selected base viscosity branch:
  - ArtificialViscosityMonaghan branch
  - ViscosityAdami and ViscosityAdamiSGS branch
- Fine-scale VMS path is active with existing stability guards (tau cap, uprime clamp, S_mag cap, nu_T cap, finite checks).

## What was validated
- ABC batch runs completed and produced:
  - out/tgv_abc_batch/summary_runs.csv
  - out/tgv_abc_batch/summary_comparison.csv
- Previous-vs-new comparison artifacts generated:
  - out/tgv_abc_batch/compare_previous/summary_runs_delta_vs_previous.csv
  - out/tgv_abc_batch/compare_previous/ke_C_old_vs_new.png
- A/B baseline KE stayed effectively unchanged.
- C (VMS) changed significantly at t=0.5 and t=1.0 after resolved-operator update.

## Important interpretation
- F(Ubar) is now closer to base WCSPH momentum physics (better consistency).
- This does not guarantee universally better metrics than classical SGS; C_s needs re-tuning with the updated operator behavior.

## Scope caveats
- Intended/validated scope: WCSPH fluid cases (e.g., TGV, DHIT workflows).
- Boundary-aware VMS coupling is not fully implemented.
- TLSPH is not targeted by this VMS path.

## Next steps for tomorrow
1. Run a focused re-tuning sweep for C_s with updated implementation (TGV first, then DHIT).
2. Rank settings by multi-metric objective:
   - KE trajectory error
   - velocity-field difference
   - stability
   - runtime
3. Produce final recommendation window for C_s and min_shepard.
4. If needed, add boundary-aware VMS contributions for wall cases.

## Suggested kickoff message for new chat
Please continue from RESUME_TOMORROW_VMSLES.md. Start by re-running a compact TGV + DHIT C_s sweep with the current src/schemes/fluid/vms_les.jl implementation and generate a ranked settings table (accuracy, stability, runtime) versus AdamiSGS baseline.
