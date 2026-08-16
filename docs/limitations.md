# Scientific and deployment limitations

1. **Simulation evidence.** Encoder responses are synthetic reference-ODE
   signals. Experimental drivetrain validation has not been performed.
2. **Measured-file scope.** Only time and electromagnetic torque originate from
   `jera1.mat`; encoder responses are not measured-file channels.
3. **Model class.** The stiffness representation is bounded and monotonically
   decreasing. Arbitrary non-monotone recovery, repeated degradation/recovery,
   and multiple simultaneous fault trajectories have not been demonstrated.
4. **Known parameters.** `Jm`, `Jl`, and `bv` are treated as known and correct.
   Uncertainty in these parameters can be confounded with stiffness.
5. **Load torque.** The current relative dynamic equation does not include an
   independently varying external load torque.
6. **Excitation.** Identifiability depends on the spectral content and amplitude
   of the applied torque profile.
7. **Sparse state quality.** With 751 labels and 1501 physics points, stiffness
   metrics remain accurate, but relative-speed reconstruction has R² 0.88969.
8. **Sampling.** The 121-point case aliases the dominant torsional band. The
   401-point case nominally satisfies Nyquist but remains too coarse for reliable
   state reconstruction and trapezoidal integration.
9. **Online state source.** The causal stiffness updater uses an
   offline-pretrained, time-conditioned `RelativeStateNet`; online retraining or
   initialization from a new live encoder stream was not evaluated.
10. **Timing claim.** The benchmark demonstrates causal near-real-time
    feasibility on one tested CPU. It is not hard-real-time certification and
    provides no worst-case scheduler guarantee.
11. **Threshold timing.** The 315 N m/rad alarm occurs approximately 83.9 ms
    early. This is detection, not unbiased transition-time estimation.
12. **Data availability.** Dataset redistribution rights are unresolved, so
    independent full retraining requires a separately authorized input copy.
13. **Downstream use.** Predictive maintenance and adaptive control are possible
    future applications of the estimated degradation signal; neither application
    is implemented or validated in this repository.
