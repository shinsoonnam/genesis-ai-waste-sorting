# Genesis robust sorting (prototype)

1. Upload the Colab notebook and `genesis_shape_sorter_robust.py`.
2. Run quick test before the full recording.
3. Read `genesis_robust_outputs/robust_sorting_results.csv`.
4. Confirm 3/3 success and 60–180 seconds of video. Do not assume success merely because code finishes.

Changes: smooth joint setpoint interpolation, higher transport waypoint, measured pick location, per-object suction approach, post-drop settle, logging and box containment check, ~60+ second recording target.

**Limitations:** Simulation-based segmentation, synthetic objects, suction weld (not finger grasp), no collision-free arm planning; Genesis/Colab integration not tested here. If object still misses bin, inspect logged positions and adjust waypoints. A shape classifier correctness result does not imply successful physical sorting.
