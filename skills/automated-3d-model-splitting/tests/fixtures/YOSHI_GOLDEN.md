# Approved Yoshi 3MF regression fixture

`yoshi_golden.json` records the approved source and output hashes, six region signatures, five part pairings, user selected tenon and mortise roles, insertion axes, colors, and geometric tolerances. `yoshi_visual_semantics.json` contains the approved region labels and visual evidence. The source and accepted output are `example/yoshisittingonledge.3mf` and `example/yoshisittingonledge_split_parts.3mf`.

Run from the repository root:

```bash
RUN_YOSHI_GOLDEN=1 .venv/bin/python -X utf8 -B -m unittest discover -s skills/automated-3d-model-splitting/tests -p test_yoshi_golden.py -v
```

The test reruns recognition, checks the six region signatures, applies the approved direction corrections that make P01 the tenon for both shoes, and restores the approved I001/P03 and I002/P02 interface IDs by part pair. It then rebuilds Stage 05 and checks roles, axes, clearance, part colors, watertightness, bounding boxes, and bidirectional sampled vertex distances against the approved 3MF. P05 must retain exactly one source degenerate face without increasing that count. The geometric comparison tolerance is 1 mm. Review confirmations and generated files exist only in the test's temporary directory.

This full model regression is opt-in. Updating its approved hashes or role mapping requires a newly reviewed result.
