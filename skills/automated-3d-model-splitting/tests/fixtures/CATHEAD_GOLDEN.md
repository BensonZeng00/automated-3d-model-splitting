# Approved cathead 3MF regression fixture

`cathead_golden.json` records the approved source hash, all 13 region signatures, 12 tenon/mortise relations and insertion axes, output colors, and geometric comparison tolerances. `cathead_visual_semantics.json` contains image-grounded labels confirmed during the split review. The source and accepted output are `example/cathead.3mf` and `example/cathead_split_parts.3mf`.

Run from the repository root:

```bash
RUN_CATHEAD_GOLDEN=1 python -X utf8 -B -m unittest discover -s skills/automated-3d-model-splitting/tests -p test_cathead_golden.py -v
```

The test reruns recognition, checks the exact region signatures before writing temporary review confirmations, replans and compares interface roles and approximate axes, then rebuilds Stage 05 from saved artifacts. It compares all output part colors, watertightness, bounding boxes and bidirectional sampled vertex distances against the approved 3MF, and checks all 12 configured 0.20 mm clearances. The geometric tolerances are 1 mm; triangle or vertex counts are not used as equality requirements. Review confirmations are confined to the test's temporary directory.

The full-model regression is opt-in locally and runs only by manual workflow dispatch. Updating this fixture requires a newly reviewed result and explicit user approval.
