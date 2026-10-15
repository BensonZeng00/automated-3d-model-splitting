# Approved 3MF regression fixture

`panda_golden.json` captures the user-approved 15 region signatures, 14 tenon/mortise roles and axes, and part colors. `panda_visual_semantics.json` supplies the previously confirmed labels. The original and accepted output are `example/Grimmiger+Panda.3mf` and `example/Grimmiger+Panda_split_parts.3mf`.

Run from the repository root:

```bash
RUN_PANDA_GOLDEN=1 python -X utf8 -B -m unittest discover -s skills/automated-3d-model-splitting/tests -p test_panda_golden.py -v
```

The test reruns recognition, checks the part signatures before creating temporary review confirmations, replans the interfaces, and rebuilds the final 3MF. It checks interface roles and approximate axes, then compares exported colors, watertightness, bounding boxes and bidirectional sampled vertex distances with the accepted model. The two geometric comparison tolerances are 1 mm; no binary equality or exact triangle count is required. Review confirmations exist only in the test's temporary directory.

The full-model case is opt-in locally because it processes a large mesh. Its GitHub Actions workflow runs only when manually started with **Run workflow**. Changes to the user's accepted result require a new explicit review before updating this fixture.
