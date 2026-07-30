# Automated 3D Model Splitting

[简体中文（Default）](README.md) | [English](README.en.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

Love 3D printing but not 3D modeling? You can still split your own parts.

Many models look fully painted while remaining a single solid mesh. Separating eyes, mouths, clothing, or decorations for printing usually means learning modeling software, cutting geometry, closing surfaces, making sockets, and repeatedly tuning assembly clearances.

This AI-assisted tool is made for everyday makers and hobbyists. Give the model a colored or painted 3MF, and it recognizes printable color regions, creates matching inserts and sockets, and produces a validated assembly 3MF ready for slicing. Single-color models are not supported yet.

No Blender, CAD, or advanced parameter archaeology required.

## What it does

- Splits independent printable parts from surface paint colors.
- Selects the main body from geometric and structural evidence.
- Builds inward inserts and matching sockets for other parts.
- Preserves the original colors and AMS filament slots.
- Keeps all printable parts in one grouped assembly.
- Handles nested parts recursively and validates every assembly step.
- Adapts insertion depth and fit clearance to parent-wall thickness.
- Checks topology, winding, colors, material slots, assembly integrity, and visible details before delivery.

## Installation

Send this prompt to your AI assistant:

```text
Install the automated-3d-model-splitting Skill from https://github.com/BensonZeng00/automated-3d-model-splitting, install the required dependencies, and run preflight. When finished, ask whether I want to test it with the cathead example; if I agree, split it automatically and report the result.
```

After installation and a successful preflight, the assistant asks whether to run the bundled `cathead` test. If you agree, it automatically obtains the exact verified example, recognizes its parts, produces the assembly 3MF, and validates the result. If you decline, it does not download or read the example.

## Example: Cat Head

The left image shows the original painted 3MF. The right image shows the result after automatic splitting: the ears, forehead marking, eyes, nose, cheeks, and other colored regions become independently printable assembly parts.

| Original cat head | Automatically split |
| :---: | :---: |
| ![Original cat head model](example/cathead.png) | ![Automatically split cat head model](example/cathead-split.png) |

[Download the cathead.3mf example](example/cathead.3mf)

## Usage

Upload a painted 3MF and say:

```text
Use $automated-3d-model-splitting to split this model, produce the grouped assembly, and retain the final validation result.
```

You can also name sensitive details:

```text
Split this model. Do not cover the mouth, tongue, or eyes, and retain the validation result.
```

## What you receive

You receive one assembly `.3mf` whose body, eyes, mouth, decorations, and other parts remain correctly positioned and colored while being independently selectable and printable.

The assistant also reports the number of parts, preserved colors, assembly layout, and validation status. If slicer confirmation is needed, open the final file yourself and provide screenshots of the model, expanded assembly tree, and sensitive details; the Skill does not control your slicer.

## Suitable models

The current release targets 3MF files with per-triangle surface paint. It stops and explains the issue instead of guessing when a model is damaged, structurally ambiguous, or contains multiple indistinguishable instances.

## Keywords

`3D model splitting` · `painted 3MF splitter` · `multicolor 3D printing` · `AMS model splitting` · `Bambu Studio model parts` · `AI 3D model splitting` · `Codex Skill`

---

Current version: `1.3.5`

Test results: [TEST_RESULTS.md](TEST_RESULTS.md)

License: [Apache License 2.0](LICENSE)
