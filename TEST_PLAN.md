# Test Plan

## Overview
This repository now includes an automated Playwright smoke test that opens the
existing `index.html` page in a headless Chromium browser and verifies the
mouse-segmentation output. The goal is to catch regressions where the red
outline disappears, becomes too thick, or shifts away from the mouse.

## Test Suite

### `tests/mouse-outline.spec.ts`
* Launches a static HTTP server that serves the repository (via Playwright
  `webServer`).
* Loads `index.html`, waits for the OpenCV processing pipeline to finish, and
  reads the `#output-canvas` pixel data.
* Computes contour metrics (red pixel count, bounding box, horizontal and
  vertical stroke width) to ensure the outline is tight, centered, and not
  excessively thick in any direction.
* Checks deterministic numeric guardrails (canvas size, red pixel totals,
  bounding box limits, stroke run lengths, and outline coverage) so the test
  stays fully text-based—no binary golden artifacts are required.

## Usage

```bash
npm install
npm test
```

`npm test` starts the local static server, runs the Playwright test, and then
shuts the server down automatically. Because the assertions rely on measured
metrics (rather than binary golden files), future updates only need to adjust
the numeric thresholds if the expected outline geometry intentionally changes.
