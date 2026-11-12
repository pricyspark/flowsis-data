# Marker detection experiments

Updated November 11, 2026. This is the experiment log and follow-up test plan;
dataset identity audits under `audits/` are historical snapshots, not the
ground truth for the updated dataset.

## Current implementation

- Quadrants remain the initial search regions. Every seeded connected blue
  component gets its own padded box and L-template evaluation. Clothing and
  tape are no longer combined into one search mask.
- Confident pixels have blue score >0.25. Connected pixels with score >0.15
  may extend a component. A component needs at least `max(3, round(0.02*k*k))`
  confident pixels, where `k` is the detection kernel width. One bright/noisy
  pixel is insufficient to validate a large weak-colored patch.
- Both grown and strict masks supply candidates. Nearby duplicate matches
  are consolidated. Up to three candidates per quadrant participate in the
  joint selection, which checks crop geometry and scores template fit,
  consistency of marker size, and proximity to predicted positions when known.
  Nearly tied, spatially distinct solutions are rejected as ambiguous.
- The L kernel remains 100/4284 of the longest image side, accounting for
  detection downsampling. Maximum normalized template error remains 0.35.
- Video tracking initializes from observed candidates on the first frame.
  Its orientation preview overlays the selected outline for review. It
  cannot establish physical identity when unrelated blue objects genuinely
  resemble markers; inspect initialization rather than accepting it blindly.
- Subsequent search boxes follow previously validated component bounds. Each
  box is translated using a timestamp-based velocity estimate, then expanded
  by 0.75 marker widths plus an allowance for recent displacement. Prediction
  is capped at one marker width and the extra motion allowance at one marker
  width. These are scale-based parameters, not fixed image locations. Searches
  also remain inside their corresponding quadrant.
- Tracker observations, timestamp, and velocity update only after all four
  markers and crop geometry pass. No unrestricted quadrant fallback or missing
  corner inference runs after a local-search failure. The whole video is
  rejected and diagnostic files are saved, including attempted search boxes.
- Velocity averaging stabilizes the search prediction only. Output corners
  are not temporally smoothed; Gaussian video stabilization remains a separate
  experiment.

The shared candidate detector also changes future image conversions and
coordinate previews. Existing images, masks, and annotations are not rewritten.
Review changed crop coordinates before forcing reconversion of annotated data.

## What has been promising, and what has not

| Approach | Evidence | Assessment |
| --- | --- | --- |
| Kernel proportional to frame size | Fixed the seven original first-frame failures. | Retained. Corrects resolution mismatch, not glare. |
| Globally lower blue cutoff | Earlier saved-frame test: 4/4 recovered at 0.225, but erased tape could be replaced by jeans. | Insufficient on its own. |
| 3x3 closing | Earlier saved-frame test: 3/4 recovered, with a clothing false acceptance. | Optional future cleanup only; not enabled. It can move boundaries. |
| Seeded weak-blue growth | Earlier saved-frame test: 4/4 recovered, with a clothing false acceptance when the search remained broad. | Pair with candidate isolation and temporal locality. |
| Local search plus growth | Earlier prototype: 4/4 saved failures recovered and all 16 erased-marker cases rejected. | Promising small-set result; not a guarantee for the revised dataset. |
| Growth alone on dark photographs | Current implementation initially regressed on `data/raw-images/2/IMG_8615.heic`: weaker pixels connected the marker to noise. | Preserve strict-color candidates too; regression test added. |
| Virtual-tripod vid.stab | Earlier clip: estimated board-center 95th-percentile step increased from 1.61 to 76.34 px/frame. | Poor fit for that scene; do not use as the default. |
| Frame-to-frame Gaussian vid.stab | Same clip: estimated 95th-percentile step reduced to 1.14 px/frame, while slow camera drift remained. | Modest useful baseline, separate from detector correctness. |

Earlier prototype measurements and images are local artifacts in
`coordinate-review/video-marker-diagnosis/`. The Gaussian comparison is under
`exports/stabilization-baseline/20261105_201240/`. These directories are ignored
by Git and may not exist in another checkout. Earlier counts and tool IDs
describe their source snapshots, which predate recent dataset changes.

## Automated tests

Run in the project environment:

```bash
conda activate flowsis-data
python -m pytest tests/test_marker_tracking.py tests/test_square_crop.py tests/test_video_convert.py tests/test_img_convert.py -q
```

The regression tests cover:

- The same marker geometry at 1440, 3024, and 4284 pixel resolutions.
- Pale connected arms with a confident blue seed, and weak candidates with
  only one confident pixel.
- Strict candidates surviving growth that fills an L's concavity.
- Independent boxes for unrelated components in one quadrant.
- Ambiguous, similarly good initial candidates.
- Local tracking with a second, perfectly L-shaped blue distractor elsewhere
  in the same quadrant; erasing each real marker must still fail.
- Missing horizontal or vertical arms, without filling them in.
- Gradual motion beyond the initial search box, variable timestamps, repeated
  timestamps, sudden jumps, and changed frame dimensions.
- No state update after failed observations; a later valid test frame can
  still be evaluated against the last valid state.
- Conversion failure cleanup, exact saved diagnostic pixels after rotation,
  preservation of frame order/timestamps, and 2160-square output.

## Current-data validation on November 11

All **356 repository tests passed** in the `flowsis-data` environment. The
full suite includes data-sync subprocess tests, so ensure the environment's
Python is also first on `PATH`, rather than invoking its interpreter while
leaving a different Python available to subprocesses.

A detection-only scan of the current 11 raw videos, in decoded orientation
(0 degrees), produced these results:

| Source | Result |
| --- | --- |
| `0/20260901_191034.mp4` | Reject at frame 544, 18.095s. Visual review shows an arm covering part of the bottom-right marker. The old frame-31 glare failure is recovered. |
| `1/20260901_191154.mp4` | All 1,035 frames pass. |
| `2/20260901_191326.mp4` | All 799 frames pass. |
| `10/20261109_170239.mp4` | All 1,683 frames pass. |
| `12/20261105_201240.mp4` | All 557 frames pass. |
| `12/20261105_201306.mp4` | All 530 frames pass. |
| `13/IMG_0965.MOV` | All 904 frames pass. |
| `14/20261105_201020.mp4` | All 714 frames pass. |
| `16/IMG_0963.MOV` | Reject at frame 1. Bottom markers are cut off by the source frame; the initial frame fails in every quarter-turn orientation. |
| `23/20261109_170101.mp4` | All 1,931 frames pass. |
| `29/20261105_200708.mp4` | All 741 frames pass. |

These are detection pass counts, not manually verified corner-accuracy labels.
No production MP4s were replaced by this scan. The interactive converter still
requires rotation/outline review, and a video with any unverified frame is
discarded in full.

The 32 main coordinate previews plus the extra low-light examples identified
35 unique raw-photo references. Of those, **34 originals are available and all
pass**, with no pass-to-fail regressions against the previous detector.
`10/IMG_0257` is no longer available under its former folder. The largest
corner-coordinate difference from the previous detector was **9.58 source
pixels**; this is a reason to check ground truth, not proof of improved accuracy.
The previously regressed low-light `2/IMG_8615.heic` passes with the strict
candidate fallback.

Local per-video JSON, rejected frames, and the photo comparison are under
`exports/marker-detection-evaluation/`. The local reproduction script can be
run on one video from the repository root:

```bash
PYTHONPATH=. python exports/marker-detection-evaluation/evaluate_video_markers.py data/raw-videos/12/20261105_201240.mp4
```

## Next experiments, in priority order

1. **Full-video and low-light regression checks.** Test the current collection,
   not only the original seven videos. Review every failure image to label it
   as genuine occlusion, glare, ambiguity, rapid motion, or another error.
   Check the original files corresponding to the low-light coordinate previews.
   Local detection-only runs save per-video JSON and failed frames under
   `exports/marker-detection-evaluation/`; they do not publish converted MP4s.
2. **Manual corner ground truth.** Mark the intended physical inner corners in
   a small set of bright, pale, dark, tilted, and noisy frames. Measure error
   in original pixels, not just the number of successful detections. Tape
   overlaps and extra protrusions can yield a low-error L at the wrong point.
3. **Hard negatives and partial occlusion.** Add blue clothes near a real
   marker, skin crossing a corner, fake Ls close to the tracked position,
   missing first-frame markers, connected blue clutter, and grayscale glare
   that erases all color. The 16 erased-marker trials are not an exhaustive
   occlusion test. Do not infer a hidden marker from the other three.
4. **Search-window sensitivity.** Compare padding in marker-width units, and
   review camera jolts, dropped-frame gaps, scale changes, and motion near
   quadrant boundaries. Current conservative local search may reject a
   legitimate large jump. Track long-run drift and components clipped by a
   search boundary; avoid gradual migration onto background blue.
5. **Native-resolution corner refinement.** Compare the current integer-grid
   template corner with local boundary fitting or subpixel refinement. One
   pixel on the 1000-pixel detection grid represents 1.44 source pixels for
   these 1440-pixel videos, which the final warp can magnify. Verify that
   refinement does not snap to an overlap, shadow, or outer tape corner.
6. **Gaussian smoothing after detection is reliable.** Compare fixed-reference
   board registration, modest temporal denoising of corner measurements, and
   the existing relative-Gaussian baseline. Measure board jitter, drift,
   instrument-motion preservation, and marker intrusion after the existing
   crop inset. Do not use smoothing to conceal an invalid observation.
7. **Capture improvements.** For new data, compare matte tape/printed
   fiducials, diffuse lighting, and controlled polarization. These do not
   recover missing information in old footage.

For each experiment, record source paths, accepted rotation, parameters,
pass/reject counts, false acceptances, localization error where labeled,
runtime, and representative diagnostic images. Full-video scans must state
their tested orientation; the current detection-only runner uses decoded
orientation (0 degrees), while the interactive converter uses user review.
