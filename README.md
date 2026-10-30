# FlowSIS Data

## Installation

You will need Python and Git to use this repo. If you do not have Python, you can try and install it. It is easiest on Linux and MacOS. On Windows, I recommend using WSL2 to emulate Linux. If you are comfortable with a terminal and command line I recommend installing Python with Miniforge, otherwise you can use Anaconda, which has a graphics user interface (GUI). Installation for both of these is relatively simple. If you cannot install Python use the lab PC, which runs Linux. Similarly, install Git based on your system.

Once you have Python and Git, in your command line (called Terminal in MacOS and Linux) type:

```shell
# Navigate to home. If you prefer to clone to somewhere else, nagivate there.
cd ~
# Clone the repo to your local machine
git clone https://github.com/pricyspark/flowsis-data.git
cd flowsis-data

conda env create -f environment.yml -y
conda activate flowsis-data
```

## Usage

Git synchronizes code and DVC version pointers. DVC synchronizes the dataset
with shared storage. Run commands from the repository root unless stated otherwise.

All source media, annotations, manifests, and review logs live under `data/`:

```text
data/
  raw-images/
  images/
  masks/images/
  videos/
  image_manifest.csv
  image_object_manifest.csv
  image_review.jsonl
```

Paths stored inside manifests remain relative to `data/`, such as
`images/12/photo.jpg` and `masks/images/12/photo.npz`. Do not prefix these values
with `data/`. Annotation commands default to `--data-dir data`; their dataset
path options are relative to that directory, while absolute paths are preserved.
Use `--data-dir /path/to/dataset` to annotate a standalone dataset.

### One-time DVC storage setup

DVC has been initialized. Before publishing the migration, configure a shared
Google Drive folder and create the first dataset snapshot:

```shell
# For existing environments; new environments include this dependency.
python -m pip install dvc-gdrive
dvc remote add -d storage gdrive://YOUR_FOLDER_ID
dvc add data
dvc push
```

Share the folder with each collaborator's Google account. Use a dedicated folder:
DVC stores content-addressed cache objects there, rather than a browsable copy
of the dataset layout. The first transfer requires authentication. If Google's
default DVC application is blocked, configure your own OAuth client following
[the DVC Google Drive guide](https://doc.dvc.org/user-guide/data-management/remote-storage/google-drive).
Keep credentials outside Git.

After a successful upload, stage the old dataset paths' deletions along with
`data.dvc`, `.dvc/config`, and ignore rules. Do not commit the contents of `data/`.
This migration does not remove historical data from Git; history cleanup is separate.

### Collect and prepare images

Download your original images, unzip them if needed, and place them in
`data/raw-images`. For single-tool captures, use the physical Tool ID as the folder
name. For multiple tools, use the reserved `multi` folder with a subdirectory
for each capture group with the same tools and overall setup:

```text
data/raw-images/
  12/                         # Single-tool captures of physical tool 12
  13/                         # Single-tool captures of physical tool 13
  multi/
    setup-001/                # Same tools/setup, randomized positions and rotations
      IMG_1001.HEIC
      IMG_1002.HEIC
    setup-002/                # Different tool set or overall capture conditions
      IMG_1010.HEIC
```

Use unique setup folder names. For multi-tool captures, moving or rotating the
included tools stays in the same setup. Changing the tool set or overall capture
conditions starts a new setup. Folder names organize captures;
physical tool identities for multi-tool images belong in the object manifest.
Collect overlapping tools so they have a consistent front-to-back order: a tool
must not pass above another in one place and below it elsewhere. Avoid interwoven
arrangements. During review, reject incompatible arrangements with
`order impossible`, or skip them with `s` for later review. Ignore regions are
for uncertain pixels, not a substitute for resolving an incompatible arrangement.
The converter preserves subdirectories, so
`raw-images/multi/setup-001/IMG_1001.HEIC` becomes
`images/multi/setup-001/IMG_1001.jpg`.

Pull incoming code and dataset changes before making local edits:

```shell
git pull
dvc pull
```

Convert and crop the raw images. The converter isolates the blue marker in each image quadrant, matches its inner corner with an L-shaped convolution kernel, applies a perspective correction, and saves the finished 2160×2160 JPG in `images`. If a marker is missing, incomplete, or does not match the kernel reliably, it does not create the JPG and lists the failed image and reason after processing finishes.

```shell
python img_convert.py
```

Existing JPGs are skipped by default. To reprocess all raw images and replace their existing JPGs, run:

```shell
python img_convert.py --force
```

If forced conversion fails, the old JPG is removed so `images` cannot retain an output that failed the current crop validation. The original file remains in `raw-images`.

To save annotated previews of the detected coordinates for visual review, run:

```shell
python review_crop_coordinates.py --count 32
```

Use `--count 0` to review every raw image. Previews are saved in `coordinate-review` and are not used by the converter.

Prepare the image and object manifests (this script was previously named `basic.py`):

```shell
python prepare_manifests.py
```

For single-tool folders, the script lists each new tool ID and asks you for its class, name, and confidence. Tool classes and names are automatically changed to lowercase with punctuation removed. Confidence must be a number from 0 to 1. If it finds undocumented images for a tool ID that is already in the manifest, it warns you and asks whether to add or skip them. Pressing Enter skips that tool. If you continue, the script reuses the tool's existing class, name, and confidence. It reads the original files in `raw-images` to fill in the capture time, camera make, model, flash, ISO, exposure, aperture, and focal length in millimeters when that metadata is available. Capture times are stored with their date and timezone, for example `2026-09-01T19:51:50.430-04:00`. The metadata is written only to the CSV; it is not copied into the JPG files.

`multi/` is reserved for explicit intake. The preparation script never treats
`multi` as a Tool ID, creates inferred instances there, or assigns sample IDs
and angles from its filenames. It reports converted multi-tool images that have
not been registered, and registered multi-tool images with no instance rows,
with instructions to run `annotate.py declare`. These images do not interrupt
single-tool preparation. Declared multi-tool images still receive available
camera metadata and refreshed navigation columns; their identities, masks,
sample IDs, and angles are preserved.

Use the two manifests as spreadsheets. The columns you need for manual review
are at the left; internal IDs and mask details are farther to the right.

| File | What to check and edit |
| --- | --- |
| `image_manifest.csv` | Sample ID, Angle, Filepath, Lights, and other capture details. |
| `image_object_manifest.csv` | Tool ID, Tool Class, Tool Name, and identification Confidence for each tool in an image. |

The image manifest starts with `Tool IDs`, `Sample ID`, `Angle`, `Filepath`, and
`Lights`. `Tool IDs` is a summary: `0` means one tool, and `0;12` means tools 0
and 12. Change physical tool identities in the **object manifest**, then run
`python prepare_manifests.py` to refresh this summary.

The object manifest starts with tool ID, class, name, confidence, sample ID,
and filepath. Its sample ID and filepath are copied from the image manifest
so you can find an image without looking up its internal ID. Edit these two
fields in the **image manifest**, then run `python prepare_manifests.py` to refresh the
copies. Do not edit internal Image ID or Instance ID values. Verification
reports summaries or references that disagree between files.

A **Tool ID** identifies one physical instrument. A **Sample ID** identifies a
related capture group with the same physical tools and overall setup. Sample IDs
are globally unique across single-tool and multi-tool captures. A **Capture
Session ID** identifies the broader photography session, which can contain
several samples.

For **single-tool samples**, keep the controlled rotation protocol: one unchanged
setup, with eight views at 0, 45, 90, 135, 180, 225, 270, and 315 degrees. Flipping
the tool, changing its opening or location, or changing the overall setup starts
a new sample. Correct recorded angles if filenames are out of order.

For **multi-tool samples**, keep the same Sample ID while randomizing the
positions and rotations of the same tools within the same overall setup. Any
number of images is valid. Start a new sample when the physical tool set or
overall capture conditions change. Leave `Angle` **blank**, rather than entering
`NA` or estimating each tool's orientation. If a shared arrangement rotation is
actually known, an optional angle from 0 to less than 360 degrees is accepted;
there is no rotation-coverage requirement. The sample type is determined from
the declared tool set, so no new manifest columns are needed.

Keep `Angle` in the manifest for controlled single-tool captures. For example:

| Tool IDs | Sample ID | Angle | Filepath |
| --- | --- | --- | --- |
| 0 | 15 | 0 | images/0/IMG_1002.jpg |
| 0 | 15 | 45 | images/0/IMG_1003.jpg |
| 0 | 15 | 90 | images/0/IMG_1001.jpg |

Run `python prepare_manifests.py` after editing. It puts unfinished images
(missing Sample ID, or missing Angle for a single-tool image) at the bottom,
and sorts each section by numeric Tool IDs. Rows with the same
tool IDs retain their existing order, regardless of sample, angle, or filename.
Object rows follow the same image order. It preserves
existing sample IDs and angles. Filename order only provides initial suggestions
for new, unassigned single-tool batches; check that each batch really is one
unchanged setup.

`prepare_manifests.py` creates a pending instance for each new single-tool image and reuses
existing tool details from object rows. Annotation fields stay blank until review;
identification `Confidence` is separate from `Annotation Confidence`. Existing
masks and review fields are preserved when adding images. Snapshot both manifests with DVC
together.

Multiple tools can be represented by additional instance rows with the same
`Image ID` and distinct `Instance ID` values. Automatic folder intake remains
for single-tool captures. For multi-tool images, use `annotate.py declare` to
add explicit tool metadata, then run `annotate.py images` without `--single-tool`
for mask-to-tool assignment. Metadata-only instance rows are excluded from mask
training. An accepted empty image retains its tool metadata without a positive
mask annotation.

For single-tool folders, raw images are considered in consecutive filename slots of eight, including newly added raw images that do not have a converted JPG yet. A complete slot receives a unique sample ID and angles from 0 to 315 degrees. If conversion or cropping fails, the raw file continues to occupy its original slot. The other images in that slot are left ungrouped for review, but the failure does not shift the angles or prevent later complete eight-image slots from being assigned automatically.

Check both `image_manifest.csv` and `image_object_manifest.csv`. Confirm instance tool identities and confidence, image filepaths, sample groupings, and angles. In particular, confirm that each recorded Angle matches its image, regardless of filename order.

You can also run an automatic verification:

```shell
python verify_data.py
```

The verifier first checks the CSV and files for missing images, images that are not in the manifest, duplicate paths, tool IDs that disagree with their folder, invalid confidence values, incomplete single-tool rotation samples, missing or repeated single-tool angles, and inconsistent tool sets within samples. Randomized multi-tool samples may have any image count and blank angles. It then asks whether Codex should visually review temporary, downscaled contact sheets. The visual review looks for a different tool set within a sample, suspicious single-tool rotation order, image-quality problems, and inconsistent class or name conventions. These findings are suggestions for a person to review; the verifier never changes the manifest. Temporary contact sheets are deleted automatically.

The visual review requires the Codex command-line program and a one-time ChatGPT sign-in. If `codex` is not installed, install it with:

```shell
curl -fsSL https://chatgpt.com/codex/install.sh | sh
codex login
```

The results are saved as `verification_report.html`, which you can open in a web browser, and `verification_report.json`, which contains the same results in a machine-readable format. To run only the CSV and file checks without Codex, use `python verify_data.py --static-only`.

Once the manifest is correct, snapshot and upload the dataset. Coordinate edits
with collaborators; resolve incoming changes before replacing local data.

```shell
git status
```

Update the dataset pointer and upload its referenced data before committing:

```shell
dvc add data
dvc push
git add data.dvc .gitignore
```

The files are now "staged". We commit staged changes so they are saved to the Git history. Each commit needs a message to describe the change. Format your commit message using your name, the tools, and the samples added:

```shell
git commit -m "Jane Doe" -m "tools: x, y, z" -m "samples: a, b, c, d, e, f"
```

Finally, push it to GitHub to upload it online so everyone can now see them.

```shell
git push
```

You can now check the repo on the GitHub website to confirm your changes.

## Mask annotation

After converting images and updating the manifest with `prepare_manifests.py`, generate and
review SAM3 masks from the `flowsis-data` repository root:

```shell
# Update an existing environment after pulling the annotation changes.
conda env update -n flowsis-data -f environment.yml
conda activate flowsis-data
python annotate.py images
```

The first run downloads `facebook/sam3` through Transformers. If access to the
checkpoint requires authentication, authenticate with Hugging Face using an
account authorized to download it. Interactive review requires a graphical
Matplotlib backend. CUDA is selected when available; use `--device cpu` to
select CPU explicitly. `--batch-size 1` reduces the image batch size.

Review mode is selected from each image's declared tools, so one command can
review both single-tool and multi-tool images. Single-tool images use a simple
mask review without Instance ID commands. Predicted masks are **not merged** by
default: accept the one correct mask. If the model returns alternatives, choose
one by its preview number; this still uses the single-tool interface.

Optional `--merge-masks` (also accepted as `--single-tool`) merges predicted
fragments into one mask during inference. It is only for captures known to need
fragment union, and cannot be used with pending multi-tool images. Prompts use `Tool Class` by default. When two or more declared tools share a
class within an image, their prompts automatically include names, such as
"iris scissors" and "mayo scissors". Classes unique within the image retain
class-only prompts. Use `--image-prompt name-and-class` to include names for all
tools. Identical resulting prompts run once, while physical identities remain
separate assignments.
A second pass refines a padded crop by default; disable it with
`--no-refine-crop`.


In the terminal review prompt:

- `a` accepts the mask when exactly one candidate is available.
- `a 0` selects candidate 0 when several alternatives were predicted. Only one
  candidate can be accepted; multiple selections do not implicitly merge masks.
- `e` accepts an empty image; `r` rejects it; `s` skips it for later.
- `p TEXT` reruns segmentation with another prompt.
- `i 0,2` marks ignore regions; `v partial 0` sets the mask's visibility
  (`clear`, `partial`, or `severe`).
- `clean` optionally removes tiny holes and specks from the sole candidate;
  `clean 0 close 3` targets candidate 0 with a 3-pixel closing kernel. `unclean 0`
  restores it. See the cleanup options below.
- `q` quits. Rerunning the same command resumes unfinished images.

Previously skipped images are shown after other unfinished images. Each group
keeps its manifest order, and each image is shown at most once per run. To work
through skipped images first, run:

```shell
python annotate.py images --skipped-first
```

Basic preparation assigns `Image ID` and `Capture Session ID`. Annotation reads
tool metadata from instance rows. Review the suggested capture sessions before splitting
training data. Outputs are `image_object_manifest.csv`, `masks/images/<tool ID>/<image stem>.npz`,
and the append-only `image_review.jsonl`. Commit these together with the updated
`image_manifest.csv` so masks and resume decisions stay synchronized. Mask
bundles retain the format consumed by `flowsis` training. `--auto-accept` bypasses
human review and should only be used when that is intended.

### Visualize saved masks

Export PNG previews without running SAM3 or opening a graphical window:

```shell
# Preview the first 32 saved image bundles.
python visualize_masks.py
# Preview every saved bundle for one tool.
python visualize_masks.py masks/images/12 --count 0
# Preview one image or extracted video frame.
python visualize_masks.py masks/images/12/IMG_1001.npz
python visualize_masks.py masks/videos --count 32
# Specify the source explicitly for a bundle in a custom location.
python visualize_masks.py /path/to/mask.npz --image /path/to/image.jpg
```

Previews are saved under `mask-review/`, preserving subdirectories of the input
directory. Each PNG shows the original image, colored overlay, and mask-only
view, with bundle indices and identities in the legend. Ignore regions appear
as black/white stripes; overlapping instances blend their colors. Empty bundles
show the unchanged image and a blank mask panel. Image dimensions must match
the masks; the script reports failures and exits with a nonzero status.
Use `--output-dir` to choose another destination, `--alpha` to adjust overlay
opacity, and `--max-size` to adjust the displayed image resolution per panel.
The script uses NumPy, Pillow, and Matplotlib from the existing environment.

### Multiple tools and overlapping tools

Before mask annotation, prepare multi-tool captures in this order:

1. Place originals under `raw-images/multi/<setup>/`, grouping images with the
   same tools and overall setup. Tool positions and rotations may vary freely.
2. Run `python img_convert.py` and inspect any conversion failures.
3. Declare the known physical tools for the entire arrangement in one command:

   ```shell
   python annotate.py declare --folder setup-001 --tool-ids 12 13
   ```

   `--folder setup-001` is shorthand for `--folder images/multi/setup-001`.
   Every image in the folder and its subfolders receives both tool declarations.
   Use a separate folder when the set of tools changes. Non-image files are
   ignored, and missing or empty folders raise an exception. The command adds
   image rows if necessary; it does not infer tool identities from folder names.

   Each Tool ID must already exist in `image_object_manifest.csv`, typically
   from single-tool preparation. Class, name, and identification confidence are
   copied from those records. Missing IDs, missing/conflicting class or name,
   and invalid confidences raise an exception before declarations are written.
   If confidence differs between existing records, the lowest value is used.
   Identify new tools in the manifest before declaring them in multi-tool images;
   the declaration command no longer accepts metadata overrides.

4. In `image_manifest.csv`, assign one globally unique Sample ID to the images
   with the same physical tools and overall setup, even when tool positions and
   rotations vary. Leave `Angle` blank for these randomized captures; any image
   count is valid. A setup folder name does not automatically assign a Sample ID.
   Review each tool's ID, class, name, and identification confidence in
   `image_object_manifest.csv`.
5. Run `python prepare_manifests.py` to fill available camera metadata and
   refresh navigation columns. It preserves your manual sample and angle values.
6. Run `python verify_data.py --static-only` and resolve manifest errors before
   annotation.
7. Run `python annotate.py images` to assign and review masks.

You may run `prepare_manifests.py` before step 3; it prepares single-tool captures
and lists the multi-tool images that still need declaration. It does not infer
how many tools are present: reviewers must declare **every** physical tool, even
if an image already has one instance row.

For individual images, use `--filepath images/multi/setup-001/IMG_1001.jpg`
(repeat `--filepath` for several images). Alternatively, `--sample-id 15` targets
every image of an existing, manually assigned sample, including a randomized
multi-tool sample. All target modes accept `--tool-ids 12 13` or repeated `--tool-id 12 --tool-id 13`.
Instance IDs are generated from the image and physical tool ID; `--instance-id`
allows an explicit identifier when declaring exactly one tool. Duplicate instance
or physical tool IDs within an image are rejected.
`declare` does not infer a rotation sequence or camera metadata.
To add a tool to an already reviewed image, `declare --reopen` clears all its
annotation fields and reopens it for review, while keeping existing identities.

To review only multi-tool images, skipping single-tool captures:

```shell
python annotate.py images --multi-tool
```

This selects images with multiple declared tools in `image_object_manifest.csv`,
regardless of their folder or manifest position. Existing review decisions still
apply; add `--skipped-first` to prioritize previously skipped multi-tool images.
Without this flag, both kinds of images are reviewed in manifest order.

Run multi-tool annotation **without** `--merge-masks`/`--single-tool` or `--auto-accept`.
The model runs each unique prompt once per image. Tools sharing a class use
`name class` prompts to distinguish variants, such as `iris scissors` and
`mayo scissors`. Tools with the same name and class still share a prompt and
require separate Tool ID assignments. Optional
`--discovery-prompt` and `--canonical-prompt` add predictions. All candidates are
retained for explicit review, including possible duplicates and overlapping
masks. Automatic crop refinement is disabled for batches containing multi-tool
images and for additional prompts entered during explicit review. After assigning
masks, use the explicit per-tool crop refinement described below.
Predictions may overlap, but accepted instance masks must not share pixels.
Resolve any remaining overlaps after corrections using the ordering step below.

The interface uses a Matplotlib image preview and terminal commands. The preview
shows colored, numbered masks, bounding boxes, prompts, model scores, and current
assignments. The terminal lists declared Tool IDs, classes, names,
identification confidence, visibility, and assigned candidate indices. Prompts
and scores are suggestions; the reviewer supplies the physical identity.

The explicit assignment interface below is used **only for multi-tool images**.
Single-tool images always use the simpler reviewer described above, including
when the model predicts several alternative masks or returns no predictions.

Use physical **Tool IDs** in commands; each can occur only once in an image.
For example, if mask candidate 0 covers tool 0 (scissors) and candidate 1 covers
tool 3 (rongeur):

```text
review> g 0 0
review> g 3 1
review> a
```

The first number after `g` is the Tool ID; the second is the mask candidate
number shown in the preview. The prompt on a candidate is a segmentation hint,
not its final tool identity. Assigned candidates are labeled `tool 0`, `tool 3`,
etc. Instance IDs remain internal CSV/mask references; legacy commands using
full Instance IDs still work when they do not match a Tool ID. A matching Tool
ID takes precedence. `n` generates its internal Instance ID automatically.

In the multi-tool reviewer, `masks []` means **no candidates are assigned** to
that tool; it does not tell you whether the model found any masks. The terminal
lists the candidate count and numbered predictions separately. Inspect the
Matplotlib preview, then assign candidate numbers to each declared Tool ID
before accepting. Use `g TOOL_ID 0,2` when candidates 0 and 2 are disconnected
pieces of that tool. Resolve other candidates by assigning them to another tool,
excluding false predictions with `x`, or marking ignore regions with `i`.
If there are **no candidates**, try `p TEXT` with a suitable tool description,
inspect the new predictions, and assign them. Use `s` to defer an image if
prediction still fails. `a` alone does not assign masks in multi-tool review.

Candidate colors are determined by their stable candidate numbers, using a
spread-out palette that does not wrap every 10 or 20 candidates. A tool's
assigned mask uses the same color in the overview and its zoom panel as its
selected candidate. Grouped fragments use the first selected candidate's color
for the combined tool mask. Reassigning to a new candidate changes that tool's
color; removing other candidates does not. Overlapping candidate pixels can
still blend their colors in the overview.

Use `remove 1,3` to discard unwanted candidates from the preview and active
list. Candidate numbers are never renumbered or reused within the current
image review session. Removing an assigned
candidate clears the affected tool's entire assignment, including grouped
fragments, refinement, and spatial prompts; assign that tool again before
accepting. It also removes any ignore-region membership. Other assignments
remain unchanged, and already-derived masks are not recomputed automatically.
Removed masks cannot be reassigned or used for subtraction. Use `x 1,3` when
you want to exclude candidates while retaining them for possible reassignment.
Removal is recorded in the review log and does not change the final mask format.

The multi-tool preview combines a full-image overview with one zoomed panel
per declared tool. Before any assignment, the overview shows each unassigned
candidate as a colored mask overlay with a numbered dotted bounding box. Colored pixels show the actual mask; the bounding box
alone does not indicate which tool pixels were segmented. Candidate numbers
remain visible in the overview. Assigned
tools use consistent colors in the overview and their crop panels; original
image context remains visible around each tool, including overlapping tools.
Panels without an assigned visible mask show a placeholder.

#### Resolve overlapping masks with tool order

After assigning and correcting the masks, the reviewer reports the number of
pixels claimed by multiple tools and lists the affected Tool IDs. Magenta
outlines mark disputed regions in the overview and tool panels. Non-overlapping
masks can be accepted directly; `a` blocks acceptance while any instance masks
still overlap.

Use `order` with physical Tool IDs **from front to back**. Include every tool
involved in any overlap; unrelated tools can be omitted. For example, tool 3
is above tool 0:

```text
review> g 0 0
review> g 3 1
review> refine all
review> order 3 0
review> a
```

At each disputed pixel, the first listed tool claiming it keeps that pixel;
lower tools lose it. Pixels claimed by only one tool remain unchanged. This
does not fill gaps, infer hidden surfaces, or change the source candidates.
Visibility remains the reviewer's decision; set `partial` or `severe` when
appropriate before ordering. Bounding boxes are recomputed from the final masks
when accepted.

Ordering updates the tool panels but **does not accept the image**. Inspect the
colored masks around the magenta outlines, then enter `a` separately. A second
window opens automatically after every successful `order ...`, showing close-ups
of the disputed regions. Each close-up has three linked panels: the original
image without overlays, the masks before ordering, and final pixel ownership
after ordering. Crops use the original image pixels with nearest-neighbor
display; they are not downscaled preview files.

Use the Matplotlib toolbar to zoom or pan: all three panels move together so
you can compare exactly the same pixels. Press `]` for the next disputed region
and `[` for the previous one. Disputes spread across the image or a large area
are split into small crops with surrounding context, so even a few conflicting
pixels can be examined closely. Press `m` to hide/show the colored masks, or
Esc to close the inspection window. These keys act on the focused figure,
not the terminal. Closing the viewer does not accept the annotation.

Enter `inspect` at the review prompt to reopen the close-ups without changing
the masks or invalidating the order. After inspecting all regions, enter `a`
in the terminal to accept, or change/reset the order or edit the masks. An
incorrect upper-tool boundary can still cut valid pixels out of the lower tool,
so correct boundaries before ordering. The tool panels show the resolved masks;
the terminal's candidate assignments still refer to the unchanged source masks.

`order reset` restores the masks from before ordering, including their current
refinements and cleanup. Enter another `order ...` to change the order; each
attempt starts from those same masks, so subtraction does not accumulate.
Starting any editing command (assignment, visibility, refinement, cleanup,
spatial edits, candidate removal/exclusion/ignore, declarations, or new prompts)
clears the order before the edit, even if the edit is cancelled or invalid.
Resolve any resulting overlaps again and review before accepting. Skip, reject,
and quit do not save the ordered masks.

An order cannot silently erase a tool's whole mask. If this would happen, the
command reports the tool and retains the previous order. Fix duplicate or
incorrect assignments, choose another physically correct order, or explicitly
mark a genuinely fully hidden tool with `v TOOL_ID occluded` and resolve its
unused candidates. Never mark a visible tool occluded just to pass validation.

If tools pass over and under each other, a global order is not valid. Use
`order impossible` to reject the image and record the reason, or `s` to leave it
unfinished. Physical depth cannot be inferred reliably from prediction overlap:
side-by-side tools with overlapping predictions need boundary corrections.

After assigning the masks, refine tools individually or together:

```text
review> g 0 0
review> g 3 1
review> refine all
review> restore 3
review> refine 3 rongeur
review> a
```

`refine TOOL_ID` uses that tool's name and class as the prompt; optional text
overrides it for that attempt. The command reruns segmentation on a padded crop
of the original pixels and maps the result back into full-image coordinates.
It selects a prediction by agreement with the assigned mask, rather than by
model score alone. Low agreement or similarly matching alternatives leave the
previous mask intact. These checks are conservative heuristics; inspect every
refinement, especially when tools overlap. They do not subtract another tool's
mask or guarantee correct identity.

Results touching an interior crop edge retry with more padding, up to three
crop sizes. A boundary at the edge of the actual image is allowed. Refinement
never merges crop predictions automatically. In each tool panel, the colored
mask is the current result; mask outlines are not drawn. `restore TOOL_ID`
returns to the original assignment;
`restore all` restores all tools. You can retry refinement or accept the current
results with `a`. Refining one tool does not change another tool's assignment.
Assigning different candidates with `g`, clearing with `u`, or marking a tool
absent/occluded discards its refinement.

Crop refinement is optional and runs only on request in multi-tool review,
including when `--no-refine-crop` disables the automatic single-tool pass.
Refinement decisions are logged; unaccepted refinement masks are held for the
current reviewer session. Accept to save them before quitting. No new manifest
columns or cropped training images are required: final masks retain the original
image dimensions and stable Instance IDs.

When one prediction covers several tools, you can derive a remainder by
subtracting independently identified visible masks. This uses the assumption
that visible pixels of different physical tools are disjoint. For a combined
candidate 0 containing tools 0 and 3:

```text
review> draw 3
review> subtract 0 0 3
review> x 0
review> a
```

Keep `subtract` for intermediate mask construction: its result is an editable
candidate that can be inspected, refined, or corrected further. `order` serves
the final ownership check across assigned tool masks and does not create a new
candidate. Using `subtract` does not bypass the final non-overlap requirement.

For model-assisted separation, **start with `click 3`** rather than tracing an
exact mask. Left-click visible pixels of tool 3 (foreground), and right-click
the neighboring tool and background (negative prompts). Include positive points
on each visible part you want retained, not only a small problem area.
Backspace undoes a point; Enter runs SAM3; Esc or closing cancels. Turn off the
Matplotlib toolbar's zoom/pan mode before placing points. This can start with
no candidate or assignment for tool 3.

```text
review> click 3
review> click 3
review> subtract 0 0 3
review> x 0
review> a
```

Repeating `click 3` reopens the active prompt with its earlier points and updates
**the same candidate**, keeping its number and color. Add foreground/background
corrections or use Backspace to undo points. The reviewer lists saved prompt
candidate numbers and click counts for each tool.

For an obscured tool with separate visible portions, use independent prompts:

```text
review> click 3
review> click 3 new
review> click 3 1
```

`click 3 new` starts with no points or polygon seed, adds a new candidate, and
keeps the tool's existing assigned portions. `click 3 1` reopens the prompt for
candidate 1 belonging to tool 3; subsequent `click 3` edits that prompt. Each
prompt has its own click history. Editing one portion preserves the others;
the final annotation combines their pixels into one tool mask, preserving gaps.
Inspect each result: independent prompts can still return overlapping or overly
broad masks. Use `g 3 INDEX[,INDEX...]` to choose which candidates belong to the
tool, or `remove INDEX` to discard a candidate. Removing an assigned candidate
clears that tool's assignment, so regroup the remaining portions afterwards.

`click 3 reset` clears the active prompt's points and polygon seed, then updates
that same candidate after you provide fresh points. Other portions remain
assigned. Esc cancels without changing the active prompt. If SAM3 cannot satisfy
the clicks, the previous mask stays assigned and the draft points remain
available for another attempt. Prompt histories last for the current reviewer
session; nonremoved prompts remain selectable by candidate number even after
clearing or changing an assignment.

The first successful click prompt creates and assigns a model-generated candidate.
Unless you use `new`, starting a prompt without an existing click history replaces
the tool's assignment. Superseded candidates generated for that tool by drawing,
subtraction, or spatial prompting are automatically excluded; other tools and
initial SAM candidates remain unchanged. Clicks and grouping actions are recorded
in the review log; the final mask format is unchanged.

Alternatively, **`polygon 3` uses a rough polygon as a SAM3 prompt**, rather than
saving the polygon as the final segmentation. The polygon editor works as below;
Enter sends the coarse mask to SAM3's image tracker as a dense mask prompt.
Inspect the returned model mask. `click 3` afterwards combines additional points
with that polygon prompt, useful for removing included background or separating
a neighboring tool. If the current assigned mask is unsuitable as an editor
starting point, clear it with `u 3` first. `draw 3` remains available for exact
manual annotation without model inference.

These commands use [SAM3's image tracker point and mask prompting](https://huggingface.co/docs/transformers/v5.13.1/en/model_doc/sam3_tracker).
The interactive model loads on first use, and repeated prompts on the same image
reuse its image features. Text and interactive models take turns on the selected
device to limit accelerator memory. A spatial prompt guides the model; it does
not guarantee a correct full boundary or complete tool separation. Inspect the
result before subtraction or acceptance. Prompt editing state is session-local;
accepted masks retain the usual full-image coordinates and Tool IDs.

`draw 3` opens a polygon editor for tool 3. Click vertices around its visible
pixels, then click the first vertex to close each polygon. Use `+` for additive
pieces and `-` to remove background or holes; multiple polygons support
separated visible pieces. `z` undoes the last completed polygon, Enter saves,
and Esc or closing the editor cancels. Zoom using the Matplotlib toolbar before
drawing as needed, then turn off zoom/pan before placing vertices. The editor starts from the tool's current mask; use `u 3`
first if you want to start blank. Saving creates a new candidate and assigns
it to that Tool ID automatically. A correct SAM candidate can be assigned with
`g 3 INDEX` instead of drawing.

`subtract 0 0 3` means **assign tool 0 the pixels of candidate 0 minus tool 3's
current assigned mask**. It creates and assigns a new candidate, retaining the
source combined candidate unchanged. Exclude unused combined or duplicate
candidates with `x` before accepting. For several known masks, use a
comma-separated Tool ID list, for example `subtract 0 0 3,7`. Refined masks are
used when available. Drawing/subtraction replaces the target's assignment and
clears its crop refinement; other tools are unchanged. Original candidate
assignments can be restored with `g`.

Subtraction cannot infer a split when both inputs still contain all the tools.
At least one independently correct tool mask is required, and the combined mask
must cover the desired remaining tool. Missing or extra pixels in either input
can affect the remainder, so inspect both colored masks before accepting. Only
visible surfaces are annotated; subtraction does not fill hidden regions.
Operations are recorded in the review log, and final mask format is unchanged.
Manual edits are saved to masks on acceptance, not when quitting an unfinished
review.

Cleanup is **optional and off by default**. After prompting, merging, and
subtraction, use `clean TOOL_ID [OPERATION [SIZE]]` to postprocess the current
full-resolution tool mask, then inspect its zoom panel before `a`:

```text
review> clean 3
review> clean 3 open 3
review> clean 3 close 3
review> unclean 3
```

| Operation | Size meaning and default | Effect |
| --- | --- | --- |
| `small` (default) | Maximum region area, 16 pixels | Remove detached foreground specks and fill enclosed holes at or below the area limit. |
| `holes` | Maximum hole area, 16 pixels | Fill only small enclosed holes; leave larger holes and background connected to an image edge. |
| `specks` | Maximum component area, 16 pixels | Remove only small disconnected foreground components. |
| `open` | Odd kernel width, 3 pixels | Erode then dilate with an elliptical kernel to remove thin slivers and protrusions. |
| `close` | Odd kernel width, 3 pixels | Dilate then erode to fill narrow gaps, notches, and small holes. |
| `smooth` | Odd kernel width, 3 pixels | Apply opening followed by closing. |

For example, `clean 3 holes 8` fills enclosed defects of at most 8 pixels;
`clean 3 open 3` can remove long pixel-wide subtraction remnants that an area
filter would keep. Sizes refer to saved image pixels, not display pixels.
Foreground components use 8-connectivity and holes use 4-connectivity.

Repeated cleanup commands act on the current result. `unclean 3` undoes **all**
cleanup for tool 3, preserving any preceding crop refinement. Cleanup acts on the
union of assigned fragments, without changing candidate numbers or click
histories. The preview marks cleaned tools and reports added/removed pixel
counts. Growing operations cannot claim pixels currently assigned to other
tools or marked as ignored; existing overlaps are not resolved automatically.
Subtraction uses the current cleaned mask of each tool being subtracted.
Reassigning, drawing, reprompting that tool with clicks/polygons, or successfully
refining it clears its previous cleanup. `restore` clears both cleanup and crop
refinement. Changes are logged and only saved to the usual mask bundle on acceptance.

Inspect tips, teeth, finger holes, and disconnected visible portions carefully:
opening can erase real narrow features, closing can join tools or bridge an
occlusion gap, and area filters can remove small real portions or fill small
real openings. Cleanup cannot determine the correct semantic boundary. Use
additional model prompts or localized drawing for those errors. For best
subtraction results, correct the known tool first, subtract its current mask,
then clean the remainder only if needed.

In the simple single-tool reviewer, use `clean [INDEX] [OPERATION [SIZE]]` and
`unclean [INDEX]`; omit the candidate index only when exactly one candidate is
available. For example, `clean holes 8` or `clean 0 open 3`. No cleanup is applied
by `--auto-accept`, and existing saved masks are not rewritten by these commands.
These controls currently apply to image review.

| Command | Action |
| --- | --- |
| `g TOOL_ID 0,2` | Assign or combine candidates 0 and 2 into this tool's mask; replaces its previous assignment. |
| `u TOOL_ID` | Clear an assignment, its refinement, and its visibility state. |
| `click TOOL_ID [new\|INDEX\|reset]` | Revise the active SAM3 click prompt in place; `new` adds an independent portion, `INDEX` selects an earlier prompt, and `reset` clears the active prompt. |
| `polygon TOOL_ID` | Draw a coarse mask prompt and let SAM3 generate the boundary; follow with `click TOOL_ID` for corrections. |
| `draw TOOL_ID` | Trace/edit visible mask pixels using additive and subtractive polygons; automatically assign the saved result. |
| `subtract TOOL_ID BASE_INDEX OTHER_TOOL_IDS` | Assign the source candidate minus one or more other tools' current masks; separate the other Tool IDs with commas. |
| `clean TOOL_ID [OPERATION [SIZE]]` | Optionally clean the current tool mask; inspect its preview before accepting. |
| `unclean TOOL_ID` | Undo all cleanup for this tool, preserving its crop refinement and assignment. |
| `refine TOOL_ID [TEXT]` / `refine all` | Refine assigned tools using crops of the original image; inspect the updated panels. |
| `restore TOOL_ID` / `restore all` | Discard cleanup and crop refinements and restore the original assigned masks. |
| `order TOOL_ID ...` | Resolve overlapping pixels from front to back; include all overlapping tools, inspect the result, then accept separately. |
| `order reset` | Undo ordering while preserving source assignments, refinements, and cleanup. |
| `order impossible` | Reject an arrangement without a consistent front-to-back order and record the reason. |
| `inspect` | Reopen the original/before/after dispute close-ups after ordering; linked zoom/pan and inspection do not alter the masks or order. |
| `v TOOL_ID partial` | Set `clear`, `partial`, `severe`, `occluded`, or `absent`. Occluded means fully hidden; absent means not in the image. |
| `remove 1,3` | Discard candidates from the preview and active list without renumbering; affected tools become unresolved. |
| `x 1,3` | Exclude false or duplicate candidates, retaining them for possible reassignment. |
| `i 1,3` | Add candidates to the shared ignore region. |
| `p forceps` | Append predictions using another prompt while preserving existing assignments. |
| `n TOOL_ID "CLASS" "NAME" CONFIDENCE` | Declare an unexpected tool with complete metadata. Quote multiword class/name values. |
| `a` | Accept after every tool and candidate is resolved and final instance masks are non-overlapping. |
| `e` | Accept an empty image: explicitly mark all declared tools absent and exclude remaining candidates. |
| `s`, `r`, `q` | Skip, reject, or quit. |

A candidate can be assigned to only one tool. Clear its assignment before moving
it to another tool or excluding it. Each candidate must be assigned, excluded,
or ignored; each tool must have a visible mask or be explicitly occluded/absent.
Use `partial` or `severe` when some of the tool remains visible. Acceptance does
not infer physical identities or create unnamed reviewed tools.

Final masks describe **visible surfaces**. Grouping disconnected fragments takes
their pixelwise union without filling gaps or inventing hidden geometry. The
result occupies one mask entry and one object row, with a bounding box around
all its visible fragments. Separate tools retain independent masks with no
shared pixels. Correct boundaries or resolve disputed ownership with `order`
before accepting; review crossing regions carefully to avoid labeling hidden
surfaces. Overlaps between fragments assigned to the same tool are combined by
union and do not require ordering.

The CSV columns and version-2 `[N, H, W]` mask-bundle layout are unchanged.
Visible rows reference their declared Instance ID and exact bundle index. Absent
and fully occluded tools retain metadata and resolved visibility but have no
positive mask, blank mask paths/boxes, and index `-1`. They use
`Annotation Complete=TRUE` and `Reviewed=FALSE`: there is no positive mask to
review, and this metadata-only convention allows the existing flowsis training
builder to consume the image's visible tools. The review log records the human
resolution of every tool, including these nonvisible tools.

Assignments and fragment grouping are recorded in `image_review.jsonl`, with
source candidate IDs, indices, prompts, scores, and per-tool visibility. The
log also records front-to-back Tool IDs and Instance IDs, the number of disputed
pixels, per-tool removed-pixel counts, resets, invalidation by later edits, and
rejection of incompatible arrangements. Accepted decisions include the final
ordered Tool IDs; that list is empty when no ordering was needed. Ordering is
session-local, just like assignments.
The final mask bundle does not preserve fragments separately. Skipped/quit images
restart candidate generation and assignments on the next run; the log retains
previous actions for audit but does not restore an unfinished editing session.
The verifier checks completed rows for unresolved instances, invalid flags,
mask identity/index mismatches, overlapping instance pixels, bounding boxes, and
image dimensions. Saving also rejects overlapping instance masks. Ignore regions
are separate from instance ownership and may overlap instance masks; the
non-overlap requirement applies between different tool instances.

Existing mask bundles are not automatically rewritten. Legacy overlapping
bundles remain readable for inspection, but verification now reports them as
invalid and they must be reannotated before use under this policy. The version-2
NPZ layout and CSV columns remain unchanged.

### Single-tool video prototype

Put MP4 originals in `data/videos/<Tool ID>/`, for example `data/videos/12/opening.mp4`.
Each video must contain at most the one known physical instrument, already
identified in `image_object_manifest.csv`. The tool can be absent or fully
occluded in individual frames. Conflicting image class/name metadata is reported
for correction; identification confidence uses the lowest existing confidence
for that physical tool. Files directly in `videos/` require an explicit
`--tool-id` during preparation.

Update the environment to install PyAV, then extract every frame:

```shell
conda env update -n flowsis-data -f environment.yml
conda activate flowsis-data
python annotate.py prepare-videos
python annotate.py verify-videos
python annotate.py videos
```

Preparation does not download model weights. It writes PNG images at
`frames/videos/<Video ID>/000000.png`, `000001.png`, etc., preserving decoded
RGB pixels without resizing or cropping. Source display-rotation metadata is
not applied. Each frame's decoded presentation timestamp is stored explicitly;
frame times are not calculated from FPS. Audio is not extracted.

Two CSVs are authoritative:

| Manifest | Rows and fields |
| --- | --- |
| `video_manifest.csv` | One row per MP4: Tool ID/class/name/confidence, Video ID, source Filepath, FPS, actual decoded Frame Count, Width/Height, Capture Session ID, Condition, source SHA256, Extraction ID, and Extraction Complete. |
| `video_frame_manifest.csv` | One row per frame: Video ID, zero-based Frame Index, Timestamp Seconds, PNG Filepath, Frame ID, PNG SHA256, bounding box, Mask Path/Index, visibility, annotation/review flags, Review Status, and optional Anchor Path. Tool identity comes from the video row. |

There is no separate video object manifest in this single-tool prototype.
`Extraction Complete` must be true before annotation. Extracted frames initially
have blank annotation fields and pending review status. Capture sessions default
to distinct video session IDs; use `--capture-session-id` during preparation or
edit the video manifest to group captures from the same actual session.
Physical Tool IDs are retained for training split grouping.

Rerunning preparation verifies and reuses unchanged PNGs. Missing PNGs can be
recreated exactly from the unchanged MP4. A changed MP4 or modified PNG requires
`python annotate.py prepare-videos --reextract`; this clears affected masks,
anchors, and review decisions before extracting again. Use `--video-id VIDEO_ID`
to restrict preparation, annotation, verification, or export to one video.
Interrupted extraction resumes on the next preparation run.

The reviewer shows one frame with a green current mask, plus numbered text
candidates when requested. Start with `p` to use the known tool's name/class,
then `g 0` to select a candidate (`g 0,2` combines fragments). Alternatively,
use mouse prompts directly. A missing prediction remains unresolved; it is never
silently labeled absent or occluded. Text prompts change segmentation suggestions,
not the physical identity or training label.

| Command | Action |
| --- | --- |
| `p [TEXT]` | Generate text candidates for this frame, or the selected repair region. Omit TEXT to use the current tool prompt. |
| `g 0,2` | Select/group candidates into the current tool mask. The union preserves disconnected visible pieces and empty gaps. |
| `click` | Open the mouse editor: left click marks tool pixels; right click marks background. Clicks accumulate for the same frame and repair region during the reviewer session. Backspace undoes the last click; Enter generates a replacement; Esc cancels. Include positive examples of all desired visible parts. |
| `box` | Draw a box to guide segmentation. Enter applies it; Esc cancels. |
| `region` or `region L T R B` | Select a repair rectangle by dragging or providing pixel coordinates. Right/bottom edges are exclusive. Text, click, and box prompts operate on this crop; only pixels inside it are replaced. |
| `whole` | Return to whole-tool correction. Whole-tool prompts can change pixels anywhere in the frame. |
| `prop both 80 160` | From the current corrected frame, propagate backward and forward within the inclusive range 80–160. `forward` and `backward` are also supported. |
| `preview 80 160` | Inspect every frame's current mask in pages of up to 12 overlays. Close each page to continue. |
| `accept 80 160` | Accept nonempty masks in that inclusive range after previewing those exact mask versions. Changed masks require another preview. |
| `v partial`, then `a` | Set `clear`, `partial`, or `severe` and accept the current visible mask. `a` defaults to clear when no state was set. |
| `absent` / `occluded` | Explicitly accept this frame with no visible mask, distinguishing absence from a fully hidden tool. |
| `reopen 80 160` | Reopen reviewed frames in this inclusive range before replacing their masks. Existing corrected anchors remain available as inputs. |
| `jump 120` | Inspect another frame directly. |
| `next` / `s`, `r`, `q` | Leave this frame unfinished and continue, reject it, or quit the entire video command. |

For example, if early masks are good but tracking drifts around frame 120:

1. `jump 120`, then `reopen 80 160` if those frames were already reviewed.
2. Use `click` or region-only repair and inspect the resulting mask.
3. Run `prop both 80 160`.
4. Run `preview 80 160`, correct any remaining failures, then accept the good
   frames or ranges. Resolve missing masks explicitly.

Propagation uses the corrected **whole-tool** mask, including region-only repairs.
A repair rectangle itself does not follow the moving tool. The model can read
neighboring frames for context, but only the requested range is replaced.
Accepted/rejected frames are protected until explicitly reopened. Corrected
anchor masks are saved separately and reused together within each model window,
including after restarting the reviewer. Click markers are retained during the
current session; after a restart, the saved correction mask remains available
but the markers are not restored in the editor. Each direction is processed in bounded
chunks with overlapping context and a carried boundary mask. Defaults are
`--chunk-size 64 --context-frames 8`; lower them to reduce memory use. Tracking
can still drift across chunk boundaries or lose the tool; new predictions remain
pending human review. Add another anchor where tracking fails.

Final masks are independent version-2 bundles at
`masks/videos/<Video ID>/000000.npz`. A visible tool has one mask and identity
`tool-<Tool ID>`; absent/occluded frames have empty bundles. Each PNG and NPZ can
be loaded independently without opening the MP4. The append-only
`video_review.jsonl` records decisions, grouping, click coordinates (relative
to the crop when repairing a region), regions, extraction generation, and
propagation ranges. Final masks do not retain fragments separately.

`--auto-accept` is an explicit bypass for existing nonempty pending predictions;
it does not generate missing masks or infer absence. The first version uses
Matplotlib windows plus terminal commands, not a timeline editor. Its segmentation
and tracker backends load `facebook/sam3` on demand.

To train from extracted frames with the existing flowsis image reader, export an
image-style snapshot:

```shell
python annotate.py export-video-training
cd data/video_training
flowsis-build-annotated-dataset \
  --image-manifest image_manifest.csv \
  --object-manifest image_object_manifest.csv \
  --review-log image_review.jsonl \
  --image-mask-dir masks/images \
  --video-review-log /dev/null \
  --output-path ../../../flowsis/data/new/video-segmentation-dataset
```

The export keeps frame IDs, physical Tool IDs, and capture sessions. It includes
accepted clear/partial frames and explicit absent frames as negatives; severe
and fully occluded frames remain in the annotation manifests but are excluded
from this base-training snapshot. It writes one PNG/NPZ per included frame,
using hard links where possible and copies across filesystems. Atomic source
replacement cannot change an existing snapshot. Re-export after corrections;
use `--replace` to refresh an existing export or choose a new `--output-dir`.
The existing builder requires enough independent tool/session groups for its
requested splits; for a one-tool prototype use `--validation-size 0 --test-size 0`.
It reads the exported frame files, without decoding or needing the original MP4.
Do not pass the new video review log to the legacy MP4 training branch.

Use `python annotate.py --help` and each command's `--help` for options.

### Build training data in flowsis

The existing training builder resolves mask paths against its working directory.
Run the builder from `flowsis-data/data`, and use the environment where
`flowsis` is installed:

```shell
conda activate flowsis
cd data
flowsis-build-annotated-dataset \
  --image-manifest image_manifest.csv \
  --object-manifest image_object_manifest.csv \
  --review-log image_review.jsonl \
  --image-mask-dir masks/images \
  --video-review-log /dev/null \
  --output-path ../../flowsis/data/new/segmentation-dataset
```

Paths stored by annotation are relative to the dataset directory. Run the
training builder from that directory so both image and mask paths resolve.

Run annotation tests without downloading model weights:

```shell
python -m pytest tests -q
```

Mask paths mirror the image tree: `images/0/IMG_8585.jpg` produces
`masks/images/0/IMG_8585.npz`. Image IDs and review-log keys remain unchanged.
Images with the same stem in the same folder but different extensions must
be renamed to avoid mask collisions.

## Uninstallation

```shell
# If you cloned into a directory other than home, go there instead.
cd ~

rm -rf flowsis-data
```
