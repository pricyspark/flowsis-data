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

We will use Git pushes and pulls to sychronize data. I recommend briefly researching the basics of what Git is, and what pushes and pulls do. If you already have a preferred Git GUI, use that. Otherwise, this guide will give command line interface (CLI) instructions to enter into the terminal.

Put all the your new images into this repository's (abbreviated as repo) temp folder. You can do this by uploading to the Google Drive under images/{your name} from your phone or personal device. Then download the files to you computer that you are currently using. When downloading multiple files from Google Drive, it will zip them into a single file. Once it is zipped and downloaded, unzip the files, and place all of the images into the local temp folder. Create subfolders within raw-images and name them based on the tool IDs that you've photographed. For example, name them raw-images/10, raw-images/11/, raw-images/12, etc. Place your images in the subfolders accordingly, for example all images of tool 10 go in raw-images/10. Afterwards, check for any new data in GitHub using a pull.

To use the CLI, enter:

```shell
git pull
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

Now run:

```shell
python basic.py
```

The script lists each new tool ID and asks you for its class, name, and confidence. Tool classes and names are automatically changed to lowercase with punctuation removed. Confidence must be a number from 0 to 1. If it finds undocumented images for a tool ID that is already in the manifest, it warns you and asks whether to add or skip them. Pressing Enter skips that tool. If you continue, the script reuses the tool's existing class, name, and confidence. It reads the original files in `raw-images` to fill in the capture time, camera make, model, flash, ISO, exposure, aperture, and focal length in millimeters when that metadata is available. Capture times are stored with their date and timezone, for example `2026-09-01T19:51:50.430-04:00`. The metadata is written only to the CSV; it is not copied into the JPG files.

Use the two manifests as spreadsheets. The columns you need for manual review
are at the left; internal IDs and mask details are farther to the right.

| File | What to check and edit |
| --- | --- |
| `image_manifest.csv` | Sample ID, Angle, Filepath, Lights, and other capture details. |
| `image_object_manifest.csv` | Tool ID, Tool Class, Tool Name, and identification Confidence for each tool in an image. |

The image manifest starts with `Tool IDs`, `Sample ID`, `Angle`, `Filepath`, and
`Lights`. `Tool IDs` is a summary: `0` means one tool, and `0;12` means tools 0
and 12. Change physical tool identities in the **object manifest**, then run
`python basic.py` to refresh this summary.

The object manifest starts with tool ID, class, name, confidence, sample ID,
and filepath. Its sample ID and filepath are copied from the image manifest
so you can find an image without looking up its internal ID. Edit these two
fields in the **image manifest**, then run `python basic.py` to refresh the
copies. Do not edit internal Image ID or Instance ID values. Verification
reports summaries or references that disagree between files.

A **Tool ID** identifies one physical instrument. A **Sample ID** identifies
one setup photographed across rotations. Flipping a tool, changing its opening,
moving it to a new location, changing the lighting, or otherwise changing the
setup starts a **new sample**. Rotating the unchanged setup stays in the same
sample. For multiple tools, Angle describes rotation of the entire arrangement.

Keep `Angle` in the image manifest. The eight views are 0, 45, 90, 135, 180,
225, 270, and 315 degrees; 360 repeats 0. If filenames are out of order, correct
the angles beside the corresponding filepaths. For example:

| Tool IDs | Sample ID | Angle | Filepath |
| --- | --- | --- | --- |
| 0 | 15 | 0 | images/0/IMG_1002.jpg |
| 0 | 15 | 45 | images/0/IMG_1003.jpg |
| 0 | 15 | 90 | images/0/IMG_1001.jpg |

Run `python basic.py` after editing. It puts unfinished images (missing Sample ID or Angle)
at the bottom, and sorts each section by numeric Tool IDs. Rows with the same
tool IDs retain their existing order, regardless of sample, angle, or filename.
Object rows follow the same image order. It preserves
existing sample IDs and angles. Filename order only provides initial suggestions
for new, unassigned batches; check that each batch really is one unchanged setup.

`basic.py` creates a pending instance for each new single-tool image and reuses
existing tool details from object rows. Annotation fields stay blank until review;
identification `Confidence` is separate from `Annotation Confidence`. Existing
masks and review fields are preserved when adding images. Commit both manifests
together.

Multiple tools can be represented by additional instance rows with the same
`Image ID` and distinct `Instance ID` values. The current automatic folder
intake and annotation identity assignment are for single-tool captures;
multi-tool intake and explicit mask-to-tool assignment require a separate
workflow. Metadata-only instance rows are excluded from mask training until
reviewed. An accepted empty image retains its tool metadata without a positive
mask annotation.

Raw images are considered in consecutive filename slots of eight, including newly added raw images that do not have a converted JPG yet. A complete slot receives a unique sample ID and angles from 0 to 315 degrees. If conversion or cropping fails, the raw file continues to occupy its original slot. The other images in that slot are left ungrouped for review, but the failure does not shift the angles or prevent later complete eight-image slots from being assigned automatically.

Check both `image_manifest.csv` and `image_object_manifest.csv`. Confirm instance tool identities and confidence, image filepaths, sample groupings, and angles. In particular, confirm that each recorded Angle matches its image, regardless of filename order.

You can also run an automatic verification:

```shell
python verify_data.py
```

The verifier first checks the CSV and files for missing images, images that are not in the manifest, duplicate paths, tool IDs that disagree with their folder, invalid confidence values, incomplete samples, and missing or repeated angles. It then asks whether Codex should visually review temporary, downscaled contact sheets. The visual review looks for a different tool within a sample, suspicious rotation order, image-quality problems, and inconsistent class or name conventions. These findings are suggestions for a person to review; the verifier never changes the manifest. Temporary contact sheets are deleted automatically.

The visual review requires the Codex command-line program and a one-time ChatGPT sign-in. If `codex` is not installed, install it with:

```shell
curl -fsSL https://chatgpt.com/codex/install.sh | sh
codex login
```

The results are saved as `verification_report.html`, which you can open in a web browser, and `verification_report.json`, which contains the same results in a machine-readable format. To run only the CSV and file checks without Codex, use `python verify_data.py --static-only`.

Once the manifest is correct, use Git to synchronize it with GitHub. Before pushing, pull again to check for incoming changes.

```shell
git pull
```

Add the files we want to synchronize to GitHub:

```shell
git add *
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

After converting images and updating the manifest with `basic.py`, generate and
review SAM3 masks from the `flowsis-data` repository root:

```shell
# Update an existing environment after pulling the annotation changes.
conda env update -n flowsis-data -f environment.yml
conda activate flowsis-data
python annotate.py images --single-tool
```

The first run downloads `facebook/sam3` through Transformers. If access to the
checkpoint requires authentication, authenticate with Hugging Face using an
account authorized to download it. Interactive review requires a graphical
Matplotlib backend. CUDA is selected when available; use `--device cpu` to
select CPU explicitly. `--batch-size 1` reduces the image batch size.

`--single-tool` merges fragments such as the two sides of open scissors into
one object. Omit it for multi-tool images. Prompts use `Tool Class` by default;
use `--image-prompt name-and-class` for prompts such as "iris scissors".
A second pass refines a padded crop by default; disable it with
`--no-refine-crop`.

In the terminal review prompt:

- `a` accepts all candidates; `a 0,2` accepts selected candidates.
- `e` accepts an empty image; `r` rejects it; `s` skips it for later.
- `p TEXT` reruns segmentation with another prompt.
- `i 0,2` marks ignore regions; `v partial 0` sets an instance's visibility
  (`clear`, `partial`, or `severe`).
- `q` quits. Rerunning the same command resumes unfinished images.

Previously skipped images are shown after other unfinished images. Each group
keeps its manifest order, and each image is shown at most once per run. To work
through skipped images first, run:

```shell
python annotate.py images --single-tool --skipped-first
```

Basic preparation assigns `Image ID` and `Capture Session ID`. Annotation reads
tool metadata from instance rows. Review the suggested capture sessions before splitting
training data. Outputs are `image_object_manifest.csv`, `masks/images/<tool ID>/<image stem>.npz`,
and the append-only `image_review.jsonl`. Commit these together with the updated
`image_manifest.csv` so masks and resume decisions stay synchronized. Mask
bundles retain the format consumed by `flowsis` training. `--auto-accept` bypasses
human review and should only be used when that is intended.

Video annotation is also available with `python annotate.py videos --prompt
"scissors"`; it reads `videos/` and writes `video_manifest.csv`,
`video_object_manifest.csv`, `masks/videos/`, and `video_review.jsonl`.
Use `python annotate.py images --help` or `python annotate.py videos --help`
for all options.

### Build training data in flowsis

The existing training builder resolves mask paths against its working directory.
Keep running from the `flowsis-data` root, including after moving or cloning this
repository, and use the environment where `flowsis` is installed:

```shell
conda activate flowsis
flowsis-build-annotated-dataset \
  --image-manifest image_manifest.csv \
  --object-manifest image_object_manifest.csv \
  --review-log image_review.jsonl \
  --image-mask-dir masks/images \
  --video-manifest video_manifest.csv \
  --video-object-manifest video_object_manifest.csv \
  --video-review-log video_review.jsonl \
  --video-mask-dir masks/videos \
  --output-path ../flowsis/data/new/segmentation-dataset
```

Paths stored by annotation are relative to the directory where it runs, so use
the repository root consistently. Custom output directories must also remain
accessible from that directory.

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
