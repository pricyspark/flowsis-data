# FlowSIS Data

Open work: [support uncertain pixels and partially valid annotations in training](#deferred-work-partial-supervision).

## Installation

You will need Python and Git to use this repo. If you do not have Python, you can try and install it. It is easiest on Linux and MacOS. On Windows, I recommend using WSL2 to emulate Linux. If you are comfortable with a terminal and command line I recommend installing Python with Miniforge, otherwise you can use Anaconda, which has a graphics user interface (GUI). Installation for both of these is relatively simple. If you cannot install Python use the lab PC, which runs Linux. Similarly, install Git based on your system.

Once you have Python and Git, in your command line (called Terminal in MacOS and Linux) type:

```shell
# Navigate to home. If you prefer to clone to somewhere else, navigate there.
cd ~
# Clone the repo to your local machine
git clone https://github.com/pricyspark/flowsis-data.git
cd flowsis-data

conda env create -f environment.yml -y
conda activate flowsis-data
```

`environment.yml` already installs `dvc` and `dvc-gdrive`; no separate pip install
is needed. To update an existing environment:

```shell
conda env update -n flowsis-data -f environment.yml
conda activate flowsis-data
```

Continue with [DVC authentication and first download](#dvc-authentication-and-first-download)
to retrieve the dataset. Cloning Git alone does not download `data/`.

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
  video_manifest.csv
```

Paths stored inside manifests remain relative to `data/`, such as
`images/12/photo.jpg` and `masks/images/12/photo.npz`. Do not prefix these values
with `data/`. Annotation commands default to `--data-dir data`; their dataset
path options are relative to that directory, while absolute paths are preserved.
Use `--data-dir /path/to/dataset` to annotate a standalone dataset.

### DVC authentication and first download

The repository already contains DVC initialization, the `storage` remote in
`.dvc/config`, and the dataset pointer `data.dvc`. On a new clone, use those
settings; there is no need to run `dvc init`, `dvc remote add`, or `dvc add`.
The remote uses Google Drive (`gdrive://`), with a custom Google OAuth desktop
application. Its Google Cloud project enables access to Drive; the data itself
is stored in the shared Drive folder, not in a Google Cloud Storage bucket.

Before connecting, ask the project maintainer for:

- Access to the shared Drive folder using your Google account. Viewers with
  downloads enabled can pull; Editors can pull and push. GitHub access does not
  grant Drive access. Share with named accounts or a Google Group;
  DVC does not guarantee support for "Anyone with the link" sharing.
- The project's OAuth client ID and client secret, supplied outside Git.
- An entry for your exact Google account in the OAuth app's test-user list if
  the app's publishing status is **Testing**.

Configure the OAuth values once per clone on each machine, replacing the
placeholders with the values supplied by the maintainer:

```shell
dvc remote modify --local storage gdrive_client_id 'YOUR_CLIENT_ID'
dvc remote modify --local storage gdrive_client_secret 'YOUR_CLIENT_SECRET'
git check-ignore .dvc/config.local
```

Always use `--local` for these values. It writes them to `.dvc/config.local`,
which `.dvc/.gitignore` excludes from Git. Without `--local`, DVC writes to the
shared `.dvc/config`; committing OAuth values there can trigger GitHub's secret
scanning. Never commit personal authentication tokens or credential files.

Download the dataset and check its local status:

```shell
python dvc_gdrive_sync.py pull
dvc status
```

On the first connection, follow the Google login link DVC prints or opens.
Choose the account granted Drive access and, when applicable, test-user access.
Complete consent and let the browser return to DVC's localhost callback while
the command is still running. DVC caches your personal authorization locally;
each collaborator signs in individually. See the
[DVC Google Drive authentication documentation](https://doc.dvc.org/user-guide/data-management/remote-storage/google-drive#authorization).

The download restores images, masks, manifests, and review logs under `data/`.
Files there appear ignored or grayed out in Git-aware editors because DVC tracks
them as a dataset; Git tracks the small `data.dvc` pointer. `git status` alone
cannot show unsnapshotted data changes; use `dvc status` too.

### Daily synchronization

Use `sync_data.py` as the standard workflow for dataset contributors. Run it
before collecting data or annotating to get the latest dataset, and again when
your captures or edits are ready to share.

```shell
python sync_data.py
```

The same command handles routine updates and divergent dataset commits. It saves
your local dataset before downloading incoming changes, combines compatible
additions and manifest edits, and publishes any resulting dataset changes.
It asks for a message when it needs to create a commit. You can also supply one
with `python sync_data.py -m "Jane Doe added tool 23 videos"`. A routine update
with no local changes can bring your checkout up to date without a new commit.

| What you want to do | Workflow |
| --- | --- |
| Get the latest dataset before working or publish your dataset edits | Run `python sync_data.py` on your normal tracking branch. This workflow needs write access to Git and the DVC remote. |
| Combine local and incoming dataset changes | Run the same command. It merges compatible changes and reports conflicts that need a decision. See [Publish dataset changes](#publish-dataset-changes). |
| Set up a fresh clone | Follow [DVC authentication and first download](#dvc-authentication-and-first-download). |
| Update a checkout with read-only access | On a checkout with no unpublished Git or dataset changes, run `git pull --ff-only` and then `python dvc_gdrive_sync.py pull`. |
| Work on code or documentation | Use your normal Git workflow for code commits and branches. Use `sync_data.py` for dataset changes. |
| Restore a specific dataset version or troubleshoot storage | Follow [DVC troubleshooting and older dataset versions](#dvc-troubleshooting-and-older-dataset-versions). |

Treat `sync_data.py` as a command that can publish. It can push existing local
commits on the tracking branch, including code commits. Use it when those
commits and any local dataset edits are ready to share. Commit code changes
separately and clear any staged changes or unfinished Git merge or rebase
before syncing. Pause collection and annotation while the command runs.

### Maintainer setup for a replacement Google Drive remote

These steps are only needed when creating a replacement remote or OAuth app.
The existing remote is already configured.

1. Create a dedicated Drive folder and share it with collaborators' Google
   accounts or a Google Group. Give contributors Editor access. DVC stores
   content-addressed cache objects there; use `dvc pull` to reconstruct the
   normal dataset layout rather than editing those remote objects in Drive.
2. In [Google Cloud Console](https://console.cloud.google.com/), select or create
   a project. Under **APIs & Services → Library**, enable **Google Drive API**.
3. Under **Google Auth Platform**, configure **Branding** with the app name and
   contact details. In **Audience**, select the audience appropriate to your
   team. For an External app in **Testing**, add each collaborator's exact login
   email under **Test users**, including your own.
4. Under **Clients**, create an OAuth client of type **Desktop app**. Obtain its
   client ID and secret. In **Data Access**, configure the scopes DVC requests:
   `https://www.googleapis.com/auth/drive` and
   `https://www.googleapis.com/auth/drive.appdata`. These allow broad Drive
   access; collaborators should understand the consent request.
5. Set the remote folder URL. `FOLDER_ID` is the ID in the Drive folder's web
   address, not the Google Cloud project ID or OAuth client ID:

   ```shell
   # For a new remote only:
   dvc remote add --default storage gdrive://FOLDER_ID
   # To change the existing storage remote instead:
   dvc remote modify storage url gdrive://FOLDER_ID
   ```

6. Configure the OAuth values with the `--local` commands above, then upload:

   ```shell
   dvc status
   python dvc_gdrive_sync.py push
   python dvc_gdrive_sync.py status --cloud
   ```

   If the dataset itself changed, run `dvc add data` before pushing. Commit the
   shared remote URL in `.dvc/config` and any updated `data.dvc`. Keep OAuth
   values in `.dvc/config.local`; distribute them separately to collaborators.

Follow the [DVC Google Drive setup guide](https://doc.dvc.org/user-guide/data-management/remote-storage/google-drive)
for additional authentication options. External apps in Testing have seven-day
authorizations, including refresh tokens, so users may need to authenticate
again. Publishing an app and completing Google's verification are separate
steps; changing publishing status does not automatically verify Drive scopes.
See [Google's audience and publishing-status documentation](https://support.google.com/cloud/answer/15549945?hl=en).

### DVC troubleshooting and older dataset versions

| Symptom | What to check |
| --- | --- |
| Google blocks the default DVC app | Configure this project's custom OAuth client with `--local`, as above. |
| `403: access_denied`, mentioning developer-approved testers | Match the login email to **Audience → Test users**. Confirm the authorization request's client ID belongs to that same Cloud project, then retry with a fresh `dvc pull` or `dvc push`. |
| An unverified-app warning appears | Allowed testers may still see a warning. Confirm it names the expected app; use the continuation option if Google offers one. A blocked page without that option needs its stated access restriction resolved. |
| Organization policy blocks authorization | Ask your Google Workspace administrator to allow the app; adding a test user does not override organization restrictions. |
| Login stops working after about seven days | The app may still be in Testing. Reauthorize; consult the DVC authentication guide if cached credentials prevent a fresh login. |
| Login succeeds but data access fails | Verify Drive folder access for the account used, and Editor access for uploads. OAuth test-user access and folder permissions are separate. |
| Missing files or cache objects during `dvc pull` | Ask the contributor who published the pointer to run `python dvc_gdrive_sync.py push` for that version. |
| GitHub blocks a push containing OAuth values | Move them from `.dvc/config` to `.dvc/config.local` and remove them from every affected outgoing commit, not only the current file. |

If Google Drive returns HTTP 400 `Bad Request` while listing `drive/v2/files`,
use the repository wrapper:

```shell
python dvc_gdrive_sync.py status --cloud
python dvc_gdrive_sync.py push
```

For this remote, PyDrive2 1.21.1 combines hundreds of parent folders into one
listing query that Drive rejects. The wrapper limits each query to 32 parents
and delegates the remaining command to DVC. It changes neither installed
packages nor credentials, and can also run `pull` or `fetch`. This workaround
was verified against the failing remote; it does not address unrelated HTTP
400 errors during uploads.

To restore a prior dataset, preserve local work, check out its Git commit, and
run `dvc pull`. Git selects the version pointer; DVC restores the matching data.
You can return to the latest version with `git switch main` followed by
`dvc pull`. Referenced objects must still exist in the remote or local cache.

The local `.dvc/cache/` may use additional disk space. Do not manually delete
objects from the shared Drive folder or run remote garbage collection without
coordinating which historical dataset versions everyone needs to retain.
Migrating to DVC does not remove old media from Git history; that is a separate,
coordinated history-cleanup operation.

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
`data/raw-images/multi/setup-001/IMG_1001.HEIC` becomes
`data/images/multi/setup-001/IMG_1001.jpg`.

Pull incoming code and dataset changes before making local edits:

```shell
python sync_data.py
```

Convert and crop the raw images. The converter isolates the blue marker in each image quadrant, matches its inner corner with an L-shaped convolution kernel, applies a perspective correction, and saves the finished 2160×2160 JPG in `images`. If a marker is missing, incomplete, or does not match the kernel reliably, it does not create the JPG and lists the failed image and reason after processing finishes.

```shell
python img_convert.py
```

Reviewed camera-orientation corrections are stored in `data/image_orientation_corrections.json`. Keys are JPG paths relative to `images/` (for example, `24/IMG_2762.jpg`); values are clockwise rotations of 0, 90, 180, or 270 degrees applied after cropping. The converter applies these corrections whenever it creates a JPG, including forced reconversion, while preserving the raw originals. When correcting an existing annotated JPG, rotate its masks and ignore regions and recompute bounding boxes with the same transformation.

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
`multi` as a Tool ID, creates inferred instances there, or assigns angles from
its filenames. It reports converted multi-tool images that have
not been registered, and registered multi-tool images with no instance rows,
with instructions to run `annotate.py declare`. These images do not interrupt
single-tool preparation. Once every image in a setup folder is declared with
the same tool set, preparation automatically assigns one globally unique
Sample ID to that folder. Added images reuse the folder's existing ID. Separate
setup folders receive separate IDs even if their tool sets match; nested image
subdirectories remain part of their top-level setup. Conflicting tool sets,
multiple existing IDs, or an ID used outside the setup are reported for review
without changing that setup's assignments. Declared images receive available
camera metadata and refreshed navigation columns; identities, masks, existing
sample IDs, and angles are preserved.

Use the two manifests as spreadsheets. The columns you need for manual review
are at the left; internal IDs and mask details are farther to the right.

| File | What to check and edit |
| --- | --- |
| `image_manifest.csv` | Sample ID, Angle, Filepath, Lights, and other capture details. |
| `image_object_manifest.csv` | Tool ID, Tool Class, Tool Name, optional Manufacturer, and identification Confidence for each tool in an image. |

The image manifest starts with `Tool IDs`, `Sample ID`, `Angle`, `Filepath`, and
`Lights`. `Tool IDs` is a summary: `0` means one tool, and `0;12` means tools 0
and 12. Change physical tool identities in the **object manifest**, then run
`python prepare_manifests.py` to refresh this summary.

The object manifest starts with tool ID, class, name, manufacturer, confidence, sample ID,
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

To renumber assigned samples consecutively from zero in numeric Tool ID order, run
`python prepare_manifests.py --renumber-samples`. This preserves sample membership,
angles, and image/instance IDs and refreshes the Sample ID copies in the object manifest.
Unassigned images remain unassigned. Visual review validity is independent of sample
numbers; the next verification run updates the active saved reviews to the new numbers.

`prepare_manifests.py` creates a pending instance for each new single-tool image and reuses
existing tool details from object rows. Annotation fields stay blank until review;
new tools prompt for an optional Manufacturer (press Enter to leave it blank).
Manufacturer is preserved as metadata and excluded from model prompts and class labels.
Keep design names in Tool Name, with lowercase words separated by underscores;
store brands separately (for example, name `leksell`, manufacturer `jarit`).
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

Visual reviews are saved after each successful batch in `data/visual_review_log.json`.
Keep this file with the manifests to share completed reviews across machines.
The log records the image IDs, paths, content hashes, physical tool IDs, labels,
angles, review policy, requested model, timestamp, and findings for each sample.
Unchanged samples reuse their saved findings, including warnings; the reports mark
them as reused and retain the original review timestamp. Static checks always run.
If every check is reusable, verification does not start Codex or ask for confirmation.

Adding, removing, or changing an image within a sample, editing its labels or angles,
or changing the review instructions or requested model triggers another review of
that entire sample. The complete sample provides context for tool consistency and
rotation checks. New samples do not invalidate other samples from the same tool.
Sample renumbering, row order, manufacturer, and mask-annotation edits do not invalidate
visual checks. Class/name taxonomy suggestions are cached separately, using individual
object labels rather than joined multi-tool labels, so taxonomy changes alone do not
repeat image reviews. The review prompt explicitly describes the surgical context,
including the surgical lineman pliers subtype for Tool ID 11.

Use `python verify_data.py --force-review` to review all assigned samples and taxonomy
again, or `--review-log PATH` to choose another log. Failed or invalid agent responses
are not cached. An unreadable or corrupt log stops visual review before any paid calls;
repair it or select a new log. Older reports without input fingerprints cannot establish
that a previous review still applies and are not imported automatically. The first
visual run establishes the fingerprinted baseline. `--static-only` leaves this log untouched.

### Publish dataset changes

Once your captures and manifest edits are ready, run:

```shell
python sync_data.py
```

Enter your name and the tools/samples added or corrected when prompted. The
script snapshots your local work before retrieving incoming data. It combines
independent file additions and uses a three-way comparison of manifest records
and fields. Image records match on `Image ID`; object records match on `Image ID`
and `Instance ID`; videos match on `Video ID`. Review logs retain distinct events.
It detects independently assigned Sample IDs that would combine different groups.

After combining data, the script checks image/object identities, file references,
and sample-ID collisions. It does not require complete eight-angle groups or
run the full annotation-quality review. Use `python verify_data.py --static-only`
separately for those checks. The script uses DVC to generate `data.dvc` and
creates a local Git commit when needed. Before pushing Git, it uploads through
`dvc_gdrive_sync.py` and verifies remote availability for the dataset snapshots
referenced by the commits it is about to publish.
If another person publishes first, it combines the new incoming changes and
retries up to three times. It never force-pushes or edits hashes by hand.

If synchronization stops, read the reported file/record and both values. Conflicting
edits, edit/delete disagreements, different captures at the same path, Sample ID
collisions, and code conflicts require a decision. Correct or rename the affected
files in `data/` and rerun; ask the maintainer if the decision is unclear. A missing
published snapshot requires its original contributor to run
`python dvc_gdrive_sync.py push` on the machine with that snapshot.

Merge attempts run in separate checkouts under `.data-sync/`; your working dataset
is not replaced until publication succeeds. The directory is ignored by Git and
contains recovery pointers and, after success, the previous dataset in
`data-before-sync/`. Allow disk space for a working copy of the combined dataset.
If the final local replacement is interrupted, the next run stops and identifies
the recovery directory; ask the maintainer to finish applying the published
snapshot before continuing. This prevents an incomplete replacement from being
interpreted as intentional deletions. Keep recovery files until you have confirmed the result. Do not run DVC cache
garbage collection during recovery. Failed attempts may also retain a combined
working copy. Local OAuth configuration is removed from temporary checkouts when
an attempt exits normally or reports an error.

For maintainers publishing manually, always upload successfully before publishing
the Git pointer:

```shell
dvc add data
python dvc_gdrive_sync.py push
git add data.dvc
git commit -m "Name: describe dataset changes"
git push
```

Manual publishing requires reconciling incoming changes first. Do not choose one
side of a `data.dvc` conflict to discard the other person's dataset.

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
select CPU explicitly. `--batch-size 1` reduces the inference batch size; all
pending images of a single-tool sample are still predicted before inspection.

Review mode is selected from each image's declared tools, so one command can
review both single-tool and multi-tool images. Single-tool images with a Sample ID
are reviewed together: SAM3 predicts every pending view, then a tiled preview
shows images numbered from **0** in their queue order. Completed images are
excluded on resume, and numbers are assigned again for the remaining views.
Each tile also labels its mask candidates separately. Images without a Sample ID
retain individual review. Neither single-tool interface needs Instance IDs.
Predicted masks are **not merged** by default: accept the one correct mask for
each image. Multi-tool images retain explicit instance assignment.

Optional `--merge-masks` merges predicted
fragments into one mask during inference. It is only for captures known to need
fragment union, and cannot be used with pending multi-tool images. Prompts use `Tool Class` by default. When two or more declared tools share a
class within an image, their prompts automatically include names, such as
"iris scissors" and "mayo scissors". Classes unique within the image retain
class-only prompts. Use `--image-prompt name-and-class` to include names for all
tools. Identical resulting prompts run once, while physical identities remain
separate assignments.
A second pass refines a padded crop by default; disable it with
`--no-refine-crop`.


In the sample review prompt:

- `a` accepts all remaining images when each has exactly one nonignored,
  nonempty candidate. Ambiguous or missing masks must be resolved first.
- `a X [MASK]` accepts image X, selecting the numbered mask when needed.
- `focus X` opens image X by itself. Use `p TEXT`, `clicks [MASK]`, `draw [MASK]`,
  or cleanup commands without an image number while focused. `a [MASK]` accepts
  just that image and returns to the overview; `s` skips it. `back` returns to
  the overview without deciding. Image numbers stay the same.
- `p TEXT` re-prompts pending images in the sample, preserving accepted images;
  `p X TEXT` explicitly re-prompts only image X, reopening it if already accepted.
- `clicks X [MASK]` opens foreground/background clicks on image X, then runs SAM3.
  `draw X [MASK]` traces a polygon prompt for SAM3 on that image. An optional mask
  number selects the initial overlay (and initial polygon drawing).
- `e X` explicitly accepts image X as empty. `r X` rejects it; `s X` skips it.
  Bare `r` or `s` applies to all remaining pending images.
- `i X 0,2` marks image X's candidates 0 and 2 as ignored; `i X` clears ignores.
  `v X partial [MASK]` sets visibility (`clear`, `partial`, or `severe`).
- `clean X [MASK] [OPERATION [SIZE]]` applies optional cleanup to an image's mask;
  `unclean X [MASK]` restores it.
- `q` quits, saving images already accepted in the sample. Pending images remain
  unfinished. Rerunning the command resumes them.

In the overview, every image-specific edit or spatial prompt requires an image
number. Focused commands apply only to the focused image. Tile coordinates are
only for display; clicks and drawings use original image pixels.

For individual images without a Sample ID, the terminal commands remain:

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

Previously skipped images are queued after other unfinished images. Single-tool
samples stay together, with sample priority determined by their first queued image;
other views of that sample join the same review even if previously skipped.
Each image is shown at most once per run. To prioritize skipped images, run:

```shell
python annotate.py images --skipped-first
```

Basic preparation assigns `Image ID` and `Capture Session ID`. Annotation reads
tool metadata from instance rows. Review the suggested capture sessions before splitting
training data. Outputs are `image_object_manifest.csv`, `masks/images/<tool ID>/<image stem>.npz`,
and the append-only `image_review.jsonl`, all under `data/`. Snapshot these together
with the updated `image_manifest.csv` using `dvc add data`, then publish the dataset
and pointer as described above so masks and resume decisions stay synchronized. Mask
bundles retain the format consumed by `flowsis` training. `--auto-accept` bypasses
human review and should only be used when that is intended.

### Visualize saved masks

Export PNG previews without running SAM3 or opening a graphical window:

```shell
# Preview the first 32 saved image bundles.
python visualize_masks.py
# Preview every saved bundle for one tool.
python visualize_masks.py data/masks/images/12 --count 0
# Preview one image or extracted video frame.
python visualize_masks.py data/masks/images/12/IMG_1001.npz
python visualize_masks.py data/masks/videos --count 32
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

1. Place originals under `data/raw-images/multi/<setup>/`, grouping images with the
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

4. Keep each setup folder limited to the same physical tools and overall setup,
   even when tool positions and rotations vary. Preparation assigns one Sample
   ID per fully declared folder; any image count is valid. Leave `Angle` blank
   for these randomized captures. Review each tool's ID, class, name, and
   identification confidence in `image_object_manifest.csv`.
5. Run `python prepare_manifests.py` to assign the setup's Sample ID, fill
   available camera metadata, and refresh navigation columns. It preserves
   existing sample IDs and angle values.
6. Run `python verify_data.py --static-only` and resolve manifest errors before
   annotation.
7. Run `python annotate.py images` to assign and review masks.

You may run `prepare_manifests.py` before step 3; it prepares single-tool captures
and lists the multi-tool images that still need declaration. It does not infer
how many tools are present: reviewers must declare **every** physical tool, even
if an image already has one instance row.

For individual images, use `--filepath images/multi/setup-001/IMG_1001.jpg`
(repeat `--filepath` for several images). Alternatively, `--sample-id 15` targets
every image of an existing sample, including a randomized
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
To review only single-tool images, skipping multi-tool captures:

```shell
python annotate.py images --single-tool
```

The two filters are mutually exclusive. Without either flag, both kinds of images
are reviewed in manifest order. `--single-tool` only filters images; add
`--merge-masks` separately if predicted fragments should be merged.

To jump directly to a tool or sample:

```shell
python annotate.py images --tool-id 12
python annotate.py images --sample-id 34
python annotate.py images --tool-id 12 --sample-id 34 --single-tool
```

Tool IDs are matched against declared instances in `image_object_manifest.csv`;
Sample IDs are matched against `image_manifest.csv`. Combining the filters selects
images that match both. Use `--tool-ids 12 13` or repeat `--tool-id` to select images
containing any requested tool. Repeat `--sample-id` to select any of several samples.
For multi-tool images, the whole image and all declared instances are reviewed.
Completed images remain skipped; `--skipped-first` still controls queue priority.
If no unfinished images match, the command exits without loading the model.

Run multi-tool annotation **without** `--merge-masks` or `--auto-accept`.
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

Multi-tool review uses a persistent Matplotlib canvas. Select the physical tool
at left, then select a proposal or choose a correction mode. The overview keeps
its zoom and tool selection between operations; the right panel shows the
selected tool at a closer scale. Tool colors remain stable across assignments.
Other views of the same sample appear as reference thumbnails, with accepted
views first. Useful text prompts from accepted views are included in subsequent
initial predictions for that sample; masks are predicted independently.
Terminal commands remain available alongside the canvas.

Every successful edit saves an atomic checkpoint in `image_review_state/` beside
the review log (a custom log name uses `<log stem>_state/`). It includes actual
candidate and edited masks, assignments, points, visibility, locks, crossing
ownership, and up to 40 undo states. Skipping, quitting, or closing the canvas
retains this draft. Rerunning the command restores it without rerunning initial
predictions. Changed source pixels, tool metadata, model, or threshold invalidate
a stale draft, as do changes to configured text prompts. Completed images retain
their normal mask bundles and manifest records; their draft is removed after
saving. Older JSONL logs alone cannot
restore edits made before mask checkpoints were introduced.

**Reset the current image:** click **Reset image** or enter `reset`. This restores
the original predictions and declarations and clears all assignments, corrections,
click histories, locks, exclusions, and crossing decisions for this image.
Reset is undoable. Click Undo/Redo, use Ctrl+Z/Ctrl+Y on the canvas, or enter
`undo`/`redo`; the bounded history also survives reopening a draft.

For routine editing, select a tool and choose a canvas mode:

- **Select proposal:** click a mask; Shift-click adds another fragment. Near-identical
  proposals share one outline and a label such as `2/5`; source numbers remain
  available in the terminal and the Masks field.
- **Correct mask:** stage positive left-clicks on missing tool pixels and negative
  right-clicks on leakage, then Apply. The current mask seeds SAM3; compatible
  consecutive corrections reuse decoder logits and cached image embeddings.
- **Add piece:** click a disconnected visible portion and Apply. Its prediction is
  added to the tool while preserving its existing visible pieces.
- **Box:** drag a rectangle around the target and Apply for a SAM3 box prompt.
- **Brush add/erase** or **Add/Erase polygon:** stage a local pixel edit and Apply.
  Brush radius is in original image pixels. These edits do not rerun segmentation.
- **Split piece:** select the source tool, enter the destination Tool ID in the
  Split to Tool field, draw a polygon, and Apply to transfer those visible pixels.
- **Crossing owner:** select the foreground tool and click a disputed region.
  Different crossings can have different owners.

Cancel clears staged geometry without altering the masks. Lock/unlock protects a
finished tool from edits; newly added local pixels cannot overwrite locked tools.
Discard unused first previews the unused proposals, then a second click excludes
them. Undo restores exclusions. Prompts and model scores remain suggestions;
the reviewer supplies physical tool identities.

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

The persistent canvas uses stable colors for physical tools. Unassigned proposal
outlines retain candidate colors and numbers; near-identical proposals are grouped
for display without deleting their source masks. The legacy tool-preview function
continues to support candidate-colored panels.

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

The overview shows current tool masks, optional unused proposal outlines, and
magenta disputed regions. The selected tool's crop shows its mask and, when
available, yellow outlines of changes from its assigned source. Toggle Show
proposals to inspect boundaries without alternative predictions in the way.

Local commands use original image pixels:

```text
review> correct 3 1020,780,1 1040,820,0
review> add 3 1200,900,1
review> box 3 700 500 1400 1300
review> brush 3 erase 4 1030,800 1040,810
review> paint 3 erase 1000,800 1050,800 1050,850 1000,850
review> split 3 0 1000,800 1050,800 1050,850 1000,850
review> lock 3
review> unlock 3
review> discard
review> discard confirm
```

Point labels are 1 for the tool and 0 for background. `correct` replaces the
selected tool's mask using its current mask as a prompt; `add` unions a new
prediction with its visible pieces. Brush and polygon edits preserve pixels
outside the selected region. General undo/redo covers these commands.

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
Successful edits retain crossing ownership wherever the claimant masks are
unchanged. Newly changed disputed pixels require inspection and a fresh ownership
decision; canceled or invalid edits preserve the previous result. Skipping or
quitting saves the unfinished state, including existing ownership decisions.
`order reset` clears both global ordering and locally selected ownership.

An order cannot silently erase a tool's whole mask. If this would happen, the
command reports the tool and retains the previous order. Fix duplicate or
incorrect assignments, choose another physically correct order, or explicitly
mark a genuinely fully hidden tool with `v TOOL_ID occluded` and resolve its
unused candidates. Never mark a visible tool occluded just to pass validation.

If tools pass over and under each other, select Crossing owner on the canvas,
or enter `own TOOL_ID X Y` at each crossing. The selected tool keeps the disputed
pixels it claims in that connected region; other crossings remain unchanged.
This avoids requiring one global order. `order impossible` still rejects an
unusable arrangement. Physical depth is supplied by the reviewer; side-by-side
tools with overlapping predictions need boundary corrections.

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

Place originals in `data/raw-videos/<Tool ID>/`, then run `python video_convert.py`.
The first-frame preview lets you rotate with `r`/`l`, accept with `a`/Enter,
or quit with `q`/Escape. Check that its green outline follows the four marker
inner corners before accepting. Converted MP4s go to `data/videos/<Tool ID>/`;
existing outputs are skipped unless `--force` is supplied.

The detector evaluates separate blue components within each quadrant. Confident
blue pixels seed growth into connected weaker-blue pixels to tolerate pale tape;
strict-color candidates are also retained for noisy or low-light captures. The
video converter then derives moving search neighborhoods from validated marker
bounds, marker size, recent motion, and decoded timestamps. Every frame still
requires four observed markers. Prediction only guides the search; it does not
replace missing observations. The crop uses each frame's detected coordinates,
with the existing inward margin; Gaussian output smoothing is not integrated.

A failed marker check rejects the whole converted video and saves the oriented,
uncropped frame plus JSON context under
`data/videos/<Tool ID>/conversion-failures/<video stem>/`. The warning prints
the PNG path; the JSON includes the attempted search bounds when tracking had
already initialized. Originals remain untouched. `--source-dir` and
`--output-dir` can override the default directories. Failed forced conversion
removes the old MP4, consistent with image conversion.

See [marker detection experiments](marker_detection_experiments.md) for the
tested approaches, known limitations, and suggested follow-up tests.

Converted MP4s use paths such as `data/videos/12/opening.mp4`.
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
anchors, and review decisions before extracting again. Use `--video NUMBER`
to restrict preparation, annotation, verification, or export to one video.
Interrupted extraction resumes on the next preparation run.

List the permanent video numbers, tool IDs, accepted-frame counts, and filenames:

```shell
python annotate.py list-videos
python annotate.py videos --video 4
python annotate.py verify-videos --video 4
python -u batch_propagate_videos.py --video 4 --auto-preview
```

Numbers are assigned on preparation, or on the first listing/selection of an
existing dataset. They are saved in `data/video_manifest.numbers.json`, keyed
by the original hash-based Video ID. Reordering the manifest does not change
numbers; removing a video reserves its old number, and new videos receive new
numbers. Keep this small registry with the dataset and include it in DVC updates
and selective collaborator downloads. Do not delete/regenerate it. A custom
manifest uses the same filename with `.csv` replaced by `.numbers.json`.
Frame IDs, mask paths, and saved annotations continue to use their existing hash
identifiers. Legacy `--video-id VIDEO_ID` selection is also supported, but cannot
be combined with `--video`.

Reviewers print a short command hint when opening an image or video, rather than
repeating the full command list at every prompt. Refer to the command tables in
this README for syntax and less common operations.

The reviewer shows one frame with a green current mask, plus numbered text
candidates when requested. The frame view and saved previews zoom to a padded
crop around the mask/candidates by default; frames without a visible mask use
the full view. Use `view full` for context and `view crop` to restore zoom.
Display zoom preserves full-frame pixel coordinates and does not set a repair
region. Start with `p` to use the known tool's name/class,
then `g 0` to select a candidate (`g 0,2` combines fragments). Alternatively,
use mouse prompts directly. Selecting text candidates runs a second SAM3 pass
on a padded crop of the selected mask, using the same overlap checks and crop
expansion as image annotation. Ambiguous or clipped refinements retain the
original proposal. Use `--no-refine-crop` to disable this second pass. Region-only
text repairs refine inside the selected repair rectangle and preserve pixels
outside it. A missing prediction remains unresolved; it is never
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
| `prop forward 10` | Propagate 10 frames beyond the current frame, stopping at the video boundary. `backward 10` goes back 10 frames; `both 10` goes up to 10 frames in each direction. `propr` is also accepted as an alias for `prop`. |
| `preview 80 160` | Save PNG contact sheets with up to 12 labeled mask overlays per page. Generate the whole range without opening windows or waiting between pages; the terminal prints the output directory. |
| `accept 80 160` | Accept pending nonempty masks and reaccept changed/unverified accepted masks in that inclusive range after previewing their current versions. Unchanged accepted frames are preserved; rejected frames must be reopened. Prints newly accepted, reaccepted, and preserved frame ranges. |
| `accept` | Accept all pending nonempty masks and reaccept changed/unverified accepted masks in the current video without requiring previews. Preserve existing visibility (default: clear). Unchanged accepted/rejected frames, empty masks, and frames without masks are skipped. |
| `v partial`, then `a` | Set `clear`, `partial`, or `severe` and accept the current visible mask. `a` defaults to clear when no state was set. |
| `absent` / `occluded` | Explicitly accept this frame with no visible mask, distinguishing absence from a fully hidden tool. |
| `reopen 80 160` | Reopen reviewed frames in this inclusive range before replacing their masks. Existing corrected anchors remain available as inputs. |
| `unanchor 120` / `unanchor` | Delete a saved correction anchor at the specified/current frame, preserving its current mask and review status. Propagate from a different, good frame to replace the bad mask; starting propagation on the unanchored frame makes its current mask an anchor again. |
| `anchor 120` / `anchor` | Use the existing nonempty mask at the specified/current frame as a saved anchor, preserving the mask and review status. |
| `jump 120` | Inspect another frame directly. |
| `view crop` / `view full` | Zoom around masks/candidates (default), or display the full frame. Also applies to saved previews. |
| `autopreview on` / `autopreview off` | Enable/disable unattended contact-sheet generation after each propagation. Can also be enabled at startup with `--auto-preview`. |
| `next-video` | Leave this video and open the next selected video, preserving all saved work. Ends the command when there are no more selected videos. |
| `stay` / `stay on` / `stay off` | Stay on the current video after completion, or restore automatic advancement. Enter `stay` before the final acceptance if you want to keep inspecting. |
| `next` / `s`, `r`, `q` | Leave this frame unfinished and continue, reject it, or quit the entire video command. |

By default, `python annotate.py videos` skips finished videos and advances after
the final acceptance/rejection. A finished video has every frame accepted with
matching acceptance checksums or explicitly rejected. Pending, changed, or
unverified accepted frames prevent it from being considered finished. The
terminal prints each video's source path and ID when opening it.

To revisit a finished video, use `python annotate.py videos --video NUMBER`.
Explicit selection also keeps that video open after completion. Alternatively,
`--stay-on-video` keeps completed videos open while browsing the normal queue;
use `next-video` to leave each one. Within a session, `stay off` resumes automatic
advancement, `next-video` leaves manually, and `q` exits the whole reviewer.

For example, if early masks are good but tracking drifts around frame 120:

1. `jump 120`, then `reopen 80 160` if those frames were already reviewed.
2. Use `click` or region-only repair and inspect the resulting mask.
3. Run `prop both 80 160`.
4. Run `preview 80 160`, correct any remaining failures, then accept the good
   frames or ranges. Resolve missing masks explicitly.

After accepting frames 0–10, continue from the accepted boundary mask with:

```text
jump 10
prop forward 10
preview 11 20
accept 11 20
```

`preview` saves all sheets before returning to the prompt. Open the PNGs in the
printed directory to inspect them before acceptance. Each sheet labels frame
indices, timestamps, review status, and whether a mask exists. A JSON manifest
records frame/mask checksums and crop bounds so each set of sheets identifies
the exact mask versions it depicts. Changed masks require fresh previews for
range acceptance; generating sheets does not accept anything.

New acceptance records include the saved NPZ checksum, mask path, visibility,
extraction ID, and source-frame checksum recorded in the manifest. Range
acceptance compares those values before skipping an already accepted frame.
Changed masks need a preview of their current version and are then automatically
reopened and reaccepted by `accept START END`. Older records without checksums
are unverified and also need one reacceptance to establish a baseline; past
changes cannot be reconstructed from those records. Anchor-only changes do not
invalidate a mask's acceptance. Verification and training export reject known
changes to masks with checksum-backed acceptance records. The frame title marks
changed/unverified accepted masks when displayed.

To combine propagation and preview generation into one unattended operation:

```shell
python annotate.py videos --chunk-size 8 --context-frames 1 --auto-preview
```

Then run, for example, `jump 300` and `prop forward 300`. Tracking saves its masks,
then generates all sheets for the range without asking the user to close windows.
Alternatively, enter `autopreview on` in an existing reviewer session. Sheets are
stored under `data/previews/videos/<Video ID>/<range>-<snapshot ID>/`; each call
creates a separate snapshot. Use `--preview-dir` to change the destination and
`--preview-size 1024` for larger inspection tiles (default: 768; allowed: 64–2048).
Snapshots can be deleted when no longer needed; they are not training assets.

For a headless batch, create and inspect at least one nonempty anchor per video.
It can be at any frame: `anchor FRAME` marks an existing good mask. The helper
chooses the highest-numbered saved anchor that is not rejected and propagates
forward from its saved mask. Use `next-video` to prepare another video, then quit the
interactive reviewer before starting the helper, since both write the manifests:

```shell
python -u batch_propagate_videos.py --chunk-size 16 --context-frames 1 --auto-preview
```

This lightweight helper shares the options from `annotate.py videos`, including
`--data-dir`, `--video NUMBER` (or legacy `--video-id`), `--device`, and preview
settings. It opens no GUI and
never accepts masks. Frames at or before the last anchor are left unchanged.
After that anchor, pending predictions are regenerated (including existing
masks), while accepted/rejected frames remain protected. Nonempty reviewed masks
can serve as checkpoints across protected stretches. A frame-0 mask is not
required. Videos without a saved anchor, or without pending frames after their
last anchor, are skipped. The chosen anchor and new/regenerated frame ranges
are printed and included in the batch report. To start a later run from newly
reviewed progress, mark its last good frame with `anchor FRAME` first.

Tracking loss stops that video gracefully; other exceptions are reported and the
batch continues to the next video. Predictions saved before the problem remain
pending. A degraded but nonempty prediction can continue through the whole video;
all generated masks still need inspection. If a protected empty/rejected frame
prevents continuing into a pending stretch, add a good anchor after it. Optional
`--auto-preview` saves contact sheets for partial results too. Per-video outcomes,
starting anchors, new/regenerated frame ranges, errors, and preview directories are written incrementally to
`data/batch_propagation_report.json` (replaced by the next run). Chunking remains
bounded; a system/driver crash can still stop the process, but already saved masks
remain available for the next run.

Propagation starts from the currently displayed frame. One number is a positive
frame count beyond that frame; two numbers are an explicit inclusive start/end
range. Range arguments do not move to another frame. An accepted starting frame
can guide tracking without being reopened or changed. Other accepted/rejected frames in the target range
still require `reopen` before replacement. Fully reviewed videos can also be
opened interactively to inspect or reopen their masks.

The frame title shows whether it has a saved anchor. If a bad manual correction
keeps reappearing during propagation, remove that reference with `unanchor FRAME`,
then propagate from a good frame through it. Removing an anchor does not reopen
an accepted mask; use `reopen FRAME FRAME` too if that mask needs replacement.

The tracker can condition on anchors before and after a target frame when both
are in its processing window; propagation direction controls traversal, not which
side may supply correction masks. For example, to repair frame 356 using good
masks on both sides:

```text
anchor 355
anchor 357
unanchor 356
jump 357
prop backward 1
preview 356 356
```

Use at least `--context-frames 1` for this example (default: 2) so frame 355 is
included in the window even though only frame 356 is being replaced. Accepted
reference frames remain accepted. If the target is already accepted, reopen it
before propagation. `prop both` alone runs away from the current frame in both
directions; it does not register neighboring masks as anchors or combine two
independent predictions.

After propagation, the terminal reports inclusive ranges for new pending masks,
regenerated pending masks, and reused correction anchors. Existing pending
predictions within the requested range are regenerated; correction anchors guide
tracking and are reused. An accepted starting mask is explicitly reported as
preserved and unchanged. Gaps are shown as separate ranges.

Propagation uses the corrected **whole-tool** mask, including region-only repairs.
A repair rectangle itself does not follow the moving tool. The model can read
neighboring frames for context, but only the requested range is replaced.
Accepted/rejected frames are protected until explicitly reopened. Corrected
anchor masks are saved separately and reused together within each model window,
including after restarting the reviewer. Tracking uses a padded crop containing
the window's anchors. If a prediction approaches an interior crop edge or becomes
empty, tracking retries on a larger crop from the last reliable mask, expanding
up to the full frame as needed. The reliable mask is reframed into the new crop;
potentially clipped predictions are not used as recovery anchors. Each new chunk
chooses a fresh crop from its anchors. Saved masks and anchors always retain
full-frame dimensions and coordinates. Crop-edge checks cannot detect an unseen,
disconnected region that appears outside the crop without changing the tracked
mask; review and add corrections for these cases. Click markers are retained during the
current session; after a restart, the saved correction mask remains available
but the markers are not restored in the editor. Each direction is processed in bounded
chunks with overlapping context and a carried boundary mask. Defaults are
`--chunk-size 16 --context-frames 2`; lower them to reduce memory use. Tracking
can still drift across chunk boundaries or lose the tool; new predictions remain
pending human review. Add another anchor where tracking fails.

Predictions are saved one frame at a time as inference produces them, before
acceptance. Restarting the reviewer retains saved pending masks. Tracking sessions
release their frame/state caches after each window, and switching between text
segmentation and tracking unloads the inactive model (switching back reloads its
weights from the local model cache). Saved previews unload both models, decode
one full-resolution frame at a time, and assemble only bounded thumbnails into
one sheet at a time. They do not create Matplotlib figures. Interactive frame
views crop and shrink images before passing a single byte raster to Matplotlib;
closed figures are collected before opening another view. These measures reduce
peak RAM/VRAM usage but do not impose a hard memory limit. For constrained systems,
try `python annotate.py videos --chunk-size 8 --context-frames 1`.

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

For combined base and temporal training, use the [shared dataset builder](#shared-image-and-video-dataset)
below. It references the existing assets directly. To export a separate image-style
snapshot for legacy image workflows:

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

### Build training data

Use the repository's shared dataset builder from the repository root; see
[Build a temporary model dataset](#build-a-temporary-model-dataset) below.
The builder includes reviewed video runs by default; `--no-include-videos` retains a still-only export.

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

## Shared image and video dataset

Build one dataset for both model stages, without copying images, frames, or masks:

```shell
python build_dataset.py --output-path exports/combined-dataset
```

Video input is each video's **earliest contiguous accepted run**. Leading
unaccepted frames are skipped; the run ends at the first subsequent unaccepted
frame. Acceptance must agree between the frame manifest and review log, and the
selected assets must still match their reviewed versions. Later accepted runs
are excluded for now. Frame indices and presentation timestamps remain unchanged.
Use `--no-include-videos` for a still-only build.

The export contains shared class ordering, source type, physical tools, capture
sessions, video/frame IDs, timestamps, interval IDs, mask paths, validity flags,
and `base_eligible`. Severe and fully occluded frames remain available to temporal
training, while base training filters them out. Absent frames are valid base
negatives. Video identity and physical tools survive even when a frame has no
visible instances. Asset paths are relative to the dataset directory; preserve
the layout when moving the repository. No original MP4 is needed by either
training view of this export.

Splits are static and shared by both training stages. The default `--split-by tool`
keeps connected physical tools and image samples together. `--split-by session`
groups capture sessions, samples, and complete videos. `--split-by sample` groups
image samples and complete videos, allowing the same physical tools in different
splits. In sample-based modes, still images without sample IDs are excluded.

For within-video development, split every accepted run into approximately equal
sections, then allocate contiguous train/validation/test intervals in each:

```shell
python build_dataset.py --output-path exports/combined-sections \
  --video-sections 3 --split-gap 8 --min-interval-frames 8
```

Split order rotates across sections with a seeded initial offset. `--split-gap`
excludes that many frames between different splits inside a section. Every
interval has a distinct ID; snippets cannot cross sections even when the same
split touches both sides of a boundary. Still images use sample grouping in this
mode. Short videos that cannot supply the requested minimum frames per interval
stay entirely in training and are flagged in the report. Use
`--short-video-policy error` to reject such builds instead.

Section splits measure performance within familiar recordings; use whole-video,
session, or tool holdouts for independent evaluation. Assign the split before
base training: temporal holdout frames must never enter base training or replay.
Rebuilding after accepting more frames can change splits, so keep the original
export fixed for an experiment. The source assets are external references, not
an immutable snapshot; re-export and re-evaluate if annotations change.

`build_report.json` lists selected runs, excluded frames, interval boundaries,
short-video fallbacks, classes, base-eligible counts, and usable snippet counts.
`--clip-lengths 4 8 --clip-strides 1 2 3` controls the geometries counted in the
report, not training. An 8-frame clip at stride 3 needs a 22-frame interval;
check the report for the geometries you intend to train on.

In the neighboring flowsis repo, train both stages from this same path:

```shell
flowsis-train-base --dataset-path ../flowsis-data/exports/combined-dataset \
  --text-embeddings-dir path/to/embeddings --video-fraction 0.5
flowsis-train-temporal --dataset-path ../flowsis-data/exports/combined-dataset \
  --base-model outputs/base/best
```

The base trainer samples dynamically with replacement. `--video-fraction 0.5`
assigns half the expected draws to each source when both are present. Within
stills, capture samples receive equal total probability; within videos, each
video receives equal total probability, then its frames are sampled uniformly.
With only one source, that source receives all draws. `--samples-per-epoch`
controls the training draw count; by default it equals the eligible row count.
These probabilities are independent of annotation-confidence loss weights.

The temporal trainer samples videos uniformly among those eligible for each clip
geometry, then valid starts inside their assigned intervals. It reads saved PNGs
and NPZs directly, with one shared spatial transform per clip. Validation uses a
fixed, evenly spread snippet schedule and reports overall and per-video losses.
Set `--validation-batches` and `--batches-per-epoch` high enough to cover all
eligible validation videos. Legacy MP4-based temporal datasets still load.

## Build a temporary model dataset

The shared dataset builder lives here in `build_dataset.py` and reads image and
video manifests plus their review logs from `data/` by default. Use `--root PATH` to select another dataset directory.
Install the updated environment above, which includes Hugging Face `datasets`:

```shell
python build_dataset.py
```

The output is a Hugging Face `DatasetDict` at `exports/segmentation-dataset`, with
train, validation, and test splits targeting 70%/15%/15%. The generated files are
ignored by Git and live outside the DVC source dataset. To choose another destination or proportions:

```shell
python build_dataset.py --output-path exports/experiment-dataset --validation-size 0.15 --test-size 0.15 --seed 42
```

The builder refuses to replace an existing output directory unless `--force`
is supplied:

```shell
python build_dataset.py --force
```

This replaces the dataset at `--output-path` (default
`exports/segmentation-dataset`). The replacement is built and saved in a temporary
directory first, preserving the existing dataset if validation or saving fails.
Source image, mask, and raw-image directories cannot be used as output paths.

Each image keeps
all its mask instances, physical identities, class labels, and original mask
indices. It uses the schema consumed by `flowsis.data.loaders` and the existing
multi-instance training collators, so no loader conversion is required. Images
and NPZ masks use paths relative to the dataset directory. Move this repository
as a unit so the dataset, images, and masks keep their relative layout. Model
commands resolve these paths when opening the dataset. Python callers should
use `flowsis.data.loaders.load_local_dataset(path)`; `datasets.load_from_disk`
returns the stored relative paths without resolving them. `asset_paths.json`
versions this path convention. This is an export with external assets, rather
than a self-contained copy of all images and masks.

Images must have a latest review decision of `accepted`. With `--split-by tool`,
images without a Sample ID remain eligible and are grouped by their Tool IDs.
With `--split-by sample`, images without a Sample ID are excluded, even if they
have valid masks; the report counts them under `skipped.missing_sample_id`.
Positive images require
all instances to have complete, reviewed, visible, class-valid and base-valid masks.
Images with ignore pixels or any instance unsuitable for base training are
excluded as a whole to avoid teaching the model that an unlabelled tool is
background. Pending and skipped images are excluded. Accepted images with broken
mask references, identities, dimensions, or bounding boxes stop the build.
Video runs follow the shared export policy described above and preserve temporal-only supervision.

Explicitly accepted background images are also supported. The annotation
workflow must save an empty mask bundle at the canonical path below
`masks/images/`, matching the image dimensions and with no ignore pixels. Missing
annotations alone never imply background. No background images are synthesized.
Object `area` is the foreground pixel count in its visible mask, rather than
bounding-box area; model augmentations recompute it from the transformed mask.

By default, `--split-by tool` groups by **physical Tool ID and Sample ID**, not
capture session.
Multi-tool images connect all their Tool IDs, keeping those tools in the same
split as their single-tool captures. All views of a sample stay together.
Capture Session ID remains available as metadata. Because whole connected groups
stay together, achieved split sizes can differ from the requested proportions.
Before allocating holdouts, the splitter reserves complete groups that cover
every exported class in training, prioritizing rare classes. A class confined
to one group forces that group into training. Training coverage and group
integrity take precedence over requested fractions; this can leave requested
holdouts empty. Empty splits are omitted from the saved dataset and listed in
`empty_splits` in `build_report.json`. This guarantees that validation and test
contain no classes absent from training. A positive training fraction is required.

To split by **Sample ID only**, allowing the same physical tools to appear in
different splits, use:

```shell
python build_dataset.py --split-by sample --force
```

All views sharing a Sample ID stay together in either mode. Sample-based
splitting shuffles sample groups with the requested seed before assigning them
by target image counts. It can improve class coverage when a class has only one
physical tool captured in several samples. Training class coverage is enforced
in both splitting modes. The report records `split_by`, grouping fields,
`split_policy`, sample counts, and `classes_absent` for each requested split.
Classes with one sample can be trained but have no independent evaluation;
two samples can support train/validation without forcing a test example.
The holdouts assess known classes. Unknown-class rejection needs a separately
defined evaluation protocol; a fixed-class detector cannot predict new labels.

The export always preserves object-level `tool_class` and `tool_name`, regardless
of which fields define `category`. Select ordered label fields at build time:

```shell
# Broad labels, such as "scissors"
python build_dataset.py --split-by sample --label-fields tool_class --output-path exports/coarse-dataset

# Default specific labels, such as "iris scissors"
python build_dataset.py --split-by sample --label-fields tool_name tool_class --output-path exports/specific-dataset
```

`--label-fields` accepts an ordered, nonempty selection of distinct fields
(`tool_name`, `tool_class`), rather than a fixed menu of labeling modes. Values
are joined with spaces, blank values are skipped, and repeated suffixes are
avoided ("iris scissors" plus "scissors" stays "iris scissors"). Objects whose
selected fields are all blank stop the export. Labels are sorted for stable
category IDs, and the chosen fields are recorded in `build_report.json`.
Rebuild and retrain after changing label fields: class IDs, training coverage,
and split assignments may change. Prompt generation is a separate downstream
step and must use the resulting ordered labels.

Before building, still-image bundles under `masks/` (excluding `masks/videos/`)
are checked for pixel overlap, including bundles outside the accepted subset.
Video masks are validated only within the selected accepted runs. A single shared pixel stops the
build and writes a `.mask-audit.json` report beside the requested output. No masks
are silently trimmed. A successful output includes `build_report.json`, listing
mask audit counts, excluded images, class coverage, and actual split sizes.

Train from the model repository using the new dataset:

```shell
cd ../flowsis
flowsis-train-base --dataset-path ../flowsis-data/exports/segmentation-dataset --text-embeddings-dir path/to/embeddings
# Or train the detector:
flowsis-train-detector --dataset-path ../flowsis-data/exports/segmentation-dataset
```

For base training, generate text embeddings for the new dataset's class labels
using the model repository's prompt/embedding workflow, and replace
`path/to/embeddings` with that directory. Embeddings for the previous dataset
may not match the collected tool labels.

The updated model training and augmentation preview commands default to 0–45°
of extra rotation, since the collected views already cover the full circle.
Synthetic overlap stays enabled with a 25% chance of one additional source image
before visibility/severity filtering. An added image may contain several tools.
These defaults correspond to `--rotation-min-degrees 0 --rotation-max-degrees 45
--overlap-min-overlays 0 --overlap-max-overlays 1 --overlap-p 0.25`.
Validation and test views do not receive synthetic overlap or random rotation.

The export stores image-level `tool_ids` and per-object `tool_id`, with no
singular image-level tool field. Each object retains the manifest's `instance_id`
(scoped to its source image) and its bundle's `mask_index`. The mask loader adds
`source_image_id` at runtime so instances remain traceable after synthetic
compositing. Image/object `modified` flags are initialized by the loaders and
updated by augmentation; they are not persisted. `video_path` and `frame_index`
remain for the shared image/video loader, and per-object `mask_path` remains for
explicit mask references. Base-model training samples Sample IDs with equal
total weight across their views; images without a Sample ID stand alone.
Detector-only training retains its existing image-shuffling policy.

`Base Valid` is an annotation eligibility flag, not an additional target tensor.
It controls whether an annotation can supervise the base detector/segmentation
model using its class, bounding box, and visible mask. The current annotation
workflow sets it identically to `Class Valid`. Ignore regions are explicitly
reviewer-marked uncertain pixels; ordinary background and occluded pixels owned
by another tool are valid negatives for the target tool. Synthetic overlap
computes visible masks rather than automatically creating ignore regions.

## Deferred work: partial supervision

This policy is intentionally unimplemented for now. The builder currently
excludes an entire positive image if any instance fails the base-training
validity checks, or if its mask bundle has ignore pixels.
Ordinary occlusion is represented by disjoint visible masks. The current accepted
dataset has no uncertainty-related exclusions.

- [ ] Preserve per-instance mask/class/base validity and image-level ignore masks
  in the dataset instead of discarding otherwise usable images. Distinguish
  absent/fully hidden metadata from uncertain visible objects.
- [ ] Transform ignore masks consistently with images and instance masks during
  cropping, resizing, rotation, and synthetic compositing, then collate them.
- [ ] Apply validity and ignore information to matching, classification,
  objectness/background supervision, and mask losses in `flowsis`. Uncertain
  visible tools must not become false background targets. Keep the matching
  evaluation policy consistent with training.
- [ ] Test mixed-validity images, uncertain boundaries, absent/fully hidden tools,
  accepted background images, and augmented samples before enabling the policy.
