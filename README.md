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

Raw images are considered in consecutive filename slots of eight, including newly added raw images that do not have a converted JPG yet. A complete slot receives a unique sample ID and angles from 0 to 315 degrees. If conversion or cropping fails, the raw file continues to occupy its original slot. The other images in that slot are left ungrouped for review, but the failure does not shift the angles or prevent later complete eight-image slots from being assigned automatically.

Check the updated `image_manifest.csv` and make sure the tool IDs, filepaths, sample groupings, and angles look correct. In particular, confirm that the filenames sort in the same order in which the tool was rotated.

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

## Uninstallation

```shell
# If you cloned into a directory other than home, go there instead.
cd ~

rm -rf flowsis-data
```
