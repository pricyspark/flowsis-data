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

Convert raw images to JPG and save them in the images folder.

```shell
python img_convert.py
```

Now run:

```shell
python basic.py
```

The script lists each new tool ID and asks you for its class, name, and confidence. Confidence must be a number from 0 to 1. It reads the original files in `raw-images` to fill in the camera make, model, flash, ISO, exposure, aperture, and focal length when that metadata is available. The metadata is written only to the CSV; it is not copied into the JPG files. If a tool has a multiple of eight images, the script treats each consecutive set of eight filenames as one sample and fills in unique sample IDs and angles from 0 to 315 degrees. Otherwise, it leaves the sample IDs and angles blank for you to review.

Check the updated `image_manifest.csv` and make sure the tool IDs, filepaths, sample groupings, and angles look correct. In particular, confirm that the filenames sort in the same order in which the tool was rotated. Once the manifest is correct, use Git to synchronize it with GitHub. Before pushing, pull again to check for incoming changes.

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
