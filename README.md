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
```

## Usage

We will use Git pushes and pulls to sychronize data. I recommend briefly researching the basics of what Git is, and what pushes and pulls do. If you already have a preferred Git GUI, use that. Otherwise, this guide will give command line interface (CLI) instructions to enter into the terminal.

Put all the your new images into this repository's (abbreviated as repo) temp folder. You can do this by uploading to the Google Drive under temp/{your name}. Create subfolders within temp and name them based on the tool IDs that you've photographed. For example, name them temp/10, temp/11/, temp/12, etc. Place your images in the subfolders accordingly, for example all images of tool 10 go in temp/10. Afterwards, sync your local repo with GitHub using a pull.

To use the CLI, enter:

```shell
git pull
```

Once the repo is synced, in the terminal run:

```shell
python basic.py
```

This updates image_manifest.csv and enters rows based on the subfolders you made in the images folder. Check through the updated image_manifest.csv, and make sure the tool ID and filepaths look good. Git lets you see exactly what changed in each line. For now, you will still need to manually enter sample IDs, tool classes, tool names, and confidences. If something bad happens, you can revert to the old version since it's saved with Git. Once the manifest is correctly filled out, use Git to sychnonize it on GitHub. We do this using a push. Before pushing, it's good practice to pull first to make sure theres no incoming changes.

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

Finally, push it to GitHub to send you new changes to the cloud so everyone can now see them.

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
