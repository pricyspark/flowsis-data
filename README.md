# FlowSIS Data

## Installation

```shell
git clone https://github.com/pricyspark/flowsis-data.git
```

## Usage

Download all of temporary images and place into the folder temp. Create subfolders within temp based on the tool IDs that you're photographed. For example, temp/10, temp/11/, temp/12, etc. Once all the images are arranged in subfolders, sync your local repository with the one on GitHub using a pull. If you have a preferred git GUI, use that. Otherwise, use the CLI with:

```shell
git pull
```

Once the repository is synced, run:

```shell
python basic.py
```

This update image_manifest.csv. It will also create previous_image_manifes.csv if it doesn't exist already. previous_image_manifest.csv will have the contents of image_manifest.csv from before you ran basic.py.

Check through the updated image_manifest.csv and make sure the tool ID and filepaths look good. For now, you will still need to manually enter sample IDs, tool classes, tool names, and confidences. If something bad happens, you can manually revert to the old version since it's saved in previous_image_manifest.csv. Once the manifest is correctly filled out, push it to GitHub to sync it. Set the commit message to your name, the tool IDs you entered, and the sample IDs you entered. For example:

```text
John Doe
tools: x, y, z
samples: a, b, c, d, e, f
```

If you do not have a preferred git GUI, you can use the CLI with:

```shell
# Sync from GitHub last time
git pull
# Sync your changes to GitHub
git add *
git commit -m "Jane Doe" -m "tools: x, y, z" -m "samples: a, b, c, d, e, f"
git push
```
