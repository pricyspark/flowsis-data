# Image dataset audit — November 6, 2026

**November 9 update:** [latest manifest review](image_dataset_audit/review_2026-11-09.md) fixes a tool 31/tool 39 sample-ID collision, synchronizes 57 object rows, and corrects tool 10 camera metadata. Earlier status counts below are historical.

**Current sample status:** [tools 24 and 28](image_dataset_audit/tools_24_28_status.md) have synchronized manifest references. Sample IDs 200+ are collision-free; current size/angle errors reflect the checker’s fixed eight-view rule. Earlier validation counts below are historical.

**Latest update:** [shadow audit and applied camera corrections](image_dataset_audit/shadow_findings.md) records 100 corrected JPGs and aligned mask updates. The earlier zero-error result below predates the ongoing tool 28 angle edits; current validation flags sample 126.

Finalized the corrected tool folders and synchronized their raw captures and both manifests. Inspected contact sheets covering all 1,497 original JPGs, compared SHA-256 hashes across both image trees, checked manifest/object/mask references with the repository validator, and compared available raw HEIC EXIF metadata with manifest values.

## Completed corrections

- Moved the 31 images from `images/13-new` into `images/13`, and the 31 from `images/14-new` into `images/14`. Removed the empty temporary folders.
- Swapped the corresponding original capture sets between `raw-images/13` and `raw-images/14`: 32 raw files in each direction. Staged the swap to preserve different photographs sharing filenames, including `IMG_0476` and `IMG_0477`.
- Consolidated 55 byte-identical JPG pairs and 55 byte-identical raw HEIC pairs from folders 14 and 17, retaining the tool 17 originals and their existing image IDs, reviewed masks, and review decisions. Removed the 55 unannotated, incorrectly labeled tool 14 manifest entries.
- Moved the three remaining scissors images and their originals from folder 14 to folder 17: `IMG_2673`, `IMG_2674`, and `IMG_2675`. Preserved their image IDs and changed their object labels to `17 / scissors / mayo / 1`. Kept their annotations pending.
- Added 62 corrected images and object records to the manifests. Tool 13 is `scissors / unnamed`, following the supplied identification; tool 14 retains its existing `needleholder / unnamed` identification. Both have confidence `0` because the specific identity remains unconfirmed.
- Registered tool 12's 32 images as `forceps / adson_needle_tissue`, confidence `0.5`, following the supplied identification. The normalized name preserves both words separated by the slash; it does not resolve their meaning.
- Added available raw-camera metadata and capture-session references for these 94 new records. Left unknown lighting, sample, angle, and annotation fields blank. Did not copy the erroneous sample assignments from the duplicate tool 14 rows onto tool 17.
- Preserved every preexisting object record outside tool 14, including all annotation fields, and preserved `image_review.jsonl` byte-for-byte.

## Remaining issues requiring review

### Sample membership and angle discrepancies

These are visual problems that the structural validator does not detect: eight rows can contain every angle while still mixing configurations. These existing assignments were reported rather than rewritten.

| Folder / tool | Samples | Evidence |
| --- | --- | --- |
| 24 | 111–115 | Samples mix open and closed needleholder views. For example, sample 111 starts with open `IMG_2724`–`IMG_2727`, then closed `IMG_2728`–`IMG_2731`. Sample 112 contains closed `IMG_2732`–`IMG_2736`, then open `IMG_2737`, `IMG_2739`, and `IMG_2740`. Split by actual unchanged setup and recheck angles. |
| 29 | 128, 129, 131, 132 | Sample 128 mixes open `IMG_2943`–`IMG_2946` with closed `IMG_2947`–`IMG_2950`; sample 129 mixes closed `IMG_2951`–`IMG_2954` with open `IMG_2955`–`IMG_2958`. The same alternating configuration problem occurs in samples 131 and 132. |
| 30 | 134, 135 | Sample 134 mixes open `IMG_2992`–`IMG_2995` with closed `IMG_2996`–`IMG_2999`. Sample 135 mixes closed `IMG_3001`–`IMG_3004` with open `IMG_3005`–`IMG_3008`. |
| 31 | 136, 137 | Sample 136 has seven closed views followed by open `IMG_3066` at recorded 315°. Sample 137 starts with open `IMG_3067`–`IMG_3069`, then switches to closed `IMG_3070`–`IMG_3074`. |
| 10 | 13 | `IMG_0959` and `IMG_0960` replace two views in an otherwise earlier capture set, but their raw metadata shows a different camera and capture date. Confirm whether these genuinely belong to the same setup/sample. Their stored metadata is also stale; see below. |
| 28 | 126 | The visible orientation sequence does not appear to follow uniform 45° increments matching the recorded angles. Recheck the source order and angle assignments; do not infer them solely from sorted filenames. |

Evidence: [tool 24](image_dataset_audit/contact_sheets/24.jpg), [tool 29](image_dataset_audit/contact_sheets/29.jpg), [tool 30](image_dataset_audit/contact_sheets/30.jpg), [tool 31](image_dataset_audit/contact_sheets/31.jpg), [tool 10](image_dataset_audit/contact_sheets/10.jpg), [tool 28](image_dataset_audit/contact_sheets/28.jpg).

### Capture metadata does not match the raw files

For both `images/10/IMG_0959.jpg` and `images/10/IMG_0960.jpg`, five nonblank manifest fields disagree with the raw EXIF:

- Model: manifest `iPhone 15 Pro`; raw `iPhone 17`.
- ISO: manifest `200`; raw `160`.
- Aperture: manifest `1.78`; raw `1.6`.
- Focal length: manifest `6.765`; raw `5.96` mm.
- Capture time: manifest August 24, 2026; raw September 8, 2026.

The existing session reference should also be reviewed once the intended source/sample is established. All other compared, nonblank raw/manifest EXIF fields agreed. Exact timestamps and differences are in [metadata_mismatches.json](image_dataset_audit/metadata_mismatches.json).

### Missing originals and converted outputs

- **32 converted tool 33 images have no raw counterpart:** `IMG_1031`–`IMG_1067`, with gaps. These cover samples 52–55. They cannot be regenerated from the current raw dataset. The exact list is in [details.json](image_dataset_audit/details.json), under `converted_without_raw`.
- **12 raw files have no JPG output:** tool 13 `IMG_0370`; tool 14 `IMG_0470`; tool 30 `IMG_3009`, `IMG_3012`, `IMG_3014`, `IMG_3019`, `IMG_3022`, `IMG_3024`, `IMG_3026`, `IMG_3027`, `IMG_3028`, and `IMG_3032`. These could be prior crop-validation failures or intentionally omitted captures; the audit does not establish why. Originals were preserved, and no replacement JPGs were fabricated.

### Incomplete intake and labels

- **16 multi-tool images remain unregistered:** eight each in `images/multi/3` and `images/multi/4`. They need explicit physical tool declarations. Their apparent silhouettes are insufficient to establish physical identities safely. See [multi-tool contact sheet](image_dataset_audit/contact_sheets/multi.jpg).
- **298 registered images have no Sample ID.** This includes the 94 newly registered images and all 58 tool 17 views. Sample-based export will omit them. Complete the actual capture groups before assigning angles; several batches have missing views or configuration changes.
- Tool 13's specific scissors name and tool 14's specific needleholder name remain unknown. Tool 12's name is provisional at the requested confidence `0.5`. Pending annotations were not marked accepted.

No additional definite folder/tool identity swaps were established from the contact sheets. Fine subtype identity and physical identity among similar instruments remain beyond what these thumbnails can establish.

## Verification and artifacts

Final inventory: **1,442 JPGs, 1,422 raw images, 1,426 image-manifest rows, and 1,450 object-manifest rows**. The difference between JPGs and manifest rows is precisely the 16 undeclared multi-tool images.

The repository's static validator returned **0 errors and 314 warnings**: 298 ungrouped images and 16 undocumented images. No byte-identical duplicate JPGs or raw files remain. All registered image paths resolve; object references and reviewed annotation/mask checks pass. These checks establish structural consistency, not correctness of the visual sample assignments flagged above.

- [Detailed static findings and missing-file lists](image_dataset_audit/details.json)
- [File moves, hashes, removed duplicate IDs, and new/relabelled IDs](image_dataset_audit/changes.json)
- [Contact sheets](image_dataset_audit/contact_sheets/): these show the **pre-correction** folder labels and sample assignments, including `13-new`, `14-new`, and the incorrect old folder 14 batch.

Pre-change manifests/review log and archived duplicate files are retained locally in `/tmp/flowsis-audit/backup`. The corrected image/raw contents are retained under their final dataset paths. Temporary backups are outside the dataset and may be cleared by the system.

Refreshed the local dataset snapshot with `dvc add data`, updating `data.dvc`. Used `DVC_SITE_CACHE_DIR=/tmp/flowsis-audit/dvc-cache` to avoid the default cache database access error. A subsequent `dvc status` reported that data and pipelines are up to date. No data or Git changes were pushed to a remote.

## Timestamp follow-up

[Timestamp analysis and candidate capture groups](image_dataset_audit/timestamp_findings.md) identifies recurring eight-closed/four-open sequences that explain much of the sample-boundary drift. Candidate groups are documented for review; the manifests have not been reassigned.
