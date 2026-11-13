"""SAM3 image segmentation from foreground/background points or a coarse mask."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from image_review import Candidate

if TYPE_CHECKING:
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor


class Sam3SpatialBackend:
    def __init__(self, model_name: str, device: torch.device):
        self.model_name = model_name
        self.device = device
        self.model: Sam3TrackerModel | None = None
        self.processor: Sam3TrackerProcessor | None = None
        self.image: Image.Image | None = None
        self.embeddings: list[torch.Tensor] | None = None
        self.previous_mask: NDArray[np.bool_] | None = None
        self.previous_logits: torch.Tensor | None = None
        self.active = False

    def offload(self) -> None:
        self.image = None
        self.embeddings = None
        self.previous_mask = None
        self.previous_logits = None
        if self.model is not None:
            self.model.to("cpu")
        self.active = False

    @torch.inference_mode()
    def segment(
        self,
        image: Image.Image,
        points: list[tuple[float, float, int]],
        seed: NDArray[np.bool_] | None = None,
        *,
        box: tuple[float, float, float, float] | None = None,
    ) -> Candidate:
        if seed is not None and (
            seed.shape != (image.height, image.width)
            or seed.dtype != np.bool_
            or not seed.any()
        ):
            raise ValueError("Polygon prompt must be a nonempty full-image mask.")
        if box is not None:
            left, top, right, bottom = box
            if not all(math.isfinite(value) for value in box) or not (
                0 <= left < right <= image.width and 0 <= top < bottom <= image.height
            ):
                raise ValueError("Box must have ordered corners inside the image.")
        if not points and seed is None and box is None:
            raise ValueError(
                "Provide foreground/background points or a polygon prompt."
            )
        if seed is None and box is None and not any(label == 1 for _, _, label in points):
            raise ValueError("Add at least one positive click on the target tool.")
        if any(
            not math.isfinite(x)
            or not math.isfinite(y)
            or not 0 <= x < image.width
            or not 0 <= y < image.height
            or label not in {0, 1}
            for x, y, label in points
        ):
            raise ValueError(
                "Point coordinates must be inside the image and labels must be 0 or 1."
            )
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor

        if self.model is None:
            print("Loading SAM3 interactive image model...")
            self.model = Sam3TrackerModel.from_pretrained(self.model_name)
            self.processor = Sam3TrackerProcessor.from_pretrained(self.model_name)
        if not self.active:
            self.model.to(self.device).eval()
            self.active = True
        assert self.processor is not None
        cached = self.image is image and self.embeddings is not None
        if not cached:
            self.previous_mask = None
            self.previous_logits = None
        inputs = self.processor(
            images=None if cached else image,
            original_sizes=[[image.height, image.width]] if cached else None,
            input_points=[[[[x, y] for x, y, _ in points]]] if points else None,
            input_labels=[[[label for _, _, label in points]]] if points else None,
            input_boxes=[[list(box)]] if box is not None else None,
            return_tensors="pt",
        ).to(self.device)
        original_sizes = inputs.pop("original_sizes")
        mask_input = None
        if (
            seed is not None and cached and self.previous_mask is not None
            and np.array_equal(seed, self.previous_mask)
        ):
            mask_input = self.previous_logits.to(self.device) if self.previous_logits is not None else None
        elif seed is not None:
            # The decoder expects a low-resolution dense prompt in logit space.
            mask_input = torch.from_numpy(seed.astype(np.float32))[None, None]
            mask_input = torch.nn.functional.interpolate(
                mask_input,
                size=self.model.prompt_encoder.mask_input_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            mask_input = (mask_input * 8 - 4).to(self.device)
        output = self.model(
            **inputs,
            image_embeddings=self.embeddings if cached else None,
            input_masks=mask_input,
            multimask_output=True,
        )
        self.image = image
        self.embeddings = output.image_embeddings
        scores = output.iou_scores[0, 0].detach().cpu()
        for index in torch.argsort(scores, descending=True).tolist():
            if not torch.isfinite(scores[index]):
                continue
            low_res = output.pred_masks[:, :, index : index + 1].detach().cpu()
            processed = self.processor.post_process_masks(
                low_res, original_sizes, binarize=True
            )[0]
            mask = (
                processed.reshape(image.height, image.width)
                .cpu()
                .numpy()
                .astype(np.bool_)
            )
            if not mask.any():
                continue
            if any(
                bool(
                    mask[
                        min(round(y), image.height - 1), min(round(x), image.width - 1)
                    ]
                )
                != bool(label)
                for x, y, label in points
            ):
                continue
            self.previous_mask = mask.copy()
            selected_logits = output.pred_masks[:, :, index : index + 1]
            self.previous_logits = selected_logits.detach().cpu().reshape(
                1, 1, *selected_logits.shape[-2:]
            )
            return Candidate(
                mask,
                float(scores[index].clamp(0, 1)),
                "SAM3 box prompt" if box is not None else (
                    "SAM3 mask correction" if seed is not None else "SAM3 point prompts"
                ),
                "spatial",
            )
        raise ValueError(
            "SAM3 returned no nonempty mask satisfying your clicks. Add or adjust foreground/background clicks and retry; the previous assignment is unchanged."
        )
