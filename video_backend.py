"""Lazy SAM3 text detection and single-object interactive video tracking."""

from __future__ import annotations

import gc
from collections.abc import Generator, Iterator, Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

from image_review import Candidate
from image_refinement import tool_crop
from video_data import Mask

if TYPE_CHECKING:
    from transformers import (
        Sam3Model,
        Sam3Processor,
        Sam3TrackerVideoModel,
        Sam3TrackerVideoProcessor,
    )
    from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import (
        Sam3TrackerVideoInferenceSession,
        Sam3TrackerVideoSegmentationOutput,
    )


class Sam3VideoBackend:
    def __init__(self, model_name: str, device: torch.device, threshold: float = 0.5):
        self.model_name = model_name
        self.device = device
        self.threshold = threshold
        self.detector: Sam3Model | None = None
        self.image_processor: Sam3Processor | None = None
        self.tracker: Sam3TrackerVideoModel | None = None
        self.video_processor: Sam3TrackerVideoProcessor | None = None

    def _release_unused_memory(self) -> None:
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def release(self) -> None:
        """Unload inference weights before CPU-only inspection/export."""
        self.detector = None
        self.tracker = None
        self._release_unused_memory()

    def _release_session(self, session: Sam3TrackerVideoInferenceSession) -> None:
        session.reset_inference_session()
        session.processed_frames = None
        self._release_unused_memory()

    @torch.inference_mode()
    def text(self, image: Image.Image, prompt: str) -> list[Candidate]:
        from transformers import Sam3Model, Sam3Processor

        if self.tracker is not None:
            self.tracker = None
            self._release_unused_memory()
        if self.detector is None:
            self.detector = (
                Sam3Model.from_pretrained(self.model_name).to(self.device).eval()
            )
            self.image_processor = Sam3Processor.from_pretrained(self.model_name)
        assert self.image_processor is not None
        inputs = self.image_processor(
            images=[image], text=[prompt], return_tensors="pt"
        ).to(self.device)
        output = self.detector(**inputs)
        result = self.image_processor.post_process_instance_segmentation(
            output,
            threshold=self.threshold,
            mask_threshold=0.5,
            target_sizes=inputs["original_sizes"].tolist(),
        )[0]
        masks = result["masks"].detach().cpu().numpy()
        if masks.ndim == 4:
            masks = masks[:, 0]
        scores = result["scores"].detach().cpu().numpy().reshape(-1)
        return [
            Candidate(mask.astype(np.bool_), float(score), prompt, f"candidate-{index}")
            for index, (mask, score) in enumerate(zip(masks, scores, strict=True))
            if mask.any()
        ]

    def _session(
        self, images: Sequence[Image.Image]
    ) -> Sam3TrackerVideoInferenceSession:
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

        if self.detector is not None:
            self.detector = None
            self._release_unused_memory()
        if self.tracker is None:
            self.tracker = (
                Sam3TrackerVideoModel.from_pretrained(self.model_name)
                .to(self.device)
                .eval()
            )
            self.video_processor = Sam3TrackerVideoProcessor.from_pretrained(
                self.model_name
            )
        assert self.video_processor is not None
        return self.video_processor.init_video_session(
            video=list(images),
            inference_device=self.device,
            processing_device=self.device,
            video_storage_device="cpu",
            inference_state_device="cpu",
        )

    def _mask(
        self, output: Sam3TrackerVideoSegmentationOutput, size: tuple[int, int]
    ) -> Mask:
        assert self.video_processor is not None
        masks = self.video_processor.post_process_masks(
            [output.pred_masks],
            original_sizes=[list(size)],
            binarize=True,
        )[0]
        return masks.detach().cpu().numpy().reshape(-1, *size)[0].astype(np.bool_)

    @torch.inference_mode()
    def clicks(
        self,
        image: Image.Image,
        points: list[tuple[float, float, int]],
        box: tuple[int, int, int, int] | None = None,
    ) -> Mask:
        if not points and box is None:
            raise ValueError("Provide positive/negative clicks or a box.")
        session = self._session([image])
        assert self.video_processor is not None and self.tracker is not None
        try:
            self.video_processor.add_inputs_to_inference_session(
                session,
                frame_idx=0,
                obj_ids=1,
                input_points=[[[[x, y] for x, y, _ in points]]] if points else None,
                input_labels=[[[label for _, _, label in points]]] if points else None,
                input_boxes=[[list(box)]] if box is not None else None,
            )
            return self._mask(
                self.tracker(inference_session=session, frame_idx=0),
                (image.height, image.width),
            )
        finally:
            self._release_session(session)

    @torch.inference_mode()
    def track(
        self,
        images: Sequence[Image.Image],
        anchors: dict[int, Mask],
        start: int,
        stop: int,
        reverse: bool,
    ) -> Iterator[tuple[int, Mask]]:
        """Track on crops, expanding and retrying from a reliable full-frame anchor."""
        shape = (images[0].height, images[0].width)
        if start not in anchors or not anchors[start].any():
            raise ValueError("Cropped tracking requires a nonempty starting anchor.")
        if any(mask.shape != shape for mask in anchors.values()):
            raise ValueError("Tracking anchors must match full-frame dimensions.")
        occupied = anchors[start].copy()
        for mask in anchors.values():
            occupied |= mask
        bounds = tool_crop(occupied, padding=0.4)
        del occupied
        current = start
        carry = anchors[start]
        while True:
            left, top, right, bottom = bounds
            cropped = [image.crop(bounds) for image in images]
            local_anchors = {
                index: mask[top:bottom, left:right]
                for index, mask in anchors.items()
            }
            local_anchors[current] = carry[top:bottom, left:right]
            predictions = self._track_window(
                cropped, local_anchors, current, stop, reverse
            )
            retry = False
            try:
                for index, mask in predictions:
                    margin = max(1, int(np.ceil(0.05 * min(mask.shape))))
                    near_edge = (
                        (left > 0 and mask[:, :margin].any())
                        or (top > 0 and mask[:margin, :].any())
                        or (right < shape[1] and mask[:, -margin:].any())
                        or (bottom < shape[0] and mask[-margin:, :].any())
                    )
                    full_view = bounds == (0, 0, shape[1], shape[0])
                    if index not in local_anchors and not full_view and (
                        near_edge or not mask.any()
                    ):
                        # Discard a possibly clipped prediction. Reframe the last
                        # reliable mask, rather than using clipped pixels as an anchor.
                        dx = max(1, (right - left + 1) // 2)
                        dy = max(1, (bottom - top + 1) // 2)
                        bounds = (
                            max(0, left - dx),
                            max(0, top - dy),
                            min(shape[1], right + dx),
                            min(shape[0], bottom + dy),
                        )
                        retry = True
                        break
                    full = np.zeros(shape, dtype=np.bool_)
                    full[top:bottom, left:right] = mask
                    if index in anchors:
                        full = anchors[index]
                    current, carry = index, full
                    yield index, full
            finally:
                predictions.close()
                del cropped, local_anchors
            if not retry:
                return

    @torch.inference_mode()
    def _track_window(
        self,
        images: Sequence[Image.Image],
        anchors: dict[int, Mask],
        start: int,
        stop: int,
        reverse: bool,
    ) -> Generator[tuple[int, Mask], None, None]:
        session = self._session(images)
        assert self.video_processor is not None and self.tracker is not None
        try:
            for index, mask in sorted(anchors.items()):
                self.video_processor.add_inputs_to_inference_session(
                    session,
                    frame_idx=index,
                    obj_ids=1,
                    input_masks=mask,
                )
                self.tracker(inference_session=session, frame_idx=index)
            for output in self.tracker.propagate_in_video_iterator(
                inference_session=session,
                start_frame_idx=start,
                max_frame_num_to_track=abs(stop - start),
                reverse=reverse,
            ):
                index = int(output.frame_idx)
                mask = (
                    anchors[index]
                    if index in anchors
                    else self._mask(output, (images[index].height, images[index].width))
                )
                yield index, mask
        finally:
            self._release_session(session)
