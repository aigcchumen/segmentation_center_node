import torch
import numpy as np
from scipy import ndimage
import json


class SegmentationCenterNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_regions": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1}),
                "points_per_region": ("INT", {"default": 2, "min": 2, "max": 10, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "find_multiple_region_points"
    CATEGORY = "chumen"

    def find_multiple_region_points(self, images, threshold, max_regions, points_per_region):
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()

        if images.ndim == 4:
            img = images[0]
        else:
            img = images

        if img.ndim > 2:
            img = img.squeeze()
            if img.ndim > 2:
                img = img.mean(axis=2)

        mask = (img > threshold).astype(np.uint8)
        labeled, num_features = ndimage.label(mask)

        all_points = []
        if num_features > 0:
            sizes = np.bincount(labeled.ravel())[1:]
            sorted_regions = np.argsort(sizes)[::-1]

            for i in range(min(num_features, max_regions)):
                region = sorted_regions[i] + 1
                y, x = np.where(labeled == region)

                if len(y) > 0 and len(x) > 0:
                    region_points = []

                    # Start point (top-left most point)
                    start_index = np.argmin(y * img.shape[1] + x)
                    region_points.append({"x": int(x[start_index]), "y": int(y[start_index])})

                    # End point (bottom-right most point)
                    end_index = np.argmax(y * img.shape[1] + x)
                    region_points.append({"x": int(x[end_index]), "y": int(y[end_index])})

                    # Additional points
                    if points_per_region > 2:
                        indices = np.linspace(0, len(x) - 1, points_per_region - 2, dtype=int)
                        for idx in indices:
                            region_points.append({"x": int(x[idx]), "y": int(y[idx])})

                    all_points.extend(region_points)

        # If no regions were found, return a list with a single default coordinate
        if not all_points:
            all_points = [{"x": -1, "y": -1}]

        return (json.dumps(all_points),)


NODE_CLASS_MAPPINGS = {
    "SegmentationCenterNode": SegmentationCenterNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SegmentationCenterNode": "Segmentation Center"
}