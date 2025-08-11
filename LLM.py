"""Local analysis of brain CT images.

This module replaces the previous Gemini API based implementation with a
purely local approach.  A very small image processing routine is used to find
bright regions in the image which may correspond to abnormalities such as a
tumour.  The code is intentionally simple and self contained so it can run in
environments without network access or external API keys.

The script loads an image, attempts to locate the most prominent bright region
and draws a bounding box around it.  A short textual report is written to
disk along with the marked image.

This implementation does **not** aim to be a medically accurate detector; it
is only a placeholder demonstrating how a local model could replace the
previous API dependent workflow.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


IMAGE_PATH = "tumor_ex.jpeg"


@dataclass
class DetectionResult:
    """Container for detection results."""

    center: Tuple[float, float]
    box: Tuple[float, float, float, float]
    label: str


def detect_bright_region(image: Image.Image, label: str) -> Optional[DetectionResult]:
    """Detect the brightest region of the image.

    The image is converted to grayscale and thresholded using a simple heuristic
    (mean + std).  The largest remaining connected region is used as the
    predicted abnormality.  If no pixels exceed the threshold a result of
    ``None`` is returned.
    """

    gray = image.convert("L")
    arr = np.asarray(gray)
    threshold = arr.mean() + arr.std()
    mask = arr > threshold

    if not mask.any():
        return None

    ys, xs = np.where(mask)
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    box = (float(xmin), float(ymin), float(xmax), float(ymax))
    center = (float(center_x), float(center_y))
    return DetectionResult(center=center, box=box, label=label)


def draw_box_and_center_on_image(
    image: Image.Image,
    result: Optional[DetectionResult],
    output_path: str,
) -> None:
    """Draw the bounding box and centre marker on ``image``.

    If ``result`` is ``None`` the original image is simply saved to
    ``output_path`` without modification.
    """

    if result is None:
        image.save(output_path)
        return

    draw = ImageDraw.Draw(image)
    xmin, ymin, xmax, ymax = result.box
    center_x, center_y = result.center

    draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="lime", width=2)
    radius = max(2, int(min(image.size) * 0.02))
    draw.ellipse(
        [(center_x - radius, center_y - radius), (center_x + radius, center_y + radius)],
        fill="red",
        outline="yellow",
    )
    draw.text((xmin, ymin), result.label, fill="yellow")
    image.save(output_path)


def generate_report(result: Optional[DetectionResult], label: str) -> str:
    """Generate a simple textual report."""

    if result is None:
        return f"No convincing {label} detected."
    return f"Possible {label} detected around the marked region."


def main() -> None:
    label = "tumor"

    if not os.path.exists(IMAGE_PATH):
        print(f"Image file not found at: {IMAGE_PATH}")
        return

    image = Image.open(IMAGE_PATH).convert("RGB")
    result = detect_bright_region(image, label)

    safe_label = label.replace(" ", "_").lower()
    output_report = f"report_for_{safe_label}_local_bbox.txt"
    output_image = f"marked_image_for_{safe_label}_local_bbox.png"

    draw_box_and_center_on_image(image.copy(), result, output_image)

    report_text = generate_report(result, label)
    with open(output_report, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Final report saved to {output_report}")
    print(f"Final marked image saved to {output_image}")


if __name__ == "__main__":
    main()

