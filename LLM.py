"""Local analysis of brain CT images using a small open model.

The previous implementation used a toy image processing routine to highlight
bright regions.  This version swaps that out for a locally executed
vision-language model (VLM).  The model we target is `Molmo`, an open-source
multimodal LLM that can be downloaded from Hugging Face and run without any
external API calls.  Given an image and a prompt asking for tumour
localisation, Molmo is expected to return bounding box coordinates which are
then drawn on the image.

**Note**: the weights for Molmo are not bundled with this repository.  They
must be downloaded separately (for example from ``allenai/Molmo-7B-DPO``) and
either cached by ``transformers`` or placed in a local directory.  When run in
an environment without the model files the script will attempt to download
them and may fail if internet access is restricted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import re

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image, ImageDraw


IMAGE_PATH = "tumor_ex.jpeg"
MODEL_NAME = os.getenv("MOLMO_MODEL", "allenai/Molmo-7B-DPO")


@dataclass
class DetectionResult:
    """Container for detection results."""

    center: Tuple[float, float]
    box: Tuple[float, float, float, float]
    label: str


def run_molmo(image: Image.Image, label: str) -> Optional[DetectionResult]:
    """Run the Molmo VLM on ``image`` to obtain a tumour bounding box.

    The function loads the model and processor from ``MODEL_NAME``.  The
    resulting text is expected to contain the coordinates in the form
    ``xmin,ymin,xmax,ymax``.  If such a pattern cannot be found the function
    returns ``None``.
    """

    try:
        processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
    except Exception as err:  # pragma: no cover - relies on external files
        print(f"Failed to load Molmo model: {err}")
        return None

    prompt = (
        "You are an expert radiologist. "
        "Analyse this brain CT image and respond with tumour bounding box as "
        "xmin,ymin,xmax,ymax. If no tumour is visible respond 'none'."
    )

    inputs = processor(prompt, images=image, return_tensors="pt").to(model.device)

    try:
        outputs = model.generate(**inputs, max_new_tokens=100)
        text = processor.decode(outputs[0], skip_special_tokens=True)
    except Exception as err:  # pragma: no cover
        print(f"Molmo generation failed: {err}")
        return None

    match = re.search(r"(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", text)
    if not match:
        return None

    xmin, ymin, xmax, ymax = map(float, match.groups())
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    return DetectionResult(center=(center_x, center_y), box=(xmin, ymin, xmax, ymax), label=label)


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
    result = run_molmo(image, label)

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

