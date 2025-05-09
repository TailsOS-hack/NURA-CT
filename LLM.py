import os
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError 
import requests
from PIL import Image, ImageDraw, ImageFont
import re
import json 

MODEL_NAME = "gemini-2.5-flash-preview-04-17"  

IMAGE_PATH_OR_URL = "tumor_ex.jpeg"  
SAMPLE_IMAGE_URL = "https://prod-images-static.radiopaedia.org/images/34839675/e0bfac0370247b75f8173d8f6c79d3_big_gallery.jpeg"


def get_finding_term(condition_name):
    if condition_name == "tumor":
        return "tumor"
    elif condition_name == "hemorrhagic stroke":
        return "hemorrhage" 
    elif condition_name == "TBI":
        return "evidence of Traumatic Brain Injury (e.g., hematoma, contusion)"
    return "significant finding"

def get_prompt_for_bounding_box(condition_name):
    finding_term = get_finding_term(condition_name)
    finding_term_upper_snake = finding_term.upper().replace(' ', '_').replace('(', '').replace(')', '').replace('.', '')

    prompt = (
        f"You are a specialized medical imaging AI analyzing a CT scan.\n"
        f"The image is suspected to show a {condition_name}.\n\n"
        f"Tasks:\n"
        f"1. Provide a concise technical radiological report based on your findings in the image. "
        f"Focus on any signs of a {finding_term}.\n\n"
        f"2. If you identify the primary location of the {finding_term}, you MUST provide its bounding box. "
        f"The bounding box coordinates should be in the format [ymin, xmin, ymax, xmax], "
        f"normalized to a 0-1000 scale (where 0,0 is the top-left corner of the image).\n"
        f"   - Output this bounding box on a new line, clearly marked like this:\n"
        f"     BOUNDING_BOX_{finding_term_upper_snake}: [ymin, xmin, ymax, xmax]\n"
        f"   - Example for a tumor: BOUNDING_BOX_TUMOR: [250, 300, 450, 500]\n"
        f"   - If multiple instances are present, identify the most prominent one.\n"
        f"   - If you cannot confidently detect a {finding_term} or provide a bounding box, "
        f"     clearly state 'No {finding_term} confidently detected for bounding box.' instead of providing a box.\n\n"
        f"Ensure your entire response, including the report and the bounding box line (if applicable), is returned."
    )
    return prompt

def extract_bounding_box_from_output(gemini_text_output, image_width, image_height, condition_name):
    if condition_name.strip().lower() == "normal":
        print("Class is 'Normal'. No bounding box should be extracted.")
        return None

    finding_term = get_finding_term(condition_name)
    finding_term_upper_snake = finding_term.upper().replace(' ', '_').replace('(', '').replace(')', '').replace('.', '')
    pattern_string = rf"BOUNDING_BOX_{re.escape(finding_term_upper_snake)}:\s*\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]"
    match = re.search(pattern_string, gemini_text_output, re.IGNORECASE | re.MULTILINE)

    if match:
        try:
            ymin_norm = float(match.group(1))
            xmin_norm = float(match.group(2))
            ymax_norm = float(match.group(3))
            xmax_norm = float(match.group(4))

            pixel_xmin = (xmin_norm / 1000.0) * image_width
            pixel_ymin = (ymin_norm / 1000.0) * image_height
            pixel_xmax = (xmax_norm / 1000.0) * image_width
            pixel_ymax = (ymax_norm / 1000.0) * image_height

            pixel_xmin = max(0, min(pixel_xmin, image_width -1))
            pixel_ymin = max(0, min(pixel_ymin, image_height -1))
            pixel_xmax = max(0, min(pixel_xmax, image_width-1))
            pixel_ymax = max(0, min(pixel_ymax, image_height-1))
            
            if pixel_xmin >= pixel_xmax or pixel_ymin >= pixel_ymax:
                print(f"Warning: Invalid bounding box dimensions after de-normalization: xmin={pixel_xmin}, ymin={pixel_ymin}, xmax={pixel_xmax}, ymax={pixel_ymax}. Skipping.")
                return None

            center_x = (pixel_xmin + pixel_xmax) / 2.0
            center_y = (pixel_ymin + pixel_ymax) / 2.0
            
            bounding_box_pixels = (pixel_xmin, pixel_ymin, pixel_xmax, pixel_ymax)
            label = finding_term
            print(f"Found and de-normalized bounding box for '{label}': {bounding_box_pixels}")
            print(f"Center point: ({center_x:.1f}, {center_y:.1f})")
            return ((center_x, center_y), bounding_box_pixels, label)
        except ValueError as e:
            print(f"Warning: Could not parse coordinates from match: {match.groups()}. Error: {e}. Skipping.")
            return None
        except Exception as e:
            print(f"Warning: An unexpected error occurred during bounding box parsing: {e}. Match: {match.groups()}. Skipping.")
            return None
            
    print(f"No bounding box line matching 'BOUNDING_BOX_{finding_term_upper_snake}: [...]' found in the model output.")
    return None


def draw_box_and_center_on_image(image_pil, center_coord, bounding_box_pixels, label, output_path):
    if not bounding_box_pixels:
        print("No bounding box provided to draw. Saving original image.")
        try:
            image_pil.save(output_path)
            print(f"Original image saved to {output_path}")
        except Exception as e:
            print(f"Error saving original image: {e}")
        return

    draw = ImageDraw.Draw(image_pil)
    width, height = image_pil.size
    
    pixel_xmin, pixel_ymin, pixel_xmax, pixel_ymax = bounding_box_pixels
    center_x, center_y = center_coord

    box_outline_color = "lime"
    box_thickness = max(1, min(width, height) // 200)
    draw.rectangle(bounding_box_pixels, outline=box_outline_color, width=box_thickness)

    marker_radius = max(2, min(width, height) // 100)
    marker_fill_color = "red"
    marker_outline_color = "yellow"
    draw.ellipse(
        (center_x - marker_radius, center_y - marker_radius,
         center_x + marker_radius, center_y + marker_radius),
        fill=marker_fill_color, outline=marker_outline_color, width=max(1, marker_radius // 3))

    if label:
        font_size = max(12, min(width, height) // 40)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()
                print("Arial/DejaVuSans font not found. Using default PIL font.")
        
        text_fill_color = "white"
        text_bg_color = "rgba(0,0,0,180)"

        text_x = pixel_xmin + box_thickness + 2
        text_y = pixel_ymin + box_thickness + 2
        
        try:
            text_bbox = draw.textbbox((text_x, text_y), label, font=font)
        except TypeError: 
            text_bbox_size = draw.textsize(label, font=font)
            text_bbox = (text_x, text_y, text_x + text_bbox_size[0], text_y + text_bbox_size[1])

        bg_rect_coords = (text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2)

        if bg_rect_coords[2] > width: 
            text_x = pixel_xmax - (text_bbox[2]-text_bbox[0]) - box_thickness - 4
        if bg_rect_coords[3] > height: 
             text_y = pixel_ymax - (text_bbox[3]-text_bbox[1]) - box_thickness - 4
        
        try:
            text_bbox = draw.textbbox((text_x, text_y), label, font=font)
        except TypeError:
            text_bbox_size = draw.textsize(label, font=font)
            text_bbox = (text_x, text_y, text_x + text_bbox_size[0], text_y + text_bbox_size[1])
        bg_rect_coords = (text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2)

        draw.rectangle(bg_rect_coords, fill=text_bg_color)
        draw.text((text_x, text_y), label, fill=text_fill_color, font=font)

    print(f"Bounding box and center marker drawn for '{label}'")

    try:
        image_pil.save(output_path)
        print(f"Image with markings saved to {output_path}")
    except Exception as e:
        print(f"Error saving marked image: {e}")

def download_image_if_not_exists(image_path, url):
    if not os.path.exists(image_path):
        print(f"Image '{image_path}' not found. Attempting to download from {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(image_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded '{image_path}' successfully.")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Failed to download image: {e}")
            return False
    return True

def classify_image_with_gemini(image_pil, model):
    prompt = (
        "You are a specialized medical imaging AI. "
        "Classify the provided CT scan image into one of the following categories: "
        "'tumor', 'hemorrhagic stroke', 'TBI', or 'Normal'. "
        "Respond with only the class name, nothing else."
    )
    generation_config = genai.types.GenerationConfig(
        max_output_tokens=10, temperature=0.1,
    )
    try:
        response = model.generate_content(
            [prompt, image_pil],
            generation_config=generation_config,
            stream=False
        )
        class_text = response.text.strip().lower()
        for c in ["tumor", "hemorrhagic stroke", "tbi", "normal"]:
            if c in class_text:
                return c
        print(f"Warning: Gemini returned unexpected class: {class_text}. Defaulting to 'Normal'.")
        return "normal"
    except Exception as e:
        print(f"Error during classification: {e}")
        return "normal"

def elimination_classify_with_gemini(image_pil, model):
    incidence_order = ["Normal", "TBI", "hemorrhagic stroke", "tumor"]
    possible_classes = ["tumor", "hemorrhagic stroke", "TBI", "Normal"]
    eliminated = set()
    for step in range(len(possible_classes) - 1):
        remaining = [c for c in possible_classes if c not in eliminated]
        if len(remaining) == 1:
            return remaining[0]
        if step == 0 and "Normal" in remaining:
            prompt = (
                f"You are a specialized medical imaging AI. The possible classes are: {', '.join(remaining)}. "
                f"If you see any abnormal mass, lesion, hemorrhage, or injury, you MUST eliminate 'Normal'. "
                f"Otherwise, eliminate the class you are most confident is NOT present. "
                f"Respond with only the class name to eliminate. Do not say 'None'. Do not refuse. Eliminate one class per step."
            )
        else:
            prompt = (
                f"You are a specialized medical imaging AI. The possible classes are: {', '.join(remaining)}. "
                f"Based on your medical knowledge and the provided CT scan image, eliminate the class you are most confident is NOT present. "
                f"If you are unsure, eliminate the class that is least likely based on real-world incidence rates (least common first: tumor < hemorrhagic stroke < TBI < Normal). "
                f"Respond with only the class name to eliminate. Do not say 'None'. Do not refuse. Eliminate one class per step."
            )
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=20, temperature=0.1,
        )
        try:
            response = model.generate_content(
                [prompt, image_pil],
                generation_config=generation_config,
                stream=False
            )
            elim_text = response.text.strip().lower()
            found = False
            for c in remaining:
                if c.lower() in elim_text and not ("not " + c.lower()) in elim_text:
                    eliminated.add(c)
                    print(f"Eliminated: {c}")
                    found = True
                    break
            if not found:
                fallback = sorted(remaining, key=lambda x: incidence_order.index(x))[0]
                eliminated.add(fallback)
                print(f"[INCIDENCE] Eliminated: {fallback}")
        except Exception as e:
            print(f"Error during elimination classification: {e}")
            fallback = sorted(remaining, key=lambda x: incidence_order.index(x))[0]
            eliminated.add(fallback)
            print(f"[INCIDENCE] Eliminated: {fallback}")
    remaining = [c for c in possible_classes if c not in eliminated]
    return remaining[0] if remaining else "Normal"

def main():
    api_key = "AIzaSyBWHjGwgjujmW5KoDd3PjmCjDtmc6rT_vg" 

    if not api_key or api_key == "AIzaSyB9Q3sQVOVzatUsteYE8nfzl3kQoOxGH7c" or api_key == "AIzaSyB9Q3sQVOVzatUsteYE8nfzl3kQoOxGH7c_OR_YOUR_PLACEHOLDER": 
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("ERROR: API Key seems to be a placeholder or is the example key.")
        print("Please replace it in the script with your actual Google API key.")
        print("Using a hardcoded API key is a security risk for non-prototype code.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return 
        
    try:
        genai.configure(api_key=api_key)
        print("Gemini API configured with hardcoded key (PROTOTYPE MODE).")
    except Exception as e:
        print(f"Error configuring Gemini API with the provided key: {e}")
        print("Please ensure the API key is correct and valid.")
        return

    print(f"Initializing Gemini model: {MODEL_NAME}")
    try:
        model = genai.GenerativeModel(MODEL_NAME)
    except Exception as e:
        print(f"Error initializing model '{MODEL_NAME}': {e}")
        print("Please ensure the model name is correct and your API key has access.")
        return

    if IMAGE_PATH_OR_URL == "tumor_ex.jpeg":
        if not download_image_if_not_exists(IMAGE_PATH_OR_URL, SAMPLE_IMAGE_URL):
            return

    raw_image = None
    if IMAGE_PATH_OR_URL.startswith("http://") or IMAGE_PATH_OR_URL.startswith("https://"):
        try:
            print(f"Loading image from URL: {IMAGE_PATH_OR_URL}")
            response = requests.get(IMAGE_PATH_OR_URL, stream=True)
            response.raise_for_status()
            raw_image = Image.open(response.raw).convert("RGB")
        except requests.exceptions.RequestException as e:
            print(f"Error loading image from URL: {e}")
            return
        except IOError as e:
            print(f"Error processing image from URL (PIL): {e}")
            return
    else:
        if not os.path.exists(IMAGE_PATH_OR_URL):
            print(f"Image file not found at: {IMAGE_PATH_OR_URL}")
            return
        try:
            print(f"Loading image from local path: {IMAGE_PATH_OR_URL}")
            raw_image = Image.open(IMAGE_PATH_OR_URL).convert("RGB")
        except FileNotFoundError:
            print(f"Image file not found at: {IMAGE_PATH_OR_URL}")
            return
        except IOError as e:
            print(f"Error loading image from local file (PIL): {e}")
            return
    if raw_image is None:
        print("Failed to load image. Exiting.")
        return

    image_width, image_height = raw_image.size

    print("\nClassifying image with Gemini using elimination method...")
    predicted_class = elimination_classify_with_gemini(raw_image, model)
    print(f"Predicted class: {predicted_class}")

    safe_class = predicted_class.replace(' ', '_').lower()
    safe_model = MODEL_NAME.replace('.', '_').replace('-', '_')
    output_report = f"report_for_{safe_class}_{safe_model}_bbox.txt"
    output_image = f"marked_image_for_{safe_class}_{safe_model}_bbox.png"

    current_image = raw_image.copy()
    current_bounding_box = None
    current_center = None
    current_label = None
    final_report = None
    gemini_radiology_report = None
    if predicted_class.strip().lower() == "normal":
        print("Class is 'Normal'. No bounding box refinement will be attempted.")
        try:
            current_image.save(output_image)
            print(f"Final marked image saved to {output_image}")
        except Exception as e:
            print(f"Error saving final marked image: {e}")
        gemini_radiology_report = "No abnormal findings."
        try:
            with open(output_report, "w", encoding="utf-8") as f:
                f.write(gemini_radiology_report)
            print(f"Final report saved to {output_report}")
        except Exception as e:
            print(f"Error saving final report: {e}")
        return
    step = 0
    max_steps = 10
    best_bbox = None
    best_bbox_area = 0
    best_bbox_center = None
    best_bbox_label = None
    best_bbox_image = None
    while step < max_steps:
        step += 1
        print(f"\n--- Gemini Bounding Box Refinement Step {step} ---")
        if step == 1:
            prompt_text = (
                f"You are a specialized medical imaging AI analyzing a CT scan.\n"
                f"The image is suspected to show a {predicted_class}.\n\n"
                f"Tasks:\n"
                f"1. Provide a concise technical radiological report based on your findings in the image. "
                f"Focus on any signs of a {get_finding_term(predicted_class)}.\n\n"
                f"2. If you identify the primary location of the {get_finding_term(predicted_class)}, you MUST provide its bounding box. "
                f"The bounding box coordinates should be in the format [ymin, xmin, ymax, xmax], "
                f"normalized to a 0-1000 scale (where 0,0 is the top-left corner of the image).\n"
                f"   - Output this bounding box on a new line, clearly marked like this:\n"
                f"     BOUNDING_BOX_{get_finding_term(predicted_class).upper().replace(' ', '_').replace('(', '').replace(')', '').replace('.', '')}: [ymin, xmin, ymax, xmax]\n"
                f"   - Example for a tumor: BOUNDING_BOX_TUMOR: [250, 300, 450, 500]\n"
                f"   - If multiple instances are present, identify the most prominent one.\n"
                f"   - If you cannot confidently detect a {get_finding_term(predicted_class)} or provide a bounding box, "
                f"     clearly state 'No {get_finding_term(predicted_class)} confidently detected for bounding box.' instead of providing a box.\n\n"
                f"If the bounding box is correct and in the right spot, reply with 'DONE'. If not, reply with the corrected bounding box.\n"
                f"Be extremely picky: only reply 'DONE' if the box is perfectly correct and tightly fits the finding.\n"
                f"Ensure your entire response, including the report and the bounding box line (if applicable), is returned."
            )
            gemini_inputs = [prompt_text, current_image]
        else:
            prompt_text = (
                f"You are a specialized medical imaging AI. The image already contains a bounding box for {predicted_class}.\n"
                f"The previous bounding box was: [ymin, xmin, ymax, xmax] = ["
                f"{int((current_bounding_box[1]/image_height)*1000)}, "
                f"{int((current_bounding_box[0]/image_width)*1000)}, "
                f"{int((current_bounding_box[3]/image_height)*1000)}, "
                f"{int((current_bounding_box[2]/image_width)*1000)}].\n"
                f"If this bounding box is correct and in the right spot, reply with 'DONE'. If not, reply with the corrected bounding box in the same format.\n"
                f"Be extremely picky: only reply 'DONE' if the box is perfectly correct and tightly fits the finding.\n"
                f"If you cannot confidently detect a {get_finding_term(predicted_class)}, reply with 'No {get_finding_term(predicted_class)} confidently detected for bounding box.'\n"
                f"Ensure your response includes only the bounding box line or 'DONE'."
            )
            gemini_inputs = [prompt_text, current_image]
        print(f"Prompt to Gemini:\n{prompt_text}\n")
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=2048, 
            temperature=0.2,
        )
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        retry_count = 0
        max_retries = 3
        while retry_count < max_retries:
            try:
                response = model.generate_content(
                    gemini_inputs,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    stream=False
                )
                gemini_output = response.text
            except Exception as e:
                print(f"Error during Gemini call: {e}")
                break
            print(f"Gemini output (step {step}, retry {retry_count+1}):\n{gemini_output}\n")
            if step == 1:
                report_lines = []
                for line in gemini_output.splitlines():
                    if line.strip().startswith("BOUNDING_BOX_") or line.strip().upper() == "DONE" or "confidently detected" in line.lower():
                        break
                    report_lines.append(line)
                gemini_radiology_report = "\n".join([l for l in report_lines if l.strip()])
            if 'done' in gemini_output.lower():
                print("Gemini indicated the bounding box is correct. Stopping refinement loop.")
                break
            if (f"no {get_finding_term(predicted_class)} confidently detected" in gemini_output.lower()):
                print("Gemini indicated no finding confidently detected. Stopping refinement loop.")
                break
            extracted_info = extract_bounding_box_from_output(gemini_output, image_width, image_height, predicted_class)
            if not extracted_info:
                print("No valid bounding box found, retrying with a more explicit prompt.")
                prompt_text += ("\n\nIMPORTANT: You MUST output a bounding box line in the format BOUNDING_BOX_{label}: [ymin, xmin, ymax, xmax] or say 'No {label} confidently detected for bounding box.' Do not skip this line.")
                gemini_inputs = [prompt_text, current_image]
                retry_count += 1
                continue
            current_center, current_bounding_box, current_label = extracted_info
            bbox_area = abs((current_bounding_box[2] - current_bounding_box[0]) * (current_bounding_box[3] - current_bounding_box[1]))
            if bbox_area > best_bbox_area:
                best_bbox = current_bounding_box
                best_bbox_area = bbox_area
                best_bbox_center = current_center
                best_bbox_label = current_label
                best_bbox_image = current_image.copy()
            print(f"Drawing bounding box: {current_bounding_box} with center {current_center} and label '{current_label}'")
            temp_image = current_image.copy()
            draw_box_and_center_on_image(temp_image, current_center, current_bounding_box, current_label, output_image)
            print(f"Saved image with bounding box to {output_image}")
            current_image = Image.open(output_image)
            break
        else:
            print("Max retries reached without a valid bounding box. Stopping refinement loop.")
            break
        if 'done' in gemini_output.lower() or (f"no {get_finding_term(predicted_class)} confidently detected" in gemini_output.lower()):
            break
    if best_bbox is not None and best_bbox_image is not None:
        print(f"Using best bounding box found: {best_bbox} with area {best_bbox_area}")
        draw_box_and_center_on_image(best_bbox_image, best_bbox_center, best_bbox, best_bbox_label, output_image)
    try:
        with open(output_report, "w", encoding="utf-8") as f:
            f.write(gemini_radiology_report if gemini_radiology_report else "No report generated.")
        print(f"Final report saved to {output_report}")
    except Exception as e:
        print(f"Error saving final report: {e}")
    try:
        current_image.save(output_image)
        print(f"Final marked image saved to {output_image}")
    except Exception as e:
        print(f"Error saving final marked image: {e}")

if __name__ == "__main__":
    main()