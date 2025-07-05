from flask import Flask, render_template, request, redirect, url_for, flash
import os
import nibabel as nib
import numpy as np
from skimage.transform import resize # Used in original segment_stroke
import tensorflow as tf
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt # For saving images

app = Flask(__name__)
app.secret_key = 'super secret key' # Replace with a real secret key in production

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'static/results' # Store results in static for easy display
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Ensure upload and results directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'nii'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

def segment_stroke_web(image_path, model_path='stroke_segmentation_model_3d.h5'):
    try:
        # Load the pre-trained model
        # Check if model exists, otherwise this will error out in production
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model = tf.keras.models.load_model(model_path)

        ct_scan_nifti = nib.load(image_path)
        ct_scan_data = ct_scan_nifti.get_fdata()

        # Preprocess: Resize, normalize (if necessary based on training)
        # Original script uses (128, 128, 64)
        target_shape = (128, 128, 64)
        ct_scan_resized = resize(ct_scan_data, target_shape, mode='constant', preserve_range=True)

        # Expand dimensions for model input (batch_size, height, width, depth, channels)
        ct_scan_model_input = np.expand_dims(ct_scan_resized, axis=(0, -1))

        # Predict mask
        predicted_mask = model.predict(ct_scan_model_input)

        # Postprocess: Apply threshold to get binary mask
        # Original script uses > 0.5
        binary_mask = (predicted_mask > 0.5).astype(np.float32)

        # Select a representative slice to return for display
        # Original script iterates to find a slice with segmentation
        # For simplicity here, let's try to find one or default
        # binary_mask shape is (1, 128, 128, 64, 1) after prediction

        # Squeeze out batch and channel dimensions for easier processing
        binary_mask_squeezed = np.squeeze(binary_mask, axis=(0, -1)) # Shape (128, 128, 64)
        ct_scan_resized_squeezed = np.squeeze(ct_scan_model_input, axis=(0,-1)) # Shape (128, 128, 64)


        best_slice_index = -1
        max_segmentation_area = 0

        # Iterate through slices along the depth axis (axis 2)
        for i in range(binary_mask_squeezed.shape[2]):
            slice_segmentation_area = np.sum(binary_mask_squeezed[:, :, i])
            if slice_segmentation_area > max_segmentation_area:
                max_segmentation_area = slice_segmentation_area
                best_slice_index = i

        if best_slice_index != -1 and max_segmentation_area > 0:
            # Return the original resized slice and the corresponding mask slice
            original_slice_for_display = ct_scan_resized_squeezed[:, :, best_slice_index]
            mask_slice_for_display = binary_mask_squeezed[:, :, best_slice_index]
            return original_slice_for_display, mask_slice_for_display, best_slice_index, None # No error
        else:
            # If no segmentation found, return a middle slice of the original scan and no mask
            middle_slice_index = ct_scan_resized_squeezed.shape[2] // 2
            original_slice_for_display = ct_scan_resized_squeezed[:, :, middle_slice_index]
            return original_slice_for_display, None, middle_slice_index, "No stroke detected or segmentation area is zero."

    except FileNotFoundError as fnf_error:
        print(f"Error in segment_stroke_web: {fnf_error}")
        return None, None, -1, str(fnf_error)
    except Exception as e:
        print(f"Error in segment_stroke_web: {e}")
        # Return None for images and the error message
        return None, None, -1, f"An error occurred during segmentation: {str(e)}"

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the file
        original_scan_slice, segmented_mask_slice, layer_index, error_message = segment_stroke_web(filepath)

        if error_message:
            # If segment_stroke_web returned an error message
            return render_template('index.html', error=error_message)

        if original_scan_slice is not None:
            # Generate unique filenames for the images
            base_filename = secure_filename(file.filename).rsplit('.', 1)[0]

            original_image_filename = f"{base_filename}_original_slice{layer_index}.png"
            original_image_path = os.path.join(app.config['RESULTS_FOLDER'], original_image_filename)

            # Save the original scan slice
            plt.imsave(original_image_path, original_scan_slice, cmap='gray')
            original_image_url = url_for('static', filename=f'results/{original_image_filename}')

            segmented_image_url = None
            message_display = f"File processed. Displaying slice {layer_index}."

            if segmented_mask_slice is not None:
                segmented_image_filename = f"{base_filename}_segmented_slice{layer_index}.png"
                segmented_image_path = os.path.join(app.config['RESULTS_FOLDER'], segmented_image_filename)

                # Overlay the mask on the original slice for better visualization
                plt.imshow(original_scan_slice, cmap='gray')
                plt.imshow(segmented_mask_slice, cmap='Reds', alpha=0.5) # Reds cmap, with alpha for transparency
                plt.axis('off') # Turn off axis numbers and ticks
                plt.savefig(segmented_image_path, bbox_inches='tight', pad_inches=0)
                plt.close() # Close the figure to free memory

                segmented_image_url = url_for('static', filename=f'results/{segmented_image_filename}')
            else:
                # No segmentation found, update message
                message_display = f"File processed. Displaying slice {layer_index}. No stroke detected in this slice or image."


            return render_template('index.html',
                                   original_image=original_image_url,
                                   segmented_image=segmented_image_url, # This will be None if no segmentation
                                   message=message_display,
                                   layer_index=layer_index) # Add layer_index here
        else:
            # This case should ideally be caught by error_message from segment_stroke_web
            return render_template('index.html', error="Failed to process the image or extract slices.")

    else:
        flash('Allowed file types are .nii') # Keep flash for this type of error
        return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
