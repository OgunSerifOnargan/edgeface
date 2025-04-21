from cv2 import cvtColor, COLOR_BGR2GRAY, threshold, THRESH_BINARY, boundingRect, resize, COLOR_GRAY2BGR
from numpy import zeros, uint8, any, mean, zeros_like, float32, clip


def center_and_crop_image(image, output_size=(150, 150)):
    # Convert the image to grayscale to detect black areas
    gray = cvtColor(image, COLOR_BGR2GRAY)

    # Threshold to create a binary mask where non-black pixels are 1, and black are 0
    _, thresh = threshold(gray, 1, 255, THRESH_BINARY)

    # Find the bounding box of the non-black region
    x, y, w, h = boundingRect(thresh)

    # Crop the image to the non-black region
    cropped_image = image[y:y+h, x:x+w]

    # Get the dimensions of the cropped image
    cropped_h, cropped_w = cropped_image.shape[:2]

    # Create a square canvas for centering the cropped image
    max_dim = max(cropped_w, cropped_h)
    centered_image = zeros((max_dim, max_dim, 3), dtype=uint8)

    # Calculate the offsets to center the cropped image
    y_offset = (max_dim - cropped_h) // 2
    x_offset = (max_dim - cropped_w) // 2

    # Place the cropped image onto the center of the new square canvas
    centered_image[y_offset:y_offset+cropped_h, x_offset:x_offset+cropped_w] = cropped_image

    # Resize the final centered image to the desired output size
    if centered_image is None or centered_image.size == 0:
        return None
    else:
        final_image = resize(centered_image, output_size)

    return final_image

def adjust_image_colors(image, target_mean=(128, 128, 128)):
    # Create a mask to ignore black pixels (assuming black means [0, 0, 0] in BGR)
    mask = any(image != [0, 0, 0], axis=-1)
    
    # Get non-black pixels
    non_black_pixels = image[mask]

    # Calculate the mean of the non-black pixels for each channel (BGR)
    mean_bgr = mean(non_black_pixels, axis=0)

    # Calculate scale factors for each channel
    scale_factors = [target_mean[i] / mean_bgr[i] if mean_bgr[i] != 0 else 1 for i in range(3)]

    # Apply the scale factors to the original image
    corrected_image = zeros_like(image, dtype=float32)
    for i in range(3):  # B, G, R channels
        corrected_image[:, :, i] = image[:, :, i] * scale_factors[i]

    # Clip the values to keep them within the valid range [0, 255]
    corrected_image = clip(corrected_image, 0, 255).astype(uint8)

    return corrected_image

def preprocess_face_img(face_img):
    face_img = cvtColor(face_img, COLOR_BGR2GRAY)
    face_img = cvtColor(face_img, COLOR_GRAY2BGR)
    face_img = center_and_crop_image(face_img, output_size=(150, 150))
    face_img = adjust_image_colors(face_img, target_mean=(128, 128, 128))
    return face_img