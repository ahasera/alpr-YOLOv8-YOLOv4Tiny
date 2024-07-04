import cv2
import numpy as np

def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))  # Increased clip limit for higher contrast
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def sharpen_image(image):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(image, -1, kernel)

def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def remove_noise(image, strength=10):
    return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)

def upscale_image(image, scale=2):
    height, width = image.shape[:2]
    return cv2.resize(image, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)

def deskew_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def preprocess_image(image_path, save_steps=False):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error reading image {image_path}")

        # Step 1: Enhance contrast
        contrast_enhanced = enhance_contrast(image)
        if save_steps:
            cv2.imwrite("contrast_enhanced.jpg", contrast_enhanced)

        # Step 2: Sharpen the image with contrast
        sharpened_image = sharpen_image(contrast_enhanced)
        if save_steps:
            cv2.imwrite("sharpened_image.jpg", sharpened_image)

        # Step 3: Adjust gamma
        gamma_corrected = adjust_gamma(sharpened_image, gamma=1.2)
        if save_steps:
            cv2.imwrite("gamma_corrected.jpg", gamma_corrected)

        # Step 4: Remove noise
        denoised_image = remove_noise(gamma_corrected)
        if save_steps:
            cv2.imwrite("denoised_image.jpg", denoised_image)

        # Step 5: Upscale the image
        upscaled_image = upscale_image(denoised_image)
        if save_steps:
            cv2.imwrite("upscaled_image.jpg", upscaled_image)

        return upscaled_image

    except Exception as e:
        print(f"Error preprocessing image {image_path}: {str(e)}")
        return None