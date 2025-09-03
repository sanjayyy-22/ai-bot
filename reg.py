import numpy as np
import cv2
import requests
from io import BytesIO

def region_growing(image, seed, threshold):
    rows, cols = image.shape
    segmented = np.zeros((rows, cols), dtype=np.uint8)
    segmented[seed] = 1
    seed_value = image[seed]

    def _get_neighbors(y, x):
        return [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]

    stack = [seed]
    while stack:
        y, x = stack.pop()
        for ny, nx in _get_neighbors(y, x):
            if 0 <= ny < rows and 0 <= nx < cols:
                if segmented[ny, nx] == 0 and abs(int(image[ny, nx]) - int(seed_value)) <= threshold:
                    segmented[ny, nx] = 1
                    stack.append((ny, nx))

    return segmented

def load_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        image = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        return image
    except Exception as e:
        print(f"Error loading image from URL: {e}")
        return None

def load_image_from_file(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return image

def main():
    url = ''  # URL of the image
    file_path = '/content/drive/MyDrive/Computer Vision/sample_image.jpg'  # Hardcoded fallback image path

    # Attempt to load the image from the URL
    image = load_image_from_url(url)

    # If URL loading failed, fall back to the hardcoded file path
    if image is None:
        print("Loading image from fallback file path.")
        image = load_image_from_file(file_path)

    # Check if the image is loaded successfully
    if image is None:
        print("Error: Image could not be loaded from both URL and file path.")
        return

    seed = (100, 100)  # Example seed point (y, x)
    threshold = 45
    result = region_growing(image, seed, threshold)

    # Save the segmented image
    cv2.imwrite('FinalSegementedRegionGrow.png', result * 255)
    print("Segmented image saved as 'FinalSegementedRegionGrow.png'")

if __name__ == "__main__":
    main()