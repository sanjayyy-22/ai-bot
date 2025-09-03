import cv2
import numpy as np
from matplotlib import pyplot as plt

sample_img = '/content/drive/MyDrive/Computer Vision/sample_image.jpg'

img = cv2.imread(sample_img)

if img is None:
    print(f"Error: Could not load image")
else:
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray_img_array = np.array(gray_img)

    gx_mask = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    gy_mask = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

    img_rows, img_cols = gray_img.shape
    mask_rows, mask_cols = gx_mask.shape

    output_rows = img_rows - mask_rows + 1
    output_cols = img_cols - mask_cols + 1

    gx = np.zeros((output_rows, output_cols))
    gy = np.zeros((output_rows, output_cols))

    for i in range(output_rows):
        for j in range(output_cols):
            roi = gray_img[i:i+mask_rows, j:j+mask_cols]
            gx[i, j] = np.sum(roi * gx_mask)
            gy[i, j] = np.sum(roi * gy_mask)

    # Calculate gradient magnitude
    magnitude = np.sqrt(gx**2 + gy**2)

    direction = np.arctan2(gy, gx) * 180 / np.pi
    direction[direction < 0] += 180


    # Non-maximum suppression
    Z = np.zeros_like(magnitude)
    rows, cols = magnitude.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            angle = direction[i, j]

            # makign the angle to 4 directions (0, 45, 90, 135)
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                neighbor1 = magnitude[i, j + 1]
                neighbor2 = magnitude[i, j - 1]
            elif (22.5 <= angle < 67.5):
                neighbor1 = magnitude[i - 1, j + 1]
                neighbor2 = magnitude[i + 1, j - 1]
            elif (67.5 <= angle < 112.5):
                neighbor1 = magnitude[i - 1, j]
                neighbor2 = magnitude[i + 1, j]
            else: # 112.5 <= angle < 157.5
                neighbor1 = magnitude[i - 1, j - 1]
                neighbor2 = magnitude[i + 1, j + 1]

            if magnitude[i, j] >= neighbor1 and magnitude[i, j] >= neighbor2:
                Z[i, j] = magnitude[i, j]
            else:
                Z[i, j] = 0

    # Hysteresis Thresholding
    high_threshold_ratio = 0.15
    low_threshold_ratio = 0.05

    print("Z",Z)

    high_threshold = np.max(Z) * high_threshold_ratio
    low_threshold = high_threshold * low_threshold_ratio

    print(high_threshold,low_threshold)

    # Create binary images for strong and weak edges
    strong_i, strong_j = np.where(Z >= high_threshold)
    weak_i, weak_j = np.where((Z >= low_threshold) & (Z < high_threshold))

    # Initialize the output edge map
    output_img = np.zeros_like(Z, dtype=np.uint8)
    output_img[strong_i, strong_j] = 255

    # output_img[weak_i, weak_j] = 100


    # Plotting the results
    plt.imshow(gray_img, cmap='gray')
    plt.title('Original Grayscale')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    plt.imshow(output_img, cmap='gray')
    plt.title('Canny Edges')
    plt.axis('off')

    plt.tight_layout()
    plt.show()