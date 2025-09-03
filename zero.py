import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("/content/drive/MyDrive/PSG/SEMESTER_9/Computer_Vision_and_Image_Analysis/images/sample_image.jpg")

if img is None:
    print("Error: Could not load image")
else:
    blue, green, red = cv2.split(img)

    l, b = red.shape
    red_new = np.zeros((l, 2*b+1), dtype=np.uint8)
    green_new = np.zeros((l, 2*b+1), dtype=np.uint8)
    blue_new = np.zeros((l, 2*b+1), dtype=np.uint8)

    for i in range(l):
        for j in range(b):
            red_new[i, 2*j+1] = red[i, j]
            green_new[i, 2*j+1] = green[i, j]
            blue_new[i, 2*j+1] = blue[i, j]

            if 2*j+2 == 2*b:
                break
            else:
                red_new[i, 2*j+2] = red[i, j]
                green_new[i, 2*j+2] = green[i, j]
                blue_new[i, 2*j+2] = blue[i, j]

    rows, cols = red_new.shape
    red_new_col = np.zeros((2*l+1, cols), dtype=np.uint8)
    green_new_col = np.zeros((2*l+1, cols), dtype=np.uint8)
    blue_new_col = np.zeros((2*l+1, cols), dtype=np.uint8)

    for i in range(l):
        for j in range(cols):
            red_new_col[2*i+1, j] = red_new[i, j]
            green_new_col[2*i+1, j] = green_new[i, j]
            blue_new_col[2*i+1, j] = blue_new[i, j]

            if 2*i+2 == 2*l:
                break
            else:
                red_new_col[2*i+2, j] = red_new[i, j]
                green_new_col[2*i+2, j] = green_new[i, j]
                blue_new_col[2*i+2, j] = blue_new[i, j]

    modified_image = cv2.merge((blue_new_col, green_new_col, red_new_col))
    cv2.imwrite("/content/drive/MyDrive/PSG/SEMESTER_9/Computer_Vision_and_Image_Analysis/images/modified1.jpg", modified_image)
    plt.figure()
    plt.axis('off')
    plt.imshow(cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB))
    plt.show()
