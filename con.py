import numpy as np
from PIL import Image

threshold = 128

binary_img = np.array([
    [0,0,1,0,0,1],
    [1,1,1,0,1,1],
    [0,1,0,0,0,0],
    [0,0,0,1,1,1],
    [0,0,0,1,0,1],
    [1,1,0,0,0,0],
  ])


binary_img = np.array([
    [1,1,1,0,0,0,0,0],
    [1,1,1,0,1,1,0,0],
    [1,1,1,0,1,1,0,0],
    [1,1,1,0,0,0,1,0],
    [1,1,1,0,0,0,1,0],
    [1,1,1,0,0,0,1,0],
    [1,1,1,0,0,0,1,0],
    [1,1,1,0,0,0,0,0],
  ])

binary_img = np.array([
    [0,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,0,0,0,0],
    [0,0,1,1,0,1,1,0,0],
    [0,1,1,1,1,1,1,1,0],
    [0,1,1,1,0,1,1,0,0],
    [0,0,0,0,0,0,0,0,0]
  ])



print("\nOriginal Image")
print(binary_img)

height, width = binary_img.shape

labels = np.zeros((height, width), dtype=np.int32)

current_label = 0
equivalence_table = {}

# Pass 1
for x in range(height):
    for y in range(width):
        if binary_img[x][y] == 1:

            left_label, top_label = 0, 0
            if(x > 0):
                top_label = labels[x-1][y]
            if(y > 0):
                left_label = labels[x][y-1]

            if left_label != 0 and top_label != 0:
                if(left_label != top_label):
                    equivalence_table[max(left_label, top_label)] = equivalence_table[min(left_label, top_label)]
                labels[x][y] = min(left_label, top_label)

            elif left_label != 0 or top_label != 0:
                if left_label != 0:
                    labels[x][y] = left_label
                else:
                    labels[x][y] = top_label
            else:
                current_label += 1
                labels[x][y] = current_label
                equivalence_table[current_label] = current_label

print("First Pass: ")
print(labels)

# Pass 2
for i in range(height):
    for j in range(width):
        if labels[i, j] > 0:
            labels[i, j] = equivalence_table[labels[i, j]]


print("Second Pass: ")
print(labels)