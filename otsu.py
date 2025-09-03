import cv2
import numpy as np
from matplotlib import pyplot as plt

sample_img_path = '/content/drive/MyDrive/PSG/SEMESTER_9/Computer_Vision_and_Image_Analysis/images/rolls.jpg'

input_img = cv2.imread(sample_img_path)
gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

arr = np.array(gray_img)

#lets take a np array for example 0 1 2 1 0 0, 0 3 4 4 1 0, 2 4 5 5 4 0, 1 4 5 5 4 1, 0 3 4 4 3 1, 0 2 3 3 2 0
#arr = np.array([[0,1,2,1,0,0],[0,3,4,4,1,0],[2,4,5,5,4,0],[1,4,5,5,4,1],[0,3,4,4,3,1],[0,2,3,3,2,0]])
pixel,count = np.unique(arr,return_counts=True)

pixel_count = np.column_stack((pixel,count))
print(pixel_count)
total_count = np.sum(count)
b_count, f_count = 0,0
max_threshold = 0
max_t = 0
for i in range(1,len(pixel)+1):
  # b count will be equal to sum of count before the ith pixel
  b_count = np.sum(count[:i])
  # f count will be equal to sum of count after the ith pixel
  f_count = np.sum(count[i:])

  wb = b_count/total_count
  wf = f_count/total_count

  #Mb will be equal to sum of pixel x count before the ith pixel
  mb = np.sum(pixel_count[:i,0]*pixel_count[:i,1])/b_count
  #Mf will be equal to sum of pixel x count after the ith pixel
  mf = np.sum(pixel_count[i:,0]*pixel_count[i:,1])/f_count

  threshold = wb*wf*(mb-mf)**2
  if threshold>max_threshold:
    max_threshold = threshold
    max_t = i

  if threshold<max_threshold:
    break

print(max_threshold)
t = max_threshold**(1/2)
print(max_t)
#now change the values of the image array where they are less than t and if greater leave them alone
l,b = arr.shape
for i in range(l):
  for j in range(b):
    if arr[i,j]<max_t:
      arr[i,j] = 0
    if arr[i,j]>=max_t:
      arr[i,j] = 255

plt.imshow(gray_img,cmap='gray')
plt.show()

plt.imshow(arr,cmap='gray')
plt.show()