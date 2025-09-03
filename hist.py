import cv2
import numpy as np
import matplotlib.pyplot as plt

#img = cv2.imread("/content/drive/MyDrive/images.jpeg", cv2.IMREAD_GRAYSCALE)

# freq = np.zeros(256, dtype=int)   # for 8-bit grayscale
# rows, cols = img.shape

# for i in range(rows):
#     for j in range(cols):
#         freq[img[i, j]] += 1

freq = np.zeros(8,dtype=int)
freq = np.array([790,1023,850,656,329,245,122,81])

total_pixels = sum(freq)

# total_pixels = rows * cols
oldpmf = freq / total_pixels
pmf = [i for i in oldpmf if i!=0]
print(pmf)

# ---------------- Step 4: CDF (Cumulative Distribution Function) ----------------
cdf = np.cumsum(pmf)
print(cdf)
# ---------------- Step 5: Mapping function ----------------
L = len(pmf)  # total gray levels
sk = np.round(cdf * (L - 1)).astype(np.uint8)
print(sk)
# ---------------- Step 6: Apply sk ----------------
equalized_res = np.zeros(L)


for i in range(L):
  for v in range(len(sk)):
    if sk[v]==i:
      # print("hoi")
      equalized_res[i]+=pmf[v]

      print(equalized_res)


# ---------------- Step 7: Visualization ----------------
plt.figure(figsize=(14, 8))



plt.subplot(2, 2, 1)
plt.title("Original Histogram")
plt.stem( np.arange(8),pmf)

plt.subplot(2,2,2)
plt.title("Equalized Histogram")
plt.stem(np.arange(8),equalized_res)




