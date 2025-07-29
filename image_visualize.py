import numpy as np
import matplotlib.pyplot as plt

path = "usa_full_img_results/dnbr_normalized_ndvi_masked.npy"
path2 = "PATCHES_BANDS/greece_patch_(6912, 9984).npy"


img = np.load(path)

print(img.shape)

# Extract RGB bands: B04 (Red), B03 (Green), B02 (Blue)
# In the band order, these are at indices [3, 2, 1]
img_rgb = img[:, :, [3, 2, 1]]  # Red, Green, Blue order

# Normalize to 0-255 range for display
img_rgb = (img_rgb - np.min(img_rgb)) / (np.max(img_rgb) - np.min(img_rgb)) * 255

img_rgb = img_rgb.astype(np.uint8)  # Use uint8 for images

plt.imshow(img_rgb)
plt.show()

