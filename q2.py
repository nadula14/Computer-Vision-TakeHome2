import numpy as np
import cv2
from collections import deque
import matplotlib.pyplot as plt

def region_growing(image, seeds, threshold=15):
    h, w = image.shape
    segmented = np.zeros((h, w), np.uint8)
    visited = np.zeros((h, w), np.bool_)

    for seed in seeds:
        queue = deque([seed])
        seed_value = image[seed]
        while queue:
            x, y = queue.popleft()
            if visited[x, y]:
                continue
            visited[x, y] = True
            pixel_value = image[x, y]
            if abs(int(pixel_value) - int(seed_value)) < threshold:
                segmented[x, y] = 255
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny]:
                            queue.append((nx, ny))
    return segmented

def create_synthetic_image():
    img = np.zeros((100, 100), dtype=np.uint8)
    img[20:40, 20:40] = 85   # Object 1
    img[60:80, 60:80] = 170  # Object 2
    return img

# --- Main Execution ---

# Create synthetic image
test_img = create_synthetic_image()

# Seed points inside both objects
seeds = [(25, 25), (65, 65)]

# Apply region growing
region_mask = region_growing(test_img, seeds, threshold=20)

# Mark seed points for visualization
temp_img = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
for x, y in seeds:
    cv2.circle(temp_img, (y, x), 2, (0, 0, 255), -1)

# Save output images
cv2.imwrite("synthetic_image_q2.png", test_img)
cv2.imwrite("seed_points_marked_q2.png", temp_img)
cv2.imwrite("region_grown_output_q2.png", region_mask)

# Plot all for visual inspection
plt.figure(figsize=(10, 3))
plt.subplot(1, 3, 1)
plt.title("Synthetic Image")
plt.imshow(test_img, cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Seed Points")
plt.imshow(temp_img)

plt.subplot(1, 3, 3)
plt.title("Region Grown")
plt.imshow(region_mask, cmap='gray')

plt.tight_layout()
plt.savefig("region_growing_plot_q2.png")
plt.show()
