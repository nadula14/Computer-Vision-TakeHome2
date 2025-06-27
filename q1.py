import cv2
import numpy as np
import matplotlib.pyplot as plt

image = np.full((200, 200), 50, dtype=np.uint8) 
cv2.circle(image, (60, 60), 30, 120, -1)         
cv2.rectangle(image, (120, 120), (170, 170), 200, -1)  


mean = 0
stddev = 20
noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8)
noisy_image = cv2.add(image, noise)

_, otsu_thresh = cv2.threshold(noisy_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imwrite("original_image.png", image)
cv2.imwrite("noisy_image.png", noisy_image)
cv2.imwrite("otsu_threshold.png", otsu_thresh)
