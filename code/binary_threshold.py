import cv2
import numpy as np

def binary_threshold(image, sobel_kernel=3, xgrad_thresh=(30, 150), s_thresh=(125, 255)):
    # Grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))

    # Caculate the direction of the gradient
    xscaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Treshold gradient
    sx_binary = np.zeros_like(xscaled_sobel)
    sx_binary[(xscaled_sobel >= xgrad_thresh[0]) & (xscaled_sobel <= xgrad_thresh[1])] = 1

    # Treshold color channel
    hls_img = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls_img[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sx_binary)
    combined_binary[(s_binary == 1) | (sx_binary == 1)] = 1

    return combined_binary
