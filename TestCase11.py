from math import sqrt
import cv2
import numpy as np
from skimage import io, color, filters, feature

import matplotlib.pyplot as plt
import cv2
import numpy as np
img = cv2.imread(r"C:\Users\nourn\OneDrive\Desktop\Semester 8\Computer Vision\Project\Test Cases\11-weewooweewooweewoo.png", cv2.IMREAD_GRAYSCALE)


def detect_noise(image):
    # Convert the image to grayscale
    if image.shape[-1] < 3:
        gray_image = color.rgb2gray(image)
    else:
        gray_image = color.rgb2gray(image[:, :, :3])  # Take only the RGB channels

    # Compute Fourier Transform to analyze frequency content
    fft_image = np.fft.fft2(gray_image)
    magnitude_spectrum = np.log(np.abs(fft_image) + 1)

    # Extract horizontal component of the magnitude spectrum
    horizontal_component = np.abs(np.fft.fftshift(fft_image)[:, :fft_image.shape[1] // 2])

    # Calculate the maximum value in the horizontal component
    max_horizontal_component = np.max(horizontal_component)

    # Define a threshold for detecting horizontal noise
    # You may need to adjust this threshold based on your images
    noise_threshold = 0.1

    # Check if the maximum horizontal component value exceeds the threshold
    print("Maximum feature value:", max_horizontal_component)

    # Define a threshold for detecting vertical noise
    # You may need to adjust this threshold based on your images
    noise_threshold = 100000

    # Check if the features indicate vertical noise
    if max_horizontal_component> noise_threshold:
        return True
    else:
        return False


# Example usage:
# Load an image
image1 = io.imread(r"C:\Users\nourn\OneDrive\Desktop\Semester 8\Computer Vision\Project\Test Cases\11-weewooweewooweewoo.png")

# Detect vertical noise in the image
if detect_noise(image1):
    print("Image contains vertical noise.")
else:
    print("Image does not contain vertical noise.")


def distance(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def butterworthLP(D0, imgShape, n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = 1 / (1 + (distance((y, x), center) / D0) ** (2 * n))
    return base



fourier_transform = np.fft.fft2(img)
center_shift = np.fft.fftshift(fourier_transform)
epsilon = 1e-10  # A small constant to avoid zero values
fourier_noisy = 20 * np.log(np.abs(center_shift)+epsilon)




rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

# horizontal mask
center_shift[crow :crow + 1, 0:ccol - 10] = 1
center_shift[crow :crow + 1, ccol + 10:] = 1
# vertical mask
# center_shift[:crow - 10, ccol - 4:ccol + 4] = 1
# center_shift[crow + 10:, ccol - 4:ccol + 4] = 1

filtered = center_shift * butterworthLP(80, img.shape, 10)

f_shift = np.fft.ifftshift(center_shift)
denoised_image = np.fft.ifft2(f_shift)
denoised_image = np.real(denoised_image)

f_ishift_blpf = np.fft.ifftshift(filtered)
denoised_image_blpf = np.fft.ifft2(f_ishift_blpf)
denoised_image_blpf = np.real(denoised_image_blpf)

fourier_noisy_noise_removed = 20 * np.log(np.abs(center_shift)+epsilon)
_, denoised_image_blpf_bin = cv2.threshold(denoised_image_blpf, 128, 255, cv2.THRESH_BINARY)

fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(2, 3, 1)
ax1.title.set_text("Original Image")
ax1.imshow(img, cmap='gray')
ax2 = fig.add_subplot(2, 3, 2)
ax2.imshow(fourier_noisy, cmap='gray')
ax2.title.set_text("Fourier Transform")
ax3 = fig.add_subplot(2, 3, 3)
ax3.imshow(fourier_noisy_noise_removed, cmap='gray')
ax3.title.set_text("Fourier Transform with mask")
ax4 = fig.add_subplot(2, 3, 4)
ax4.imshow(denoised_image, cmap='gray')
ax4.title.set_text("Denoised and unfiltered image")
ax5 = fig.add_subplot(2, 3, 5)
ax5.imshow(denoised_image_blpf, cmap='gray')
ax5.title.set_text("Before Binarization")

ax5 = fig.add_subplot(2, 3, 6)
ax5.imshow(denoised_image_blpf_bin, cmap='gray')
ax5.title.set_text("Denoised and filtered image")
#cv2.imwrite("testCase11.png", denoised_image_blpf_bin)
#plt.show()