import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\nourn\OneDrive\Desktop\Semester 8\Computer Vision\Project\Test Cases\02-Matsawar-3edel-ya3am.png", cv2.IMREAD_GRAYSCALE)

# Apply thresholding to convert to binary image
_, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
edges = cv2.Canny(img, 100, 200)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(largest_contour)
qr_code_region = img[y:y + h, x:x + w]
cv2.imshow("QR Code Region", qr_code_region)
# Apply edge detection

edges = cv2.Canny(qr_code_region, 50, 150, apertureSize=3)

# Detect lines using Hough Transform
lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

# Initialize variables to store angles of detected lines
vertical_angles = []
horizontal_angles = []

# Loop through each detected line
for line in lines:
    rho, theta = line[0]
    if np.pi / 4 < theta < 3 * np.pi / 4:  # Vertical lines have theta close to pi/2
        vertical_angles.append(theta)
    else:  # Horizontal lines have theta close to 0 or pi
        horizontal_angles.append(theta)

# Calculate skewness angles based on detected lines
vertical_skewness = np.mean(vertical_angles) if vertical_angles else 0
horizontal_skewness = np.mean(horizontal_angles) if horizontal_angles else 0

# Convert skewness angles to degrees
vertical_skewness_deg = np.rad2deg(vertical_skewness)
horizontal_skewness_deg = np.rad2deg(horizontal_skewness)

# Print the skewness angles
print("Vertical skewness (degrees):", vertical_skewness_deg)
print("Horizontal skewness (degrees):", horizontal_skewness_deg)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

# Initialize variables to store angles of detected lines
vertical_angles = []
horizontal_angles = []

# Loop through each detected line
for line in lines:
    rho, theta = line[0]
    if np.pi / 4 < theta < 3 * np.pi / 4:  # Vertical lines have theta close to pi/2
        vertical_angles.append(theta)
    else:  # Horizontal lines have theta close to 0 or pi
        horizontal_angles.append(theta)

# Calculate skewness angles based on detected lines
vertical_skewness = np.mean(vertical_angles) if vertical_angles else 0
horizontal_skewness = np.mean(horizontal_angles) if horizontal_angles else 0

# Convert skewness angles to degrees
vertical_skewness_deg = np.rad2deg(vertical_skewness)
horizontal_skewness_deg = np.rad2deg(horizontal_skewness)

# Calculate the rotation angle to correct skewness
rotation_angle_deg = 180+horizontal_skewness_deg  # Start with the horizontal skewness angle

# Perform the rotation
rows, cols = img.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle_deg, 1)
img = cv2.warpAffine(img, rotation_matrix, (cols, rows), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)

qr_code_region = cv2.warpAffine(qr_code_region, rotation_matrix, (cols, rows), flags=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)
# Display the rotated image
cv2.imshow("Rotated Image", img)
cv2.imshow("Rotated Qr code", qr_code_region)

cv2.waitKey(0)
cv2.destroyAllWindows()
contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get the largest contour (assumed to be the QR code)
largest_contour = max(contours, key=cv2.contourArea)

# Get the minimum area rectangle bounding the contour
rect = cv2.minAreaRect(largest_contour)

# Get the angle of rotation
angle = rect[-1]

# Ensure the angle is within the range (-45, 45)
if angle < -45:
    angle += 90

# Rotate the image to correct skewness
rows, cols = img.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D(rect[0], angle, 1)
rotated_image = cv2.warpAffine(img, rotation_matrix, (cols, rows), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)

# Add a white border around the rotated image
border_width = 20  # Adjust the border width as needed
border_color = (255, 255, 255)  # White color
img = cv2.copyMakeBorder(img, border_width, border_width, border_width, border_width,
                                    cv2.BORDER_CONSTANT, value=border_color)

# Display the bordered image
cv2.imshow("Bordered Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

