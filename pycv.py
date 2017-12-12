# coding=utf-8

# PyCV: Python + OpenCV test

# Developed and tested with:
#
# os:                 Windows 10 Pro
# environment:        miniconda3
# python version:     3.6.2
# opencv version:     3.3.0
# matplotlib version: 2.0.2
# numpy version:      1.13.1

# Notes:
#
#    This is a solution-oriented approach considering 2 flat images.
#    Filter ordering (morphological opening/closing) as well as filter
#    and transform parameters were fine tuned to the problem.

# Use:
#
#    python pycv.py img/[filename]


import sys
import os.path
import platform
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import cv2


# Constants
RADIUS = 10
CC_LABELS = 1
CC_CENTROIDS = 3
CIRCLE_RADIUS = 2
MARKER_COLOR = (255,255,0)
MARKER_RADIUS = 4


# Print versions
def print_env():
    # Print versions
    print("Python version:\t\t" + platform.python_version())
    print("OpenCV version:\t\t" + cv2.__version__)
    print("Matplotlib version:\t" + mpl.__version__)
    print("Numpy version:\t\t" + np.__version__)


# Read input from argv and convert it to an RGB image
def read_input():
    # Assert user input is valid
    assert(len(sys.argv) == 2)
    file = sys.argv[1]
    assert(os.path.exists(file))

    # Return an RGB image read from file
    image = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
    return image


# Convert an RGB image to binary using Otsu's threshold
def convert_to_binary(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY);
    _, bw_image = cv2.threshold(gray_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return bw_image


# Apply morphological filtering and Gaussian blur to the image
def filter_image(image, radius):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius,radius))
    filtered_image = image.copy()
    filtered_image = cv2.morphologyEx(filtered_image, cv2.MORPH_OPEN, kernel)
    filtered_image = cv2.morphologyEx(filtered_image, cv2.MORPH_CLOSE, kernel)
    blur_image = cv2.GaussianBlur(filtered_image, (5,5), 0)
    return blur_image


# Apply Hough Circle Transform and return the circles center position and radius
def apply_hct(image, min_radius):
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, min_radius,
                               param1 = 1,
                               param2 = 35,
                               minRadius = min_radius,
                               maxRadius = 0)
    return circles


# Plot over the circles over the original image
def mark_image(image, circles):
    result_image = image.copy()
    bw_image = convert_to_binary(image)
    
    # Check if there is any circle
    if circles is not None:
        # Round the center position
        circles = np.uint16(np.around(circles[0,:]))

        # Check if any circle is in a black component
        for c in circles:
            if bw_image[c[1], c[0]] == 0:
                bw_image = (255 - bw_image)
                break

        # Get the connected components
        cc = cv2.connectedComponentsWithStats(bw_image, 8)

        # Plot over the original image
        for c in circles:
            label = cc[CC_LABELS][c[1], c[0]]
            centroid = np.uint16(np.around(cc[CC_CENTROIDS][label]))
            cv2.circle(result_image, (centroid[0], centroid[1]),
                       c[CIRCLE_RADIUS], MARKER_COLOR, MARKER_RADIUS)

    return result_image


# Show results
def show_results(image, result):
    _, ax = plt.subplots(1,2)
    ax[0].set_title('Input')
    ax[0].imshow(image)
    ax[0].axis('off')
    ax[1].set_title('Output')
    ax[1].imshow(result)
    ax[1].axis('off')
    plt.show()


# Main function
def main():
    print("\tPython + OpenCV")
    print("------------------------------")
    print_env()
    print("------------------------------")
    print("Reading image...")
    image = read_input()
    print("Applying threshold...")
    bw_image = convert_to_binary(image)
    print("Filtering...")
    filtered_image = filter_image(bw_image, RADIUS)
    print("Finding circles...")
    circles = apply_hct(filtered_image, RADIUS)
    print("Plotting...") 
    result_image = mark_image(image, circles)
    print("Showing results...")
    show_results(image, result_image)
    print("Done!")


# Run main
if __name__=="__main__":
    main()