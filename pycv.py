# coding=utf-8

# Developed and tested with:
#
# os:                 Windows 10 Pro
# environment:        miniconda3
# jupyter version:    4.3.0
# python version:     3.6.2
# opencv version:     3.3.0
# matplotlib version: 2.0.2
# numpy version:      1.13.1

# Notes:
#
#

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

def print_env():
    # Print versions
    print("Python version:\t\t" + platform.python_version())
    print("OpenCV version:\t\t" + cv2.__version__)
    print("Matplotlib version:\t" + mpl.__version__)
    print("Numpy version:\t" + np.__version__)

def read_input():
    assert(len(sys.argv) == 2)

    file = sys.argv[1]
    assert(os.path.exists(file))

    return cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)

def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY);
    
def convert_to_binary(image):
    _, bw_image = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return bw_image

def filter_image(image, radius):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius,radius))
    filtered_image = image.copy()
    filtered_image = cv2.morphologyEx(filtered_image, cv2.MORPH_OPEN, kernel)
    filtered_image = cv2.morphologyEx(filtered_image, cv2.MORPH_CLOSE, kernel)
    return filtered_image

def blur_image(image):
    return cv2.GaussianBlur(image, (5,5), 0)

def apply_hct(image, min_radius):
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 10,
                               param1 = 1,
                               param2 = 35,
                               minRadius = min_radius,
                               maxRadius = 0)
    return circles

def mark_image(input, circles):
    marked_image = input.copy()
    gray_image = convert_to_gray(input)
    bw_image = convert_to_binary(gray_image)

    if circles is not None:
        circles = np.uint16(np.around(circles))

    for c in circles[0,:]:
        if bw_image[c[1], c[0]] == 0:
            bw_image = (255 - bw_image)
            break

    cc = cv2.connectedComponentsWithStats(bw_image, 8)

    for c in circles[0,:]:
        label = cc[1][c[1], c[0]]
        centroid = np.uint16(np.around(cc[3][label]))
        cv2.circle(marked_image, (centroid[0], centroid[1]), c[2], (255,0,128), 4)
        
    return marked_image
    
def main():
    print_env()
    image = read_input()
    gray_image = convert_to_gray(image)
    bw_image = convert_to_binary(gray_image)
    filtered_image = filter_image(bw_image, 10)
    smooth_image = blur_image(filtered_image)
    circles = apply_hct(smooth_image, 10)
    marked_image = mark_image(image, circles)
    plt.figure()
    plt.imshow(marked_image)
    plt.show()

if __name__=="__main__":
    main()