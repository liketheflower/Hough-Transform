
# coding: utf-8


import re
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
import assignment2_p1
import assignment3_h1


def canny_algo(binary_edge_image_high_threshold,binary_edge_image_low_threshold):
    assignment3_h1.zero_padding(binary_edge_image_high_threshold)
    assignment3_h1.zero_padding(binary_edge_image_low_threshold)
    for i in range(binary_edge_image_low_threshold.shape[0]-2):
        for j in range(binary_edge_image_low_threshold.shape[0]-2):
            mask=binary_edge_image_high_threshold[i:i+3,j:j+3]
            if math.isclose(sum(sum(mask)),0.0):
                binary_edge_image_low_threshold[i+1,j+1] = 0.0
    assignment3_h1.un_zero_padding(binary_edge_image_high_threshold)
    assignment3_h1.un_zero_padding(binary_edge_image_low_threshold)
    return binary_edge_image_low_threshold







def h2(input_gray_level_image_name, threshold=128):
    output_gray_level_edge_image = assignment3_h1.h1(input_gray_level_image_name)
    #double threshold is used here as shown in canny algo
    binary_edge_image_high_threshold= assignment2_p1.get_binary_image(output_gray_level_edge_image)
    assignment3_h1.plot_image(binary_edge_image_high_threshold, input_gray_level_image_name,"binary_edge_image_with_high_threshold")
    low_threshold_indi = True
    binary_edge_image_low_threshold = assignment2_p1.get_binary_image(output_gray_level_edge_image, low_threshold_indi)
    assignment3_h1.plot_image(binary_edge_image_low_threshold,input_gray_level_image_name, "binary_edge_image_with_low_threshold")
    binary_edge_image = canny_algo(binary_edge_image_high_threshold,binary_edge_image_low_threshold)
    assignment3_h1.plot_image(binary_edge_image, input_gray_level_image_name,"binary_edge_image_with_canny_double_threshold_combination")
    return binary_edge_image

if __name__ == '__main__':
    # Get the input gray level image based on the file name
    input_gray_level_image_name = "hough_simple_1.pgm"
    """
    This is related to the h2 of the assignment 3 of the computer vision course
    Output_gray_level_edge_image is the out put image
    The input grey level image is the input
    """
    # get the binary image based on the edge image detected by the sobel filter
    # threshold = 50
    output_binary_edge_image = h2(input_gray_level_image_name)
    



