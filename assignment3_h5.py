
# coding: utf-8
import numpy as np
import math
import matplotlib.pyplot as plt
import assignment3_h1
import assignment3_h2
import assignment3_h3
import assignment3_h4


def h5(input_gray_level_image_name, resolution_level,threshold_of_hough_voting):
    #input_gray_level_image_name = "hough_simple_1.pgm"
    """
        This is related to the h5 of the assignment 3 of the computer vision course
        Output_gray_level_edge_image is the out put image
        The input grey level image is the input
        """
    binary_edge_image=assignment3_h2.h2(input_gray_level_image_name, threshold=128)
    line_image = assignment3_h4.h4(input_gray_level_image_name, resolution_level,threshold_of_hough_voting)
    line_image = assignment3_h2.canny_algo(binary_edge_image , line_image)
    if resolution_level==0:
        image_title="detected_lines_not_extend_beyond_the_object_LOW_resolu_hough_space"
    else:
        image_title="detected_lines_not_extend_beyond_the_object_HIGH_resolu_hough_space"

    assignment3_h1.plot_image(line_image,input_gray_level_image_name,image_title)
    original_image = assignment3_h1.read_pgm(input_gray_level_image_name)
    if resolution_level==0:
        image_title="image_with_detected_lines_not_extend_beyond_object_LOW_resolu_hough_space"
    else:
        image_title="image_with_detected_lines_not_extend_beyond_object_HIGH_resolu_hough_space"
    
    assignment3_h4.plot_original_image_with_detected_line(original_image, line_image,image_title,input_gray_level_image_name)
    return line_image



if __name__ == '__main__':
    # Get the input gray level image based on the file name
    #input_gray_level_image_name = "hough_simple_1.pgm"
    #input_gray_level_image_name = "hough_simple_2.pgm"
    input_gray_level_image_name = "hough_complex_1.pgm"
    """
    This is related to the extra credit of the assignment 3 of the computer vision course
    Output_gray_level_edge_image is the out put image
    The input grey level image is the input
    """
    # get the binary image based on the edge image detected by the sobel filter
    #pick up top N most important lines
    threshold_of_hough_voting = 20
    resolution_level = 1
    h5(input_gray_level_image_name, resolution_level,threshold_of_hough_voting)
