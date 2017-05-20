
# coding: utf-8
import numpy as np
import math
import matplotlib.pyplot as plt
import assignment3_h2
import assignment3_h1

def get_max_rou(output_binary_edge_image):
    return int(math.sqrt(output_binary_edge_image.shape[0]**2+output_binary_edge_image.shape[1]**2))
def get_y_hough_array(rou,hight_of_hough_image, maximum_rou):
    return int((rou+maximum_rou)/(2*maximum_rou/hight_of_hough_image))
def get_hough_image_size(resolution_level):
    if resolution_level== 0 :
        width = 100
        hight = 100
    if resolution_level== 1 :
        width = 800
        hight = 800
    return width,hight


def h3(input_gray_level_image_name, resolution_level):
    '''this function is used to generate the hough voting array and also visualize the hough space image.
        the input is the original gray level image and the threshold for the hough voting. Also the resolution level is one of the inputs: 0 means low level and 1 means high level'''
    
    width_of_hough_image,hight_of_hough_image =get_hough_image_size(resolution_level)
    min_index=hight_of_hough_image
    max_index=0
   
    output_binary_edge_image = assignment3_h2.h2(input_gray_level_image_name, 50)
   
   
    hough_array = np.zeros((width_of_hough_image,hight_of_hough_image))
    #sita is the parameter of rou for the hough space
    sita=np.arange(0,np.pi,np.pi/width_of_hough_image)
    '''sita can be calculated based on the value of rou and the data points in the image space.
    '''
    maximum_rou = get_max_rou(output_binary_edge_image)
    for i in range(output_binary_edge_image.shape[0]):
        for j in range(output_binary_edge_image.shape[1]):
            #print("output_binary_edge_image[:, 10]",output_binary_edge_image[:, 10])
            if int(output_binary_edge_image[i,j])==255:
                #print("output_binary_edge_image[i,j]")
                for k in range(width_of_hough_image):
                    #x and y is using the same value as the index of the original image array
                    rou = i*np.cos(sita[k])+j*np.sin(sita[k])
                    #print("rou",rou)
                    #print("maxrou",maximum_rou)

                    if rou<=maximum_rou and rou>=-maximum_rou:
                        
                        y_index=get_y_hough_array(rou,hight_of_hough_image, maximum_rou)
                        
                        min_index=min(min_index,y_index)
                        max_index=max(max_index,y_index)
                        #print("min_index,max_index",min_index,max_index)
                        if y_index<0:
                            y_index=0
                        
                        if y_index>=hight_of_hough_image:
                            y_index = hight_of_hough_image-1
                        #print("y_index",y_index)
                        hough_array[k,y_index]+=1
    return hough_array, maximum_rou




def get_the_maximum_from_2darray(array):
    return float(max(array.max(axis=1)))



if __name__ == '__main__':
    # Get the input gray level image based on the file name
    input_gray_level_image_name = "hough_simple_1.pgm"
    """
    This is related to the h2 of the assignment 3 of the computer vision course
    Output_gray_level_edge_image is the out put image
    The input grey level image is the input
    """
    # get the binary image based on the edge image detected by the sobel filter
    #threshold_of_hough_voting = 10
    resolution_level = 0
    hough_array, maximum_rou = h3(input_gray_level_image_name, resolution_level)
    
    print ("the hough array is:\n",hough_array)
    #plot the hough array
    assignment3_h1.plot_image(hough_array,input_gray_level_image_name,"hough_image")
    '''
    print(hough_array.max(axis=0))
    print(hough_array.max(axis=1))
    max_value = get_the_maximum_from_2darray(hough_array)
    
    hist, bin_edges = np.histogram(hough_array, bins=256, density=False)
    plt.figure(1)
    plt.plot(hist, 'r')
    plt.title('original hist')
    plt.show()
    '''






