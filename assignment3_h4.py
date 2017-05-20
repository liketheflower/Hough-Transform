
# coding: utf-8
import numpy as np
import math
import matplotlib.pyplot as plt
import assignment3_h1
import assignment3_h2
import assignment3_h3

def get_index_of_a_specified_vlaue_from_2darray(array2d,value):
    for i in range(array2d.shape[0]):
       for j in range(array2d.shape[1]):
           if int(10000*array2d[i,j])==int(10000*value):
              return i,j
    else:
        print ("error")

def generate_one_line_image(line_image,input_gray_level_image_name,sita,rou):
    one_line_image = np.zeros(line_image.shape)
    x_=[]
    y_=[]
    if math.isclose(np.sin(sita),0.0):
        if int(rou)>=0 and int(rou)<=line_image.shape[0]-1 :
            one_line_image[int(rou),:] = 255
        else:
            print ("error")

    else:
        for i in range(line_image.shape[0]):
            x_.append(i)
            y = (rou-i*np.cos(sita))/(np.sin(sita))
            y_.append(y)
           
            if y<=one_line_image.shape[1]-1 and y>=0:
                y = int(y)
                one_line_image[i,y] = 255.0
       #plt.plot(x_,y_)
       #plt.show()
    #print ("line_image[i,y]",one_line_image[i,y])
    sorted_line_image=np.sort(one_line_image, axis=None)
    print ("sorted_line_image[-4:]",sorted_line_image[-4:])
    #assignment3_h1.plot_image(one_line_image,input_gray_level_image_name,title_of_the_image='one line image ')
    return one_line_image
def max_pooling(array,filter_width):
    max_pooling_array = np.zeros(array.shape)
    for i in range(int(array.shape[0]/filter_width)):
         for j in range(int(array.shape[1]/filter_width)):
             max_poolint_kernel =array[i*filter_width:i*filter_width+4,j*filter_width:j*filter_width+4]
             max_value = assignment3_h1.get_max_vlaue_of_2darray(max_poolint_kernel)
             i_,j_ = get_index_of_a_specified_vlaue_from_2darray(max_poolint_kernel,max_value)
             max_pooling_array[i*filter_width+i_,j*filter_width+j_] = max_value
    return   max_pooling_array



def plot_original_image_with_detected_line(original_image, line_image,image_title,input_gray_level_image_name):
    '''this function is used to display the original grey image with a colored detected lines '''
    im = original_image.astype(float)
    I = np.dstack([im, im, im])
    for i in range(line_image.shape[0]):
        for j in range(line_image.shape[1]):
            if line_image[i,j] > 0:
                I[i, j, :] = [1.0, 0.0, 0.0]

    plt.imshow(I, interpolation='nearest' )
    plt.title(image_title)
    plt.savefig("/Users/jimmy/Dropbox/001 Courses/2016 fall/CV/assignment/hw3/"+input_gray_level_image_name[:-4]+image_title+'.png')
    plt.show()

def h4(input_gray_level_image_name, resolution_level,line_numer):
    hough_array, maximum_rou = assignment3_h3.h3(input_gray_level_image_name, resolution_level)
    
    #gaussian_kernel=np.array([[1, 2 ,1],[2, 4 ,2],[1, 2 ,1]])
    #gaussian_kernel=gaussian_kernel/16
    #y_direction_sobel_kernel=np.array([[-1, 0 ,1],[-2, 0 ,2],[-1, 0 ,1]])
    #x_direction_edge= assignment3_h1.some_filter(hough_array, x_direction_sobel_kernel)
    #y_direction_edge= assignment3_h1.some_filter(hough_array, y_direction_sobel_kernel)
    #hough_array = assignment3_h1.non_max(hough_array,x_direction_edge,y_direction_edge)
    #hough_array = assignment3_h1.some_filter(hough_array, gaussian_kernel)
    if resolution_level==0:
        hough_image_title="hough_image_with_LOW_resolu"
    else:
        hough_image_title="hough_image_with_HIGH_resolu"
    assignment3_h1.plot_image(hough_array,input_gray_level_image_name,hough_image_title)
    hough_array = max_pooling(hough_array, 4)
    input_gray_level_image = assignment3_h1.read_pgm(input_gray_level_image_name)
    print ("the hough array is:",hough_array)
    #plot the hough array
    if resolution_level==0:
        image_title="LOW_resolu_after_max_pooling"
    else:
        image_title="HIGH_resolu_after_max_pooling"
    assignment3_h1.plot_image(hough_array,input_gray_level_image_name,image_title)
    sorted_hough_flat_array=np.sort(hough_array, axis=None)
    line_image = np.zeros((input_gray_level_image.shape))
    #line_image = np.zeros((200,200))
    width,height=assignment3_h3.get_hough_image_size(resolution_level)
    #print ("sorted_hough_flat_array[-4:]",sorted_hough_flat_array[-10:])

    for i in range(line_numer):
        #print ("line number,sorted_hough_flat_array[-(i+1)]",i,sorted_hough_flat_array[-(i+1)])
        x,y=get_index_of_a_specified_vlaue_from_2darray(hough_array,sorted_hough_flat_array[-(i+1)])
        hough_array[x,y] = 0.0
        #print (" x,y", x,y)
        sita = (np.pi)*(x/width)
        rou = -maximum_rou+2*maximum_rou*(y/height)
        #print ("sita,rou",sita,rou)
        one_line_image=generate_one_line_image(line_image,input_gray_level_image_name,sita,rou)
        line_image+=one_line_image
    for i in range(line_image.shape[0]):
    	for j in range(line_image.shape[1]):
    		if line_image[i,j]>255.0:
    			line_image[i,j]=255.0
    sorted_line_image=np.sort(line_image, axis=None)
#print ("sorted_line_image_combined[-4:]",sorted_line_image[-4:])
    if resolution_level==0:
        image_title="detected_lines_based_on_LOW_resolu_hough_space"
    else:
        image_title="detected_lines_based_on_HIGH_resolu_hough_space"
    assignment3_h1.plot_image(line_image,input_gray_level_image_name,image_title)
    original_image = assignment3_h1.read_pgm(input_gray_level_image_name)
    if resolution_level==0:
        image_title="image_with_detected_lines_LOW_resolu_hough_space"
    else:
        image_title="image_with_detected_lines_HIGH_resolu_hough_space"
    plot_original_image_with_detected_line(original_image, line_image,image_title,input_gray_level_image_name)
    return line_image



if __name__ == '__main__':
    # Get the input gray level image based on the file name
    input_gray_level_image_name = "hough_simple_1.pgm"
    """
    This is related to the h2 of the assignment 3 of the computer vision course
    Output_gray_level_edge_image is the out put image
    The input grey level image is the input
    """
    # get the binary image based on the edge image detected by the sobel filter
    #pick up top N most important lines
    threshold_of_hough_voting = 20
    resolution_level = 0
    h4(input_gray_level_image_name, resolution_level,threshold_of_hough_voting)




