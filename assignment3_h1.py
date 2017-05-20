
# coding: utf-8



import re
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
import os
import copy
import collections
#from matplotlib import pyplot

# read pgm file
def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=58
                            ).reshape((int(height), int(width)))



def plot_image(image,input_gray_level_image_name,title_of_the_image=' '):
    plt.imshow(image, cmap='gray')
    plt.title(title_of_the_image)
    plt.savefig("/Users/jimmy/Dropbox/001 Courses/2016 fall/CV/assignment/hw3/"+input_gray_level_image_name[:-4]+title_of_the_image+'.png')
    plt.show()

def zero_padding(input_array):
    #insert zeros to the first column and first row
    output_array=np.insert(input_array, 0, values=0, axis=1)
    output_array=np.insert(output_array, 0, values=0, axis=0)
    z = np.zeros((len(output_array[:,0]),1))
    output_array= np.append(output_array, z, axis=1)
    z = np.zeros((1, len(output_array[0,:])))
    output_array= np.append(output_array, z, axis=0)
    #print(output_array)
    return output_array

def un_zero_padding(input_array):
    p=scipy.delete(input_array, 0, 1)
    p=scipy.delete(p, -1, 1)
    p=scipy.delete(p, 0, 0)
    p=scipy.delete(p, -1, 0)
    return p

    
    
def some_filter(original_image, filter):
    #zero padding is used for the boundary
    #here the image is the image_after_zero_padding
    #print("original_image",original_image.shape)
    image=zero_padding(original_image)
    #print("image_after_zero_padding.shape",image.shape)
    image_after_filter = np.zeros(image.shape)
    #print("image_after_filter.shape",image_after_filter.shape)
    for i in range (len(image[:,0])-2):
        for j in range(len(image[0,:])-2):
            image_after_filter[i+1][j+1]=image[i][j]*filter[0][0]+image[i][j+1]*filter[0][1]+image[i][j+2]*filter[0][2]+image[i+1][j]*filter[1][0]+image[i+1][j+1]*filter[1][1]+image[i+1][j+2]*filter[1][2]+image[i+2][j]*filter[2][0]+image[i+2][j+1]*filter[2][1]+image[i+2][j+2]*filter[2][2]
    image_after_filter=un_zero_padding(image_after_filter)
    #print(image_after_filter.shape)
    return image_after_filter


def combine_x_y_edge_image(x, y):
    """This function is used to combine the x direction edge image
    and the y direction image.
    The formular used here is sqrt(x^2+y^2)"""
    com_output = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            com_output[i,j]=math.sqrt(x[i,j]**2+y[i,j]**2)
    return com_output

def pure_bilinear_interpola(x,y,f00,f01,f10,f11):
    a =  np.matrix((1-x, x ))
    b =  np.matrix(((f00,f01),(f10, f11)))
    c =  np.matrix(((1-y,),(y,)))
    return  float(a*b*c)
def bilinear_interpola(i,j,thine_edge,x_direction_edge,y_direction_edge):
    if int(1000*x_direction_edge[i,j])==0:
        #print "warning the dy is almost 0."
        if i<4 and j<4:
            print ("warning the dx is almost 0.")

        return thine_edge[i,j-1],thine_edge[i,j+1]
    elif int(1000*y_direction_edge[i,j])==0:
        if i<4 and j<4:
            print ("warning the dy is almost 0.")
            print (thine_edge[i-1,j],thine_edge[i+1,j])
        return thine_edge[i-1,j],thine_edge[i+1,j]

    elif int(1000*x_direction_edge[i,j]*1000*y_direction_edge[i,j])>0:
        if int(1000*abs(y_direction_edge[i,j]))>int(1000*abs(x_direction_edge[i,j])):
            x = abs(x_direction_edge[i,j])/abs(y_direction_edge[i,j])
            y = 1.0
        else:
            x = 1.0
            y = abs(y_direction_edge[i,j])/abs(x_direction_edge[i,j])
        
        neg_x = 1-x
        neg_y = 1-y
            #if i<4 and j<4:
        
        positive_direction = pure_bilinear_interpola(x     ,     y,thine_edge[i,j],thine_edge[i,j+1],thine_edge[i+1,j],thine_edge[i+1,j+1])
        if i<4 and j<4:
            print ("I am here x*y>0 positive",i,j,x     ,     y,thine_edge[i,j],thine_edge[i,j+1],thine_edge[i+1,j],thine_edge[i+1,j+1])
            print ("I am here negative",i,j,neg_x , neg_y,thine_edge[i-1,j-1],thine_edge[i-1,j],thine_edge[i,j-1],thine_edge[i,j])
        negative_direction = pure_bilinear_interpola(neg_x , neg_y,thine_edge[i-1,j-1],thine_edge[i-1,j],thine_edge[i,j-1],thine_edge[i,j])
    else:
        if int(1000*abs(y_direction_edge[i,j]))>int(1000*abs(x_direction_edge[i,j])):
            x = 1-abs(x_direction_edge[i,j])/abs(y_direction_edge[i,j])
            y = 1.0
        
        else:
            x = 0.0
            y = abs(y_direction_edge[i,j])/abs(x_direction_edge[i,j])
    

        neg_x = 1-x
        neg_y = 1-y
            
        positive_direction = pure_bilinear_interpola(x     ,     y,thine_edge[i-1,j],thine_edge[i-1,j+1],thine_edge[i,j],thine_edge[i,j+1])
        if i<4 and j<4:
            print ("I am here x*y<0 positive",i,j,x     ,     y,thine_edge[i-1,j],thine_edge[i-1,j+1],thine_edge[i,j],thine_edge[i,j+1])
            print ("I am here negative",i,j,neg_x , neg_y,thine_edge[i,j-1],thine_edge[i,j],thine_edge[i+1,j-1],thine_edge[i+1,j])
        negative_direction = pure_bilinear_interpola(neg_x , neg_y,thine_edge[i,j-1],thine_edge[i,j],thine_edge[i+1,j-1],thine_edge[i+1,j])
    if i<4 and j<4:
  
        print ("positive_direction,negative_direction",positive_direction,negative_direction)

    return positive_direction,negative_direction


def non_max(blur_edge,x_direction_edge,y_direction_edge):
    '''do the non_max compression based on the canney edge detector algo'''

    blur_edge=zero_padding(blur_edge)
    thine_edge = copy.deepcopy(blur_edge)
    
    x_direction_edge=zero_padding(x_direction_edge)
    y_direction_edge=zero_padding(y_direction_edge)
    for i in range(thine_edge.shape[0]-2):
        for j in range(thine_edge.shape[1]-2):
            positive_direction,negative_direction=bilinear_interpola(i+1,j+1,blur_edge,x_direction_edge,y_direction_edge)
            non_max_indicator=thine_edge[i+1,j+1]<=positive_direction or thine_edge[i+1,j+1]<=negative_direction
            if non_max_indicator:
                thine_edge[i+1,j+1]=0
    thine_edge=un_zero_padding(thine_edge)
    return thine_edge


def non_max_old(blur_edge, non_max_direction,non_max_wide=1):
    for i in range(non_max_wide):
        thine_edge=zero_padding(blur_edge)
    if (non_max_direction == 'x'):
        for i in range(blur_edge.shape[0]-non_max_wide*2):
           for j in range(blur_edge.shape[1]-non_max_wide*2):
               non_max_indicator = False
               for k in range(non_max_wide):
                   non_max_indicator=non_max_indicator or blur_edge[i+2,j+2]<=blur_edge[i+2+k+1,j+2] or blur_edge[i+2,j+2]<blur_edge[i+2-k-1,j+2]
               if non_max_indicator:
                   blur_edge[i+2,j+2]=0

    if (non_max_direction == 'y'):
        for i in range(blur_edge.shape[0]-non_max_wide*2):
           for j in range(blur_edge.shape[1]-non_max_wide*2):
               non_max_indicator = False
               for k in range(non_max_wide):
                   non_max_indicator=non_max_indicator or blur_edge[i+2,j+2]<=blur_edge[i+2,j+2+k+1] or blur_edge[i+2,j+2]<blur_edge[i+2,j+2-k-1]
                   #if blur_edge[i+2,j+2]<blur_edge[i+2,j] or blur_edge[i+2, j+2]<blur_edge[i+2,j+1] or blur_edge[i+2,j+2]<=blur_edge[i+1,j+3] or blur_edge[i+2, j+4]<=blur_edge[i+4,j+2] :
                   if non_max_indicator:
                       blur_edge[i+2,j+2]=0
    for i in range(non_max_wide):
        thine_edge=un_zero_padding(thine_edge)
    return thine_edge
def get_counter_info(a):
    b = a.astype(int)
    b = b.reshape(1,len(b)*len(b[0]))
    print (collections.Counter(b[0]))
def get_max_vlaue_of_2darray(a):
    return max(a.max(axis=1))
def h1(input_gray_level_image_name,sigma_value=10):
    """This function is used to resolve the programming homework h1. It is used 
    to do the edge detection based on the sobel filter."""
    input_gray_level_image = read_pgm(input_gray_level_image_name)
    plot_image(input_gray_level_image,input_gray_level_image_name, 'Input_original_image')
    
    # sobel filter is used here to detect the edge 
    #x-direction kernel
    
    
    x_direction_sobel_kernel=np.array([[-1, -2 ,-1],[0, 0 ,0],[1, 2 ,1]])
    y_direction_sobel_kernel=np.array([[-1, 0 ,1],[-2, 0 ,2],[-1, 0 ,1]])
    x_direction_edge= some_filter(input_gray_level_image, x_direction_sobel_kernel)
    y_direction_edge= some_filter(input_gray_level_image, y_direction_sobel_kernel)
    output_gray_level_edge_image = combine_x_y_edge_image(x_direction_edge,y_direction_edge)
    plot_image(x_direction_edge,input_gray_level_image_name,'X_direction_edge_image')
    plot_image(y_direction_edge,input_gray_level_image_name,'Y_direction_edge_image')
   

    print ('Output_combined_edge_image_before_non_max[0:20,0:20]',output_gray_level_edge_image[0:8,0:5])
    plot_image(output_gray_level_edge_image,input_gray_level_image_name,'Output_combined_edge_image_before_non_max_compression')
    #non_max_wide = 3
    #x_direction_edge=non_max(x_direction_edge, 'y',non_max_wide)
    #y_direction_edge=non_max(y_direction_edge, 'x',non_max_wide)
    #output_gray_level_edge_image = combine_x_y_edge_image(x_direction_edge,y_direction_edge)
   

#x_direction_edge= some_filter(output_gray_level_edge_image, x_direction_sobel_kernel)
#y_direction_edge= some_filter(output_gray_level_edge_image, y_direction_sobel_kernel)
#x_direction_edge= -2*np.ones((x_direction_edge.shape))
#   y_direction_edge= np.ones((y_direction_edge.shape))
    output_gray_level_edge_image = non_max(output_gray_level_edge_image,x_direction_edge,y_direction_edge)
    get_counter_info(output_gray_level_edge_image)
    #plot_image(output_gray_level_edge_image,input_gray_level_image_name,'Output_combined_edge_image_after_non_max_compression')
    #normaize the value to 0~255
    output_gray_level_edge_image=(255/get_max_vlaue_of_2darray(output_gray_level_edge_image))*output_gray_level_edge_image
    #print ("++++++++++++")
    #get_counter_info(output_gray_level_edge_image)

    
    #print ('Output_combined_edge_image_after_non_max[0:8,0:5]',output_gray_level_edge_image[0:8,0:5])
    plot_image(output_gray_level_edge_image,input_gray_level_image_name,'Output_combined_edge_image_after_non_max_compression')
    return output_gray_level_edge_image
    

if __name__ == '__main__':
    # Get the input gray level image based on the file name
    input_gray_level_image_name = "hough_simple_1.pgm"
    """
    This is related to the h1 of the assignment 3 of the computer vision course
    Output_gray_level_edge_image is the out put image
    The input grey level image is the input
    """
    output_gray_level_edge_image = h1(input_gray_level_image_name)
   
    
    



