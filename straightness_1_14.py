######################
from __future__ import division
import math
import numpy
#import imageio
import numpy.linalg
from scipy.special import gamma
from scipy.ndimage.filters import gaussian_filter
import scipy.misc
import scipy.io
import skimage.transform
import tensorflow as tf
import numpy as np
from past.builtins import xrange
#from IPython.core.debugger import Pdb
#pdb = Pdb()
def downsample(x):
    with tf.name_scope('downsample'):
        x = tf.identity(x)
        return tf.add_n([x[:,::2,::2,:], x[:,1::2,::2,:],
                         x[:,::2,1::2,:], x[:,1::2,1::2,:]]) / 4.

def upsample(x):
    with tf.name_scope('upsample'):
        x = tf.identity(x)
        x = tf.concat([x, x, x, x], axis=-1)
        return tf.depth_to_space(x, 2)
#config = tf.ConfigProto()
#import tensorflow_graphics as tfg
def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)
def filter_create():
	"""Function to mimic the 'fspecial' gaussian MATLAB function
	"""
	#x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
	filter1 = [.05, .25, .4, .25, .05]
	filter2 = tf.convert_to_tensor(filter1)
	filter2_3 = tf.reshape(filter2,[1,5])
	filter22 = filter2
	#print(filter2.get_shape())
	filter_3 = tf.reshape(filter22,[5,1])
	mat_mul = tf.matmul(filter_3,filter2_3)
	
	return mat_mul
def laplacian_pyramid(image,res,batch_size):
	#print(image.get_shape())
	num_levels = 1
	image2 = tf.expand_dims(image, 0)
	window = filter_create()
	window2 = tf.expand_dims(window, -1)
	window_final = tf.expand_dims(window2, -1)
	all_levels = []
	N = 3
	sd =0.5
	#window = _tf_fspecial_gauss(N, sd)
	#print(window.get_shape())
	#image_ori = image
	img_ori =tf.reshape(tf.image.resize_images(tf.image.rgb_to_grayscale(image),[res,res]),[batch_size,res,res,1])
	#pdb.set_trace()
	for i in range(6):
		image2_filtered = tf.nn.conv2d(img_ori, window_final, strides=[1, 1, 1, 1], padding='SAME')
		####################
		pyramid = downsample(image2_filtered)
		##################### up sampling and filtering again 
		pyramid2 = upsample(pyramid)	
		image_upsamples = tf.nn.conv2d(pyramid2, window_final, strides=[1, 1, 1, 1], padding='SAME')
		lappyramid = img_ori - image_upsamples
		img_ori = pyramid
		all_levels.append(lappyramid)
	return all_levels
def DN_filters_dom(laplace,level,res,batch_size):
	sigmas = [0.0248,0.0185,0.0179,0.0191,0.0220,0.2782,0.0248,0.0185,0.0179,0.0191,0.0220,0.2782]
	filter_list = []
	DN_Filter1 = np.array([[0,0,0,0,0], [0,0,0.1011,0,0],[0,0.1493,0,0.1460,0.0072],[0,0,0.1015,0,0],[0,0,0,0,0]])
	filter_list.append(DN_Filter1)
	#print(DN_Filter1.shape)
	DN_Filter2 = np.zeros((5,5),dtype=float)
	DN_Filter2[1:4,1:4] = np.array([[0,0.0757,0],[0.1986,0,0.1846],[0,0.0837,0]])
	filter_list.append(DN_Filter2)
	DN_Filter3 = np.zeros((5,5),dtype=float)
	DN_Filter3[1:4,1:4] = np.array([[0,0.0477,0],[0.2138,0,0.2243],[0,0.0467,0]])
	filter_list.append(DN_Filter3)
	DN_Filter4 = np.zeros((5,5),dtype=float)
	DN_Filter4[1:4,1:4] = np.array([[0,0,0],[ 0.2503,0,0.2616],[0,0,0]])
	filter_list.append(DN_Filter4)
	DN_Filter5 = np.zeros((5,5),dtype=float)
	DN_Filter5[1:4,1:4] = np.array([[0,0,0],[0.2598,0,0.2552],[0,0,0]])
	filter_list.append(DN_Filter5)
	DN_Filter6 = np.zeros((5,5),dtype=float)
	DN_Filter6[1:4,1:4] = np.array([[0,0,0],[0.2215,0,0.0717],[ 0,0,0]])
	filter_list.append(DN_Filter6)
	############################
	DN_dom_list = []
	res_list = [res,int(res/2),int(res/4),int(res/8),int(res/16),int(res/32)]
	for k in range(6):
		window = tf.convert_to_tensor(filter_list[k],dtype=tf.float32)
		window2 = tf.expand_dims(window, -1)
		window_final = tf.expand_dims(window2, -1)
		A2 = tf.nn.conv2d(tf.abs(laplace[k]),window_final,strides=[1, 1, 1, 1], padding='SAME')
		denominator = sigmas[k] + A2
		DN_dom = tf.divide(laplace[k],denominator)
		#print(tf.shape(DN_dom)[1])
		#print(tf.shape(DN_dom)[2])
		#DN_dom_reshaped = tf.reshape(DN_dom, [batch_size,res_list[k]*res_list[k]])
		DN_dom_list.append(DN_dom)
	return DN_dom_list[level]
#import sys
#img = imageio.imread('lena.png').astype(numpy.float)/255.0
#img2=tf.compat.v2.convert_to_tensor(img,dtype=tf.float32)
#print(type(img2))
#laplace = laplacian_pyramid(img2,res)
#DN_dom_2 = DN_filters_dom(laplace,level,res)
#print(DN_dom_2[0].shape)
#imageio.imwrite('DN_domain.png',DN_dom_2[0][0,:,:,0])
#print(DN_dom_2[0].shape)
#imageio.imwrite('DN_domain1.png',DN_dom_2[1][0,:,:,0])
#print(DN_dom_2[0].shape)
#imageio.imwrite('DN_domain2.png',DN_dom_2[2][0,:,:,0])
#print(DN_dom_2[0].shape)
#imageio.imwrite('DN_domain3.png',DN_dom_2[3][0,:,:,0])
#imageio.imwrite('DN_domain1.png',DN_dom_2[1][:,:,0])
#imageio.imwrite('DN_domain2.png',DN_dom_2[2][:,:,0])
#imageio.imwrite('DN_domain3.png',DN_dom_2[3][:,:,0])
#imageio.imwrite('DN_domain4.png',DN_dom_2[4][:,:,0])
