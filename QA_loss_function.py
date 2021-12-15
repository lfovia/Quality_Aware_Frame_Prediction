##### Author : ParimalaKancharla #########
############ This is the basic code for Video Generation with Normal - 3D conv generator and 3D conv discriminator #####
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import division, print_function, absolute_import

import os
from os import listdir
import urllib
import math
import numpy as np
import numpy.linalg
from scipy.special import gamma
from scipy.ndimage.filters import gaussian_filter
import scipy.misc
import scipy.io
import random
from straightness_1_14 import *
#from IPython.core.debugger import Pdb
#pdb = Pdb()
from past.builtins import xrange
import skimage.transform
#from ops import *
from glob import glob
#from ops import *
from glob import glob
import numpy as np
import tarfile
import pickle
from tensorflow.python.platform import gfile
import tensorflow as tf

import sys
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance

import tensorflow as tf
import numpy as np

import functools
def Straight_ness_double_derivatives_deriv_gen0(videos,res,batch_size):
	level = 0
	no_frames = 4
	#Double_straight_ness = DN_filters_dom(videos,level)
	#mean_Double_straight_ness = mean(Double_straight_ness)
	#derivative_list = []
	list_lgn_frames = []
	#pdb.set_trace()
	#for ii in range(4):
	img2 = videos
	img22 = tf.expand_dims(img2, 1, name=None)
        #pdb.set_trace()
	laplace = laplacian_pyramid(img2,res,batch_size)
	print(laplace)		
	################
	DN_dom_2 = DN_filters_dom(laplace,level,res,batch_size)
	list_lgn_frames.append(DN_dom_2)
        #pdb.set_trace()
	print(list_lgn_frames)
		
	return list_lgn_frames
def Straight_ness_double_derivatives_deriv_gen1(videos,res,batch_size):
	level = 1
	no_frames = 4
	#Double_straight_ness = DN_filters_dom(videos,level)
	#mean_Double_straight_ness = mean(Double_straight_ness)
	#derivative_list = []
	list_lgn_frames = []
	#pdb.set_trace()
	#for ii in range(4):
	img2 = videos
	img22 = tf.expand_dims(img2, 1, name=None)
        #pdb.set_trace()
	laplace = laplacian_pyramid(img2,res,batch_size)
	print(laplace)		
	################
	DN_dom_2 = DN_filters_dom(laplace,level,res,batch_size)
	list_lgn_frames.append(DN_dom_2)
        #pdb.set_trace()
	print(list_lgn_frames)
		
	return list_lgn_frames
def Straight_ness_double_derivatives_deriv_gen2(videos,res,batch_size):
	level = 2
	no_frames = 4
	#Double_straight_ness = DN_filters_dom(videos,level)
	#mean_Double_straight_ness = mean(Double_straight_ness)
	#derivative_list = []
	list_lgn_frames = []
	#pdb.set_trace()
	#for ii in range(4):
	img2 = videos
	img22 = tf.expand_dims(img2, 1, name=None)
        #pdb.set_trace()
	laplace = laplacian_pyramid(img2,res,batch_size)
	print(laplace)		
	################
	DN_dom_2 = DN_filters_dom(laplace,level,res,batch_size)
	list_lgn_frames.append(DN_dom_2)
        #pdb.set_trace()
	print(list_lgn_frames)
		
	return list_lgn_frames
def Straight_ness_double_derivatives_deriv_gen3(videos,res,batch_size):
	level = 4
	no_frames = 4
	#Double_straight_ness = DN_filters_dom(videos,level)
	#mean_Double_straight_ness = mean(Double_straight_ness)
	#derivative_list = []
	list_lgn_frames = []
	#pdb.set_trace()
	#for ii in range(4):
	img2 = videos
	img22 = tf.expand_dims(img2, 1, name=None)
        #pdb.set_trace()
	laplace = laplacian_pyramid(img2,res,batch_size)
	print(laplace)		
	################
	DN_dom_2 = DN_filters_dom(laplace,level,res,batch_size)
	list_lgn_frames.append(DN_dom_2)
        #pdb.set_trace()
	print(list_lgn_frames)
		
	return list_lgn_frames
def Straight_ness_double_derivatives_deriv_gen4(videos,res,batch_size):
	level = 5
	no_frames = 4
	#Double_straight_ness = DN_filters_dom(videos,level)
	#mean_Double_straight_ness = mean(Double_straight_ness)
	#derivative_list = []
	list_lgn_frames = []
	#pdb.set_trace()
	#for ii in range(4):
	img2 = videos
	img22 = tf.expand_dims(img2, 1, name=None)
        #pdb.set_trace()
	laplace = laplacian_pyramid(img2,res,batch_size)
	print(laplace)		
	################
	DN_dom_2 = DN_filters_dom(laplace,level,res,batch_size)
	list_lgn_frames.append(DN_dom_2)
        #pdb.set_trace()
	print(list_lgn_frames)
		
	return list_lgn_frames

def Straight_ness_double_derivatives_deriv_input_frames1(input_frames,res,batch_size):
	level = 1
	no_frames = 4
	#Double_straight_ness = DN_filters_dom(videos,level)
	#mean_Double_straight_ness = mean(Double_straight_ness)
	#derivative_list = []
	list_lgn_frames = []
	#pdb.set_trace()
	for ii in range(0,12,3):
		img2 = input_frames[:,:,:,ii:ii+3]
		img22 = tf.expand_dims(img2, 1, name=None)
		    #pdb.set_trace()
		laplace = laplacian_pyramid(img2,res,batch_size)
		#print(laplace)		
		################
		DN_dom_2 = DN_filters_dom(laplace,level,res,batch_size)
		list_lgn_frames.append(DN_dom_2)
		    #pdb.set_trace()
		#print(list_lgn_frames)
		
	return list_lgn_frames
def Straight_ness_double_derivatives_deriv_input_frames2(input_frames,res,batch_size):
	level = 2
	no_frames = 4
	#Double_straight_ness = DN_filters_dom(videos,level)
	#mean_Double_straight_ness = mean(Double_straight_ness)
	#derivative_list = []
	list_lgn_frames = []
	#pdb.set_trace()
	for ii in range(0,12,3):
		img2 = input_frames[:,:,:,ii:ii+3]
		img22 = tf.expand_dims(img2, 1, name=None)
		    #pdb.set_trace()
		laplace = laplacian_pyramid(img2,res,batch_size)
		#print(laplace)		
		################
		DN_dom_2 = DN_filters_dom(laplace,level,res,batch_size)
		list_lgn_frames.append(DN_dom_2)
		    #pdb.set_trace()
		#print(list_lgn_frames)
		
	return list_lgn_frames
def Straight_ness_double_derivatives_deriv_input_frames3(input_frames,res,batch_size):
	level = 3
	no_frames = 4
	#Double_straight_ness = DN_filters_dom(videos,level)
	#mean_Double_straight_ness = mean(Double_straight_ness)
	#derivative_list = []
	list_lgn_frames = []
	#pdb.set_trace()
	for ii in range(0,12,3):
		img2 = input_frames[:,:,:,ii:ii+3]
		img22 = tf.expand_dims(img2, 1, name=None)
		    #pdb.set_trace()
		laplace = laplacian_pyramid(img2,res,batch_size)
		#print(laplace)		
		################
		DN_dom_2 = DN_filters_dom(laplace,level,res,batch_size)
		list_lgn_frames.append(DN_dom_2)
		    #pdb.set_trace()
		#print(list_lgn_frames)
		
	return list_lgn_frames
def Straight_ness_double_derivatives_deriv_input_frames4(input_frames,res,batch_size):
	level = 4
	no_frames = 4
	#Double_straight_ness = DN_filters_dom(videos,level)
	#mean_Double_straight_ness = mean(Double_straight_ness)
	#derivative_list = []
	list_lgn_frames = []
	#pdb.set_trace()
	for ii in range(0,12,3):
		img2 = input_frames[:,:,:,ii:ii+3]
		img22 = tf.expand_dims(img2, 1, name=None)
		    #pdb.set_trace()
		laplace = laplacian_pyramid(img2,res,batch_size)
		#print(laplace)		
		################
		DN_dom_2 = DN_filters_dom(laplace,level,res,batch_size)
		list_lgn_frames.append(DN_dom_2)
		    #pdb.set_trace()
		#print(list_lgn_frames)
		
	return list_lgn_frames
def Straight_ness_double_derivatives_deriv_input_frames(input_frames,res,batch_size):
	level = 4
	no_frames = 4
	#Double_straight_ness = DN_filters_dom(videos,level)
	#mean_Double_straight_ness = mean(Double_straight_ness)
	#derivative_list = []
	list_lgn_frames = []
	#pdb.set_trace()
	for ii in range(0,12,3):
		img2 = input_frames[:,:,:,ii:ii+3]
		img22 = tf.expand_dims(img2, 1, name=None)
		    #pdb.set_trace()
		laplace = laplacian_pyramid(img2,res,batch_size)
		#print(laplace)		
		################
		DN_dom_2 = DN_filters_dom(laplace,level,res,batch_size)
		list_lgn_frames.append(DN_dom_2)
		    #pdb.set_trace()
		#print(list_lgn_frames)
		
	return list_lgn_frames
