'''
mymaps.py
Author: Andrea Frank
Date: November 30, 2020

Module for defining a rectangular environment in terms of world coordinates and other coordinates (e.g. robot-POV, configuration space). 
Created for use in UCSD CSE 276C: Mathematics of Robotics course, Fall 2020 with Henrik Christensen.
Code modified from my code for HW 4 in CSE 276A: Introduction to Robotics, Fall 2019 with Henrik Christensen.
'''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy import ndimage as ndi
from skimage.morphology import medial_axis
from skimage.draw import line
from math import sqrt
from copy import copy, deepcopy
import itertools as it


######################################################################################################
#           RECT CLASS 
######################################################################################################

class Rect():
    ''' 
    This class keeps track of the parameters of a rectangle for easy querying. 

    Attributes:

    self.width          float           - Length of the rectangle along the x-dimension in arbitrary units.
    self.height         float           - Length of the rectangle along the y-dimension in arbitrary units.
    self.anchor         2-D np.array    - The point of reference on the rectangle in the world coordinate system.
                                            By default, the anchor is set to (0,0).
    self.anchor_type    string          - What the anchor point refers to on the rectangle, can be 'origin' (default)
                                            (i.e. the lower left corner of the rectangle) or 'center'.
            Note:  The anchor is used for specifying the location of the rectangle within a larger context. Thus, a 
                    10x20 rectangle with anchor (3,2) of type "origin" has corners at [(3,2),(13,2),(13,22),(3,22)].
    self.opts           any             - Optional other information to store in rectangle for later retrieval.


    '''
    def __init__(self, width, height, anchor=np.array(0,0), anchor_type='origin', **kwargs):
        anchor = np.array(anchor)
        self.width  = width
        self.height = height
        if anchor_type  == 'origin':
            self.left   = anchor[0]
            self.bottom = anchor[1]
            self.center = anchor + np.array([self.width, self.height])/2
        elif anchor_type == 'center':
            self.center = anchor
            self.left   = self.center[0] - self.width/2
            self.bottom = self.center[1] - self.height/2
        else:
            raise Exception('anchor_type argument can only accept values'
                            '\'origin\' or \'center\' (\'origin\' by default)')
            
        self.right  = self.left + width
        self.top    = self.bottom + height
        self.corners =  np.array([ [self.left,  self.bottom], 
                                   [self.left,  self.top   ], 
                                   [self.right, self.top   ], 
                                   [self.right, self.bottom]
                            ])
        self.opts = kwargs



######################################################################################################
#           GENERAL HELPER METHODS
######################################################################################################

def array_dist2d(pt, inds):
    ''' 
    Returns the 2D Euclidean distance between two points in an array, as if the array were a grid
    where each element occupies a 1x1 square. 
    '''
    return np.sqrt( np.sum( ((pt-inds) ** 2), axis=-1) ) 

def dist2d(p,q):
    ''' 
    Returns the elementwise 2D Euclidean distance between two arrays of points in a Cartesian 
    coordinate system.

    p and q must either have the same dimensional or be broadcastable (e.g. p is 10x3 and q is 10x1).    
    '''
    p = np.array(p)
    q = np.array(q)
    while np.ndim(p)<2: p = p[None]
    while np.ndim(p)<2: p = p[None]
    coord_diffs = np.sum((p,-q),axis=-1)   
    return np.sqrt( np.sum(coord_diffs ** 2) )

def normalize(arr):
    '''
    Normalize an array by the highest magnitude value in the array.
    NOTE: Does not center values around 0.
    '''
    return arr / np.nanmax(np.abs(arr))