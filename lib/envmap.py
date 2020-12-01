'''
Filename: envmap.py
Description: My implementation of a environmental map representation.
Author: Andi Frank
E-mail: a2frank@eng.ucsd.edu
Purpose: UCSD CSE 276C - Mathematics of Robotics; Fall 2020 (with Prof. Henrik Christensen)
NOTE: Modified from costmap.py by Andi Frank (me), written for UCSD CSE 276A - Intro to Robotics; Fall 2019 (also with Prof. Henrik Christensen).
'''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import medial_axis
from skimage.draw import line
from copy import copy, deepcopy
import itertools as it
from . import myhelpers as hlp


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
    self.opts           any             - Optional other information to store in rectangle for later retrieval.


    '''
    def __init__(self, width, height, anchor=np.array([0,0]), anchor_type='origin', **kwargs):
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
#           MAP CLASS 
######################################################################################################

class Map():
    '''
    Rectangular environmental map represented as a 2D grid array. Handles translating from (x,y) coordinate space
    to (row,col) indexing of arrays. Grid elements may represent, for example, cost of being at location.

    NOTE: When querying Map, use (x,y) indexing, not (row,col) indexing like np.ndarrays.
    '''


    ########################################
    #           INITIALIZE 
    ########################################
    def __init__(self, origin, xlim, ylim, resolution=1, fill=0, lefthanded=False):
        '''
        NOTE: Map uses (x,y) indexing, not (row,col) indexing like np.ndarrays
        '''
        self.origin     = np.array(origin, dtype=float)
        self.xlim       = np.array(xlim, dtype=float)
        self.ylim       = np.array(ylim, dtype=float)
        self.res        = resolution
        self.lefthanded = lefthanded  # Marks whether coordinate system is right-handed or left-handed
        
        width  = int(np.ceil(np.diff(xlim)/self.res)) + 1
        height = int(np.ceil(np.diff(ylim)/self.res)) + 1
        
        # Grid set up so nrows=height, ncols=width
        self.grid = np.full([height, width], fill)

        # Offset gridline coordinates so Cartesian origin lies at the center of a grid square
        self.grid_offset = self.res/2     





    ######################################################
    #           INTERNAL CARTESIAN-TO-INDEX FUNCTIONS 
    ######################################################

    def _to_ind(self, coord, xy='xy', dtype=None):
        # Make sure input is an array and add singleton axes as needed
        coord = np.array(coord)
        while np.ndim(coord)<2:
            coord = coord[None]
        if   xy=='x':
            # Translate origin to [0,0] AND apply grid offset
            coord = coord - self.origin[0] #+ self.grid_offset[0]
        elif xy=='y':
            # Translate origin to [0,0] AND apply grid offset
            coord = coord - self.origin[1] #+ self.grid_offset[1]
        elif xy=='xy':
            # Translate origin to [0,0] AND apply grid offset
            coord = coord - self.origin #+ self.grid_offset
            # Switch order of coords: matrix indexes (vert,horz), not (horz,vert)
            coord = np.roll(coord, 1, axis=1)
        else:
            raise Exception('xy parameter can only accept inputs \'x\', \'y\', or \'xy\'.') 
         # Scale by resolution and cast to integers.
        ind = np.floor(coord/self.res).astype(int)
        # If in right-handed coordinates, need to invert y-axis
        if 'y' in xy and not self.lefthanded: 
            # Invert rows axis -> coord (0,0) is at bottom left, grid (0,0) is at top left
            grid_top = self.grid.shape[0]-1
            ind[:,0] = grid_top - ind[:,0]
        # Remove any added unnecessary dimensions
        ind = np.squeeze(ind).astype(int)
        # Cast to desired datatype. Or, return as list of arrays if input as list.
        if dtype is not None:
            ind = (ind).astype(dtype)
        return ind

    
    def _to_coord(self, ind, xy='xy', dtype=None):
        # Make sure input is an array and add singleton axes as needed
        ind = np.array(ind)
        while np.ndim(ind)<2:
            ind = ind[None]
        # If in right-handed coords, flip y-axis
        if 'y' in xy and not self.lefthanded: 
            # Invert rows axis -> coord (0,0) is at bottom left, grid (0,0) is at top left
            grid_top = self.grid.shape[0]-1
            ind[:,0] = grid_top - ind[:,0]
        # Scale by resolution (now as floats) 
        coord = ind.astype(float)*self.res
        # Add in origin
        if   xy=='x':
            coord = coord + self.origin[0] 
        elif xy=='y':
            coord = coord + self.origin[1] 
        elif xy=='xy':
            # Switch order of coords: matrix indexes (vert,horz), not (horz,vert)
            coord = np.roll(coord, 1, axis=1)
            coord = coord + self.origin 
        else:
            raise Exception('xy parameter can only accept inputs \'x\', \'y\', or \'xy\'.') 
        # Remove any added unnecessary dimensions
        coord = np.squeeze(coord).astype(int)
        # Cast to desired datatype. Or, return as list of arrays if input as list.
        if dtype is not None:
            coord = (coord).astype(dtype)
        return coord

    ######################################################
    #            ASSIGN VALUES TO LOCATIONS ON MAP
    ######################################################
    
    def draw_points(self, pts, value=1, units='world'):
        '''
        Assigns element value <value> to points in <pts>.
        By default, points in <pts> are assumed to be in world coordinates, not grid indices.
        '''
        if units is not 'grid':
            pts = self._to_ind(pts)
        if np.ndim(pts)==1:
            self.grid[pts[0],pts[1]] = value
        else:
            self.grid[pts[...,0],pts[...,1]] = value
        return self.grid
    
    def fill_rect(self, rect, value=1):
        '''
        Assigns element value <value> to the rectangular area described by Rect object <rect>.
        '''
        left   = self._to_ind(rect.left,   xy='x')
        right  = self._to_ind(rect.right,  xy='x')
        bottom = self._to_ind(rect.bottom, xy='y')
        top    = self._to_ind(rect.top,    xy='y')

        # Direction to iterate to go from top->bottom in rows and left->right in columns
        row_dir = hlp.sign(bottom-top)
        col_dir = hlp.sign(right-left)
        for row in range(top, bottom+row_dir, row_dir):
            for col in range(left, right+col_dir, col_dir):
                try:
                    self.grid[row,col] = value
                except IndexError:
                    pass # if out of bounds of map, just skip this assignment
                
        return self.grid

    def draw_line(self,p,q,value=1,units='world'):
        '''
        Assigns element value <value> to points along the line segment connecting points <p> and <q>.
        By default, points are assumed to be in world coordinates, not grid indices.
        '''
        if units is not 'grid':
            p,q = self._to_ind((p,q))
        r,c = line(*p,*q)
        self.grid[r,c] = value
        return self.grid

    def outline_rect(self, rect, value=1):
        '''
        Assigns element value <value> to the perimeter of rectangular area described by Rect object <rect>.
        '''
        pts = rect.corners
        for _ in range(len(pts)):
            pts = np.roll(pts,1,axis=0)
            self.draw_line(pts[0],pts[1])
        return self.grid