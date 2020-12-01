'''
Filename: myhelpers.py
Description: Module for common helper functions I don't want to have to define every file.
Author: Andi Frank
E-mail: a2frank@eng.ucsd.edu
Purpose: Convenience
Last updated: November 30, 2020
'''

##############################################################
#                       IMPORTS
##############################################################

from math import pi, sin, cos, tan, atan2, sqrt


##############################################################
#                       CUSTOM ERRORS
##############################################################

class InputError(Exception):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self,*args,**kwargs):
        Exception.__init__(self,*args,**kwargs)

    #def __init__(self, expression, message):
    #    self.expression = expression
    #    self.message = message


##############################################################
#                       MATH
##############################################################
def sign(n):
    if n > 0:
        return 1
    if n < 0:
        return -1
    else: # should only be in 0 case
        return 0

def clip(x, xmin=0, xmax=None):
    if x < xmin:
        return xmin
    if x > xmax:
        return xmax
    return x


def within_pi(theta):
    '''
    Bound angle to between [-pi, pi]
    Based off:     
    http://blog.lexique-du-net.com/index.php?post/Calculate-the-real-difference-between-two-angles-keeping-the-sign
    '''
    while theta > pi:  
        theta = theta - 2*pi
    while theta < -pi:
        theta = theta + 2*pi
    return theta


def angle_a2b(a,b):
    '''
    Shortest distance (angular) between two angles.
    '''
    return within_pi(b-a)            
    