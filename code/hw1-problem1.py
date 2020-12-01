#!/usr/bin/env python
# coding: utf-8

# # CSE 276C - Mathematics of Robotics
# ## HW 1 Problem 1
# ### Andrea Frank

# Python Imports
import numpy as np



###############################################################################
########################## UTILITY FUNCTIONS ##################################
###############################################################################

def swap_rows(M,row1,row2):
    '''
    Swap the rows of 2D matrix M in place.
    '''
    M[[row1,row2]] = M[[row2,row1]]
    
def find_pivot_row(X,n=0):
    '''
    Get index of row with the largest (by magnitude) element in column n.
    '''
    return np.argsort(-np.abs(X[n:,n]))[0] + n


def perm_mat(N,row1,row2):
    '''
    Return the permutation matrix that Swaps row1 and row2 for an NxN matrix.
    '''
    P = np.eye(N)
    swap_rows(P,row1,row2)
    return P



###############################################################################
################## TESTING FUNCTIONS ##########################################
###############################################################################


def sp_ldu(A):
    '''
    Perform lu decomposition from built-in scipy.lu() function, then extract
    D using my factor_out_D to get ground-truth p,l,d,u.
    Note: scipy considers the decomposition to be A = PLDU, which is why we
    use the inverse of the output p from lu().
    '''
    from scipy.linalg import lu
    inv_p,l,du = lu(A)
    p = np.linalg.inv(inv_p)
    d,u = factor_out_D(du)
    return p,l,d,u

def test_decomp(A,P,L,D,U):
    '''
    Test that P,L,D,U extracted by my functions match those calculated
    by the built-in scipy.lu() function (applied in sp_ldu()).
    '''
    p,l,d,u = sp_ldu(A)
    assert (p == P).all()
    assert (l == L).all()
    assert (d == D).all()
    assert (u == U).all()
    print("Success!")
    
def print_LDU(A,P,L,D,U):
    '''
    Print results of decomposition for debugging.
    '''
    print('A')
    print(A)
    print()
    print('P')
    print(P)
    print('DU')
    print(np.matmul(D,U))
    print('L')
    print(L)
    print('D')
    print(D)
    print('U')
    print(U)
    print()
    print('PA')
    print(np.matmul(P,A))
    print('LDU')
    print(np.matmul(L,np.matmul(D,U)))



###############################################################################
########################## MAIN ALGORITHM #####################################
###############################################################################


def permutation_step(DU,L=None,P=None,n=0):
    '''
    Permute X to place highest magnitude pivot of X in current row (n) by swapping,
    and permute L to keep row correspondence and lower triangular form. Record 
    permutation in P.
    
    PA = LDU
    this step permutation: Pn
    (Pn P)A = (Pn L Pn)(Pn DU)
    '''
    if P is None:
        P = np.eye(len(DU))
    if L is None:
        L = np.eye(len(DU))
        
    pivot = find_pivot_row(X=DU,n=n)
    if pivot != n:
        # Find permutation for this step
        Pn = perm_mat(len(DU),pivot,n)
        # Update total permutation matrix -> Pn(PA)
        P = np.matmul(Pn,P)
        DU = np.matmul(Pn,DU)
        L = np.matmul(Pn,np.matmul(L,Pn))
    
    return P,L,DU


def elimination_step(DU,L=None,n=0):
    '''
    Perform Gaussian elimination on rows below pivot row (n) such that
    the only nonzero element of the pivot column is the pivot.
    '''
    if L is None:
        L = np.eye(len(DU))

    for nn in range(n+1,len(DU)):
        if DU[n,n] != 0:
            factor = DU[nn,n]/DU[n,n]
            DU[nn] -= factor*DU[n]  # eliminate element in pivot column by subtracting 
                                            # scaled pivot row from each lower row
            L[nn,n] = factor        # Store (inverse) elementary operations in L

    return L,DU
   
def factor_out_D(DU):
    '''
    DU should be an upper triangular matrix. This function will 
    factor out the elements on the diagonal so they are set to 1.
    The diagonal elements are returned in D, the remaining upper
    triangular matrix (with 1s on the diag) in U.
    '''    
    diag = np.diag(DU)  # extract the diagonal of DU
    D = np.diag(diag)  # make into a diagonal matrix
    U = DU / diag[:,None]  # factor out D from DU
    
    return D,U


# Define a function to perform the steps of the LDU decomposition algorithm, calling the steps from above. Permuation and elimation are called for each pivot to extract P, L, and DU from the input matrix, and then the final factoring step is called to extract D and U from DU.


def LDU(A):
    # P and L begin as identity matrices
    P = np.eye(len(A)) # assuming square matrix
    L = np.eye(len(A))
    DU = A.copy() # this matrix will have L factored out until it is DU

    for n in range(len(A-1)):
        P,L,DU = permutation_step(DU=DU,L=L,P=P,n=n)
        L,DU   = elimination_step(DU=DU,L=L,n=n)
    D,U = factor_out_D(DU)
    
    return P, L, D, U




###############################################################################
################################ MAIN() #######################################
###############################################################################


if __name__ == "__main__":

    ## Matrices from hw1 problem 2

    A = np.array(
        [[4, 7, 0],
         [3, 2, 1],
         [2, 2, -6]
        ],
        dtype='double'
    )

    B = np.array(
        [[1,0,0,0,1],
         [0,0,1,0,0],
         [0,1,0,1,0],
         [0,1,0,0,0],
         [1,0,0,0,0],
        ],
        dtype='double'
    )


    C = np.array(
        [[2, 2, 5],
         [3, 2, 5],
         [1, 1, 5]
        ],
        dtype='double'
    )

    # Test decomposition on A,B,C against scipy implementation

    X = [A,B,C]
    for x in X:
        P,L,D,U = LDU(x)
        test_decomp(A=x,P=P,L=L,D=D,U=U) # Check against built-in method


