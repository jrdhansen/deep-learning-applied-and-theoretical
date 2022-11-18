# Author: Jared Hansen
# Date: Thursday, 01/10/2019


#==============================================================================
#======= Deep Learning, homework 1 ============================================
#==============================================================================






#===== Problem 2.3 ============================================================

# We'll need the numpy library and the linalg sub-library
import numpy as np
from numpy import linalg as LA


class MatrixCheck:
    """
    This is a class for determining whether 2D NumPy arrays are PD matrices,
    PSD matrics, or neither.
    
    Attributes:
        matrix (numpy.matrixlib.defmatrix.matrix):
            contains the values of a given matrix.
    """
    
    def __init__(self, orig_matrix):
        """
        The constructor for the MatrixCheck class.
        
        Parameters:
            orig_matrix (numpy.matrixlib.defmatrix.matrix):
                the original 2D numpy array (matrix).
        
        Returns:
            n/a
        """
        self.matrix = orig_matrix

    def is_square(self):
        """
        Determines whether or not the matrix is square.
        
        Parameters:
            n/a
            
        Returns:
            bool: The return value. True for square matrices, False otherwise.
        """
        dims = np.shape(self.matrix)
        if(dims[0] == dims[1]):
            return True
        else:
            return False

    def evals_ge0(self):
        """
        Determines whether all eigenvalues of the matrix are >= 0.
        In order to account for values very close to 0, all eigenvalues are
        rounded to the fifth decimal place.
        
        Parameters:
            n/a
            
        Returns:
            bool: The return value. True if all eigenvalues are >= 0, False
                otherwise.
        """
        evals = np.round(LA.eigvals(self.matrix), 5)
        counter = 0
        for val in np.nditer(evals):
            if(val >= 0):
                counter +=1
        if(counter == len(evals)):
            return True
        else:
            return False
    
    def evals_gt0(self):
        """
        Determines whether all eigenvalues of the matrix are > 0.
        In order to account for values very close to 0, all eigenvalues are
        rounded to the fifth decimal place.
        
        Parameters:
            n/a
            
        Returns:
            bool: The return value. True if all eigenvalues are > 0, False
                otherwise.
        """
        evals = np.round(LA.eigvals(self.matrix), 5)
        counter = 0
        for val in np.nditer(evals):
            if(val > 0):
                counter +=1
        if(counter == len(evals)):
            return True
        else:
            return False

    def check_PD_PSD_neither(self):
        """
        Determines whether the matrix is PD, PSD, or neither. Prints a message
        to the console indicating which criteria the matrix satisfies.
        
        Parameters:
            n/a
            
        Returns:
            n/a
        """        
        if(self.is_square() and
           self.evals_gt0()):
            print("The matrix is PD, and is therefore also PSD")
        elif(self.is_square() and
             self.evals_ge0()):
            print("The matrix is PSD")
        else:
            print("The matrix is neither PSD nor PD")



def main():
    
    # Hard-code the given matrices from the prompt.
    A = np.matrix([[1, -2], [3, 4], [-5, 6]])
    B = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    C = np.matrix([[2, 2, 1], [2, 3, 2], [1, 2, 2]])
    
    # Check the matrix [A] -------------------------------------------- NEITHER
    A_matrix = MatrixCheck(A)
    A_matrix.check_PD_PSD_neither()
    print("No need to check this matrix for evals since it's not square.")
    
    # Check the matrix [A^T][A] ------------------------------- PD ==> PSD also
    AtransA_matrix = MatrixCheck(np.dot(A.T, A))
    AtransA_matrix.check_PD_PSD_neither()
    print(np.round(LA.eigvals(np.dot(A.T, A)), 5))

    
    # Check the matrix [A][A^T] ------------------------------------------- PSD
    AAtrans_matrix = MatrixCheck(np.dot(A, A.T))
    AAtrans_matrix.check_PD_PSD_neither()
    print(np.round(LA.eigvals(np.dot(A, A.T)), 5))

    
    # Check the matrix [B] ------------------------------------ PD ==> PSD also
    B_matrix = MatrixCheck(B)
    B_matrix.check_PD_PSD_neither()
    print(np.round(LA.eigvals(B), 5))

    
    # Check the matrix [-B] ------------------------------------------- NEITHER
    negB_matrix = MatrixCheck(-B)
    negB_matrix.check_PD_PSD_neither()
    print(np.round(LA.eigvals(-B), 5))

    
    # Check the matrix [C] ------------------------------------ PD ==> PSD also
    C_matrix = MatrixCheck(C)
    C_matrix.check_PD_PSD_neither()
    print(np.round(LA.eigvals(C), 5))

    
    # Check the matrix [C - 0.1*B] ---------------------------- PD ==> PSD also
    CminB_matrix = MatrixCheck(C - 0.1 * B)
    CminB_matrix.check_PD_PSD_neither()
    print(np.round(LA.eigvals(C - 0.1 * B), 5))

    
    # Check the matrix [C - 0.01 * [A][A^T]] -------------------------- NEITHER
    CminAAtrans_matrix = MatrixCheck(C - 0.01 * np.dot(A, A.T))
    CminAAtrans_matrix.check_PD_PSD_neither()
    print(np.round(LA.eigvals(C - 0.01 * np.dot(A, A.T)), 5))
    
main()