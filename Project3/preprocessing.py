import numpy as np

def constructA(h, K):
    """
    input: h is the dimension of the matrix, K is a 1D array of length 3
    output: A band matrix
    """
    A = np.zeros((h,h))
    K = np.append(K[::-1][:-1], K)
    for i in range(h):
        for j in range(-2, 3):
            if i + j >= 0 and i + j <= h-1:
                A[i,i+j] = K[j+2]

    return A