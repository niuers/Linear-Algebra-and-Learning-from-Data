import os
import sys
cur_dir = os.path.dirname(os.path.realpath(__file__))
if cur_dir not in sys.path:
    sys.path.append(cur_dir)

import pandas as pd
import numpy as np
import math

def outer_product(col_v, row_v):
    #Do an outer product on column vector times row vector
    m = col_v.shape[0]
    p = row_v.shape[0]
    C = np.zeros((m,p))
    for i in np.arange(m):
        for j in np.arange(p):
            C[i,j] = col_v[i]*row_v[j]
    return C
    
def columns_times_rows_outer(A, B):
    m, n1 = A.shape
    n2, p = B.shape
    if n1 != n2:
        raise ValueError("Can't multiply two matrices")
        
    n = n1
    C = np.zeros((m, p))
    for j in np.arange(n):
        sub_C = np.zeros((m,p))
        col_a = A[:, j]
        row_b = B[j, :]
        sub_C = outer_product(col_a, row_b)
        C += sub_C
    return C

def rows_times_columns(A, B):
    m, n1 = A.shape
    n2, p = B.shape
    if n1 != n2:
        raise ValueError("Can't multiply two matrices")
        
    n = n1
    C = np.zeros((m, p))
    for i in np.arange(m):
        for j in np.arange(p):
            for k in np.arange(n):
                C[i, j] += A[i,k]*B[k,j]
    return C

def columns_times_rows(A, B):
    m, n1 = A.shape
    n2, p = B.shape
    if n1 != n2:
        raise ValueError("Can't multiply two matrices")
        
    n = n1
    C = np.zeros((m, p))
    for k in np.arange(n):
        for i in np.arange(m):
            for j in np.arange(p):
                C[i,j] += A[i,k]*B[k,j]
    return C   