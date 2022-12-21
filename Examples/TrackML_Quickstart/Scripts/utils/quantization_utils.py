import numpy as np
import brevitas.nn as qnn
import math
import torch
import pandas as pd
import sys

"""
following the example of FINN https://github.com/Xilinx/finn/blob/feature/tutorial_march21/notebooks/end2end_example/cybersecurity/1-train-mlp-with-brevitas.ipynb
taking code snippets from here https://github.com/Xilinx/finn/blob/34e910eacf1e7dfa991ea0de5bbabad872885c0b/notebooks/end2end_example/cybersecurity/dataloader_quantized.py#L80
"""

def dec2bin(
    column, number_of_bits: int, left_msb: bool = True
) -> pd.Series:
    """Convert a decimal pd.Series to binary pd.Series with numbers in their
    # base-2 equivalents.
    The output is a numpy nd array.
    # adapted from: https://stackoverflow.com/q/51471097/1520469
    Parameters
    ----------
     column: pd.Series
        Series wit all decimal numbers that will be cast to binary
     number_of_bits: str
        The desired number of bits for the binary number. If bigger than
        what is needed then those bits will be 0.
        The number_of_bits should be >= than what is needed to express the
        largest decimal input
     left_msb: bool
        Specify that the most significant digit is the leftmost element.
        If this is False, it will be the rightmost element.
    Returns
    -------
    numpy.ndarray
       Numpy array with all elements in binary representation of the input.
    """

    def my_binary_repr(number, nbits):
        return np.binary_repr(number, nbits)[::-1]

    func = my_binary_repr if left_msb else np.binary_repr

    return np.vectorize(func)(column.values, number_of_bits)
     

def findMinDiff(arr_in, n, threshold = 0):
    #print(arr)
    # Initialize difference as infinite
    #arr = sorted(np.absolute(arr))
    arr = arr_in.abs().sort_values(0)
    #print(arr)
    diff = 10
 
    # Find the min diff by comparing adjacent
    # pairs in sorted array
    for i in range(n-1):
        if ((arr[i+1] - arr[i] < diff) and (arr[i+1] - arr[i] > threshold)):
            diff = arr[i+1] - arr[i]
            val0 = arr[i]
            val1 = arr[i+1]
 
    # Return min diff
    return diff, val0, val1

# computes the maximum required bits necessary to represent each number
# from a vector of numbers
def get_max_bits(vector):
    return math.ceil(math.log2(float(vector.max()) + 1.0))

def char_split(s):
    return np.array([ch for ch in s])

def learn_quantization(trainset, threshold = 0):

    for event in trainset:
        try:
            x
        except NameError:
            x = torch.clone(event.x)
            c = torch.clone(event.cell_data)
        else:
            x = torch.cat((x,event.x))
            c = torch.cat((c,event.cell_data))

    x = torch.cat((x,c),1)
    sum_maxbits = 0

    print(x)


    for ax in range(x.size(dim=1)):
        #print(ax)
        column_data = pd.DataFrame(x[:,ax])[0]
        #print(column_data)
        m, val0, val1 = findMinDiff(column_data, len(column_data), threshold)
        m_inv = 1.0 / m
        if m_inv > 1:
            column_data = column_data * m_inv
        column_data = column_data.astype(np.int32)
        #n, k, l = findMinDiff(column_data, len(column_data))
        maxbits = get_max_bits(column_data)
        # account for sign!
        if(column_data.min()<0): 
            maxbits = maxbits + 1
        
        print(f"{ax} {maxbits} {m} {val0} {val1}")
        sum_maxbits = sum_maxbits + maxbits


def quantize_features(features):

    sum_maxbits = 0
    #sum_quant   = 0
    #print(features)
    quantized_features = pd.DataFrame(features)
    #print(quantized_features)

    for ax in range(features.size(dim=1)):
        column_data = pd.DataFrame(features[:,ax])[0]
        #print(column_data)
        #print(len(column_data))
        m, val0, val1 = findMinDiff(column_data, len(column_data))
        m_inv = 1.0 / m
        if m_inv > 1:
            column_data = column_data * m_inv
        column_data = column_data.astype(np.int32)
        n, x, y = findMinDiff(column_data, len(column_data))
        #print(f"{n} is ideally 1")
        maxbits = get_max_bits(column_data)
        # account for sign!
        if(column_data.min()<0): 
            maxbits = maxbits + 1
        with open('testquantization.txt','a') as f:
            print(f"{ax} {maxbits}", file=f)# {m} {val0} {val1} {m_inv} {get_min_positive_number(column_data)}")
        sum_maxbits = sum_maxbits + maxbits
        #print(example_data_df[ax])
        #print(column_data)
        
        quantized_features[ax] = (dec2bin(column_data, maxbits, left_msb=False))#.reshape((-1,1)).flatten()
        #print(quantized_df[ax])  
        #     
    with open('testquantization.txt','a') as f:
        print(f"{sum_maxbits}", file=f)
    print(f"{sum_maxbits}")
    #print(quantized_features)
            
    for column in quantized_features.columns:
        #print(quantized_df[column])
        quantized_features[column] = quantized_features[column].apply(char_split).values
    #    print(quantized_df[column])
        
    #print(quantized_df)

    quantized_features_separated = np.column_stack(quantized_features.values.T.tolist())
    #print(quantized_features_separated)
    features = torch.from_numpy(quantized_features_separated.astype(np.int8))
    return features
    #print(features)