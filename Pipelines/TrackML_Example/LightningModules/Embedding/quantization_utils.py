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

def bin2dec(column, sign: bool):
    
    results_col = column.copy()
    for idx, x in enumerate(column):
#        print(idx, x)
        for idy, y in enumerate(x):
#            print(idx, x, idy, y)
            if (idy == 0):
                if sign:
                    result = int(y)*(-1)*2**(len(x)-1)
                else:
                    result = int(y)*2**(len(x)-1)
            else:
                result = result + int(y)*2**(len(x)-(1+idy))

        results_col[idx] = result
    return results_col
        
     

def findMinDiff(arr_in, n, threshold = 0):
    #print(arr)
    # Initialize difference as infinite
    #arr = sorted(np.absolute(arr))
    arr = arr_in.abs().sort_values(0)
    #print(arr)
    diff = 10
 
    # Find the min diff by comparing adjacent
    # pairs in sorted array
    absolute_threshold = threshold * arr[n-1]
    for i in range(n-1):
        if ((arr[i+1] - arr[i] < diff) and (arr[i+1] - arr[i] > absolute_threshold)):
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

    #print(x)

    quantizers = []
    with open('testquantization.txt','w') as f:
        f.close()
        
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
        sign = False
        if(column_data.min()<0): 
            maxbits = maxbits + 1
            sign = True
        
        print(f"{ax} {maxbits} {m} {val0} {val1} {sign}")
        with open('testquantization.txt','a') as f:
            print(f"{maxbits}, {m_inv}, {sign}", file=f)
        sum_maxbits = sum_maxbits + maxbits
        quantizers.append([maxbits, m_inv, sign])

    return quantizers


def quantize_features(features, quantizers, verbose=False, fixed_point=False, pre_point : int = 0, post_point : int = 0):

    quantized_features = pd.DataFrame(features)
#    restored_features  = pd.DataFrame(features)

    if fixed_point :
        for ax in range(features.size(dim=1)):
            column_data = pd.DataFrame(features[:,ax])[0]
            # first we apply clipping of data for max values, signed, pre_point does not include the sign bit
            column_data =np.clip(column_data,-(2**pre_point), 2**pre_point-2**(-post_point))
            # then we round, multiply each number by a 2**bits, round to integer, and divide back down
            column_data = column_data * (2**post_point)
            column_data = column_data.astype(np.int32)
            column_data = column_data.astype(np.float32)
            column_data = column_data / (2**post_point)
            quantized_features[ax] = column_data
            norm_difference = sum((quantized_features[ax] - pd.DataFrame(features[:,ax])[0]).abs())            
            if verbose:
                print(ax, fixed_point, pre_point, post_point, norm_difference)
        features = torch.from_numpy(quantized_features[:].values.astype(np.float32))
        # removed binary quantization below
    
    return features
"""   else:
           
        for ax in range(features.size(dim=1)):
            column_data = pd.DataFrame(features[:,ax])[0]
            maxbits = quantizers[ax][0]
            m_inv = quantizers[ax][1]
            sign = quantizers[ax][2]
            if m_inv > 1:
                column_data = column_data * m_inv
            column_data = column_data.astype(np.int32)
            
            #here we add clipping for max/min values, we need information if there is a sign involved for clipping (int or uint)!
            if(sign):
                column_data =np.clip(column_data,-2**(maxbits-1), 2**(maxbits-1)-1)
            else:
                column_data =np.clip(column_data,0, 2**(maxbits)-1)
                
            quantized_features[ax] = (dec2bin(column_data, maxbits, left_msb=False))#.reshape((-1,1)).flatten()
    #        print(quantized_features[ax])
            restored_features[ax] = bin2dec(quantized_features[ax], sign)
    #        print(restored_features[ax], sign)
            norm_difference = sum((restored_features[ax]*1.0/m_inv - pd.DataFrame(features[:,ax])[0]).abs())
            if verbose:
                print(ax, maxbits, m_inv, norm_difference)

        for column in quantized_features.columns:
            quantized_features[column] = quantized_features[column].apply(char_split).values
            
        quantized_features_separated = np.column_stack(quantized_features.values.T.tolist())
        #print(quantized_features_separated)
        ## below returns binarized
        features = torch.from_numpy(quantized_features_separated.astype(np.float32))
        ## below returns quantized, but non binarized
        #features = torch.tensor(restored_features[:].values.astype(np.float32))
"""