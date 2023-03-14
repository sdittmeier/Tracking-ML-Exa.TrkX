import numpy as np
import torch
import pandas as pd
import logging

def quantize_features(features, verbose=False, fixed_point=False, pre_point : int = 0, post_point : int = 0):

    quantized_features = pd.DataFrame(features)

    if fixed_point :
        for ax in range(features.size(dim=1)):
            column_data = pd.DataFrame(features[:,ax])[0]
            # first we apply clipping of data for max values, signed, pre_point does not include the sign bit
            column_data = np.clip(column_data,-(2**pre_point), 2**pre_point-2**(-post_point))
            # then we round, multiply each number by a 2**bits, round to integer, and divide back down
            column_data = column_data * (2**post_point)
            column_data = column_data.astype(np.int32)
            column_data = column_data.astype(np.float32)
            column_data = column_data / (2**post_point)
            quantized_features[ax] = column_data
            norm_difference = sum((quantized_features[ax] - pd.DataFrame(features[:,ax])[0]).abs())            
            if verbose:
                logging.info(ax, fixed_point, pre_point, post_point, norm_difference)
        features = torch.from_numpy(quantized_features[:].values.astype(np.float32))
    
    return features
