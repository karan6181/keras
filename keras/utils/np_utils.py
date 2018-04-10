"""Numpy-related utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .. import backend as K
import warnings

def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def normalize(x, axis=-1, order=2):
    """Normalizes a Numpy array.

    # Arguments
        x: Numpy array to normalize.
        axis: axis along which to normalize.
        order: Normalization order (e.g. 2 for L2 norm).

    # Returns
        A normalized copy of the array.
    """
    l2 = np.atleast_1d(np.linalg.norm(x, order, axis))
    l2[l2 == 0] = 1
    return x / np.expand_dims(l2, axis)

def mxnet_image_channels_conversion(np_data):
    """
    Transform the image tensor data from channels_last to channels_first
    if using MXNet backend

    # Arguments
        np_data: image data tensor

    # Returns
        A image data tensor with channels_first format
    """
    data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('MXNet Backend: Unknown data_format ' +
                         str(data_format))
    if K.backend() == 'mxnet':
        if data_format == 'channels_first':
            shape = np_data.shape
            if len(shape) == 5:
                if shape[1] < shape[4]:
                    warnings.warn('Image data tensor format is already '
                                  '`channels_first`', stacklevel=2)
                    return np_data
                np_data = np.transpose(np_data, (0, 4, 1, 2, 3))
            elif len(shape) == 4:
                if shape[1] < shape[3]:
                    warnings.warn('Image data tensor format is already '
                                  '`channels_first`', stacklevel=2)
                    return np_data
                np_data = np.transpose(np_data, (0, 3, 1, 2))
            elif len(shape) == 3:
                raise ValueError(
                    'Your data is either a textual data of shape '
                    '`(num_sample, step, feature)` or a grey scale image of '
                    'shape `(num_sample, height, width)`. '
                    'Case 1: If your data is time-series or a textual data'
                    '(probably you are using Conv1D), then there is no need of '
                    'channel conversion. MXNet backend internally handles '
                    'tensor dimensions. '
                    'Case 2: If your data is image(probably you are using '
                    'Conv2D), then you need to reshape the tension dimensions '
                    'as follows:'
                    '`shape = x_input.shape`'
                    '`x_input = x_input.reshape(shape[0], 1, shape[1], shape[2])`'
                    'Note: Do not use `mxnet_image_channels_conversion()` in '
                    'above cases.')
            else:
                raise ValueError('Your input dimension tensor is incorrect.')
            return np_data
        else:
            raise ValueError('MXNet Backend support `channels_first` format.'
                             'Please change the `image_data_format` in '
                             '`keras.json` to `channels_first`')
    else:
        warnings.warn('Conversion of data from channels_last to channels_first' 
                      'happens only for MXNet backend. Does not support other '
                      'backends. Hence returned original tensor.', stacklevel=2)
        return np_data
