import numpy as np
import cv2 as cv
import os

def im2col_strided(x, field_height, field_width, padding=0, stride=1):
    m, H, W, C = x.shape
    out_h = (H + 2 * padding - field_height) // stride + 1
    out_w = (W + 2 * padding - field_width) // stride + 1
    if padding > 0:
        x_padded = np.pad(x, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')
    else:
        x_padded = x
    shape = (m, out_h, out_w, field_height, field_width, C)
    strides = (x_padded.strides[0],
               stride * x_padded.strides[1],
               stride * x_padded.strides[2],
               x_padded.strides[1],
               x_padded.strides[2],
               x_padded.strides[3])
    return np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides), out_h, out_w

def fast_maxpool(x, pool_height, pool_width, stride=1, padding=0):
    x_strided, out_h, out_w = im2col_strided(x, pool_height, pool_width, padding, stride)
    out = np.max(x_strided, axis=(3, 4))
    return out, x_strided

def fast_maxpool_backprop(x, pool_height, pool_width, stride, padding, dout, x_strided):
    m, out_h, out_w, ph, pw, C = x_strided.shape

    max_val = np.max(x_strided, axis=(3, 4), keepdims=True)
    mask = (x_strided == max_val)
    mask = mask / np.sum(mask, axis=(3, 4), keepdims=True)
    dout_expanded = dout[:, :, :, None, None, :]

    dpatch = mask * dout_expanded
    m, H, W, C = x.shape
    H_padded = H + 2 * padding
    W_padded = W + 2 * padding

    dx_padded = np.zeros((m, H_padded, W_padded, C), dtype=x.dtype)

    for i in range(pool_height):
        for j in range(pool_width):
            dx_padded[:, i: i + stride * out_h: stride, j: j + stride * out_w: stride, :] += dpatch[:, :, :, i, j, :]
    
    if padding > 0:
        dx = dx_padded[:, padding:-padding, padding:-padding, :]
    else:
        dx = dx_padded
    return dx

def fast_convolution(x, W, b, padding=1, stride=1):
    m, H, W_in, C = x.shape
    fh, fw, _, K = W.shape
    x_strided, out_h, out_w = im2col_strided(x, fh, fw, padding, stride)

    out = np.einsum('mxyhwc,hwck->mxyk', x_strided, W, optimize='optimal')
    out += b
    return out.reshape(m, out_h, out_w, K)

def fast_convolution_backprop(x, W, dout, padding=1, stride=1):

    m, H, W_in, C = x.shape
    fh, fw, _, K = W.shape
    x_strided, out_h, out_w = im2col_strided(x, fh, fw, padding, stride)
    dout_reshaped = dout.reshape(m, out_h, out_w, K)
    
    dW = np.einsum('mxyhwc,mxyk->hwck', x_strided, dout_reshaped, optimize='optimal')
    
    db = np.sum(dout_reshaped, axis=(0, 1, 2), keepdims=True)
    
    dx_strided = np.einsum('mxyk,hwck->mxyhwc', dout_reshaped, W, optimize='optimal')
    dx_padded = np.zeros((m, H + 2*padding, W_in + 2*padding, C), dtype=x.dtype)
    
    for h in range(fh):
        for w in range(fw):
            dx_padded[:, 
                      h: h + stride*out_h: stride,
                      w: w + stride*out_w: stride,
                      :] += dx_strided[:, :, :, h, w, :]

    return (dx_padded[:, padding:-padding, padding:-padding, :] if padding > 0 else dx_padded), dW, db

def one_hot(data, num_classes):
    new_data = np.zeros((data.shape[0], num_classes))
    new_data[np.arange(data.shape[0]), data] = 1
    return new_data

def shuffle_data(x, y=None):
    KEYS = np.arange(x.shape[0])
    np.random.shuffle(KEYS)
    x = x[KEYS]

    if y is not None:
        y = y[KEYS]
        return x, y
    
    return x

def split_data(x, y=None, split=0.2):
    x_test, x_train = x[:int(x.shape[0]*split)], x[int(x.shape[0]*split):]

    if y is not None:
        y_test, y_train = y[:int(y.shape[0]*split)], y[int(y.shape[0]*split):]
        return x_train, y_train, x_test, y_test
    
    return x_train, x_test

def load_data(path, label_dict, image_size, split=1):

    data, labels = [], []

    for label in os.listdir(path):

        path_label = os.path.join(path, label)
        images = os.listdir(path_label)
        total_images = int(len(images)*split)
        labels.extend([label_dict[label] for _ in range(total_images)])

        for idx, image in enumerate(images, start=1):

            path_image = os.path.join(path_label, image)
            image = cv.resize(cv.cvtColor(cv.imread(path_image), cv.COLOR_BGR2GRAY), (image_size, image_size))
            data.append(image)
            print(f'Images: [{idx}/{total_images}] of {label}', end='\r')

    return data, labels