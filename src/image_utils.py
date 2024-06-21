
import numpy as np
from math import ceil
from sklearn.linear_model import LinearRegression

# source: https://github.com/fatheral/matlab_imresize


def deriveSizeFromScale(img_shape, scale):
    output_shape = []
    for k in range(2):
        output_shape.append(int(ceil(scale[k] * img_shape[k])))
    return output_shape

def deriveScaleFromSize(img_shape_in, img_shape_out):
    scale = []
    for k in range(2):
        scale.append(1.0 * img_shape_out[k] / img_shape_in[k])
    return scale

def triangle(x):
    x = np.array(x).astype(np.float64)
    lessthanzero = np.logical_and((x>=-1),x<0)
    greaterthanzero = np.logical_and((x<=1),x>=0)
    f = np.multiply((x+1),lessthanzero) + np.multiply((1-x),greaterthanzero)
    return f

def cubic(x):
    x = np.array(x).astype(np.float64)
    absx = np.absolute(x)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    f = np.multiply(1.5*absx3 - 2.5*absx2 + 1, absx <= 1) + np.multiply(-0.5*absx3 + 2.5*absx2 - 4*absx + 2, (1 < absx) & (absx <= 2))
    return f

def contributions(in_length, out_length, scale, kernel, k_width):
    if scale < 1:
        h = lambda x: scale * kernel(scale * x)
        kernel_width = 1.0 * k_width / scale
    else:
        h = kernel
        kernel_width = k_width
    x = np.arange(1, out_length+1).astype(np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    P = int(ceil(kernel_width)) + 2
    ind = np.expand_dims(left, axis=1) + np.arange(P) - 1 # -1 because indexing from 0
    indices = ind.astype(np.int32)
    weights = h(np.expand_dims(u, axis=1) - indices - 1) # -1 because indexing from 0
    weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1))
    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]
    return weights, indices

def imresizemex(inimg, weights, indices, dim):
    in_shape = inimg.shape
    w_shape = weights.shape
    out_shape = list(in_shape)
    out_shape[dim] = w_shape[0]
    outimg = np.zeros(out_shape)
    if dim == 0:
        for i_img in range(in_shape[1]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[ind, i_img].astype(np.float64)
                outimg[i_w, i_img] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)
    elif dim == 1:
        for i_img in range(in_shape[0]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[i_img, ind].astype(np.float64)
                outimg[i_img, i_w] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)        
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg

def imresizevec(inimg, weights, indices, dim):
    wshape = weights.shape
    if dim == 0:
        weights = weights.reshape((wshape[0], wshape[2], 1, 1))
        outimg =  np.sum(weights*((inimg[indices].squeeze(axis=1)).astype(np.float64)), axis=1)
    elif dim == 1:
        weights = weights.reshape((1, wshape[0], wshape[2], 1))
        outimg =  np.sum(weights*((inimg[:, indices].squeeze(axis=2)).astype(np.float64)), axis=2)
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg

def resizeAlongDim(A, dim, weights, indices, mode="vec"):
    if mode == "org":
        out = imresizemex(A, weights, indices, dim)
    else:
        out = imresizevec(A, weights, indices, dim)
    return out

def imresize(I, scalar_scale=None, method='bicubic', output_shape=None, mode="vec"):
    if method == 'bicubic':
        kernel = cubic
    elif method == 'bilinear':
        kernel = triangle
    else:
        print ('Error: Unidentified method supplied')
        
    kernel_width = 4.0
    # Fill scale and output_size
    if scalar_scale is not None:
        scalar_scale = float(scalar_scale)
        scale = [scalar_scale, scalar_scale]
        output_size = deriveSizeFromScale(I.shape, scale)
    elif output_shape is not None:
        scale = deriveScaleFromSize(I.shape, output_shape)
        output_size = list(output_shape)
    else:
        print ('Error: scalar_scale OR output_shape should be defined!')
        return
    scale_np = np.array(scale)
    order = np.argsort(scale_np)
    
    weights, indices= [], []
    for k in range(2):
        w, ind = contributions(I.shape[k], output_size[k], scale[k], kernel, kernel_width)
        weights.append(w)
        indices.append(ind)
    B = np.copy(I) 
    flag2D = False
    if B.ndim == 2:
        B = np.expand_dims(B, axis=2)
        flag2D = True
    for k in range(2):
        dim = order[k]
        B = resizeAlongDim(B, dim, weights[dim], indices[dim], mode)
    if flag2D:
        B = np.squeeze(B, axis=2)
    return B

def convertDouble2Byte(I):
    B = np.clip(I, 0.0, 1.0)
    B = 255*B
    return np.around(B).astype(np.uint8)

def outOfGamutClipping(I):
    """ Clips out-of-gamut pixels. """
    I[I > 1] = 1  # any pixel is higher than 1, clip it to 1
    I[I < 0] = 0  # any pixel is below 0, clip it to 0
    return I


def kernelP(I):
    """ Kernel function: kernel(r, g, b) -> (r,g,b,rg,rb,gb,r^2,g^2,b^2,rgb,1)
        Ref: Hong, et al., "A study of digital camera colorimetric characterization
        based on polynomial modeling." Color Research & Application, 2001. """
    return (np.transpose(
        (I[:, 0], I[:, 1], I[:, 2], I[:, 0] * I[:, 1], I[:, 0] * I[:, 2],
        I[:, 1] * I[:, 2], I[:, 0] * I[:, 0], I[:, 1] * I[:, 1],
        I[:, 2] * I[:, 2], I[:, 0] * I[:, 1] * I[:, 2],
        np.repeat(1, np.shape(I)[0]))))


def get_mapping_func(image1, image2):
    """ Computes the polynomial mapping """
    image1 = np.reshape(image1, [-1, 3])
    image2 = np.reshape(image2, [-1, 3])
    m = LinearRegression().fit(kernelP(image1), image2)
    return m


def apply_mapping_func(image, m):
    """ Applies the polynomial mapping """
    sz = image.shape
    image = np.reshape(image, [-1, 3])
    result = m.predict(kernelP(image))
    result = np.reshape(result, [sz[0], sz[1], sz[2]])
    return result


def resize_image(im, target_size):
    """ Resizes a given image to a target size.

    Args:
        im: input ndarray image (height x width x channel).
        target_size: target size (list) in the format [target_height, target_width].

    Returns:
        results the resized image (target_height x target_width x channel).
    """

    h, w, c = im.shape
    if h != target_size[1] or w != target_size[0]:
        im = imresize.imresize(im, output_shape=(target_size[0], target_size[1]))
    if c == 1:
        im = np.expand_dims(im, axis=-1)
    return im

def aspect_ratio_imresize(im, max_output=256):
    h, w, c = im.shape
    if max(h, w) > max_output:
        ratio = max_output / max(h, w)
        im = imresize.imresize(im, scalar_scale=ratio)
        h, w, c = im.shape

    if w % (2 ** 4) == 0:
        new_size_w = w
    else:
        new_size_w = w + (2 ** 4) - w % (2 ** 4)

    if h % (2 ** 4) == 0:
        new_size_h = h
    else:
        new_size_h = h + (2 ** 4) - h % (2 ** 4)

    new_size = (new_size_h, new_size_w)
    if not ((h, w) == new_size):
        im = imresize.imresize(im, output_shape=new_size)

    return im


def im2double(im):
    """ Converts an uint image to floating-point format [0-1].

    Args:
        im: image (uint ndarray); supported input formats are: uint8 or uint16.

    Returns:
        input image in floating-point format [0-1].
    """

    if im[0].dtype == 'uint8' or im[0].dtype == 'int16':
        max_value = 255
    elif im[0].dtype == 'uint16' or im[0].dtype == 'int32':
        max_value = 65535
    return im.astype('float') / max_value