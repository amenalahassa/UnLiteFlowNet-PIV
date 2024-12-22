# This files containg all signal processing functions used in the project

import numpy as np
from scipy.ndimage import gaussian_filter as gf
from skimage.restoration import denoise_wavelet

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from skimage.restoration import denoise_wavelet, denoise_nl_means, denoise_bilateral

def denoise_piv_images(inputs_data, method='gaussian', **kwargs):
    """
    Apply denoising to PIV image data using the specified method.

    Parameters:
    - inputs_data: numpy array of images, shape can be:
        (num_samples, height, width) or
        (num_samples, channels, height, width) or
        (num_samples, channels, num_images_per_sample, height, width)
    - method: string, specifying the denoising method to use ('gaussian', 'median', 'wavelet', 'non_local_means', 'bilateral', 'fft')
    - kwargs: additional parameters for the denoising function

    Returns:
    - denoised_data: numpy array of denoised images
    """
    inputs_shape = inputs_data.shape
    num_dims = inputs_data.ndim

    def apply_denoising(image, method, **kwargs):
        if method == 'gaussian':
            sigma = kwargs.get('sigma', 1)
            denoised_image = gaussian_filter(image, sigma=sigma)
        elif method == 'median':
            size = kwargs.get('size', 3)
            denoised_image = median_filter(image, size=size)
        elif method == 'wavelet':
            wavelet = kwargs.get('wavelet', 'db1')
            mode = kwargs.get('mode', 'soft')
            channel_axis = kwargs.get('channel_axis', None)
            denoised_image = denoise_wavelet(image, wavelet=wavelet, mode=mode, channel_axis=channel_axis)
        elif method == 'non_local_means':
            h = kwargs.get('h', 1.15)
            fast_mode = kwargs.get('fast_mode', True)
            patch_size = kwargs.get('patch_size', 5)
            patch_distance = kwargs.get('patch_distance', 6)
            channel_axis = kwargs.get('channel_axis', None)
            denoised_image = denoise_nl_means(image, h=h, fast_mode=fast_mode,
                                              patch_size=patch_size, patch_distance=patch_distance, channel_axis=channel_axis)
        elif method == 'bilateral':
            sigma_color = kwargs.get('sigma_color', 0.05)
            sigma_spatial = kwargs.get('sigma_spatial', 15)
            channel_axis = kwargs.get('channel_axis', None)
            denoised_image = denoise_bilateral(image, sigma_color=sigma_color, sigma_spatial=sigma_spatial, channel_axis=channel_axis)

        elif method == 'fft':
            threshold = kwargs.get('threshold', 0.1)
            fft_image = np.fft.fft2(image)
            fft_shift = np.fft.fftshift(fft_image)
            magnitude = np.abs(fft_shift)
            mask = magnitude > (threshold * np.max(magnitude))
            fft_shift_filtered = fft_shift * mask
            fft_filtered = np.fft.ifftshift(fft_shift_filtered)
            denoised_image = np.abs(np.fft.ifft2(fft_filtered))
        else:
            raise ValueError(f"Unknown denoising method: {method}")
        return denoised_image

    if num_dims == 3:
        # inputs_data shape: (num_samples, height, width)
        denoised_data = np.empty_like(inputs_data)
        num_samples = inputs_data.shape[0]
        for i in range(num_samples):
            image = inputs_data[i]
            denoised_image = apply_denoising(image, method, **kwargs)
            denoised_data[i] = denoised_image
    elif num_dims == 4:
        # inputs_data shape: (num_samples, channels, height, width)
        denoised_data = np.empty_like(inputs_data)
        num_samples, num_channels = inputs_data.shape[:2]
        for i in range(num_samples):
            for c in range(num_channels):
                image = inputs_data[i, c]
                denoised_image = apply_denoising(image, method, **kwargs)
                denoised_data[i, c] = denoised_image
    elif num_dims == 5:
        # inputs_data shape: (num_samples, channels, num_images_per_sample, height, width)
        denoised_data = np.empty_like(inputs_data)
        num_samples, num_channels, num_images = inputs_data.shape[:3]
        for i in range(num_samples):
            for c in range(num_channels):
                for j in range(num_images):
                    image = inputs_data[i, c, j]
                    denoised_image = apply_denoising(image, method, **kwargs)
                    denoised_data[i, c, j] = denoised_image
    else:
        raise ValueError(f"Unsupported input data dimensions: {num_dims}")

    return denoised_data



# Function to apply gaussian filter to the image
def gaussian_filter(image, sigma):
    """
    This function applies a gaussian filter to the image.
    It's based on the Gaussian (or normal) distribution, a bell-shaped curve, which gives the filter its name.

    Here's a breakdown of how it works:

    Gaussian Distribution: The filter uses a Gaussian function (bell curve) to assign weights to each pixel or data point in the kernel.
    The central point (the "mean") has the highest weight, and the weights decrease symmetrically as you move further from the center.

    Convolution Process: To apply the filter, you take a "kernel" (a small matrix) shaped according to the Gaussian distribution, then "convolve" it over the image or signal.
    This means you slide the kernel across each pixel or point, calculate a weighted sum based on nearby values, and replace the center value with this sum.
    The result is a smoothing effect that blurs or reduces sharp edges and noise.

    :param image:   The image to be filtered
    :param sigma:  The standard deviation of the gaussian filter
    :return:
    """
    return gf(image, sigma)


# Function to apply wavelet denoising to the image
def wavelet_denoising(image):
    """
    This function applies wavelet denoising to the image.
    Wavelet denoising is a method of reducing noise in an image by applying wavelet transforms and thresholding.

    Here's how it works:

    Wavelet Transform: The image is decomposed into wavelet coefficients using a wavelet transform.
    This transform separates the image into different frequency bands, with high-frequency bands capturing details and noise.

    Thresholding: The wavelet coefficients are thresholded to remove noise.
    Coefficients below a certain threshold are set to zero, effectively removing noise while preserving important image features.

    Inverse Transform: The denoised wavelet coefficients are then used to reconstruct the image using an inverse wavelet transform.

    :param image: The image to be denoised
    :return:
    """
    return denoise_wavelet(image, channel_axis=None)