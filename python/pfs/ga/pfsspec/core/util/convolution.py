# This is a collection of old functions, now superseeded with PSF plugins.

# @staticmethod
# def convolve_vector(wave, data, kernel_func, dlambda=None, vdisp=None, wlim=None):
#     """
#     Convolve a data vector (flux) with a wavelength-dependent kernel provided as a callable.
#     This is very generic, assumes no regular binning and constant-width kernel but bins should
#     be similarly sized.
#     :param data:
#     :param kernel:
#     :param dlam:
#     :param wlim:
#     :return:
#     """

#     # TODO: if bins are not similarly sized we need to compute bin widths and take that into
#     # account

#     def get_dispersion(dlambda=None, vdisp=None):
#         if dlambda is not None and vdisp is not None:
#             raise Exception('Only one of dlambda and vdisp can be specified')
#         if dlambda is not None:
#             if isinstance(dlambda, collections.abc.Iterable):
#                 return dlambda, None
#             else:
#                 return [dlambda, dlambda], None
#         elif vdisp is not None:
#             z = Physics.vel_to_z(vdisp)
#             return None, z
#         else:
#             z = Physics.vel_to_z(Constants.DEFAULT_FILTER_VDISP)
#             return None, z

#     # Start and end of convolution
#     if wlim is not None:
#         idx_lim = np.digitize(wlim, wave)
#     else:
#         idx_lim = [0, wave.shape[0] - 1]

#     # Dispersion determines the width of the kernel to be taken into account
#     dlambda, z = get_dispersion(dlambda, vdisp)

#     # Construct results and fill-in parts that are not part of the original
#     if not isinstance(data, tuple) and not isinstance(data, list):
#         data = [data, ]
#     res = [ np.zeros(d.shape) for d in data ]
#     for d, r in zip(data, res):
#         r[:idx_lim[0]] = d[:idx_lim[0]]
#         r[idx_lim[1]:] = d[idx_lim[1]:]

#     # Compute convolution
#     for i in range(idx_lim[0], idx_lim[1]):
#         if dlambda is not None:
#             idx = np.digitize([wave[i] - dlambda[0], wave[i] + dlambda[0]], wave)
#         else:
#             idx = np.digitize([(1 - z) * wave[i], (1 + z) * wave[i]], wave)

#         # Evaluate kernel
#         k = kernel_func(wave[i], wave[idx[0]:idx[1]])

#         # Sum up
#         for d, r in zip(data, res):
#             r[i] += np.sum(k * d[idx[0]:idx[1]])

#     return res

# @staticmethod
# def convolve_vector_varying_kernel(wave, data, kernel_func, size=None, wlim=None):
#     """
#     Convolve with a kernel that varies with wavelength. Kernel is computed
#     by a function passed as a parameter.
#     """

#     # Get a test kernel from the middle of the wavelength range to have its size
#     kernel = kernel_func(wave[wave.shape[0] // 2], size=size)

#     # Start and end of convolution
#     # Outside this range, original values will be used
#     if wlim is not None:
#         idx = np.digitize(wlim, wave)
#     else:
#         idx = [0, wave.shape[0]]

#     idx_lim = [
#         max(idx[0], kernel.shape[0] // 2),
#         min(idx[1], wave.shape[0] - kernel.shape[0] // 2)
#     ]

#     # Construct results
#     if not isinstance(data, tuple) and not isinstance(data, list):
#         data = [data, ]
#     res = [np.zeros(d.shape) for d in data]

#     # Do the convolution
#     # TODO: can we optimize it further?
#     offset = 0 - kernel.shape[0] // 2
#     for i in range(idx_lim[0], idx_lim[1]):
#         kernel = kernel_func(wave[i], size=size)
#         s = slice(i + offset, i + offset + kernel.shape[0])
#         for d, r in zip(data, res):
#             z = d[i] * kernel
#             r[s] += z

#     # Fill-in parts that are not convolved
#     for d, r in zip(data, res):
#         r[:idx_lim[0] - offset] = d[:idx_lim[0] - offset]
#         r[idx_lim[1] + offset:] = d[idx_lim[1] + offset:]

#     return res

def convolve_gaussian_log(self, spec, lambda_ref=5000,  dlambda=None, vdisp=None, wlim=None):
    """
    Convolve with a Gaussian filter, assume logarithmic binning in wavelength
    so the same kernel can be used across the whole spectrum
    :param dlambda:
    :param vdisp:
    :param wlim:
    :return:
    """

    def convolve_vector(wave, data, kernel, wlim=None):
        # Start and end of convolution
        if wlim is not None:
            idx = np.digitize(wlim, wave)
            idx_lim = [max(idx[0] - kernel.shape[0] // 2, 0), min(idx[1] + kernel.shape[0] // 2, wave.shape[0] - 1)]
        else:
            idx_lim = [0, wave.shape[0] - 1]

        # Construct results
        if not isinstance(data, tuple) and not isinstance(data, list):
            data = [data, ]
        res = [np.empty(d.shape) for d in data]

        for d, r in zip(data, res):
            r[idx_lim[0]:idx_lim[1]] = np.convolve(d[idx_lim[0]:idx_lim[1]], kernel, mode='same')

        # Fill-in parts that are not convolved to save a bit of computation
        if wlim is not None:
            for d, r in zip(data, res):
                r[:idx[0]] = d[:idx[0]]
            r[idx[1]:] = d[idx[1]:]

        return res

    data = [spec.flux, ]
    if spec.cont is not None:
        data.append(spec.cont)

    if dlambda is not None:
        pass
    elif vdisp is not None:
        dlambda = lambda_ref * vdisp / Physics.c * 1000 # km/s
    else:
        raise Exception('dlambda or vdisp must be specified')

    # Make sure kernel size is always an odd number, starting from 3
    idx = np.digitize(lambda_ref, spec.wave)
    i = 1
    while lambda_ref - 4 * dlambda < spec.wave[idx - i] or \
            spec.wave[idx + i] < lambda_ref + 4 * dlambda:
        i += 1
    idx = [idx - i, idx + i]

    wave = spec.wave[idx[0]:idx[1]]
    k = 0.3989422804014327 / dlambda * np.exp(-(wave - lambda_ref) ** 2 / (2 * dlambda ** 2))
    k = k / np.sum(k)

    res = convolve_vector(spec.wave, data, k, wlim=wlim)

    spec.flux = res[0]
    if spec.cont is not None:
        spec.cont = res[1]

def convolve_varying_kernel(self, spec, kernel_func, size=None, wlim=None):
    # NOTE: this function does not take model resolution into account!

    def convolve_vectors(wave, data, kernel_func, size=None, wlim=None):
        """
        Convolve with a kernel that varies with wavelength. Kernel is computed
        by a function passed as a parameter.
        """

        # Get a test kernel from the middle of the wavelength range to have its size
        kernel = kernel_func(wave[wave.shape[0] // 2], size=size)

        # Start and end of convolution
        # Outside this range, original values will be used
        if wlim is not None:
            idx = np.digitize(wlim, wave)
        else:
            idx = [0, wave.shape[0]]

        idx_lim = [
            max(idx[0], kernel.shape[0] // 2),
            min(idx[1], wave.shape[0] - kernel.shape[0] // 2)
        ]

        # Construct results
        if not isinstance(data, tuple) and not isinstance(data, list):
            data = [data, ]
        res = [np.zeros(d.shape) for d in data]

        # Do the convolution
        # TODO: can we optimize it further?
        offset = 0 - kernel.shape[0] // 2
        for i in range(idx_lim[0], idx_lim[1]):
            kernel = kernel_func(wave[i], size=size)
            s = slice(i + offset, i + offset + kernel.shape[0])
            for d, r in zip(data, res):
                z = d[i] * kernel
                r[s] += z

        # Fill-in parts that are not convolved
        for d, r in zip(data, res):
            r[:idx_lim[0] - offset] = d[:idx_lim[0] - offset]
            r[idx_lim[1] + offset:] = d[idx_lim[1] + offset:]

        return res

    data = [spec.flux, ]
    if spec.cont is not None:
        data.append(spec.cont)

    res = convolve_vectors(spec.wave, data, kernel_func, size=size, wlim=wlim)

    spec.flux = res[0]
    if spec.cont is not None:
        spec.cont = res[1]

# TODO: Is it used with both, a Psf object and a number? Separate!
def convolve_psf(self, spec, psf, model_res, wlim):

    # NOTE: This a more generic convolution step than what's done
    #       inside the observation model which only accounts for
    #       the instrumental effects.

    # TODO: what to do with model resolution?
    # TODO: add option to do convolution pre/post resampling
    raise NotImplementedError()

    # Note, that this is in addition to model spectrum resolution
    # if isinstance(psf, Psf):
    #     spec.convolve_psf(psf, wlim)
    # elif isinstance(psf, numbers.Number) and model_res is not None:
    #     sigma = psf
    #     spec.convolve_gaussian()

    #     if model_res is not None:
    #         # Resolution at the middle of the detector
    #         fwhm = 0.5 * (wlim[0] + wlim[-1]) / model_res
    #         ss = fwhm / 2 / np.sqrt(2 * np.log(2))
    #         sigma = np.sqrt(sigma ** 2 - ss ** 2)
    #     self.convolve_gaussian(spec, dlambda=sigma, wlim=wlim)
    # elif isinstance(psf, numbers.Number):
    #     # This is a simplified version when the model resolution is not known
    #     # TODO: review this
    #     self.convolve_gaussian_log(spec, 5000, dlambda=psf)
    # else:
    #     raise NotImplementedError()