import numpy as np
import xarray as xr
import pynusinov._misc as _m


class Fuvt2021:
    '''
    Class of the model of the spectrum of far ultraviolet radiation of the Sun (FUV) in
    the wavelength range of 115-242 nm
    '''

    def __init__(self):
        self._dataset = _m.get_nusinov_fuvt()
        self._coeffs = np.vstack((np.array(self._dataset['B0'], dtype=np.float64), np.array(self._dataset['B1'], dtype=np.float64))).transpose()

    def _get_nlam(self, nlam):
        '''
        A method for preparing data. It creates a two-dimensional array, the first column of which is filled with ones,
        the second with the values of the fluxes in the Lyman-alpha line
        :param nlam: single value or list of flux values
        :return: numpy-array for model calculation
        '''
        if isinstance(nlam, float):
            array = np.array([1., nlam], dtype=np.float64)
            return array.reshape((1, 2))
        tmp = np.array(nlam, dtype=np.float64)
        tmp = tmp.reshape((tmp.size, 1))
        array = np.ones((tmp.size, 1), dtype=np.float64)
        return np.hstack([array, tmp])

    def get_spectra(self, lyman_alpha_composite):
        '''
        Model calculation method. Returns the values of radiation fluxes in all intervals
        of the spectrum of the interval 115-242 nm
        :param lyman_alpha_composite: single value or list of flux values
        :return: xarray Dataset [fuv_flux_spectra, lband, uband, fuv_band_width]
        '''
        nlam = self._get_nlam(lyman_alpha_composite)
        res = np.array(np.dot(self._coeffs, nlam.T), dtype=np.float64) * 1.e15
        return xr.Dataset(data_vars={'fuv_flux_spectra': (('band_center', 'lyman_alpha_composite'), res),
                                     'lband' : ('band_number', np.arange(115, 242, 1)),
                                     'uband' : ('band_number', np.arange(116, 243, 1)),
                                     'fuv_band_width': ('band_number', np.ones(127))},
                          coords={'band_center': np.arange(115.5, 242.5, 1),
                                  'lyman_alpha_composite': nlam[:, 1],
                                  'band_number': np.arange(127)})
