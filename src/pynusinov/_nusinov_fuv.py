import numpy as np
import xarray as xr
import pynusinov._misc as _m


class NusinovFUV:
    '''
    Class of the model of the spectrum of far ultraviolet radiation of the Sun (FUV) in
    the wavelength range of 115-242 nm
    '''

    def __init__(self):
        self._dataset = _m.get_nusinov_fuv()
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

    def calc_spectra(self, Nlam):
        '''
        Model calculation method. Returns the values of radiation fluxes in all intervals
        of the spectrum of the interval 115-242 nm
        :param Nlam: single value or list of flux values
        :return: xarray Dataset [fuv_flux, lband, uband, fuv_line_width]
        '''
        nlam = self._get_nlam(Nlam)
        res = np.array(np.dot(self._coeffs, nlam.T), dtype=np.float64) * 1.e15
        return xr.Dataset(data_vars={'fuv_flux': (('band_center', 'lyman_alpha'), res),
                                     'lband' : ('lambda', np.arange(115, 242, 1)),
                                     'uband' : ('lambda', np.arange(116, 243, 1)),
                                     'fuv_line_width': ('lambda', np.ones(127))},
                          coords={'band_center': np.arange(115.5, 242.5, 1),
                                  'lyman_alpha': nlam[:, 1],
                                  'lambda': np.arange(127)})
