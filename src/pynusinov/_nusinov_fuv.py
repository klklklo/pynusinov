import numpy as np
import xarray as xr
import pynusinov._misc as _m


class Fuvt2021:
    '''
    FUVT model class.
    '''

    def __init__(self):
        self._dataset = _m.get_nusinov_fuvt_coeffs()
        self._coeffs = np.vstack((np.array(self._dataset['B0'], dtype=np.float64),
                                  np.array(self._dataset['B1'], dtype=np.float64))).transpose()

    def _get_nlam(self, lac):
        '''
        A method for preparing data. It creates a two-dimensional array, the first column of which is filled with ones,
        the second with the values of the fluxes in the Lyman-alpha line.
        :param lac: single value or list of flux values.
        :return: numpy-array for model calculation.
        '''

        try:
            if isinstance(lac, float) or isinstance(lac, int):
                return np.array([1., lac], dtype=np.float64).reshape(1, 2)
            return np.vstack([np.array([1., x]) for x in lac], dtype=np.float64)
        except TypeError:
            raise TypeError('Only int, float or array-like object types are allowed.')

    def get_spectral_bands(self, lac):
        '''
        Model calculation method. Returns the values of radiation fluxes in all intervals
        of the spectrum of the interval 115-242 nm.
        :param lac: single value or list of flux values.
        :return: xarray Dataset [fuv_flux_spectra, lband, uband, fuv_band_width].
        '''

        nlam = self._get_nlam(lac)
        res = np.array(np.dot(self._coeffs, nlam.T), dtype=np.float64) * 1.e15
        return xr.Dataset(data_vars={'fuv_flux_spectra': (('band_center', 'lac'), res),
                                     'lband': ('band_number', np.arange(115, 242, 1)),
                                     'uband': ('band_number', np.arange(116, 243, 1)),
                                     'fuv_band_width': ('band_number', np.ones(127))},
                          coords={'band_center': np.arange(115.5, 242.5, 1),
                                  'lac': nlam[:, 1],
                                  'band_number': np.arange(127)})

    def get_spectra(self, lac):
        return self.get_spectral_bands(lac)
