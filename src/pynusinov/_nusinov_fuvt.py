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

    def _check_types(self, lac):
        if isinstance(lac, (float, int, np.integer, list, np.ndarray)):
            if isinstance(lac, (list, np.ndarray)):
                if not all([isinstance(x, (float, int, np.integer,)) for x in lac]):
                    raise TypeError(
                        f'Only float and int types are allowed in array.')
        else:
            raise TypeError(f'Only float, int, list and np.ndarray types are allowed. f107 was {type(lac)}')
        return True

    def _get_nlam(self, lac):
        try:
            if isinstance(lac, float) or isinstance(lac, int):
                return np.array([1., lac], dtype=np.float64).reshape(1, 2)
            return np.vstack([np.array([1., x]) for x in lac], dtype=np.float64)
        except TypeError:
            raise TypeError('Only int, float or array-like object types are allowed.')

    def _predict(self, matrix_a, vector_x):
        return np.dot(matrix_a, vector_x) * 1.e15

    def get_spectral_bands(self, lac):
        if self._check_types(lac):
            nlam = self._get_nlam(lac)

        res = self._predict(self._coeffs, nlam.T)
        return xr.Dataset(data_vars={'fuv_flux_spectra': (('band_center', 'lac'), res),
                                     'lband': ('band_number', np.arange(115, 242, 1)),
                                     'uband': ('band_number', np.arange(116, 243, 1))},
                          coords={'lac': nlam[:, 1],
                                  'band_center': np.arange(115.5, 242.5, 1),
                                  'band_number': np.arange(127)},
                          attrs={'lac units': '10^15 photons · m^-2 · s^-1',
                                 'spectra units': '10^15 photons · m^-2 · s^-1',
                                 'wavelength units': 'nm',
                                 'euv_flux_spectra': 'modeled EUV solar irradiance',
                                 'lband': 'lower boundary of wavelength interval',
                                 'uband': 'upper boundary of wavelength interval'})

    def get_spectra(self, lac):
        return self.get_spectral_bands(lac)

    def predict(self, lac):
        return self.get_spectral_bands(lac)
