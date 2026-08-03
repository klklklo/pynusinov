import numpy as np
import xarray as xr
import pynusinov._misc as _m


class Fuvt2021:
    '''
    2021 FUVT model class.
    '''

    def __init__(self):
        self._bands_coeffs = _m.get_fuvt2021_coeffs()

    @staticmethod
    def scale(si_input):
        return si_input * 1.e15

    @staticmethod
    def unscale(scaled_input):
        return scaled_input * 1.e-15

    def _check_types(self, lac):
        lac = np.array(lac).reshape(-1, )
        for l in lac:
            if not isinstance(l, (int, float, np.integer)):
                raise TypeError(f'lac must be int or float, but it was {type(l).__name__}')
        return True

    def _get_nlam(self, lac):
        if isinstance(lac, float) or isinstance(lac, int):
            return np.array([1., lac], dtype=np.float64).reshape(1, 2)
        return np.vstack([np.array([1., x]) for x in lac], dtype=np.float64)

    def _predict(self, matrix_a, vector_x):
        return np.dot(matrix_a, vector_x) * 1.e15

    def get_spectral_bands(self, lac):
        if self._check_types(lac):
            nlam = self._get_nlam(lac)

        coeffs = np.vstack((np.array(self._bands_coeffs['B0'], dtype=np.float64),
                            np.array(self._bands_coeffs['B1'], dtype=np.float64))).T

        res = self._predict(coeffs, nlam.T)
        return xr.Dataset(data_vars={'fuv_flux_spectra': (('band_center', 'lac'), res),
                                     'lband': ('band_number', self._bands_coeffs['lband'].data),
                                     'uband': ('band_number', self._bands_coeffs['uband'].data)},
                          coords={'lac': nlam[:, 1],
                                  'band_center': self._bands_coeffs['center'].data,
                                  'band_number': np.arange(127)})

    def get_spectra(self, lac):
        return self.get_spectral_bands(lac)

    def predict(self, lac):
        return self.get_spectral_bands(lac)
