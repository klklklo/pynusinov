import numpy as np
import xarray as xr
import pynusinov._misc as _m


class Euvt2021:
    '''
    2021 EUVT model class.
    '''

    def __init__(self):
        self._bands_coeffs, self._lines_coeffs, self._full_coeffs = _m.get_euvt2021_coeffs()

    def _check_types(self, lac):
        lac = np.array(lac).reshape(-1, )
        for l in lac:
            if not isinstance(l, (int, float, np.integer)):
                raise TypeError(f'lac must be int or float, but it was {type(l).__name__}')
        return True

    def _get_nlam(self, lac):
        if isinstance(lac, float) or isinstance(lac, int):
            return np.array([lac, lac ** 2], dtype=np.float64).reshape(1, 2)
        return np.vstack([np.array([x, x ** 2]) for x in lac], dtype=np.float64)

    def _predict(self, matrix_a, vector_x):
        return np.dot(matrix_a, vector_x) * 1.e15

    def get_spectral_lines(self, lac):
        if self._check_types(lac):
            nlam = self._get_nlam(lac)

        coeffs = np.vstack((np.array(self._lines_coeffs['B0'], dtype=np.float64),
                            np.array(self._lines_coeffs['B1'], dtype=np.float64))).T

        res = self._predict(coeffs, nlam.T)
        return xr.Dataset(data_vars={'euv_flux_spectra': (('line_wavelength', 'lac'), res),
                                     'wavelength': ('line_number', self._lines_coeffs['lambda'].data)},
                          coords={'lac': nlam[:, 0],
                                  'line_wavelength': self._lines_coeffs['lambda'].data,
                                  'line_number': np.arange(16)})

    def get_spectral_bands(self, lac):
        if self._check_types(lac):
            nlam = self._get_nlam(lac)

        coeffs = np.vstack((np.array(self._bands_coeffs['B0'], dtype=np.float64),
                            np.array(self._bands_coeffs['B1'], dtype=np.float64))).T

        res = self._predict(coeffs, nlam.T)
        return xr.Dataset(data_vars={'euv_flux_spectra': (('band_center', 'lac'), res),
                                     'lband': ('band_number', self._bands_coeffs['lband'].data),
                                     'uband': ('band_number', self._bands_coeffs['uband'].data)},
                          coords={'lac': nlam[:, 0],
                                  'band_center': self._bands_coeffs['center'].data,
                                  'band_number': np.arange(20)})

    def get_spectra(self, lac):
        return self.get_spectral_bands(lac), self.get_spectral_lines(lac)

    def predict(self, lac):
        if self._check_types(lac):
            nlam = self._get_nlam(lac)

        coeffs = np.vstack((np.array(self._full_coeffs['B0'], dtype=np.float64),
                            np.array(self._full_coeffs['B1'], dtype=np.float64))).T

        res = self._predict(coeffs, nlam.T)
        return xr.Dataset(data_vars={'euv_flux_spectra': (('band_center', 'lac'), res),
                                     'lband': ('band_number', self._full_coeffs['lband'].values),
                                     'uband': ('band_number', self._full_coeffs['uband'].values)},
                          coords={'lac': nlam[:, 0],
                                  'band_center': self._full_coeffs['center'].values,
                                  'band_number': np.arange(36)})
    