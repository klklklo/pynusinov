import numpy as np
import xarray as xr
import pynusinov._misc as _m


class Euvt2021:
    '''
    EUVT model class.
    '''

    def __init__(self):
        self._bands_dataset, self._lines_dataset, self._full_dataset = _m.get_nusinov_euvt_coeffs()

        self._full_coeffs = np.vstack((np.array(self._full_dataset['B0'], dtype=np.float64),
                                        np.array(self._full_dataset['B1'], dtype=np.float64))).transpose()
        self._bands_coeffs = np.vstack((np.array(self._bands_dataset['B0'], dtype=np.float64),
                                        np.array(self._bands_dataset['B1'], dtype=np.float64))).transpose()
        self._lines_coeffs = np.vstack((np.array(self._lines_dataset['B0'], dtype=np.float64),
                                        np.array(self._lines_dataset['B1'], dtype=np.float64))).transpose()

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
                return np.array([lac, lac ** 2], dtype=np.float64).reshape(1, 2)
            return np.vstack([np.array([x, x ** 2]) for x in lac], dtype=np.float64)
        except TypeError:
            raise TypeError('Only int, float or array-like object types are allowed.')

    def _predict(self, matrix_a, vector_x):
        return np.dot(matrix_a, vector_x) * 1.e15

    def get_spectral_lines(self, lac):
        if self._check_types(lac):
            nlam = self._get_nlam(lac)

        res = self._predict(self._lines_coeffs, nlam.T)
        return xr.Dataset(data_vars={'euv_flux_spectra': (('line_wavelength', 'lac'), res),
                                     'wavelength': ('line_number', self._lines_dataset['lambda'].values)},
                          coords={'lac': nlam[:, 0],
                                  'line_wavelength': self._lines_dataset['lambda'].values,
                                  'line_number': np.arange(16)},
                          attrs={'lac units': '10^15 photons · m^-2 · s^-1',
                                 'spectra units': '10^15 photons · m^-2 · s^-1',
                                 'wavelength units': 'nm',
                                 'euv_flux_spectra': 'modeled EUV solar irradiance',
                                 'wavelength': 'the wavelength of a discrete line'})

    def get_spectral_bands(self, lac):
        if self._check_types(lac):
            nlam = self._get_nlam(lac)

        res = self._predict(self._bands_coeffs, nlam.T)
        return xr.Dataset(data_vars={'euv_flux_spectra': (('band_center', 'lac'), res),
                                     'lband': ('band_number', self._bands_dataset['lband'].values),
                                     'uband': ('band_number', self._bands_dataset['uband'].values)},
                          coords={'lac': nlam[:, 0],
                                  'band_center': self._bands_dataset['center'].values,
                                  'band_number': np.arange(20)},
                          attrs={'lac units': '10^15 photons · m^-2 · s^-1',
                                 'spectra units': '10^15 photons · m^-2 · s^-1',
                                 'wavelength units': 'nm',
                                 'euv_flux_spectra': 'modeled EUV solar irradiance',
                                 'lband': 'lower boundary of wavelength interval',
                                 'uband': 'upper boundary of wavelength interval'})

    def get_spectra(self, lac):
        return self.get_spectral_bands(lac), self.get_spectral_lines(lac)

    def predict(self, lac):
        if self._check_types(lac):
            nlam = self._get_nlam(lac)

        res = self._predict(self._full_coeffs, nlam.T)
        return xr.Dataset(data_vars={'euv_flux_spectra': (('band_center', 'lac'), res),
                                     'lband': ('band_number', self._full_dataset['lband'].values),
                                     'uband': ('band_number', self._full_dataset['uband'].values)},
                          coords={'lac': nlam[:, 0],
                                  'band_center': self._full_dataset['center'].values,
                                  'band_number': np.arange(36)})
