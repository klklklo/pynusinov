import numpy as np
import xarray as xr
import pynusinov._misc as _m


class Euvn1992:
    '''
    1992 Nusinov EUV model class.
    '''

    def __init__(self):
        self._bands_coeffs, self._lines_coeffs, self._full_coeffs = _m.get_euvn1992_coeffs()

    def _check_types(self, hei):
        if isinstance(hei, (float, int, np.integer, list, np.ndarray)):
            if isinstance(hei, (list, np.ndarray)):
                if not all([isinstance(x, (float, int, np.integer,)) for x in hei]):
                    raise TypeError(
                        f'Only float and int types are allowed in array.')
        else:
            raise TypeError(f'Only float, int, list and np.ndarray types are allowed. hei was {type(hei)}')
        return True

    def get_Fb(self, t):
        a = [82.1, -19.6, 1.778, 2.59, -2.33]
        b = [0, 10.55, -7.956, 3.104, -0.925]
        fb = 0
        for i in range(5):
            fb += a[i] * np.cos(2 * np.pi * i * t / 10.2) + b[i] * np.sin(2 * np.pi * i * t / 10.2)
        return fb

    def get_hei(self, f107, t):
        fb = self.get_Fb(t)
        return 1.38 + 0.111 * np.power(fb - 60, 2 / 3) + 0.0583 * np.power(f107 - fb, 2 / 3)

    def _prepare_X(self, hei):
        if isinstance(hei, float) or isinstance(hei, int):
            return np.array([hei, hei * hei], dtype=np.float64).reshape(1, 2)
        return np.vstack([np.array([x, x * x]) for x in hei], dtype=np.float64)

    def get_spectral_bands(self, _hei):
        if self._check_types(_hei):
            hei = self._prepare_X(_hei)

        coeffs = np.vstack((np.array(self._bands_coeffs['B0'], dtype=np.float64),
                            np.array(self._bands_coeffs['B1'], dtype=np.float64))).T

        spectra = np.dot(coeffs, hei.T) * 1e13

        return xr.Dataset(data_vars={'euv_flux_spectra': (('band_center', 'hei'), spectra),
                                     'lband': ('band_number', self._bands_coeffs['lband'].data),
                                     'uband': ('band_number', self._bands_coeffs['uband'].data)},
                          coords={'hei': hei[:, 0],
                                  'band_center': self._bands_coeffs['center'].data,
                                  'band_number': np.arange(19)})

    def get_spectral_lines(self, _hei):
        if self._check_types(_hei):
            hei = self._prepare_X(_hei)

        coeffs = np.vstack((np.array(self._lines_coeffs['B0'], dtype=np.float64),
                            np.array(self._lines_coeffs['B1'], dtype=np.float64))).T

        spectra = np.dot(coeffs, hei.T) * 1e13

        return xr.Dataset(data_vars={'euv_flux_spectra': (('line_wavelength', 'hei'), spectra),
                                     'wavelength': ('line_number', self._lines_coeffs['lambda'].data)},
                          coords={'hei': hei[:, 0],
                                  'line_wavelength': self._lines_coeffs['lambda'].data,
                                  'line_number': np.arange(16)})

    def get_spectra(self, _hei):
        return self.get_spectral_bands(_hei), self.get_spectral_lines(_hei)

    def predict(self, _hei):
        if self._check_types(_hei):
            hei = self._prepare_X(_hei)

        coeffs = np.vstack((np.array(self._full_coeffs['B0'], dtype=np.float64),
                            np.array(self._full_coeffs['B1'], dtype=np.float64))).T

        spectra = np.dot(coeffs, hei.T) * 1e13

        return xr.Dataset(data_vars={'euv_flux_spectra': (('band_center', 'hei'), spectra),
                                     'lband': ('band_number', self._full_coeffs['lband'].data),
                                     'uband': ('band_number', self._full_coeffs['uband'].data)},
                          coords={'hei': hei[:, 0],
                                  'band_center': self._full_coeffs['center'].data,
                                  'band_number': np.arange(35)})
