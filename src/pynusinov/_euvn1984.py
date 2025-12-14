import numpy as np
import xarray as xr
import pynusinov._misc as _m


class Euvn1984:
    '''
    1984 Nusinov EUV model class.
    '''

    def __init__(self):
        self._bands_dataset, self._lines_dataset, self._full_dataset = _m.get_euvn1984_coeffs()

    def _check_types(self, lac):
        if isinstance(lac, (float, int, np.integer, list, np.ndarray)):
            if isinstance(lac, (list, np.ndarray)):
                if not all([isinstance(x, (float, int, np.integer,)) for x in lac]):
                    raise TypeError(
                        f'Only float and int types are allowed in array.')
        else:
            raise TypeError(f'Only float, int, list and np.ndarray types are allowed. lac was {type(lac)}')
        return True

    def _get_i584_t(self, f107, t, T):
        fb = self._get_Fb(t, T)
        return self._get_hei(f107, fb)

    def _get_hei(self, f107, fb):
        return (0.725 + 0.160 * pow((fb - 60), (2/3)) + 0.0592*(f107 - fb)**(2/3))

    def _get_Fb(self, t, T):
        return 63 + 482 * np.power(np.sin(np.pi * t / T), 3.7) * np.exp(-5.2 * t / T)

    def _prepare_X(self, hei):
        if isinstance(hei, float) or isinstance(hei, int):
            return np.array([hei, hei * hei], dtype=np.float64).reshape(1, 2)
        return np.vstack([np.array([x, x * x]) for x in hei], dtype=np.float64)

    def get_spectral_bands_2(self, f107, fb):
        hei = self._prepare_X(self._get_hei(f107, fb))

        coeffs = np.vstack((np.array(self._bands_dataset['B0'], dtype=np.float64),
                            np.array(self._bands_dataset['B1'], dtype=np.float64))).T

        spectra = np.dot(coeffs, hei.T) * 1e13

        return xr.Dataset(data_vars={'euv_flux_spectra': (('band_center', 'hei'), spectra),
                                     'lband': ('band_number', self._bands_dataset['lband'].data),
                                     'uband': ('band_number', self._bands_dataset['uband'].data)},
                          coords={'hei': hei[:, 0],
                                  'band_center': self._bands_dataset['center'].data,
                                  'band_number': np.arange(19)})

    def get_spectral_lines_2(self, f107, fb):
        hei = self._prepare_X(self._get_hei(f107, fb))

        coeffs = np.vstack((np.array(self._lines_dataset['B0'], dtype=np.float64),
                            np.array(self._lines_dataset['B1'], dtype=np.float64))).T

        spectra = np.dot(coeffs, hei.T) * 1e13

        return xr.Dataset(data_vars={'euv_flux_spectra': (('line_wavelength', 'hei'), spectra),
                                     'wavelength': ('line_number', self._lines_dataset['lambda'].data)},
                          coords={'hei': hei[:, 0],
                                  'line_wavelength': self._lines_dataset['lambda'].data,
                                  'line_number': np.arange(16)})

    def get_spectra_2(self, f107, fb):
        return (self.get_spectral_bands_2(f107, fb), self.get_spectral_lines_2(f107, fb))

    def predict_2(self, f107, fb):
        hei = self._prepare_X(self._get_hei(f107, fb))

        coeffs = np.vstack((np.array(self._full_dataset['B0'], dtype=np.float64),
                            np.array(self._full_dataset['B1'], dtype=np.float64))).T

        spectra = np.dot(coeffs, hei.T) * 1e13

        return xr.Dataset(data_vars={'euv_flux_spectra': (('band_center', 'hei'), spectra),
                                     'lband': ('band_number', self._full_dataset['lband'].data),
                                     'uband': ('band_number', self._full_dataset['uband'].data)},
                          coords={'hei': hei[:, 0],
                                  'band_center': self._full_dataset['center'].data,
                                  'band_number': np.arange(35)})

    def get_spectral_bands(self, _hei):
        if self._check_types(_hei):
            hei = self._prepare_X(_hei)

        coeffs = np.vstack((np.array(self._bands_dataset['B0'], dtype=np.float64),
                            np.array(self._bands_dataset['B1'], dtype=np.float64))).T

        spectra = np.dot(coeffs, hei.T) * 1e13

        return xr.Dataset(data_vars={'euv_flux_spectra': (('band_center', 'hei'), spectra),
                                     'lband': ('band_number', self._bands_dataset['lband'].data),
                                     'uband': ('band_number', self._bands_dataset['uband'].data)},
                          coords={'hei': hei[:, 0],
                                  'band_center': self._bands_dataset['center'].data,
                                  'band_number': np.arange(19)})

    def get_spectral_lines(self, _hei):
        if self._check_types(_hei):
            hei = self._prepare_X(_hei)

        coeffs = np.vstack((np.array(self._lines_dataset['B0'], dtype=np.float64),
                            np.array(self._lines_dataset['B1'], dtype=np.float64))).T

        spectra = np.dot(coeffs, hei.T) * 1e13

        return xr.Dataset(data_vars={'euv_flux_spectra': (('line_wavelength', 'hei'), spectra),
                                     'wavelength': ('line_number', self._lines_dataset['lambda'].data)},
                          coords={'hei': hei[:, 0],
                                  'line_wavelength': self._lines_dataset['lambda'].data,
                                  'line_number': np.arange(16)})

    def get_spectra(self, _hei):
        return (self.get_spectral_bands(_hei), self.get_spectral_lines(_hei))

    def predict(self, _hei):
        if self._check_types(_hei):
            hei = self._prepare_X(_hei)

        coeffs = np.vstack((np.array(self._full_dataset['B0'], dtype=np.float64),
                            np.array(self._full_dataset['B1'], dtype=np.float64))).T

        spectra = np.dot(coeffs, hei.T) * 1e13

        return xr.Dataset(data_vars={'euv_flux_spectra': (('band_center', 'hei'), spectra),
                                     'lband': ('band_number', self._full_dataset['lband'].data),
                                     'uband': ('band_number', self._full_dataset['uband'].data)},
                          coords={'hei': hei[:, 0],
                                  'band_center': self._full_dataset['center'].data,
                                  'band_number': np.arange(35)})
