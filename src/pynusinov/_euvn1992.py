import numpy as np
import xarray as xr
import pynusinov._misc as _m


class Euvn1992:
    '''
    1992 Nusinov EUV model class.
    '''

    def __init__(self):
        self._bands_dataset, self._lines_dataset, self._full_dataset = _m.get_euvn1992_coeffs()

    def _check_types(self, lac):
        if isinstance(lac, (float, int, np.integer, list, np.ndarray)):
            if isinstance(lac, (list, np.ndarray)):
                if not all([isinstance(x, (float, int, np.integer,)) for x in lac]):
                    raise TypeError(
                        f'Only float and int types are allowed in array.')
        else:
            raise TypeError(f'Only float, int, list and np.ndarray types are allowed. lac was {type(lac)}')
        return True

    def _prepare_X(self, hei):
        if isinstance(hei, float) or isinstance(hei, int):
            return np.array([hei, hei * hei], dtype=np.float64).reshape(1, 2)
        return np.vstack([np.array([x, x * x]) for x in hei], dtype=np.float64)

    def _get_i584_t(self, f107, t, T):
        fb = self._get_Fb(t, T)
        return self._get_i584(f107, fb)

    def get_i584(self, f107, fb):
        return (1.38 + 0.111 * pow((fb - 60), (2/3)) + (f107 - fb)**(2/3)) *1e13


    def _get_Fb(self, t, T):
        a = [82.1, -19.6, 1.778, 2.59, -2.33]
        b = [0, 10.55, -7.956, 3.104, -0.925]
        fb = 0
        for i in range(5):
            fb += a[i]*np.cos(2*np.pi * i*t/T) + b[i]*np.sin(2*np.pi * i*t/T)
        return fb

    def _predict(self, matrix_a, vector_x):
        return np.dot(matrix_a, vector_x)


    def get_spectral_bands(self, f107, fb):
        i584 = self._get_i584(f107, fb)


    def get_spectral_bands(self, _hei):
        if self._check_types(_hei):
            hei = self._prepare_X(_hei)

        print(hei.shape)

        coeffs = np.vstack((np.array(self._bands_dataset['B0'], dtype=np.float64),
                                  np.array(self._bands_dataset['B1'], dtype=np.float64))).T

        print(coeffs.shape)

        spectra = self._predict(coeffs, hei.T)

        return xr.Dataset(data_vars={'euv_flux_spectra': (('band_center', 'hei'), spectra),
                                     'lband': ('band_number', self._bands_dataset['lband'].data),
                                     'uband': ('band_number', self._bands_dataset['uband'].data)},
                          coords={'hei': hei[:, 0],
                                  'band_center': self._bands_dataset['center'].data,
                                  'band_number': np.arange(19)})


    def get_spectral_lines(self, _hei):
        if self._check_types(_hei):
            hei = self._get_hei(_hei)

        coeffs = np.vstack((np.array(self._lines_dataset['B0'], dtype=np.float64),
                            np.array(self._lines_dataset['B1'], dtype=np.float64))).T

        # print(coeffs[:, 1])
        print(sum(coeffs[:, 0]))
        print(sum(coeffs[:, 1]))

        spectra = self._predict(coeffs, hei.T)

        return xr.Dataset(data_vars={'euv_flux_spectra': (('line_wavelength', 'hei'), spectra),
                                     'wavelength': ('line_number', self._lines_dataset['lambda'].values)},
                          coords={'hei': hei[:, 0],
                                  'line_wavelength': self._lines_dataset['lambda'].values,
                                  'line_number': np.arange(16)})

    def get_spectra(self, _hei):
        return (self.get_spectral_bands(_hei), self.get_spectral_lines(_hei))

    def predict(self, _hei):
        if self._check_types(_hei):
            hei = self._get_hei(_hei)

        coeffs = np.vstack((np.array(self._full_dataset['B0'], dtype=np.float64),
                            np.array(self._full_dataset['B1'], dtype=np.float64))).T

        # print(coeffs[:, 1])
        print(sum(coeffs[:, 0]))
        print(sum(coeffs[:, 1]))

        spectra = self._predict(coeffs, hei.T)

        return xr.Dataset(data_vars={'euv_flux_spectra': (('band_center', 'hei'), spectra),
                                     'lband': ('band_number', self._bands_dataset['lband'].data),
                                     'uband': ('band_number', self._bands_dataset['uband'].data)},
                          coords={'hei': hei[:, 0],
                                  'band_center': self._bands_dataset['center'].data,
                                  'band_number': np.arange(19)})
