import numpy as np
import xarray as xr
import pynusinov._misc as _m
from importlib_metadata import version


class Euvn1984:
    '''
    1984 Nusinov EUV model class.
    '''

    def __init__(self):
        self._bands_dataset, self._lines_dataset, self._full_dataset = _m.get_euvn1984_coeffs()

    class HeI1984:
        @staticmethod
        def predict(f107, t):
            f107 = np.array(f107).reshape(-1, )
            t = np.array(t).reshape(-1, )

            for f in f107:
                if not isinstance(f, (int, float, np.integer)):
                    raise TypeError(f'f107 must be int or float, but it was {type(f).__name__}')

            for _t in t:
                if not isinstance(_t, (int, float, np.integer)):
                    raise TypeError(f't must be int or float, but it was {type(_t).__name__}')

            fb = 63 + 482 * np.power(np.sin(np.pi * t / 10.2), 3.7) * np.exp(-5.2 * t / 10.2)
            return np.array(0.725 + 0.160 * np.power(fb - 60, 2. / 3) + 0.0592 * np.power(f107 - fb, 2. / 3)) * 1.e4 * 1e9


    @staticmethod
    def scale(si_input):
        return si_input * 1.e-13

    @staticmethod
    def unscale_model_input(scaled_input):
        return scaled_input * 1.e13

    def _check_types(self, hei):
        hei = np.array(hei).reshape(-1, )
        for h in hei:
            if not isinstance(h, (int, float, np.integer)):
                raise TypeError(f'hei must be int or float, but it was {type(h).__name__}')
        return True

    def _prepare_X(self, hei):
        if isinstance(hei, float) or isinstance(hei, int):
            return np.array([hei, hei * hei], dtype=np.float64).reshape(1, 2)
        return np.vstack([np.array([x, x * x]) for x in hei], dtype=np.float64)

    def get_spectral_bands(self, _hei, scale_si_input=False):
        if self._check_types(_hei):
            hei = self._prepare_X(_hei)

        coeffs = np.vstack((np.array(self._bands_dataset['B0'], dtype=np.float64),
                            np.array(self._bands_dataset['B1'], dtype=np.float64))).T

        if scale_si_input:
            hei = self.scale(hei)

        spectra = np.dot(coeffs, hei.T) * 1e13

        return xr.Dataset(data_vars={'euv_flux_spectra': (('band_center', 'hei'), spectra),
                                     'lband': ('band_number', self._bands_dataset['lband'].data),
                                     'uband': ('band_number', self._bands_dataset['uband'].data)},
                          coords={'hei': hei[:, 0],
                                  'band_center': self._bands_dataset['center'].data,
                                  'band_number': np.arange(19)})

    def get_spectral_lines(self, _hei, scale_si_input=False):
        if self._check_types(_hei):
            hei = self._prepare_X(_hei)

        coeffs = np.vstack((np.array(self._lines_dataset['B0'], dtype=np.float64),
                            np.array(self._lines_dataset['B1'], dtype=np.float64))).T

        if scale_si_input:
            hei = self.scale(hei)

        spectra = np.dot(coeffs, hei.T) * 1e13

        return xr.Dataset(data_vars={'euv_flux_spectra': (('line_wavelength', 'hei'), spectra),
                                     'wavelength': ('line_number', self._lines_dataset['lambda'].data)},
                          coords={'hei': hei[:, 0],
                                  'line_wavelength': self._lines_dataset['lambda'].data,
                                  'line_number': np.arange(16)})

    def get_spectra(self, _hei, scale_si_input=False):
        return self.get_spectral_bands(_hei, scale_si_input), self.get_spectral_lines(_hei, scale_si_input)

    def predict(self, _hei, scale_si_input=False):
        if self._check_types(_hei):
            hei = self._prepare_X(_hei)

        coeffs = np.vstack((np.array(self._full_dataset['B0'], dtype=np.float64),
                            np.array(self._full_dataset['B1'], dtype=np.float64))).T

        if scale_si_input:
            hei = self.scale(hei)

        spectra = np.dot(coeffs, hei.T) * 1.e13

        return xr.Dataset(data_vars={'euv_flux_spectra': (('band_center', 'hei'), spectra),
                                     'lband': ('band_number', self._full_dataset['lband'].data),
                                     'uband': ('band_number', self._full_dataset['uband'].data)},
                          coords={'hei': hei[:, 0],
                                  'band_center': self._full_dataset['center'].data,
                                  'band_number': np.arange(35)})
