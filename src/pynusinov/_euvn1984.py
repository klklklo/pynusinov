import numpy as np
import xarray as xr
import pynusinov._misc as _m


class Euvn1984:
    '''
    1984 Nusinov EUV model class.
    '''

    def __init__(self):
        self._bands_dataset, self._lines_dataset, self._full_dataset = _m.get_euvn1984_coeffs()

    class HeI:
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
            hei = np.array(0.725 + 0.160 * np.power(fb - 60, 2 / 3) + 0.0592 * np.power(f107 - fb, 2 / 3))
            return xr.Dataset(data_vars={'hei': ('_hei', hei),
                                             'f107': ('_f107', f107),
                                             'time': ('_time', t)},
                                  coords={'_hei': np.arange(len(hei)),
                                          '_f107': np.arange(len(f107)),
                                          '_time': np.arange(len(t))},
                                  attrs={'Name': 'Calсulate He I values from daily F10.7 (in s.f.u.) and time (in years)',
                                         'He I units': '10^9 photons · cm^-2 · s^-1',
                                         'F10.7 units': 's.f.u., (1 s.f.u. = 10^-22 · W · m^-2 · Hz^-1)',
                                         'Time units': 'The time from the moment the 20 or 21 solar cycle begins (in years)'})

    def _check_types(self, lac):
        lac = np.array(lac).reshape(-1, )
        for l in lac:
            if not isinstance(l, (int, float, np.integer)):
                raise TypeError(f'lac must be int or float, but it was {type(l).__name__}')
        return True

    def _prepare_X(self, hei):
        if isinstance(hei, float) or isinstance(hei, int):
            return np.array([hei, hei * hei], dtype=np.float64).reshape(1, 2)
        return np.vstack([np.array([x, x * x]) for x in hei], dtype=np.float64)

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
        return self.get_spectral_bands(_hei), self.get_spectral_lines(_hei)

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
