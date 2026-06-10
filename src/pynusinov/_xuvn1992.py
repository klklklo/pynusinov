import numpy as np
import xarray as xr
import pynusinov._misc as _m


class Xuvn1992:
    '''
    1992 XUV model class.
    '''

    def __init__(self):
        self._bands_coeffs = _m.get_xuvn1992_coeffs()

    class I082:
        @staticmethod
        def predict(f107):
            f107 = np.array(f107).reshape(-1, )
            for f in f107:
                if not isinstance(f, (int, float, np.integer)):
                    raise TypeError(f'f107 must be int or float, but it was {type(f).__name__}')

            i082 = (0.29 * np.array(f107).reshape(-1, ) - 18) * 1.e-6
            return xr.Dataset(data_vars={'i082': ('_i082', i082),
                                         'f107': ('_f107', f107)},
                              coords={'_i082': np.arange(len(i082)),
                                      '_f107': np.arange(len(f107))},
                              attrs={'Name': 'Calсulate I 0.8-2.0 values from daily F10.7 (in s.f.u.)',
                                     'I 0.8-2.0 units': 'W · m^-2',
                                     'F10.7 units': 's.f.u., (1 s.f.u. = 10^-22 · W · m^-2 · Hz^-1)'})

    @staticmethod
    def scale_si_input(input):
        return input * 1.e4

    @staticmethod
    def unscale(input):
        return input * 1.e-4

    def get_spectral_bands(self, i082, scale_si_input=False):
        i082 = np.array(i082).reshape(-1,)

        for i in i082:
            if not isinstance(i, (int, float, np.integer)):
                raise TypeError(f'i082 must be int or float, but it was {type(i).__name__}')

        if scale_si_input:
            i082 = self.scale_si_input(i082)

        d = 1.56 / self._bands_coeffs['uband'].data + 0.22

        spectra = np.repeat(self._bands_coeffs['I'].data.reshape(-1, 1), i082.size, axis=1)

        for i, f in enumerate(i082):
            x = np.power(f / 1.35, d)
            spectra[:, i] = (spectra[:, i] * x) * 1.e11

        return xr.Dataset(data_vars={'xuv_flux_spectra': (('band_center', 'i082'), spectra),
                                     'lband': ('band_number', self._bands_coeffs['lband'].data),
                                     'uband': ('band_number', self._bands_coeffs['uband'].data)},
                          coords={'i082': i082,
                                  'band_center': self._bands_coeffs['center'].data,
                                  'band_number': np.arange(13)})

    def get_spectra(self, i082, scale_si_input=False):
        return self.get_spectral_bands(i082, scale_si_input)

    def predict(self, i082, scale_si_input=False):
        return self.get_spectral_bands(i082, scale_si_input)
