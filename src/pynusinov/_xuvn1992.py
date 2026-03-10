import numpy as np
import xarray as xr
import pynusinov._misc as _m


class Xuvn1992:
    '''
    1992 XUV model class.
    '''

    def __init__(self):
        self._bands_coeffs = _m.get_xuvn1992_coeffs()

    def get_i0820(self, f107):
        h = 6.62607015e-34
        c = 299792458
        l = 1.4e-9
        return (0.29 * np.array(f107).reshape(-1, ) - 18) / (h*c / l) * 1.e-17

    def get_spectral_bands(self, i082):
        i082 = np.array(i082).reshape(-1,)
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

    def get_spectra(self, i082):
        return self.get_spectral_bands(i082)

    def predict(self, i082):
        return self.get_spectral_bands(i082)
