import numpy as np
import xarray as xr
import pynusinov._misc as _m


class Xuv1986:
    '''
    1986 XUV model class.
    '''

    def __init__(self):
        self._xuv_dataset = _m.get_xuv1986_coeffs()

    def get_spectral_bands(self, _f107):
        f107 = 0.29 * np.array(_f107).reshape(-1, ) - 18
        d = 1.56 / self._xuv_dataset['uband'].data + 0.22
        spectra = np.repeat(self._xuv_dataset['I'].data.reshape(-1, 1), f107.size, axis=1)

        for i, f in enumerate(f107):
            X = pow(f / 1.35, d)
            spectra[:, i] = (spectra[:, i] * X)

        return xr.Dataset(data_vars={'xuv_flux_spectra': (('band_center', 'f107'), spectra),
                                     'lband': ('band_number', self._xuv_dataset['lband'].data),
                                     'uband': ('band_number', self._xuv_dataset['uband'].data)},
                          coords={'f107': f107,
                                  'band_center': self._xuv_dataset['center'].data,
                                  'band_number': np.arange(13)})

    def get_spectra(self, _f107):
        return self.get_spectral_bands(_f107)

    def predict(self, _f107):
        return self.get_spectral_bands(_f107)
