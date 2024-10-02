import numpy as np
import xarray as xr
import pynusinov._misc as _m


class Euvt2021:
    '''
    Class of the model of the spectrum of extra ultraviolet radiation of the Sun (EUV) in
    the wavelength range of 10-105 nm
    '''
    def __init__(self):
        self._bands_dataset, self._lines_dataset = _m.get_nusinov_euvt()
        self._bands_coeffs = np.vstack((np.array(self._bands_dataset['B0'], dtype=np.float64),
                                        np.array(self._bands_dataset['B1'], dtype=np.float64))).transpose()
        self._lines_coeffs = np.vstack((np.array(self._lines_dataset['B0'], dtype=np.float64),
                                        np.array(self._lines_dataset['B1'], dtype=np.float64))).transpose()

    def _get_nlam(self, lyman_alpha_composite):
        '''
        A method for preparing data. It creates a two-dimensional array, the first column of which is filled with ones,
        the second with the values of the fluxes in the Lyman-alpha line
        :param lyman_alpha_composite: single value or list of flux values
        :return: numpy-array for model calculation
        '''
        if isinstance(lyman_alpha_composite, float):
            return np.array([lyman_alpha_composite, lyman_alpha_composite ** 2], dtype=np.float64)[None, :]
        tmp = np.array(lyman_alpha_composite, dtype=np.float64)[:, None]
        tmp1 = np.array([x ** 2 for x in tmp], dtype=np.float64)
        return np.hstack([tmp, tmp1])

    def get_spectral_lines(self, lyman_alpha_composite):
        '''
        Model calculation method. Returns the values of radiation fluxes in all intervals
        of the spectrum of the interval 10-105 nm
        :param lyman_alpha_composite: single value or list of flux values
        :return: xarray Dataset [euv_flux]
        '''
        nlam = self._get_nlam(lyman_alpha_composite)
        res = np.dot(self._lines_coeffs, nlam.T) * 1.e15
        return xr.Dataset(data_vars={'euv_flux_spectra': (('line', 'lyman_alpha_composite'), res)},
                          coords={'line': self._lines_dataset['line'].values,
                                  'lyman_alpha_composite': nlam[:, 0],
                                  })

    def get_spectral_bands(self, lyman_alpha_composite):
        '''
        Model calculation method. Returns the xarray dataset values of radiation fluxes in all intervals
        of the spectrum of the interval 10-105 nm
        :param lyman_alpha_composite: single value or list of flux values
        :return: xarray Dataset [euv_flux_spectra, lband, uband, center]
        '''
        nlam = self._get_nlam(lyman_alpha_composite)
        res = np.dot(self._bands_coeffs, nlam.T) * 1.e15
        return xr.Dataset(data_vars={'euv_flux_spectra': (('band_center', 'lyman_alpha_composite'), res),
                                     'lband': ('band_number', self._bands_dataset['start'].values),
                                     'uband': ('band_number', self._bands_dataset['stop'].values),
                                     'center': ('band_number', self._bands_dataset['center'].values)},
                          coords={'band_center': self._bands_dataset['center'].values,
                                  'lyman_alpha_composite': nlam[:, 0],
                                  'band_number': np.arange(20)})

    def get_spectra(self, lyman_alpha_composite):
        '''
        Model calculation method. Combines the get_spectra_lines() and get_spectral_bands() methods
        :param lyman_alpha_composite: single value or list of flux values
        :return: xarray Dataset [euv_flux, lband, uband], xarray Dataset [euv_flux_spectra, lband, uband, center],
        '''
        return self.get_spectral_lines(lyman_alpha_composite), self.get_spectral_bands(lyman_alpha_composite)