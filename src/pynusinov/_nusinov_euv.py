import numpy as np
import xarray as xr
import pynusinov._misc as _m


class Euvt2021:
    '''
    Class of the model of the spectrum of extra ultraviolet radiation of the Sun (EUV) in
    the wavelength range of 10-105 nm
    '''
    def __init__(self):
        self._bands_dataset, self._lines_dataset = _m.get_nusinov_euvt_coeffs()
        self._bands_coeffs = np.vstack((np.array(self._bands_dataset['B0'], dtype=np.float64),
                                        np.array(self._bands_dataset['B1'], dtype=np.float64))).transpose()
        self._lines_coeffs = np.vstack((np.array(self._lines_dataset['B0'], dtype=np.float64),
                                        np.array(self._lines_dataset['B1'], dtype=np.float64))).transpose()

    def _get_nlam(self, lac):
        '''
        A method for preparing data. It creates a two-dimensional array, the first column of which is filled with ones,
        the second with the values of the fluxes in the Lyman-alpha line
        :param lac: single value or list of flux values
        :return: numpy-array for model calculation
        '''
        if isinstance(lac, float):
            return np.array([lac, lac ** 2], dtype=np.float64)[None, :]
        tmp = np.array(lac, dtype=np.float64)[:, None]
        tmp1 = np.array([x ** 2 for x in tmp], dtype=np.float64)
        return np.hstack([tmp, tmp1])

    def get_spectral_lines(self, lac):
        '''
        Model calculation method. Returns the values of radiation fluxes in all lines
        of the spectrum of the interval 10-105 nm
        :param lac: single value or list of flux values
        :return: xarray Dataset [euv_flux_spectra]
        '''
        nlam = self._get_nlam(lac)
        res = np.dot(self._lines_coeffs, nlam.T) * 1.e15
        return xr.Dataset(data_vars={'euv_flux_spectra': (('line', 'lac'), res)},
                          coords={'line': self._lines_dataset['line'].values,
                                  'lac': nlam[:, 0],
                                  })

    def get_spectral_bands(self, lac):
        '''
        Model calculation method. Returns the xarray dataset values of radiation fluxes in all intervals
        of the spectrum of the interval 10-105 nm
        :param lac: single value or list of flux values
        :return: xarray Dataset [euv_flux_spectra, lband, uband, center]
        '''
        nlam = self._get_nlam(lac)
        res = np.dot(self._bands_coeffs, nlam.T) * 1.e15
        return xr.Dataset(data_vars={'euv_flux_spectra': (('band_center', 'lac'), res),
                                     'lband': ('band_number', self._bands_dataset['start'].values),
                                     'uband': ('band_number', self._bands_dataset['stop'].values),
                                     'center': ('band_number', self._bands_dataset['center'].values)},
                          coords={'band_center': self._bands_dataset['center'].values,
                                  'lac': nlam[:, 0],
                                  'band_number': np.arange(20)})

    def get_spectra(self, lac):
        '''
        Model calculation method. Combines the get_spectra_lines() and get_spectral_bands() methods
        :param lac: single value or list of flux values
        :return: xarray Dataset [euv_flux_spectra], xarray Dataset [euv_flux_spectra, lband, uband, center]
        '''
        return self.get_spectral_bands(lac), self.get_spectral_lines(lac)