import numpy as np
import xarray as xr
import pynusinov._misc as _m


class NusinovEUV:
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

    def _get_nlam(self, nlam):
        '''
        A method for preparing data. It creates a two-dimensional array, the first column of which is filled with ones,
        the second with the values of the fluxes in the Lyman-alpha line
        :param nlam: single value or list of flux values
        :return: numpy-array for model calculation
        '''
        if isinstance(nlam, float):
            return np.array([nlam, nlam ** 2], dtype=np.float64)[None, :]
        tmp = np.array(nlam, dtype=np.float64)[:, None]
        tmp1 = np.array([x ** 2 for x in tmp], dtype=np.float64)
        return np.hstack([tmp, tmp1])

    def get_spectral_lines(self, Nlam):
        '''
        Model calculation method. Returns the values of radiation fluxes in all intervals
        of the spectrum of the interval 10-105 nm
        :param Nlam: single value or list of flux values
        :return: xarray Dataset [euv_flux, lband, uband]
        '''
        nlam = self._get_nlam(Nlam)
        res = np.dot(self._lines_coeffs, nlam.T) * 1.e15
        return xr.Dataset(data_vars={'euv_flux_spectra': (('band_center', 'lyman_alpha'), res),
                                     'line' : ('line_number', self._lines_dataset['line'].values)},
                          coords={'line_number': np.arange(16),
                                  'lyman_alpha': nlam[:, 0],
                                  })

    def get_spectral_bands(self, Nlam):
        '''
        Model calculation method. Returns the values of radiation fluxes in all intervals
        of the spectrum of the interval 10-105 nm
        :param Nlam: single value or list of flux values
        :return: xarray Dataset [euv_flux, lband, uband]
        '''
        nlam = self._get_nlam(Nlam)
        res = np.dot(self._bands_dataset, nlam.T) * 1.e15
        return xr.Dataset(data_vars={'euv_flux_spectra': (('band_center', 'lyman_alpha'), res),
                                     'lband' : ('lambda', self._bands_dataset['start'].values),
                                     'uband' : ('lambda', self._bands_dataset['stop'].values),
                                     'center' : ('band_center', self._bands_dataset['center'].values)},
                          coords={'band_center': self._bands_dataset['center'].values,
                                  'lyman_alpha': nlam[:, 0],
                                  'lambda': np.arange(36)})


n = NusinovEUV()
res = n.get_spectral_lines(3.31)

print(res)