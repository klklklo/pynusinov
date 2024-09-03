import functools
import xarray as xr
from importlib_resources import files


@functools.cache
def read_coeffs(file):
    return xr.open_dataset(files('pynusinov._coeffs').joinpath(file))


def get_nusinov_fuvt():
    return read_coeffs('nusinov_fuv.nc').copy()


def get_nusinov_euvt():
    return read_coeffs('euvt_spectral_bands.nc').copy(), read_coeffs('euvt_spectral_lines.nc').copy()


def convert_LaCtoLaT(LaC):
    LaC['euv_flux'] = 0.865 * LaC['euv_flux']


def convert_LaTtoLaC(LaT):
    LaT['euv_flux'] = LaT['euv_flux'] / 0.865
