import functools
import xarray as xr
from importlib_resources import files


@functools.cache
def read_coeffs(file):
    return xr.open_dataset(files('pynusinov._coeffs').joinpath(file))


def get_xuv1986_coeffs():
    return read_coeffs('xuv1986_coeffs.nc').copy()


def get_fuvt2019_coeffs():
    return read_coeffs('fuvt2019_bands_coeffs.nc').copy()


def get_fuvt2021_coeffs():
    return read_coeffs('fuvt2021_bands_coeffs.nc').copy()


def get_euvt2021_coeffs():
    return (read_coeffs('euvt2021_bands_coeffs.nc').copy(), read_coeffs('euvt2021_lines_coeffs.nc').copy(),
            read_coeffs('euvt2021_full_coeffs.nc').copy())


def convert_lac_to_lat(lac):
    return 0.865 * lac['euv_flux_spectra']


def convert_lat_to_lac(lat):
    return lat['euv_flux_spectra'] / 0.865
