# pynusinov
<!--Basic information-->
Implementation of models of the ultraviolet radiation spectra of the Sun described by A.A. Nusinov. 
Aeronomic models of variations in the radiation flux in the region of extreme (EUV) and far (FUV) ultraviolet radiation.

If you use this package, please, cite in your research the following paper:

1. Nusinov, A.A., Kazachevskaya, T.V., Katyushina, V.V. - Solar Extreme and Far Ultraviolet Radiation Modeling for Aeronomic
Calculations. Remote Sens. 2021, 13, 1454. https://doi.org/10.3390/rs13081454

# User's guide

<!--Users guide-->

Installation

The following command is used to install the package:
```
python -m pip install pynusinov
```

pynusinov is the name of the package.

The package contains two classes: NusinovEUV and NusinovFUV.

## NusinovFUV

Implementation of the Nusinov model for calculating the spectrum of far ultraviolet radiation from the Sun (FUV)
in the wavelength range 115-242 nm. The model is based on the idea of a linear dependence of radiation fluxes in
1 nm wide intervals on the intensity in the Lyman-alpha hydrogen line (l = 121.6 nm).

Input parameters:
- flow in the Lyman-alpha line Nla (in units of 10^15 m^-2 * s^-1). You can set one or more Nla values.
Use a list to pass multiple values.

Output parameters:
- xarray dataset

```
<xarray.Dataset>
Dimensions:         (band_center: 127, lyman_alpha: 1, lambda: 127)
Coordinates:
  * band_center     (band_center) float64 115.5 116.5 117.5 ... 240.5 241.5
  * lyman_alpha     (lyman_alpha) float64 <Nla input values>
  * lambda          (lambda) int32 0 1 2 3 4 5 6 ... 120 121 122 123 124 125 126
Data variables:
    fuv_flux        (band_center, lyman_alpha) float64 <calculated spectrum>
    lband           (lambda) int32 115 116 117 118 119 ... 237 238 239 240 241
    uband           (lambda) int32 116 117 118 119 120 ... 238 239 240 241 242
    fuv_line_width  (lambda) float64 1.0 1.0 1.0 1.0 1.0 ... 1.0 1.0 1.0 1.0 1.0
```

## Usage example

- import the pynusinov package;
- create an instance of the NusinovFUV class;
- perform calculations with the created instance.

The following is an example of performing the described steps:

```
>>> # importing a package with the alias p
>>> import pynusinov as p
>>> # creating an instance of the NusinovFUV class
>>> ex = p.NusinovFUV()
>>> # calculate the spectrum values at Nla = 3.31 (10^15) using calc_spectra()
>>> spectra = ex.calc_spectra(3.31)
>>> # output the resulting FUV-spectrum
>>> print(red['fuv_flux']

<xarray.DataArray 'fuv_flux' (band_center: 127, lyman_alpha: 1)>
array([[1.0226240e+13],
       [1.3365010e+13],
       [4.3559230e+13],
...
       [4.5222314e+16],
       [5.3300029e+16]])
Coordinates:
  * band_center  (band_center) float64 115.5 116.5 117.5 ... 239.5 240.5 241.5
  * lyman_alpha  (lyman_alpha) float64 3.31
```

If you need to calculate the spectrum for several Na values, pass them using a list:

```
>>> # calculate the spectrum values at Nla1 = 3.31 (10^15) and Nla2 = 7.12 (10^15) using calc_spectra()
>>> spectra = ex.calc_spectra([3.31, 7.12])
>>> # output the resulting FUV-spectrum
>>> print(red['fuv_flux']

<xarray.DataArray 'fuv_flux' (band_center: 127, lyman_alpha: 2)> Size: 2kB
array([[1.0226240e+13, 1.7099480e+13],
       [1.3365010e+13, 1.7826520e+13],
...
       [4.5222314e+16, 4.7239328e+16],
       [5.3300029e+16, 5.5418008e+16]])
Coordinates:
  * band_center  (band_center) float64 1kB 115.5 116.5 117.5 ... 240.5 241.5
  * lyman_alpha  (lyman_alpha) float64 16B 3.31 7.12
```

## NusinovEUV

Implementation of the Nusinov model for calculating the spectrum of the extreme ultraviolet radiation of the Sun (EUV)
in the wavelength range of 10-105 nm. The model is based on the idea of a linear dependence of radiation fluxes in intervals
of unequal width on the intensity in the HeI helium line (l = 58.4 nm).

Input parameters:
- the flow in the HeI line Nl (in units of 10^15 m^-2 * s^-1)

Output parameters:
- xarray dataset

```
<xarray.Dataset> Size: 1kB
Dimensions:      (band_center: 36, lyman_alpha: 1, lambda: 36)
Coordinates:
  * band_center  (band_center) float64 288B 7.5 12.5 17.5 ... 102.6 103.2 102.5
  * lyman_alpha  (lyman_alpha) float64 8B <Nl input values>
  * lambda       (lambda) int32 144B 0 1 2 3 4 5 6 7 ... 28 29 30 31 32 33 34 35
Data variables:
    euv_flux     (band_center, lyman_alpha) float64 288B <calculated spectrum>
    lband        (lambda) float64 288B 5.0 10.0 15.0 20.0 ... 102.6 103.2 100.0
    uband        (lambda) float64 288B 10.0 15.0 20.0 25.0 ... 102.6 103.2 105.0
```

## Usage example

- import the pynusinov package;
- create an instance of the NusinovEUV class;
- perform calculations with the created instance.

The following is an example of performing the described steps:

```
>>> # importing a package with the alias p
>>> import pynusinov as p
>>> # creating an instance of the NusinovFUV class
>>> ex = p.NusinovEUV()
>>> # calculate the spectrum values at Nl = 3.31 (10^15) using calc_spectra()
>>> spectra = ex.calc_spectra(3.31)
>>> # >>> # output the resulting EUV-spectrum
>>> print(res['euv_flux'])

<xarray.DataArray 'euv_flux' (band_center: 36, lyman_alpha: 1)> Size: 288B
array([[ 2.52122700e+12],
       [ 2.59186240e+12],
...
       [ 5.22986620e+12],
       [ 9.57620734e+13]])
Coordinates:
  * band_center  (band_center) float64 288B 7.5 12.5 17.5 ... 102.6 103.2 102.5
  * lyman_alpha  (lyman_alpha) float64 8B 3.31
```

If you need to calculate the spectrum for several Na values, pass them using a list:

```
>>> # calculate the spectrum values at Nl = 3.31 (10^15) and Nla2 = 7.12 (10^15)  using calc_spectra()
>>> spectra = ex.calc_spectra([3.31, 7.12])
>>> # >>> # output the resulting EUV-spectrum
>>> print(res['euv_flux'])

<xarray.DataArray 'euv_flux' (band_center: 36, lyman_alpha: 2)> Size: 576B
array([[ 2.52122700e+12,  3.44494080e+13],
       [ 2.59186240e+12,  2.14175296e+13],
...
       [ 5.22986620e+12,  1.51018048e+13],
       [ 9.57620734e+13,  2.62794074e+14]])
Coordinates:
  * band_center  (band_center) float64 288B 7.5 12.5 17.5 ... 102.6 103.2 102.5
  * lyman_alpha  (lyman_alpha) float64 16B 3.31 7.12
  ```