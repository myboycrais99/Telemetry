"""
This program will calculate the geoid separation from the WGS-84 spheroid using
a table lookup and bivariant spline 2d interpolation scheme.

High resolution data found online at:
    https://www.ngs.noaa.gov/GEOID/GEOID12B/GEOID12B_CONUS.shtml


Course data found online in "TSPI Data Reduction"
http://www.wsmr.army.mil/RCCsite/Documents/265-06(Supp)_Radar%20Transponder
    %20Antenna%20Sys%20Eval%20Handbook/tspi.pdf
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline
import struct

filename = 'g2012_coarse.bin'


def read_data(filename):
    """This function reads in data from binary data files.

    The data files use the standard NGS/NOAA format:
    The files (ASCII and binary) follow the same structure of a one-line
    header followed by the data in row-major format. The one-line header
    contains four double (real*8) words followed by three long (int*4) words.

    These parameters define the geographic extent of the area:
    SLAT:  Southernmost North latitude in whole degrees.
           Use a minus sign (-) to indicate South latitudes.
    WLON:  Westernmost East longitude in whole degrees.
    DLAT:  Distance interval in latitude in whole degrees
           (point spacing in E-W direction)
    DLON:  Distance interval in longitude in whole degrees
           (point spacing in N-S direction)
    NLAT:  Number of rows
           (starts with SLAT and moves northward DLAT to next row)
    NLON:  Number of columns
           (starts with WLON and moves eastward DLON to next column)
    IKIND: Always equal to one (indicates data are real*4 and endian condition)

    The data follows after this one-line header. The first row represents the
    southernmost row of data, with the first data point being in the SW corner.
    The row is NLON values wide spaced at DLAT intervals, and then increments
    to the next row which is DLAT to the north.

    This continues until the last row where the last value represents the
    northeast corner.

    The easternmost longitude is = WLON + (NLON - 1) * DLON, while the
    northernmost latitude is = SLAT + (NLAT - 1) * DLAT.
    """
    with open(filename, 'rb') as f:
        data = f.read()

    slat, wlon, dlat, dlon = struct.unpack_from('4d', data, offset=0)
    nlat, nlon, ikind = struct.unpack_from('3i', data, offset=32)
    goeid_separation = np.asarray(struct.unpack_from(
        '{}f'.format(nlat * nlon), data, offset=44)).reshape(nlat, nlon)

    lat = slat + np.arange(0, nlat * dlat, dlat)

    if wlon > 180:
        wlon = wlon - 360
    lon = wlon + np.arange(0, nlon * dlon, dlon)

    return RectBivariateSpline(lat, lon, goeid_separation)


def determine_datafile(lat, lon):
    """Determine which datafile to use to calculate geoid separation.

    To improve performance, geoid data are broken up into nine files. The first
    eight files provide a high accuracy map over the contiguous United States
    broken up into sections. The final datafile is a course mapping of Earth
    and will be used for all LLAs not in the contiguous United States.

    Parameters
    ----------
    lat : float array
        Latitude coordinate of an object
    lon : float array
        Longitude coordinate of an object

    Return
    ------
    out : float
        The separation, in meters, between the geoid and the WGS84 ellipsoid

    Raises
    ------
    None
    """
    # lat, lon = LLA[0], LLA[1]

    file_num = None
    filename = None

    if 41 <= lat <= 58:
        file_num = 0
    elif 24 <= lat < 41:
        file_num = 4
    else:
        filename = 'g2012_coarse.bin'

    if -130 <= lon < -112:
        file_num += 1
    elif -112 <= lon < -95:
        file_num += 2
    elif -95 <= lon < -78:
        file_num += 3
    elif -78 <= lon < -60:
        file_num += 4
    else:
        filename = 'g2012_coarse.bin'

    if filename is None:
        filename = 'g2012bu{}.bin'.format(file_num)

    return filename


def calculate_separation(lat, lon):
    """Uses """
    filename = determine_datafile(lat, lon)
    interp = read_data(filename)
    return float(interp(lat, lon))


if __name__ == '__main__':
    pass

