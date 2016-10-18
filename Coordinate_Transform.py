"""
This script contains several functions for coordinate transformation in 3D
space for telemetry systems.
"""

from __future__ import division, print_function
import numpy as np


def lla2ecef(lat, lon, alt, ellipse='WGS84'):
    """Convert latitude, longitude, and altitude to earth-centered, earth-fixed
    (ECEF) cartesian

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians
    lon : float
        Longitude in radians
    alt: float
        Altitude above reference ellipsoid in meters
    ellipse: str, optional
        Dictionary keyword to define which reference ellipsoid to use in
        calculations

    Returns
    -------
    out : array
        ECEF cartesian coordinates (E, F, G)

    Raises
    ------
    None
    """

    ellipses = {
        'WGS84': {'a': 6378137, 'inv_f': 298.257223563, 'unit': 'm'}
    }

    a = float(ellipses[ellipse]['a'])
    f = 1 / float(ellipses[ellipse]['inv_f'])

    b = a * (1 - f)
    e = np.sqrt(1 - (b / a) ** 2)

    # Intermediate calculation (prime vertical radius of curvature)
    n = a / np.sqrt(1 - (e * np.sin(lat) ** 2))

    x = (n + alt) * np.cos(lat) * np.cos(lon)
    y = (n + alt) * np.cos(lat) * np.sin(lon)
    z = ((1 - e) ** 2 * n + alt) * np.sin(lat)

    return np.array([x, y, z], dtype=float)

def lla2ten(lat, lon):
    """Rotation matrix for converting from North-East-Down (NED) reference
    frame to Earth-Centered Earth-Fixed frame from Latitude and Longitude.

    Parameters
    ----------
    lat : float
        Geodetic latitude in radians
    lon : float
        Longitude in radians

    Returns
    -------
    ten : array
        A 3x3 matrix representing the NED to ECEF rotation from LLA

    Raises
    ------
    None
    """

    ten = np.array([
        [-np.sin(lat) * np.cos(lon), -np.sin(lon), -np.cos(lat) * np.cos(lon)],
        [-np.sin(lat) * np.sin(lon), np.cos(lon), -np.cos(lat) * np.sin(lon)],
        [np.cos(lat), 0.0, -np.sin(lat)]
    ])

    return ten


if __name__ == '__main__':
    pass
