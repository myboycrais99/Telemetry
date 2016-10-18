"""
This script contains several functions for coordinate transformation in 3D
space for telemetry systems.
"""

from __future__ import division, print_function
import numpy as np

d2r = np.pi / 180
r2d = 180 / np.pi

def lla2ecef(LLA, ellipse='WGS84'):
    """Convert latitude, longitude, and altitude to earth-centered, earth-fixed
    (ECEF) cartesian

    Parameters
    ----------
    LLA : float array
        Geodetic latitude in radians, longitude in radians, and altitude above
        reference ellipsoid in meters
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
    lat, lon, alt = LLA[0] * d2r, LLA[1] * d2r, LLA[2]

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

def lla2ten(LLA):
    """Rotation matrix for converting from North-East-Down (NED) reference
    frame to Earth-Centered Earth-Fixed frame from Latitude and Longitude.

    Parameters
    ----------
    LLA : float array
        Geodetic latitude in radians, longitude in radians, and altitude above
        reference ellipsoid in meters

    Returns
    -------
    ten : array
        A 3x3 matrix representing the NED to ECEF rotation from LLA

    Raises
    ------
    None
    """
    lat, lon = LLA[0] * d2r, LLA[1] * d2r

    ten = np.array([
        [-np.sin(lat) * np.cos(lon), -np.sin(lon), -np.cos(lat) * np.cos(lon)],
        [-np.sin(lat) * np.sin(lon), np.cos(lon), -np.cos(lat) * np.sin(lon)],
        [np.cos(lat), 0.0, -np.sin(lat)]
    ])

    return ten

def relA2B_RAE(A, B):
    """

    Parameters
    ----------
    A : float array
        LLA coordinates of an object
    B : float array
        LLA coordinates of an object

    Return
    ------
    out : float array

    Raises
    ------
    None
    """

    ecef1 = lla2ecef(A)
    ecef2 = lla2ecef(B)

    diff_ecef = ecef2 - ecef1
    ten = lla2ten(A)

    relA2B_ned = np.dot(ten, diff_ecef)

    return np.array([
        np.linalg.norm(relA2B_ned),
        np.arctan2(relA2B_ned[1], relA2B_ned[0]) * r2d,
        np.arctan2(-relA2B_ned[2],
                   np.sqrt(relA2B_ned[0] ** 2 + relA2B_ned[1] ** 2)) * r2d
    ])

if __name__ == '__main__':
    pass
