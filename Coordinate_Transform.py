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
    N = a / np.sqrt(1 - (e * np.sin(lat)) ** 2)

    X = (N + alt) * np.cos(lat) * np.cos(lon)
    Y = (N + alt) * np.cos(lat) * np.sin(lon)
    Z = ((b / a) ** 2 * N + alt) * np.sin(lat)

    return np.array([X, Y, Z], dtype=float)


def rotation_ned2ecef(LLA):
    """Rotation matrix for converting from North-East-Down (NED) reference
    frame to Earth-Centered Earth-Fixed frame using Latitude and Longitude.

    Parameters
    ----------
    LLA : float array
        Geodetic latitude in radians, longitude in radians, and altitude above
        reference ellipsoid in meters

    Returns
    -------
    out : array
        A 3x3 matrix representing the NED to ECEF rotation from LLA

    Raises
    ------
    None
    """
    lat, lon = LLA[0] * d2r, LLA[1] * d2r

    return np.array([
        [-np.sin(lat) * np.cos(lon), -np.sin(lon), -np.cos(lat) * np.cos(lon)],
        [-np.sin(lat) * np.sin(lon), np.cos(lon), -np.cos(lat) * np.sin(lon)],
        [np.cos(lat), 0.0, -np.sin(lat)]
    ])


def rotation_ecef2ned(LLA):
    """Rotation matrix for converting from Earth-Centered Earth-Fixed frame to
    North-East-Down (NED) reference using Latitude and Longitude.

    Parameters
    ----------
    LLA : float array
        Geodetic latitude in radians, longitude in radians, and altitude above
        reference ellipsoid in meters

    Returns
    -------
    out : array
        A 3x3 matrix representing the ECEF to NED rotation from LLA

    Raises
    ------
    None
    """
    lat, lon = LLA[0] * d2r, LLA[1] * d2r

    return np.array([
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
        [-np.sin(lon), np.cos(lon), 0.0],
        [-np.cos(lat) * np.cos(lon), -np.cos(lat) * np.sin(lon), -np.sin(lat)]
    ])


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

    relA2B_ned = np.dot(rotation_ecef2ned(A), lla2ecef(B) - lla2ecef(A))

    r = np.linalg.norm(relA2B_ned)
    az = np.arctan2(relA2B_ned[1], relA2B_ned[0]) * r2d
    el = np.arcsin(-relA2B_ned[2] / r) * r2d

    return np.array([r, az, el])

if __name__ == '__main__':
    pass
