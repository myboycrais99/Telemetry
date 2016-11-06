"""
This script contains several functions for coordinate transformation in 3D
space for telemetry systems.
"""

from __future__ import division, print_function
import numpy as np
from numpy import cos, sin

d2r = np.pi / 180
r2d = 180 / np.pi

ellipses = {
    'WGS84': {'a': 6378137, 'inv_f': 298.257223563, 'unit': 'm'}
    }


def degrees2dms(deg):
    """"Convert from decimal degrees to degrees, minutes, seconds"""
    d = int(deg)
    m = int((deg - d) * 60)
    s = ((deg - d) * 60 - m) * 60

    return d, m, s


def dms2degrees(dms):
    """Convert from degrees, minutes, seconds to decimal degrees"""
    return dms[0] + (dms[1] + dms[2] / 60) / 60


def geodetic2ecef(LLA, ellipse='WGS84'):
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

    a = float(ellipses[ellipse]['a'])
    f = 1 / float(ellipses[ellipse]['inv_f'])

    b = a * (1 - f)
    e = np.sqrt(1 - (b / a) ** 2)

    # Intermediate calculation (prime vertical radius of curvature)
    N = a / np.sqrt(1 - (e * sin(lat)) ** 2)

    X = (N + alt) * cos(lat) * cos(lon)
    Y = (N + alt) * cos(lat) * sin(lon)
    Z = ((b / a) ** 2 * N + alt) * sin(lat)

    return np.array([X, Y, Z], dtype=float)


def ecef2geodetic(XYZ, ellipse='WGS84'):
    """Convert from ECEF to geodetic coordinates.

    This function uses the Ferrari solution to convert from ECEF to geodetic
    (latitude, longitude, altitude) coordinates using a 15-step process. This
    method was found at:
    https://en.wikipedia.org/wiki/Geographic_coordinate_conversion

    Parameters
    ----------
    XYZ : float array
        The ECEF coordinates of an object

    Return
    ------
    out : float array
        Return the geodetic coordinates of an object

    Raises
    ------
    None
    """
    X, Y, Z = XYZ

    a = float(ellipses[ellipse]['a'])
    f = 1 / float(ellipses[ellipse]['inv_f'])

    b = a * (1 - f)
    e = np.sqrt(1 - (b / a)**2)
    e1_2 = (a / b)**2 - 1

    r = np.sqrt(X**2 + Y**2)                                               # 1
    E_2 = a**2 - b**2                                                      # 2
    F = 54 * b**2 * Z**2                                                   # 3
    G = r**2 + (1 - e**2) * Z**2 - e**2 * E_2                              # 4
    C = e**4 * F * r**2 / G**3                                             # 5
    s = 1 + C + np.sqrt(C**2 + 2 * C)                                      # 6a

    if 0 <= s:
        S = s**(1 / 3)                                                     # 6b
    else:
        S = -(-s)**(1 / 3)                                                 # 6c

    P = F / (3 * (S + (1 / S) + 1)**2 * G**2)                              # 7
    Q = np.sqrt(1 + 2 * e**4 * P)                                          # 8
    r0 = -((P * e**2 * r) / (1 + Q)) + np.sqrt((a**2 / 2) * (1 + 1 / Q) -
        (P * (1 - e**2) * Z**2) / (Q * (1 + Q)) - P * r**2 / 2)            # 9
    U = np.sqrt((r - e**2 * r0)**2 + Z**2)                                 # 10
    V = np.sqrt((r - e**2 * r0)**2 + (1 - e**2) * Z**2)                    # 11
    Z0 = (b**2 * Z) / (a * V)                                              # 12

    h = U * (1 - (b**2 / (a * V)))                                         # 13
    lat = np.arctan((Z + e1_2 * Z0) / r) * r2d                             # 14
    lon = np.arctan2(Y, X) * r2d                                           # 15

    return lat, lon, h


def ned2ecef(LLA):
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
        [-sin(lat) * cos(lon), -sin(lon), -cos(lat) * cos(lon)],
        [-sin(lat) * sin(lon), cos(lon), -cos(lat) * sin(lon)],
        [cos(lat), 0.0, -sin(lat)]
    ])


def ecef2ned(LLA):
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
    return ned2ecef(LLA).T


def enu2ecef(LLA):
    """Rotation matrix for converting from East-North-Up (ENU) frame to
    Earth-Centered Earth Fixed reference frame using latitude and longitude.

    Parameters
    ----------
    LLA : float array
        Geodetic latitude in radians, longitude in radians, and altitude above
        reference ellipsoid in meters

    Returns
    -------
    out : array
        A 3x3 matrix representing the ENU to ECEF rotation from LLA

    Raises
    ------
    None
    """
    return ecef2enu(LLA).T


def ecef2enu(LLA):
    """Rotation matrix for converting from Earth-Centered Earth-Fixed frame to
    East-North-Up (ENU) reference frame using geodetic latitude and longitude

    Parameters
    ----------
    LLA : float array
        Geodetic latitude in radians, longitude in radians, and altitude above
        reference ellipsoid in meters

    Returns
    -------
    out : array
        A 3x3 matrix representing the ECEF to ENU rotation from LLA

    Raises
    ------
    None
    """
    lat, lon = LLA[0] * d2r, LLA[1] * d2r

    return([
        [-sin(lon), cos(lon, 0.0), 0.0],
        [-sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat)],
        [cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat)]
    ])


def enu2aer(enu):
    """Convert from ENU coordinate frame to AER

    Parameters
    ----------
    enu : float array
        Coordinates of an object in the East-North-Up frame

    Returns
    -------
    out : array
        The AER coordinates of an object

    Raises
    ------
    None
    """
    xEast, yNorth, zUp = enu[0], enu[1], enu[2]

    deg360 = 2 * np.arctan2(0, -1)
    r = np.hypot(xEast, yNorth)

    slant_r = np.hypot(r, zUp)
    el = np.arctan2(zUp, r) * r2d
    az = np.mod(np.arctan2(xEast, yNorth), deg360) * r2d

    return az, el, slant_r


def aer2enu(aer):
    """Convert from AER coordinate frame to ENU

    Parameters
    ----------
    aer : float array
        Coordinates of an object in the Azimuth, Elevation, Slant Range frame

    Returns
    -------
    out : array
        The ENU coordinates of an object

    Raises
    ------
    None
    """
    az, el, slant_r = aer[0] * d2r, aer[1] * d2r, aer[2]

    zUp = slant_r * sin(el)
    r = slant_r * cos(el)
    xEast = r * sin(az)
    yNorth = r * cos(az)

    return xEast, yNorth, zUp


def relA2B_RAE(A, B):
    """Determine the relative azimuth and elevation from object A to object B.

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
    relA2B_ned = np.dot(ecef2ned(A), geodetic2ecef(B) - geodetic2ecef(A))

    r = np.linalg.norm(relA2B_ned)
    az = np.arctan2(relA2B_ned[1], relA2B_ned[0]) * r2d
    el = np.arcsin(-relA2B_ned[2] / r) * r2d

    return np.array([r, az, el])


def platform_dcm(orientation):
    """Calculate direction cosine matrix relative to a body's orientation.

    Parameters
    ----------
    orientation : float array
        Heading, Pitch, and Roll of platform in degrees

    Return
    ------
    out : float array
        A 3x3 dimensioned array cooresponding to the direction cosine matrix

    Raises
    ------
    None
    """
    hdg, pitch, roll = np.asarray(orientation) * d2r

    return np.array([
        [cos(pitch) * cos(hdg), cos(pitch) * sin(hdg), -sin(pitch)],
        [-cos(roll) * sin(hdg) + sin(roll) * sin(pitch) * cos(hdg),
            cos(roll) * cos(hdg) + sin(roll) * sin(pitch) * sin(hdg),
            sin(roll) * cos(pitch)],
        [sin(roll) * sin(hdg) + cos(roll) * sin(pitch) * cos(hdg),
            -sin(roll) * cos(hdg) + cos(roll) * sin(pitch) * sin(hdg),
            cos(roll) * cos(pitch)]
    ])


def LLA_platform_offset(platform, orientation, dB):
    """Determine the LLA offset between two objects

    Parameters
    ----------
    platform : float array
        LLA coordinates of a platform object
    orientation : float array
        Heading, Pitch, Roll of platform object in degrees
    B : float array
        x, y, z offset distances from platform to object B in meters

    Return
    ------
    out : float array
        LLA coordinates of B

    Raises
    ------
    None
    """
    lat, lon = platform[0] * d2r, platform[1] * d2r
    dcm = platform_dcm(orientation)
    dB = np.asarray(dB).reshape(3, 1)

    # For some reason, there is a switch from ENU to NEU coordinates
    d_neu = np.dot(dcm.T, dB)
    vNorth, uEast, wUp = d_neu

    x0, y0, z0 = geodetic2ecef(platform)
    t = float(cos(lat) * wUp - sin(lat) * vNorth)

    dx = float(cos(lon) * t - sin(lon) * uEast)
    dy = float(sin(lon) * t + cos(lon) * uEast)
    dz = float(sin(lat) * wUp + cos(lat) * vNorth)

    return ecef2geodetic([x0 + dx, y0 + dy, z0 + dz])


if __name__ == '__main__':
    pass
