"""
This script contains several functions for coordinate transformation in 3D
space for telemetry systems.
"""

from __future__ import division, print_function
import numpy as np
from numpy import arcsin, arctan, arctan2, cos, sin, radians
from reference_ellipsoid import ReferenceEllipsoid


def get_trig(in_degrees):
    if in_degrees:
        def _arcsin(angle): return arcsin(radians(angle))

        def _arctan(angle): return arctan(radians(angle))

        def _arctan2(angle1, angle2):
            return arctan2(radians(angle1), radians(angle2))

        def _sin(angle): return sin(radians(angle))

        def _cos(angle): return cos(radians(angle))

    else:
        def _arcsin(angle): return arcsin(angle)

        def _arctan(angle): return arctan(angle)

        def _arctan2(angle1, angle2): return arctan2(angle1, angle2)

        def _sin(angle): return sin(angle)

        def _cos(angle): return cos(radians(angle))

    return _arcsin, _arctan, _arctan2, _cos, _sin


def degrees2dms(deg):
    """"Convert from decimal degrees to degrees, minutes, seconds"""
    d = int(deg)
    m = int((deg - d) * 60)
    s = ((deg - d) * 60 - m) * 60

    deg_sign = u"\N{DEGREE SIGN}".encode('ascii', 'ignore')

    return '{:.0f}{} {:02.0f}\' {:02.4f}\"'.format(d, deg_sign, abs(m), abs(s))


def dms2degrees(dms):
    """Convert from degrees, minutes, seconds to decimal degrees"""
    return dms[0] + (dms[1] + dms[2] / 60) / 60


def aer2ecef(az, elev, slant_r, lat0, lon0, h0, spheroid,
             angle_unit='degrees'):
    """
    Convert local spherical AER to geocentric ECEF

    [X, Y, Z] = AER2ECEF(AZ, ELEV, SLANTRANGE, LAT0, LON0, H0, SPHEROID)
    transforms point locations in 3-D from local spherical coordinates
    (azimuth angle, elevation angle, slant range) to geocentric Earth-Centered
    Earth-Fixed (ECEF) coordinates (X, Y, Z), given a local coordinate system
    defined by the geodetic coordinates of its origin (LAT0, LON0, H0).  The
    geodetic coordinates refer to the reference body specified by the spheroid
    object, SPHEROID.  The slant range and ellipsoidal height H0 must be
    expressed in the same length unit as the spheroid.  Outputs X, Y, and Z
    will be expressed in this unit, also. The input azimuth, elevation,
    latitude, and longitude angles are in degrees by default.

    Parameters
    ----------
    az : float
    elev : float
    slant_r : float
    lat0 : float
    lon0 : float
    h0 : float
    spheroid : float
    angle_unit : float

    Return
    ------
    x : float
    y : float
    z : float

    Raises
    ------
    None
    """
    assert isinstance(spheroid, ReferenceEllipsoid)
    assert angle_unit in ['degrees', 'radians']
    in_degrees = angle_unit == 'degrees'

    x, y, z = _aer2ecef_formula(az, elev, slant_r, lat0, lon0, h0, spheroid,
                                in_degrees)

    return x, y, z


def aer2enu(az, elev, slant_r, angle_unit='degrees'):
    """
    Convert local spherical AER to Cartesian ENU

    This function transforms point locations in 3-D from local spherical
    coordinates (azimuth angle, elevation angle, slantRange) to local Cartesian
    coordinates (xEast, yNorth, zUp). The input angles are assumed to be in
    degrees.

    Parameters
    ----------
    az : float
    elev : float
    slant_r : float
    angle_unit : float

    Return
    ------
    x_east
    y_north
    z_up

    Raises
    ------
    None
    """
    assert angle_unit in ['degrees', 'radians']
    in_degrees = angle_unit == 'degrees'

    _, _, _, _cos, _sin = get_trig(in_degrees)

    x_east, y_north, z_up = _aer2enu_formula(az, elev, slant_r, _sin, _cos)
    return x_east, y_north, z_up


def aer2geodetic(az, elev, slant_r, lat0, lon0, h0, spheroid,
                 angle_unit='degrees'):
    """
    Convert from focal spherical AER to geodetic

    This function transforms point locations in 3-D from local spherical
    coordinates (azimuth angle, elevation angle, slant range) to geodetic
    coordinates (LAT, LON, H), given a local coordinate system defined by the
    geodetic coordinates of its origin (LAT0, LON0, H0).  The geodetic
    coordinates refer to the reference body specified by the spheroid object,
    SPHEROID. The slant range and ellipsoidal height H0 must be expressed in
    the same length unit as the spheroid.  Ellipsoidal height H will be
    expressed in this unit, also.  The input azimuth and elevation angles, and
    input and output latitude and longitude angles, are in degrees by default.

    Parameters
    ----------
    az : float
    elev : float
    slant_r : float
    lat0 : float
    lon0 : float
    h0 : float
    spheroid : ReferenceEllipsoid
    angle_unit : string

    Return
    ------
    lat : float
    lon : float
    h : float

    Raises
    ------
    None
    """
    assert angle_unit in ['degrees', 'radians']
    assert isinstance(spheroid, ReferenceEllipsoid)

    in_degrees = angle_unit == 'degrees'

    x, y, z = _aer2ecef_formula(az, elev, slant_r, lat0, lon0, h0, spheroid,
                                in_degrees=True)

    lat, lon, h = ecef2geodetic(x, y, z, spheroid, in_degrees)

    return lat, lon, h


def aer2ned(az, elev, slant_r, angle_unit='degrees'):
    """
    Convert from local spherical AER to Cartesian NED

    This function transforms point locations in 3-D from local spherical
    coordinates (azimuth angle, elevation angle, slant range) to local
    Cartesian coordinates (x_north, y_east, z_down). The input angles are
    assumed to be in degrees.

    Parameters
    ----------
    az : float
    elev : float
    slant_r : float
    angle_unit : float

    Return
    ------
    x_north : float
    y_east : float
    z_down : float

    Raises
    ------
    None
    """
    assert angle_unit in ['degrees', 'radians']
    in_degrees = angle_unit == 'degrees'

    _, _, _, _cos, _sin = get_trig(in_degrees)

    y_east, x_north, z_up = _aer2enu_formula(az, elev, slant_r, _sin, _cos)
    z_down = -z_up

    return x_north, y_east, z_down


def ecef2aer(x, y, z, lat0, lon0, h0, spheroid, angle_unit='degrees'):
    """
    Convert from geocentric ECEF to local spherical AER

    This function transforms point locations from geocentric Earth-Centered
    Earth-Fixed (ECEF) coordinates (X, Y, Z) to local spherical coordinates
    (azimuth angle, elevation angle, slant range), given a local coordinate
    system defined by the geodetic coordinates of its origin (LAT0, LON0, H0).
    The geodetic coordinates refer to the reference body specified by the
    spheroid object, SPHEROID.  Inputs X, Y, Z, and ellipsoidal height H0 must
    be expressed in the same length unit as the spheroid.  The slant range will
    be expressed in this unit, also.  The input latitude and longitude angles,
    and output azimuth and elevation angles, are in degrees by default.

    Parameters
    ----------
    x : float
    y : float
    z : float
    lat0 : float
    lon0 : float
    h0 : float
    spheroid : ReferenceEllipsoid
    angle_unit : string

    Return
    ------
    az : float
    elev : float
    slant_r : float

    Raises
    ------
    """
    assert angle_unit in ['degrees', 'radians']
    assert isinstance(spheroid, ReferenceEllipsoid)

    in_degrees = angle_unit == 'degrees'

    _, _, _arctan2, _cos, _sin = get_trig(in_degrees)

    x_east, y_north, z_up = _ecef2enu_formula(x, y, z, lat0, lon0, h0,
                                              spheroid, in_degrees)

    az, elev, slant_r = _enu2aer_formula(x_east, y_north, z_up, _arctan2)

    if in_degrees:
        return np.rad2deg(az), np.rad2deg(elev), slant_r
    else:
        return az, elev, slant_r


def ecef2enu(x, y, z, lat0, lon0, h0, spheroid, angle_unit='degrees'):
    """
    Converts from geocentric ECEF to local Cartesian ENU

    This function transforms point locations from geocentric Earth-Centered
    Earth-Fixed (ECEF) coordinates (X, Y, Z) to local Cartesian coordinates
    (x_east, y_north, z_up), given a local coordinate system defined by the
    geodetic coordinates of its origin (LAT0, LON0, H0).  The geodetic
    coordinates refer to the reference body specified by the spheroid object,
    SPHEROID. Inputs X, Y, Z, and ellipsoidal height H0 must be expressed in
    the same length unit as the spheroid.  Outputs xEast, yNorth, and zUp will
    be expressed in this unit, also.  The input latitude and longitude angles
    are in degrees by default.

    Parameters
    ----------
    x : float
    y : float
    z : float
    lat0 : float
    lon0 : float
    h0 : float
    spheroid : ReferenceEllipsoid
    angle_unit : string

    Return
    ------
    x_east : float
    y_north : float
    z_up : float

    Raises
    ------
    None
    """
    assert angle_unit in ['degrees', 'radians']
    assert isinstance(spheroid, ReferenceEllipsoid)

    in_degrees = angle_unit == 'degrees'

    x_east, y_north, z_up = _ecef2enu_formula(x, y, z, lat0, lon0, h0,
                                              spheroid, in_degrees)
    return x_east, y_north, z_up


def ecef2enuv(u, v, w, lat0, lon0, angle_unit='degrees'):
    """
    Rotate vector from ECEF to local ENU

    This function rotates a Cartesian 3-vector with components U, V, W from a
    geocentric Earth-Centered, Earth-Fixed (ECEF) system to a local
    east-north-up (ENU) system with origin at latitude LAT0 and longitude LON0.
    The origin latitude and longitude are assumed to be in units of degrees.

    Parameters
    ----------
    u : float
    v : float
    w : float
    lat0 : float
    lon0 : float
    angle_unit : float

    Return
    ------
    u_east : float
    v_north : float
    w_up : float

    Raises
    ------
    None
    """
    assert angle_unit in ['degrees', 'radians']
    in_degrees = angle_unit == 'degrees'

    _, _, _, _cos, _sin = get_trig(in_degrees)

    u_east, v_north, w_up = _ecef2enuv_formula(u, v, w, lat0, lon0, _sin, _cos)

    return u_east, v_north, w_up


def ecef2geodetic(x, y, z, spheroid, in_degrees=True):
    """Convert from ECEF to geodetic coordinates.

    This function uses the Ferrari solution to convert from ECEF to geodetic
    (latitude, longitude, altitude) coordinates using a 15-step process. This
    method was found at:
    https://en.wikipedia.org/wiki/Geographic_coordinate_conversion

    Parameters
    ----------
    x : float
    y : float
    z : float
    spheroid : ReferenceEllipsoid
    in_degrees : bool

    Return
    ------
    lat : float
    lon : float
    h : float

    Raises
    ------
    None
    """
    assert isinstance(spheroid, ReferenceEllipsoid)

    _, _arctan, _arctan2, _, _ = get_trig(in_degrees)

    a = spheroid.a
    b = spheroid.b
    e = spheroid.e
    e1_2 = spheroid.e1_2

    r = np.sqrt(x**2 + y**2)                                               # 1
    e_2 = a**2 - b**2                                                      # 2
    f = 54 * b**2 * z**2                                                   # 3
    g = r**2 + (1 - e**2) * z**2 - e**2 * e_2                              # 4
    c = e**4 * f * r**2 / g**3                                             # 5
    tmp = 1 + c + np.sqrt(c**2 + 2 * c)                                    # 6a

    if 0 <= tmp:
        s = tmp**(1 / 3)                                                   # 6b
    else:
        s = -(-tmp)**(1 / 3)                                               # 6c

    p = f / (3 * (s + (1 / s) + 1)**2 * g**2)                              # 7
    q = np.sqrt(1 + 2 * e**4 * p)                                          # 8
    r0 = -((p * e**2 * r) / (1 + q)) + (
        np.sqrt((a**2 / 2) * (1 + 1 / q) - (p * (1 - e**2) * z**2) /
                (q * (1 + q)) - p * r**2 / 2))                             # 9
    u = np.sqrt((r - e**2 * r0)**2 + z**2)                                 # 10
    v = np.sqrt((r - e**2 * r0)**2 + (1 - e**2) * z**2)                    # 11
    z0 = (b**2 * z) / (a * v)                                              # 12

    h = u * (1 - (b**2 / (a * v)))                                         # 13
    lat = np.rad2deg(_arctan((z + e1_2 * z0) / r))                         # 14
    lon = np.rad2deg(_arctan2(y, x))                                       # 15

    return lat, lon, h


def ecef2ned(x, y, z, lat0, lon0, h0, spheroid, angle_unit='degrees'):
    """
    Convert from geocentric ECEF to local Cartesian NED

    This function transforms point locations from geocentric Earth-Centered
    Earth-Fixed (ECEF) coordinates (X, Y, Z) to local Cartesian coordinates
    (x_north, y_east, z_down), given a local coordinate system defined by the
    geodetic coordinates of its origin (LAT0, LON0, H0).  The geodetic
    coordinates refer to the reference body specified by the spheroid object,
    SPHEROID. Inputs X, Y, Z, and ellipsoidal height H0 must be expressed in
    the same length unit as the spheroid.  Outputs x_north, y_east, and z_down
    will be expressed in this unit, also.  The input latitude and longitude
    angles are in degrees by default.

    Parameters
    ----------
    x : float
    y : float
    z : float
    lat0 : float
    lon0 : float
    h0 : float
    spheroid : ReferenceEllipsoid
    angle_unit : string

    Return
    ------
    x_north : float
    y_east : float
    z_down : float

    Raises
    ------
    None
    """

    assert isinstance(spheroid, ReferenceEllipsoid)
    assert angle_unit in ['degrees', 'radians']
    in_degrees = angle_unit == 'degrees'

    y_east, x_north, z_up = _ecef2enu_formula(x, y, z, lat0, lon0, h0,
                                              spheroid, in_degrees)
    z_down = -z_up

    return x_north, y_east, z_down


def ecef2nedv(u, v, w, lat0, lon0, angle_unit='degrees'):
    """Rotate vector from ECEF to local NED

    This function rotates a Cartesian 3-vector with components U, V, W from a
    geocentric Earth-Centered, Earth-Fixed (ECEF) system to a local
    north-east-down (NED) system with origin at latitude LAT0 and longitude
    LON0. The origin latitude and longitude are assumed to be in units of
    degrees.

    Parameters
    ----------
    u : float
    v : float
    w : float
    lat0 : float
    lon0 : float
    angle_unit : string

    Return
    ------
    u_north : float
    v_east : float
    w_down : float

    Raises
    ------
    None
    """
    assert angle_unit in ['degrees', 'radians']
    in_degrees = angle_unit == 'degrees'

    _, _, _, _cos, _sin = get_trig(in_degrees)

    v_east, u_north, w_up = _ecef2enuv_formula(u, v, w, lat0, lon0, _sin, _cos)
    w_down = -w_up

    return u_north, v_east, w_down


def ecef_offset(spheroid, lat1, lon1, h1, lat2, lon2, h2,
                angle_unit='degrees'):
    """
    In order to minimize numerical round off for points that are closely-spaced
    relative to the dimensions of the reference ellipsoid, use a sequence of
    computations that avoids computing a small difference by subtracting two
    large numbers. Rather than convert each point to ECEF,then subtract, note
    that each ECEF coordinate in the following formulas for x, y, and z:

        N = a. / sqrt(1 - e2 * sin(phi). ^ 2);

        x = (N + h). * cos(phi). * cos(lambda );
        y = (N + h).* cos(phi).* sin( lambda );
        z = (N * (1 - e2) + h).* sin(phi);

    contains a term that is a multiple of a and a term that is a multiple of h.
    For example,

        x = a * cos(phi) * cos( lambda ) * w + h * cos(phi) * cos( lambda );

    where w = 1./ sqrt(1 - e2 * sin(phi).^ 2).Constructing x this
    way for both x1 and x2, then taking the difference and factoring out a
    gives:

        dx = a * (cos(phi2) * cos(lambda2) * w1
            - cos(phi1) * cos(lambda1) * w1)
            + (h2 * cos(phi2) * cos(lambda2)
            - h1 * cos(phi1) * cos(lambda1))

    Parameters
    ----------
    spheroid : SPHEROID
    lat1 : float
    lon1 : float
    h1 : float
    lat2 : float
    lon2 : float
    h2 : float
    angle_unit : float

    Return
    ------
    deltaX : float
    deltaY : float
    deltaZ : float

    Raises
    ------
    None
    """
    assert angle_unit in ['degrees', 'radians']
    assert isinstance(spheroid, ReferenceEllipsoid)

    in_degrees = angle_unit == 'degrees'
    _, _, _, _cos, _sin = get_trig(in_degrees)

    e2 = spheroid.e ** 2

    s1 = _sin(lat1)
    c1 = _cos(lat1)

    s2 = _sin(lat2)
    c2 = _cos(lat2)

    p1 = c1 * _cos(lon1)
    p2 = c2 * _cos(lon2)

    q1 = c1 * _sin(lon1)
    q2 = c2 * _sin(lon2)

    w1 = 1. / np.sqrt(1 - e2 * s1 ** 2)
    w2 = 1. / np.sqrt(1 - e2 * s2 ** 2)

    delta_x = spheroid.a * (p2 * w2 - p1 * w1) + (h2 * p2 - h1 * p1)
    delta_y = spheroid.a * (q2 * w2 - q1 * w1) + (h2 * q2 - h1 * q1)
    delta_z = (1 - e2) * spheroid.a * (s2 * w2 - s1 * w1) + (h2 * s2 - h1 * s1)

    return delta_x, delta_y, delta_z


def enu2aer(x_east, y_north, z_up, angle_unit='degrees'):
    """Convert local Cartesian ENU to spherical AER

    This function transforms point locations in 3-D from local Cartesian
    coordinates (x_east, y_north, z_up) to local spherical coordinates
    (azimuth angle, elevation angle, slant range). The output angles are
    returned in degrees. East-north-up (ENU) system is a right-handed local
    Cartesian system with the X-axis directed east and parallel to the local
    tangent plane, the Y-axis directed north, and the Z-axis directed upward
    along the local normal to the ellipsoid.

    As always, the azimuth angle is measured clockwise (east) from north,
    from the perspective of a viewer looking down on the local horizontal
    plane. Equivalently, in the case of ENU, it is measured clockwise from the
    positive Y-axis in the direction of the positive X-axis. The elevation
    angle is the angle between a vector from the origin and the local
    horizontal plane. The slant range is the 3-D Euclidean distance from the
    origin.

    Parameters
    ----------
    x_east : float
    y_north : float
    z_up : float
    angle_unit : string

    Return
    ------
    az : float
    elev : float
    slant_r : float

    Raises
    ------
    None
    """

    assert angle_unit in ['degrees', 'radians']
    in_degrees = angle_unit == 'degrees'

    _, _, _arctan2, _, _ = get_trig(in_degrees)

    az, elev, slant_r = _enu2aer_formula(x_east, y_north, z_up, _arctan2)

    if in_degrees:
        return np.rad2deg(az), np.rad2deg(elev), slant_r
    else:
        return az, elev, slant_r


def enu2ecef(x_east, y_north, z_up, lat0, lon0, h0, spheroid,
             angle_unit='degrees'):
    """Convert local Cartesian ENU to geocentric ECEF

    This function transforms point locations in 3-D from local Cartesian
    coordinates (x_east, y_north, z_up) to geocentric Earth-Centered
    Earth-Fixed (ECEF) coordinates (X, Y, Z), given a local coordinate system
    defined by the geodetic coordinates of its origin (LAT0, LON0, H0).  The
    geodetic coordinates refer to the reference body specified by the spheroid
    object, SPHEROID.  Inputs x_east, y_north, z_np, and ellipsoidal height H0
    must be expressed in the same length unit as the spheroid.  Outputs X, Y,
    and Z will be expressed in this unit, also.  The input latitude and
    longitude angles are in degrees by default.

    Parameters
    ----------
    x_east : float
    y_north : float
    z_up : float
    lat0 : float
    lon0 : float
    h0 : float
    spheroid : ReferenceEllipsoid
    angle_unit : string

    Return
    ------
    x : float
    y : float
    z : float

    Raises
    ------
    None
    """

    assert isinstance(spheroid, ReferenceEllipsoid)
    assert angle_unit in ['degrees', 'radians']
    in_degrees = angle_unit == 'degrees'

    x, y, z = _enu2ecef_formula(x_east, y_north, z_up, lat0, lon0, h0,
                                spheroid, in_degrees)

    return x, y, z


def enu2ecefv(u_east, v_north, w_up, lat0, lon0, angle_unit='degrees'):
    """Rotate vector from local ENU to ECEF

    This function rotates a Cartesian 3-vector with components u_east, v_north,
    w_up from a local east-north-up (ENU) system with origin at latitude LAT0
    and longitude LON0 to a geocentric Earth-Centered, Earth-Fixed (ECEF)
    system. The origin latitude and longitude are assumed to be in units of
    degrees.

    Parameters
    ----------
    u_east : float
    v_north : float
    w_up : float
    lat0 : float
    lon0 : float
    angle_unit : string

    Return
    ------
    u : float
    v : float
    w : float

    Raises
    ------
    None
    """

    assert angle_unit in ['degrees', 'radians']
    in_degrees = angle_unit == 'degrees'

    _, _, _, _cos, _sin = get_trig(in_degrees)

    u, v, w = _enu2ecefv_formula(u_east, v_north, w_up, lat0, lon0, _sin, _cos)

    return u, v, w


def enu2geodetic(x_east, y_north, z_up, lat0, lon0, h0, spheroid,
                 angle_unit='degrees'):
    """Convert from local Cartesian ENU to geodetic

    This function transforms point locations in 3-D from local Cartesian
    coordinates (x_east, y_north, z_up) to geodetic coordinates (LAT, LON, H),
    given a local coordinate system defined by the geodetic coordinates of its
    origin (LAT0, LON0, H0).  The geodetic coordinates refer to the reference
    body specified by the spheroid object, SPHEROID.  Inputs x_east, y_north,
    z_up, and ellipsoidal height H0 must be expressed in the same length unit
    as the spheroid.  Ellipsoidal height H will be expressed in this unit,
    also. The input and output latitude and longitude angles are in degrees by
    default.

    Parameters
    ----------
    x_east : float
    y_north : float
    z_up : float
    lat0 : float
    lon0 : float
    h0 : float
    spheroid : ReferenceEllipsoid
    angle_unit : string

    Return
    ------
    lat : float
    lon : float
    h : float

    Raises
    ------
    None
    """

    assert isinstance(spheroid, ReferenceEllipsoid)
    assert angle_unit in ['degrees', 'radians']
    in_degrees = angle_unit == 'degrees'

    _, _, _, _cos, _sin = get_trig(in_degrees)

    x, y, z = _enu2ecef_formula(x_east, y_north, z_up, lat0, lon0, h0,
                                spheroid, in_degrees)

    lat, lon, h = ecef2geodetic(x, y, z, spheroid, in_degrees)

    return lat, lon, h


def geocentric_latitude(phi, f, angle_unit='degrees'):
    """Convert geodetic to geocentric latitude

    This function returns the geocentric latitude corresponding to geodetic
    latitude PHI on an ellipsoid with flattening F.

    Parameters
    ----------
    phi : float
        Geodetic latitude of one or more points, specified as a scalar value,
        vector, matrix, or N-D array. Values must be in units that match the
        input argument angleUnit, if supplied, and in degrees, otherwise.
    f : float
        Flattening of reference spheroid, specified as a scalar value.
    angle_unit : string
        Unit of angle, specified as 'degrees' (default), or 'radians'

    Return
    ------
    psi : float
        Geocentric latitudes of one or more points, returned as a scalar value,
        vector, matrix, or N-D array. Values are in units that match the input
        argument angle_unit, if supplied, and in degrees, otherwise.

    Raises
    ------
    None
    """
    assert angle_unit in ['degrees', 'radians']
    in_degrees = angle_unit == 'degrees'

    _, _, _arctan2, _cos, _sin = get_trig(in_degrees)

    if f == 0:
        # Perfect sphere: Avoid round off in the trig round - trip and ensure
        # an exact identity.
        psi = phi
    else:
        t = (1 - f) ** 2
        psi = _arctan2(t * _sin(phi), _cos(phi))
    return psi


def geodetic2aer(lat, lon, h, lat0, lon0, h0, spheroid, angle_unit='degrees'):
    """Convert geodetic to local spherical AER

    This function transforms point locations from geodetic coordinates
    (LAT, LON, H) to local spherical coordinates (azimuth angle,
    elevation angle, slant range), given a local coordinate system defined by
    the geodetic coordinates of its origin (LAT0, LON0, H0). The geodetic
    coordinates refer to the reference body specified by the spheroid object,
    SPHEROID. Ellipsoidal heights H and H0 must be expressed in the same
    length unit as the spheroid.  The slant range will be expressed in this
    unit, also. The input latitude and longitude angles, and output azimuth and
    elevation angles, are in degrees by default.

    Parameters
    ----------
    lat : float
    lon  : float
    h : float
    lat0  : float
    lon0  : float
    h0 : float
    spheroid : ReferenceEllipsoid
    angle_unit : string

    Return
    ------
    az : float
    elev : float
    slant_r : float

    Raises
    ------
    None
    """
    assert isinstance(spheroid, ReferenceEllipsoid)
    assert angle_unit in ['degrees', 'radians']
    in_degrees = angle_unit == 'degrees'

    _, _, _arctan2, _, _ = get_trig(in_degrees)

    x_east, y_north, z_up = _geodetic2enu_formula(lat, lon, h, lat0, lon0, h0,
                                                  spheroid, in_degrees)
    az, elev, slant_r = _enu2aer_formula(x_east, y_north, z_up, arctan2)

    if in_degrees:
        return np.rad2deg(az), np.rad2deg(elev), slant_r
    else:
        return az, elev, slant_r


def geodetic2ecef(lat0, lon0, h0, spheroid, angle_unit='degrees'):
    """Convert latitude, longitude, and altitude to earth-centered, earth-fixed
    (ECEF) cartesian

    Parameters
    ----------
    lat0 : float
        Geodetic latitude in radians, and
    lon0 : float
        Longitude in radians
    h0 : float
        Altitude above reference ellipsoid in meters
    spheroid: str, optional
        Dictionary keyword to define which reference ellipsoid to use in
        calculations
    angle_unit: str, optional
        Define the unit of the angle. Default is degrees.

    Returns
    -------
    out : array
        ECEF cartesian coordinates (X, Y, Z)

    Raises
    ------
    None
    """
    assert isinstance(spheroid, ReferenceEllipsoid)
    assert angle_unit in ['degrees', 'radians']
    in_degrees = angle_unit == 'degrees'

    _, _, _, _cos, _sin = get_trig(in_degrees)

    a = spheroid.a
    b = spheroid.b
    e = spheroid.e

    # Intermediate calculation (prime vertical radius of curvature)
    n = a / np.sqrt(1 - (e * _sin(lat0)) ** 2)

    x = (n + h0) * _cos(lat0) * _cos(lon0)
    y = (n + h0) * _cos(lat0) * _sin(lon0)
    z = ((b / a) ** 2 * n + h0) * _sin(lat0)

    return x, y, z


def geodetic2enu(lat, lon, h, lat0, lon0, h0, spheroid, angle_unit='degrees'):
    """Convert from geodetic to local Cartesian ENU

    This function transforms point locations from geodetic coordinates
    (LAT, LON, H) to local Cartesian coordinates (x_east, y_north, z_up), given
    a local coordinate system defined by the geodetic coordinates of its origin
    (LAT0, LON0, H0).  The geodetic coordinates refer to the reference body
    specified by the spheroid object, SPHEROID.  Ellipsoidal heights H and H0
    must be expressed in the same length unit as the spheroid.  Outputs
    x_east, y_north, and z_up will be expressed in this unit, also. The input
    latitude and longitude angles are in degrees by default.

    Parameters
    ----------
    lat : float
    lon : float
    h : float
    lat0 : float
    lon0 : float
    h0 : float
    spheroid : ReferenceEllipsoid
    angle_unit : string

    Return
    ------
    x_east : float
    y_north : float
    z_up : float

    Raises
    ------
    None
    """

    assert isinstance(spheroid, ReferenceEllipsoid)
    assert angle_unit in ['degrees', 'radians']
    in_degrees = angle_unit == 'degrees'

    x_east, y_north, z_up = _geodetic2enu_formula(lat, lon, h, lat0, lon0, h0,
                                                  spheroid, in_degrees)

    return x_east, y_north, z_up


def geodetic2ned(lat, lon, h, lat0, lon0, h0, spheroid, angle_unit='degrees'):
    """Convert geodetic to local Cartesian NED

    This function transforms point locations from geodetic coordinates
    (LAT, LON, H) to local Cartesian coordinates (xNorth, yEast, zDown), given
    a local coordinate system defined by the geodetic coordinates of its origin
    (LAT0, LON0, H0).  The geodetic coordinates refer to the reference body
    specified by the spheroid object, SPHEROID.  Ellipsoidal heights H and H0
    must be expressed in the same length unit as the spheroid.  Outputs
    x_north, y_east, and z_down will be expressed in this unit, also.  The
    input latitude and longitude angles are in degrees by default.

    Parameters
    ----------
    lat : float
    lon  : float
    h : float
    lat0  : float
    lon0  : float
    h0 : float
    spheroid : ReferenceEllipsoid
    angle_unit : string

    Return
    ------
    x_north : float
    y_east : float
    z_down : float

    Raises
    ------
    None
    """
    assert isinstance(spheroid, ReferenceEllipsoid)
    assert angle_unit in ['degrees', 'radians']
    in_degrees = angle_unit == 'degrees'

    y_east, x_north, z_up = _geodetic2enu_formula(lat, lon, h, lat0, lon0, h0,
                                                  spheroid, in_degrees)
    z_down = -z_up

    return x_north, y_east, z_down


def geodetic_latitude_from_geocentric(psi, f, spheroid, angle_unit='degrees'):
    """Convert geocentric to geodetic latitude

    This function returns the geodetic latitude corresponding to geocentric
    latitude PSI on an ellipsoid with flattening F.

    Parameters
    ----------
    psi : float
        Geocentric latitude of one or more points, specified as a scalar value,
        vector, matrix, or N-D array. Values must be in units that match the
        input argument angleUnit, if supplied, and in degrees, otherwise.
    f : float
        Flattening of reference spheroid, specified as a scalar value.
    spheroid : ReferenceEllipsoid

    angle_unit : float
        Unit of angle, specified as 'degrees' (default), or 'radians'

    Return
    ------
    phi : float
        Geodetic latitudes of one or more points, returned as a scalar value,
        vector, matrix, or N-D array. Values are in units that match the input
        argument angleUnit, if supplied, and in degrees, otherwise.

    Raises
    ------
    None
    """
    assert isinstance(spheroid, ReferenceEllipsoid)
    assert angle_unit in ['degrees', 'radians']
    in_degrees = angle_unit == 'degrees'

    _, _, _arctan2, _cos, _sin = get_trig(in_degrees)

    if f == 0:
        # Perfect sphere: Avoid round off in the trig round - trip and ensure
        # an exact identity.
        phi = psi
    else:
        t = (1 - spheroid.f) ** 2
        phi = _arctan2(_sin(psi), t * _cos(psi))

    return phi


def geodetic_latitude_from_parametric(beta, f, angle_unit='degrees'):
    """Convert parametric to geodetic latitude

    This function returns the geodetic latitude corresponding to parametric
    latitude BETA on an ellipsoid with flattening F.

    Parameters
    ----------
    beta : float
        Parametric latitude of one or more points, specified as a scalar value,
        vector, matrix, or N-D array. Values must be in units that match the
        input argument angleUnit, if supplied, and in degrees, otherwise.
    f : float
        Flattening of reference spheroid, specified as a scalar value.
    angle_unit : float
        Unit of angle, specified as 'degrees' (default), or 'radians'

    Return
    ------
    psi : float
        Geodetic latitudes of one or more points, returned as a scalar value,
        vector, matrix, or N-D array. Values are in units that match the input
        argument angleUnit, if supplied, and in degrees, otherwise.

    Raises
    ------
    None
    """
    assert angle_unit in ['degrees', 'radians']
    in_degrees = angle_unit == 'degrees'

    _, _, _arctan2, _cos, _sin = get_trig(in_degrees)

    if f == 0:
        # Perfect sphere: Avoid round off in the trig round - trip and ensure
        # an exact identity.
        phi = beta
    else:
        phi = _arctan2(_sin(beta), (1 - f) * _cos(beta))

    return phi


def ned2aer(x_north, y_east, z_down, angle_unit='degrees'):
    """Convert from local Cartesian NED to spherical AER

    This function transforms point locations in 3-D from local Cartesian
    coordinates (x_north, y_east, z_down) to local spherical coordinates
    (azimuth angle, elevation angle, and slant range). The output angles are
    returned in degrees. North-east-down (NED) is a right-handed local
    Cartesian system with the X-axis directed north and parallel to the local
    tangent plane, the Y-axis directed east, and the Z-axis directed downward
    along the local normal to the ellipsoid.

    As always, the azimuth angle is measured clockwise (east) from north, from
    the perspective of a viewer looking down on the local horizontal plane.
    Equivalently, in the case of NED, it is measured clockwise from the
    positive X-axis in the direction of the positive Y-axis. The elevation
    angle is the angle between a vector from the origin and the local
    horizontal plane. The slant range is the 3-D Euclidean distance from the
    origin.

    Parameters
    ----------
    x_north : float
    y_east : float
    z_down : float
    angle_unit : string

    Return
    ------
    az : float
    elev : float
    slant_r : float

    Raises
    ------
    None
    """
    assert angle_unit in ['degrees', 'radians']
    in_degrees = angle_unit == 'degrees'

    _, _, _arctan2, _, _ = get_trig(in_degrees)

    az, elev, slant_r = _enu2aer_formula(y_east, x_north, -z_down, arctan2)

    if in_degrees:
        return np.rad2deg(az), np.rad2deg(elev), slant_r
    else:
        return az, elev, slant_r


def ned2ecef(x_north, y_east, z_down, lat0, lon0, h0, spheroid,
             angle_unit='degrees'):
    """Convert local Cartesian NED to geocentric ECEF

    This function transforms point locations in 3-D from local Cartesian
    coordinates (x_north, y_east, z_down) to geocentric Earth-Centered
    Earth-Fixed (ECEF) coordinates (X, Y, Z), given a local coordinate system
    defined by the geodetic coordinates of its origin (LAT0, LON0, H0).
    The geodetic coordinates refer to the reference body specified by the
    spheroid object, SPHEROID.  Inputs x_north, y_east, z_down, and ellipsoidal
    height H0 must be expressed in the same length unit as the spheroid.
    Outputs X, Y, and Z will be expressed in this unit, also.  The input
    latitude and longitude angles are in degrees by default.

    Parameters
    ----------
    x_north : float
    y_east : float
    z_down : float
    lat0 : float
    lon0 : float
    h0 : float
    spheroid : ReferenceEllipsoid
    angle_unit : string

    Return
    ------
    x : float
    y : float
    z : float

    Raises
    ------
    None
    """
    assert isinstance(spheroid, ReferenceEllipsoid)
    assert angle_unit in ['degrees', 'radians']
    in_degrees = angle_unit == 'degrees'

    x, y, z = _enu2ecef_formula(y_east, x_north, -z_down, lat0, lon0, h0,
                                spheroid, in_degrees)

    return x, y, z


def ned2ecefv(u_north, v_east, w_down, lat0, lon0, angle_unit='degrees'):
    """Rotate vector from local NED to ECEF

    This function rotates a Cartesian 3-vector with components
    u_north, v_east, w_down from a local north-east-down (NED) system with
    origin at latitude LAT0 and longitude LON0 to a geocentric Earth-Centered,
    Earth-Fixed (ECEF) system. The origin latitude and longitude are assumed to
    be in units of degrees.

    Parameters
    ----------
    u_north : float
    v_east : float
    w_down : float
    lat0 : float
    lon0 : float
    angle_unit : string

    Return
    ------
    u : float
    v : float
    w : float

    Raises
    ------
    None
    """
    assert angle_unit in ['degrees', 'radians']
    in_degrees = angle_unit == 'degrees'

    _, _, _, _cos, _sin = get_trig(in_degrees)

    u, v, w = _enu2ecefv_formula(v_east, u_north, -w_down, lat0, lon0, _sin,
                                 _cos)

    return u, v, w


def ned2geodetic(x_north, y_east, z_down, lat0, lon0, h0, spheroid,
                 angle_unit='degrees'):
    """Convert local Cartesian NED to geodetic

    This function transforms point locations in 3-D from local Cartesian
    coordinates (x_north, y_east, z_down) to geodetic coordinates
    (LAT, LON, H), given a local coordinate system defined by the geodetic
    coordinates of its origin (LAT0, LON0, H0). The geodetic coordinates refer
    to the reference body specified by the spheroid object, SPHEROID.
    Inputs x_north, y_east, z_down, and ellipsoidal height H0 must be expressed
    in the same length unit as the spheroid. Ellipsoid height H will be
    expressed in this unit, also. The input and output latitude and longitude
    angles are in degrees by default.

    Parameters
    ----------
    x_north : float
    y_east : float
    z_down : float
    lat0 : float
    lon0 : float
    h0 : float
    spheroid : ReferenceEllipsoid
    angle_unit : string

    Return
    ------
    lat : float
    lon : float
    h : float

    Raises
    ------
    None
    """

    assert isinstance(spheroid, ReferenceEllipsoid)
    assert angle_unit in ['degrees', 'radians']
    in_degrees = angle_unit == 'degrees'

    x, y, z = _enu2ecef_formula(y_east, x_north, -z_down, lat0, lon0, h0,
                                spheroid, in_degrees)

    lat, lon, h = ecef2geodetic(x, y, z, spheroid, in_degrees)

    return lat, lon, h


def parametric_latitude(phi, f, angle_unit='degrees'):
    """Convert geodetic to parametric latitude

    This function returns the parametric latitude corresponding to geodetic
    latitude PHI on an ellipsoid with flattening F.

    Parameters
    ----------
    phi : float
        Geodetic latitude of one or more points, specified as a scalar value,
        vector, matrix, or N-D array. Values must be in units that match the
        input argument angleUnit, if supplied, and in degrees, otherwise.
    f : float
        Flattening of reference spheroid, specified as a scalar value.
    angle_unit  : string
         Unit of angle, specified as 'degrees' (default), or 'radians'.

    Return
    ------
    Beta : float
        Parametric latitudes of one or more points, returned as a scalar value,
        vector, matrix, or N-D array. Values are in units that match the input
        argument angleUnit, if supplied, and in degrees, otherwise.

    Raises
    ------
    None
    """
    assert angle_unit in ['degrees', 'radians']
    in_degrees = angle_unit == 'degrees'

    _, _, _arctan2, _cos, _sin = get_trig(in_degrees)

    if f == 0:
        # Perfect sphere: Avoid round off in the trig round - trip and ensure
        # an exact identity.
        beta = phi
    else:
        beta = _arctan2((1 - f) * _sin(phi), _cos(phi))


# ned2geodetic
# parametricLatitude

    return beta


# Private functions
def _aer2ecef_formula(az, elev, slant_r, lat0, lon0, h0, spheroid,
                      in_degrees=True):
    """
    Transform position from local spherical (AER) to geocentric (ECEF)

    Parameters
    ----------
    az : float
    elev : float
    slant_r : float
    lat0 : float
    lon0 : float
    h0 : float
    spheroid : spheroid
    in_degrees : bool

    Returns
    -------
    x : float
    y : float
    z : float

    Raises
    ------
    None
    """
    assert isinstance(spheroid, ReferenceEllipsoid)
    _, _, _, _cos, _sin = get_trig(in_degrees)

    [x0, y0, z0] = geodetic2ecef(lat0, lon0, h0, spheroid)

    # Transform local spherical AER to Cartesian ENU
    [x_east, y_north, z_up] = _aer2enu_formula(az, elev, slant_r, _sin, _cos)

    # Offset vector from local system origin, rotated from ENU to ECEF.
    [dx, dy, dz] = _enu2ecefv_formula(x_east, y_north, z_up, lat0, lon0, _sin,
                                      _cos)

    # Origin + offset from origin equals position in ECEF.
    x = x0 + dx
    y = y0 + dy
    z = z0 + dz

    return x, y, z


def _aer2enu_formula(az, elev, slant_r, _sin, _cos):
    """
    Transform local spherical (AER convention) to Cartesian (ENU convention).

    Parameters
    ----------
    az : float
    elev : float
    slant_r : float
    _sin : function
    _cos : function

    Return
    ------
    out : float array

    Raises
    ------
    None
    """

    z_up = slant_r * _sin(elev)
    r = slant_r * _cos(elev)
    x_east = r * _sin(az)
    y_north = r * _cos(az)

    return x_east, y_north, z_up


def _ecef2enu_formula(x, y, z, lat0, lon0, h0, spheroid, in_degrees=True):
    """
    Transform position from geocentric (ECEF) to local Cartesian(ENU)

    Parameters
    ----------
    x : float
    y : float
    z : float
    lat0 : float
    lon0 : float
    h0 : float
    spheroid : spheroid, optional
    in_degrees : bool, optional
    """
    assert isinstance(spheroid, ReferenceEllipsoid)

    _, _, _, _cos, _sin = get_trig(in_degrees)

    # Origin of the local system in geocentric coordinates.
    [x0, y0, z0] = geodetic2ecef(lat0, lon0, h0, spheroid)

    # Offset vector from local system origin, rotated from ECEF to ENU.
    [x_east, y_north, z_up] = _ecef2enuv_formula(x - x0, y - y0, z - z0, lat0,
                                                 lon0, _sin, _cos)

    return x_east, y_north, z_up


def _ecef2enuv_formula(u, v, w, lat0, lon0, _sin, _cos):
    """
    Rotate Cartesian 3-vector from ECEF to ENU

    Parameters
    ----------
    u :
    v :
    w :
    lat0 :
    lon0 :
    _sin :
    _cos :

    Returns
    -------
    out :

    Raises
    ------
    None
    """
    cos_phi = _cos(lat0)
    sin_phi = _sin(lat0)
    cos_lambda = _cos(lon0)
    sin_lambda = _sin(lon0)

    t = cos_lambda * u + sin_lambda * v
    u_east = -sin_lambda * u + cos_lambda * v

    w_up = cos_phi * t + sin_phi * w
    v_north = -sin_phi * t + cos_phi * w

    return u_east, v_north, w_up


def _enu2aer_formula(x_east, y_north, z_up, _arctan2):
    """
    Transform local Cartesian (ENU convention) to spherical (AER convention)

    Parameters
    ----------
    x_east : float
    y_north : float
    z_up : float
    _arctan2 : function

    Returns
    -------
    az : float
    elev : float
    slant_r : float
    _arctan2 : function
    """

    deg360 = 2 * _arctan2(0, -1)
    r = np.hypot(x_east, y_north)
    slant_r = np.hypot(r, z_up)
    elev = _arctan2(z_up, r)
    az = np.mod(_arctan2(x_east, y_north), deg360)

    return az, elev, slant_r


def _enu2ecef_formula(x_east, y_north, z_up, lat0, lon0, h0, spheroid,
                      in_degrees=True):
    """
    Transform position from local Cartesian (ENU) to geocentric (ECEF)

    Parameters
    ----------
    x_east : float
    y_north : float
    z_up : float
    lat0 : float
    lon0 : float
    h0 : float
    spheroid : spheroid, optional
    in_degrees : bool, optional

    Return
    ------
    x_east : float
    y_north : float
    z_up : float
    """
    assert isinstance(spheroid, ReferenceEllipsoid)
    
    _, _, _, _cos, _sin = get_trig(in_degrees)

    # Origin of local system in geocentric coordinates.
    [x0, y0, z0] = geodetic2ecef(lat0, lon0, h0, spheroid)

    # Offset vector from local system origin, rotated from ENU to ECEF.
    [dx, dy, dz] = _enu2ecefv_formula(x_east, y_north, z_up, lat0, lon0, _sin,
                                      _cos)

    # Origin + offset from origin equals position in ECEF.
    x = x0 + dx
    y = y0 + dy
    z = z0 + dz

    return x, y, z


def _enu2ecefv_formula(u_east, v_north, w_up, lat0, lon0, _sin, _cos):
    """
    Rotate Cartesian 3-vector from ENU to ECEF

    Parameters
    ----------
    u_east : float
    v_north : float
    w_up : float
    lat0 : float
    lon0 : float
    _sin : function
    _cos : function

    Return
    ------
    u : float
    v : float
    w : float

    Raises
    ------
    None

    """

    cos_phi = _cos(lat0)
    sin_phi = _sin(lat0)
    cos_lambda = _cos(lon0)
    sin_lambda = _sin(lon0)

    t = cos_phi * w_up - sin_phi * v_north
    w = sin_phi * w_up + cos_phi * v_north

    u = cos_lambda * t - sin_lambda * u_east
    v = sin_lambda * t + cos_lambda * u_east

    return u, v, w


def _geodetic2enu_formula(lat, lon, h, lat0, lon0, h0, spheroid, 
                          in_degrees=True):
    """
    Transform position from geodetic to local Cartesian(ENU)

    Parameters
    ----------
    lat : float
    lon : float
    h : float
    lat0 : float
    lon0 : float
    h0 : float
    spheroid : SPHEROID, optional
    in_degrees : bool, optional
    """
    assert isinstance(spheroid, ReferenceEllipsoid)
    
    _, _, _, _cos, _sin = get_trig(in_degrees)
    # Cartesian offset vector from local origin to(LAT, LON, H).
    [dx, dy, dz] = ecef_offset(spheroid, lat0, lon0, h0, lat, lon, h)

    # Offset vector from local system origin, rotated from ECEF to ENU.
    x_east, y_north, z_up = _ecef2enuv_formula(dx, dy, dz, lat0, lon0, _sin,
                                               _cos)

    return x_east, y_north, z_up


'''
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

    return np.asarray([
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

    return np.asarray([
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

    return np.asarray([az, el, slant_r])


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

    return np.asarray([xEast, yNorth, zUp])


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

    return np.asarray([r, az, el])


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

    return np.asarray([
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
'''


if __name__ == '__main__':
    pass
