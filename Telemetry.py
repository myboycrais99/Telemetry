"""


"""

# module_name, package_name, ClassName, method_name, ExceptionName,
# function_name, GLOBAL_CONSTANT_NAME, global_var_name, instance_var_name,
# function_parameter_name, local_var_name.

from __future__ import division, print_function
from Coordinate_Transform import *
from geoid_separation import calculate_separation


class TelemetryObject(object):
    """ """
    # Try this instead:
    # init(1, 2, 3, 'opt')
    # where 1,2,3 are the three coordinates of any frame and 'opt'
    # is where the user defines which coordinate set they are using.
    # Then use if-then to see what the opt specified is and call each
    # conversion function to get other coordinates.
    def __init__(self, lla=None):
        self._lla = np.array(lla)

    @property
    def lla(self):
        return self._lla

    @lla.setter
    def lla(self, value):
        self._lla = value

    @property
    def ecef(self):
        return geodetic2ecef(self._lla)

    @ecef.setter
    def ecef(self, value):
        self._lla = ecef2geodetic(value)

    def rae_to_object(self, B):
        assert isinstance(B, TelemetryObject)
        return relA2B_RAE(self._lla, B.lla)


if __name__ == '__main__':

    lat_AK = 61.172680669444446
    lon_AK = -150.00374667777777
    alt_AK = 0

    lat_EUR = 38.50302497777778
    lon_EUR = -90.63047571111112
    alt_EUR = 0

    lat0 = 35.057992802777775
    lon0 = -106.53280289722223
    alt0 = 0

    lat1 = 35.058248
    lon1 = -106.531279
    alt1 = 4

    lat2 = 35.120378
    lon2 = -106.515037
    alt2 = 40

    # TODO - Ryan: consider using the larger geoid2012 data set but only load
    # once on program initialization. Will probably save time over reading a
    # small table every time. At the very least, load all the small maps
    # initially

    # TODO - Ryan: need to create function to calculate H based on Lat and Lon
    H = 1670.9
    N = calculate_separation(lat1, lon1)
    h = H + N

    A = TelemetryObject([lat1, lon1, alt1 + h])
    B = TelemetryObject([lat2, lon2, alt2 + h])

    # Camera offset from known GPS measurement point in the local cartesian
    # reference frame, measured in meters
    dx = 2.5        # Forward is Positive
    dy = -2.0       # Right is Positive
    dz = -3.0       # Up is Positive
    d_camera = [dx, dy, dz]

    # Relative orientation of the pedestal
    orientation = [350, 0, 0]
    camera = LLA_platform_offset(A.lla, orientation, d_camera)

    print("Camera LLA: {}".format(camera))

    print(A.rae_to_object(B))
    print(B.rae_to_object(A))
