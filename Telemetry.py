"""


"""

# module_name, package_name, ClassName, method_name, ExceptionName,
# function_name, GLOBAL_CONSTANT_NAME, global_var_name, instance_var_name,
# function_parameter_name, local_var_name.

from __future__ import division, print_function
from Coordinate_Transform import lla2ecef, relA2B_RAE
from geoid_separation import calculate_separation


class TelemetryObject(object):
    """ """
    # Try this instead:
    # init(1, 2, 3, 'opt')
    # where 1,2,3 are the three coordinates of any frame and 'opt'
    # is where the user defines which coordinate set they are using.
    # Then use if-then to see what the opt specified is and call each
    # conversion function to get other coordinates.
    def __init__(self, lat, lon, alt):
        self.lat = lat
        self.lon = lon
        self.alt = alt

        self.lat_dms = self.deg_2_dms(lat)
        self.lon_dms = self.deg_2_dms(lon)

    @staticmethod
    def deg_2_dms(deg):
        """Convert from decimal degrees to degrees, minutes, seconds format."""
        d = int(deg)
        tmp = abs(deg - d) * 60
        m = int(tmp)
        s = (tmp - m) * 60
        return d, m, s


if __name__ == '__main__':

    lat1 = 35.058248
    lon1 = -106.531279
    alt1 = 2

    lat2 = 35.120378
    lon2 = -106.515037
    alt2 = 400

    # TODO - Ryan: need to create function to calculate N based on Lat and Lon
    H = 1670.9
    N = calculate_separation(lat1, lon1)
    h = H + N

    A = (lat1, lon1, alt1 + h)
    B = (lat2, lon2, alt2 + h)

    RAE = relA2B_RAE(A, B)

    print(RAE)

    print(calculate_separation(lat1, lon1))
    print(calculate_separation(lat2, lon2))
