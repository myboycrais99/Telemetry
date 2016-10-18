"""


"""

# module_name, package_name, ClassName, method_name, ExceptionName,
# function_name, GLOBAL_CONSTANT_NAME, global_var_name, instance_var_name,
# function_parameter_name, local_var_name.

# needs to have LLA, REA, NED, ENU, XYZ (ecef)

from __future__ import division, print_function
from Coordinate_Transform import relA2B_RAE
import numpy as np


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

    # 35 03 28.77409(N) 106 31 58.09043(W)
    # FO1194 = TelemetryObject(35.057993, -106.532803, 1670.9-20+2)
    # FO1194.lat_dms = (35, 03, 28.77409)

    # print(FO1194.lat, FO1194.lat_dms)
    # print(FO1194.lon, FO1194.lon_dms)

    d2r = np.pi / 180
    r2d = 180 / np.pi

    lat1 = 35.058248
    lon1 = -106.531279
    alt1 = 2

    lat2 = 35.120378
    lon2 = -106.515037
    alt2 = 400

    # TODO - Ryan: need to create function to calculate N based on Lat and Lon
    H = 1670.9
    N = -20.685
    h = H + N

    A = (lat1, lon1, alt1 + h)
    B = (lat2, lon2, alt2 + h)

    RAE = relA2B_RAE(A, B)

    print(RAE)
