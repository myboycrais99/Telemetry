"""

"""
from __future__ import division, print_function
import numpy as np

ELLIPSOIDS = [
    {'epsg': [], 'name': 'unitsphere', 'prop1': 'semimajor_axis', 'value1': 1, 'prop2': 'eccentricity', 'value2': 0, 'full_name': 'Unit Sphere'},
    {'epsg': 7035, 'name': 'sphere', 'prop1': 'semimajor_axis', 'value1': 6371000, 'prop2': 'eccentricity', 'value2': 0, 'full_name': 'Spherical Earth'},
    {'epsg': 7019, 'name': 'grs80', 'prop1': 'semimajor_axis', 'value1': 6378137, 'prop2': 'inverse_flattening', 'value2': 298.257222101, 'full_name': 'Geodetic Reference System 1980'},
    {'epsg': 7030, 'name': 'wgs84', 'prop1': 'semimajor_axis', 'value1': 6378137, 'prop2': 'inverse_flattening', 'value2': 298.257223563, 'full_name': 'World Geodetic System 1984'},
    {'epsg': 7015, 'name': 'everest', 'prop1': 'semimajor_axis', 'value1': 6377276.345, 'prop2': 'inverse_flattening', 'value2': 300.8017, 'full_name': 'Everest 1830'},
    {'epsg': 7004, 'name': 'bessel', 'prop1': 'semimajor_axis', 'value1': 6377397.155, 'prop2': 'inverse_flattening', 'value2': 299.1528128, 'full_name': 'Bessel 1841'},
    {'epsg': 7001, 'name': 'airy1830', 'prop1': 'semimajor_axis', 'value1': 6377563.396, 'prop2': 'inverse_flattening', 'value2': 299.3249646, 'full_name': 'Airy 1830'},
    {'epsg': 7002, 'name': 'airy1849', 'prop1': 'semimajor_axis', 'value1': 6377340.189, 'prop2': 'inverse_flattening', 'value2': 299.3249646, 'full_name': 'Airy Modified 1849'},
    {'epsg': 7008, 'name': 'clarke66', 'prop1': 'semimajor_axis', 'value1': 6378206.4, 'prop2': 'semiminor_axis', 'value2': 6356583.8, 'full_name': 'Clarke 1866'},
    {'epsg': 7012, 'name': 'clarke80', 'prop1': 'semimajor_axis', 'value1': 6378249.145, 'prop2': 'inverse_flattening', 'value2': 293.465, 'full_name': 'Clarke 1880'},
    {'epsg': 7022, 'name': 'international', 'prop1': 'semimajor_axis', 'value1': 6378388, 'prop2': 'inverse_flattening', 'value2': 297.0, 'full_name': 'International 1924'},
    {'epsg': 7024, 'name': 'krasovsky', 'prop1': 'semimajor_axis', 'value1': 6378245, 'prop2': 'inverse_flattening', 'value2': 298.3, 'full_name': 'Krasovsky 1940'},
    {'epsg': 7043, 'name': 'wgs72', 'prop1': 'semimajor_axis', 'value1': 6378135, 'prop2': 'inverse_flattening', 'value2': 298.26, 'full_name': 'World Geodetic System 1972'},
    {'epsg': [], 'name': 'wgs60', 'prop1': 'semimajor_axis', 'value1': 6378165, 'prop2': 'inverse_flattening', 'value2': 298.3, 'full_name': 'World Geodetic System 1960'},
    {'epsg': [], 'name': 'iau65', 'prop1': 'semimajor_axis', 'value1': 6378160, 'prop2': 'inverse_flattening', 'value2': 298.25, 'full_name': 'International Astronomical Union 1965'},
    {'epsg': [], 'name': 'wgs66', 'prop1': 'semimajor_axis', 'value1': 6378145, 'prop2': 'inverse_flattening', 'value2': 298.25, 'full_name': 'World Geodetic System 1966'},
    {'epsg': [], 'name': 'iau68', 'prop1': 'semimajor_axis', 'value1': 6378160, 'prop2': 'inverse_flattening', 'value2': 298.2472, 'full_name': 'International Astronomical Union 1968'},
    {'epsg': 7030, 'name': 'earth', 'prop1': 'semimajor_axis', 'value1': 6378137, 'prop2': 'inverse_flattening', 'value2': 298.257223563, 'full_name': 'World Geodetic System 1984'},
    {'epsg': [], 'name': 'sun', 'prop1': 'semimajor_axis', 'value1': 694460000, 'prop2': 'eccentricity', 'value2': 0, 'full_name': 'Sun'},
    {'epsg': [], 'name': 'moon', 'prop1': 'semimajor_axis', 'value1': 1738000, 'prop2': 'eccentricity', 'value2': 0, 'full_name': 'Moon'},
    {'epsg': [], 'name': 'mercury', 'prop1': 'semimajor_axis', 'value1': 2439000, 'prop2': 'eccentricity', 'value2': 0, 'full_name': 'Mercury'},
    {'epsg': [], 'name': 'venus', 'prop1': 'semimajor_axis', 'value1': 6051000, 'prop2': 'eccentricity', 'value2': 0, 'full_name': 'Venus'},
    {'epsg': [], 'name': 'mars', 'prop1': 'semimajor_axis', 'value1': 3396900, 'prop2': 'eccentricity', 'value2': 0.1105, 'full_name': 'Mars'},
    {'epsg': [], 'name': 'jupiter', 'prop1': 'semimajor_axis', 'value1': 71492000, 'prop2': 'eccentricity', 'value2': 0.3574, 'full_name': 'Jupiter'},
    {'epsg': [], 'name': 'saturn', 'prop1': 'semimajor_axis', 'value1': 60268000, 'prop2': 'eccentricity', 'value2': 0.4317, 'full_name': 'Saturn'},
    {'epsg': [], 'name': 'uranus', 'prop1': 'semimajor_axis', 'value1': 25559000, 'prop2': 'inverse_flattening', 'value2': 1/0.0229, 'full_name': 'Uranus'},
    {'epsg': [], 'name': 'neptune', 'prop1': 'semimajor_axis', 'value1': 24764000, 'prop2': 'eccentricity', 'value2': 0.1843, 'full_name': 'Neptune'},
    {'epsg': [], 'name': 'pluto', 'prop1': 'semimajor_axis', 'value1': 1151000, 'prop2': 'eccentricity', 'value2': 0, 'full_name': 'Pluto'},
    ]

class ReferenceEllipsoid(object):
    """

    """

    def __init__(self, arg1):
        # Column 1 - epsg code
        # Column 2 - Short(convenience) name string
        # Column 3 - Name of first defining property
        # Column 4 - Value of first defining property
        # Column 5 - Name of second defining property
        # Column 6 - Value of second defining property
        # Column 7 - Full name string

        if isinstance(arg1, int):
            ellipse = next(
                (item for item in ELLIPSOIDS if item['epsg'] ==
                 arg1))
        elif isinstance(arg1, str):
            ellipse = next(
                (item for item in ELLIPSOIDS if item['name'] ==
                 arg1.lower()))

        self._a = None
        self._b = None
        self._e = None
        self._e1_2 = None
        self._f = None
        self._inv_f = None

        self._epsg = ellipse['epsg']
        self._name = ellipse['name']

        if ellipse['prop1'] == 'semimajor_axis':
            self._a = ellipse['value1']
        elif ellipse['prop2'] == 'semimajor_axis':
            self._a = ellipse['value2']

        if ellipse['prop1'] == 'semiminor_axis':
            self._b = ellipse['value1']
        elif ellipse['prop2'] == 'semiminor_axis':
            self._b = ellipse['value2']

        if ellipse['prop1'] == 'inverse_flattening':
            self._inv_f = ellipse['value1']
        elif ellipse['prop2'] == 'inverse_flattening':
            self._inv_f = ellipse['value2']

        if ellipse['prop1'] == 'eccentricity':
            self._e = ellipse['value1']
        elif ellipse['prop2'] == 'eccentricity':
            self._e = ellipse['value2']

        while None in [self._a, self._b, self._e, self._e1_2, self._inv_f]:

            if self._a is None and self._b is not None and self._inv_f is not None:
                self._a = self._b / (1 - 1 / self._inv_f)

            if self._b is None and self._a is not None and self._inv_f is not None:
                self._b = self._a * (1 - 1 / self._inv_f)

            if self._inv_f is None:
                if self._a is not None and self._b is not None:
                    self._inv_f = self._a / (self._a - self._b)
                elif self._e is not None:
                    if self._e == 0:
                        self._inv_f = 1 / np.finfo(float).eps
                    else:
                        self._inv_f = 1 / (1 - np.sqrt(1 - self._e ** 2))

            if self._a is not None and self._b is not None:
                if self._e is None:
                    self._e = np.sqrt(self._a ** 2 - self._b ** 2) / self._a
                if self._e1_2 is None:
                    self._e1_2 = (self._a / self._b) ** 2 - 1

    @property
    def name(self):
        return self._name

    @property
    def epsg(self):
        return self._epsg

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def e(self):
        return self._e

    @property
    def e1_2(self):
        return self._e1_2

    @property
    def f(self):
        return 1/self._inv_f

    @property
    def inv_f(self):
        return self._inv_f


if __name__ == '__main__':
    pass

    # e = ReferenceEllipsoid(7030)
    # print('epsg: {}'.format(e.epsg))
    # print('name: {}'.format(e.name))
    # print('a: {}'.format(e.a))
    # print('b: {}'.format(e.b))
    # print('e: {}'.format(e.e))
    # print('e1_2: {}'.format(e.e1_2))
    # print('f: {}'.format(e.f))
    # print('inv_f: {}'.format(e.inv_f))
