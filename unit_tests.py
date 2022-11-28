import unittest
import datetime
import numpy as np
import numpy.testing as npt
from gregorian import Granularities as T, TimePoint, TimeInterval, Scope


class TestLevels(unittest.TestCase):

    def test_highest(self):
        self.assertEqual(T.highest(T.CENTURY|T.YEAR|T.DAY), T.CENTURY)
        self.assertEqual(T.highest(T.DECADE|T.MONTH), T.DECADE)
        self.assertEqual(T.highest(T.YEAR), T.YEAR)

    def test_lowest(self):
        self.assertEqual(T.lowest(T.CENTURY|T.YEAR|T.DAY), T.DAY)
        self.assertEqual(T.lowest(T.DECADE|T.MONTH), T.MONTH)
        self.assertEqual(T.lowest(T.YEAR), T.YEAR)

    def test_next(self):
        self.assertEqual(T.CENTURY.next(T.CENTURY|T.YEAR|T.DAY), T.YEAR)
        self.assertEqual(T.YEAR.next(T.CENTURY|T.YEAR|T.DAY), T.DAY)
        self.assertEqual(T.DAY.next(T.CENTURY|T.YEAR|T.DAY), None)

    def test_previous(self):
        self.assertEqual(T.CENTURY.previous(T.CENTURY|T.YEAR|T.DAY), None)
        self.assertEqual(T.YEAR.previous(T.CENTURY|T.YEAR|T.DAY), T.CENTURY)
        self.assertEqual(T.DAY.previous(T.CENTURY|T.YEAR|T.DAY), T.YEAR)

    def test_prefix(self):
        self.assertEqual(T.CENTURY.prefix(T.CENTURY|T.YEAR|T.DAY), None)
        self.assertEqual(T.YEAR.prefix(T.CENTURY|T.YEAR|T.DAY), T.CENTURY)
        self.assertEqual(T.DAY.prefix(T.CENTURY|T.YEAR|T.DAY), T.YEAR|T.CENTURY)

class TestTimePoint(unittest.TestCase):

    def test_parse(self):
        self.assertEqual(TimePoint.parse('Y1984'), TimePoint(T.YEAR, 1984))
        self.assertEqual(TimePoint.parse('Y-1984'), TimePoint(T.YEAR, -1984))
        self.assertEqual(TimePoint.parse('Y0'), TimePoint(T.YEAR, 0))
        self.assertEqual(TimePoint.parse('M1984_2'), TimePoint(T.MONTH, 1984, month=2))
        self.assertEqual(TimePoint.parse('d1984_2_3'), TimePoint(T.DAY, 1984, month=2, day=3))

    def test_value(self):
        self.assertEqual(TimePoint.parse('C1984').value(T.CENTURY), 19)
        self.assertEqual(TimePoint.parse('D1984').value(T.DECADE), 198)
        self.assertEqual(TimePoint.parse('Y1984').value(T.YEAR), 1984)
        self.assertEqual(TimePoint.parse('M1984').value(T.MONTH), None)
        self.assertEqual(TimePoint.parse('d1984').value(T.DAY), None)
        self.assertEqual(TimePoint.parse('C1984_2_3').value(T.CENTURY), 19)
        self.assertEqual(TimePoint.parse('D1984_2_3').value(T.DECADE), 198)
        self.assertEqual(TimePoint.parse('Y1984_2_3').value(T.YEAR), 1984)
        self.assertEqual(TimePoint.parse('M1984_2_3').value(T.MONTH), 2)
        self.assertEqual(TimePoint.parse('d1984_2_3').value(T.DAY), 3)

    def test_str(self):
        self.assertEqual(str(TimePoint(T.YEAR, 1984)), 'Y1984')
        self.assertEqual(str(TimePoint(T.MONTH, 1984, month=2)), 'M1984_2')
        self.assertEqual(str(TimePoint(T.DAY, 1984, month=2, day=3)), 'd1984_2_3')

    def test_before(self):
        self.assertTrue(TimePoint(T.YEAR, 1984).before(TimePoint(T.YEAR, 1985)))
        self.assertTrue(TimePoint(T.YEAR, 1984).before(TimePoint(T.YEAR, 1984)))
        self.assertTrue(TimePoint(T.YEAR, 1984).before(TimePoint(T.MONTH, 1985, month=2)))
        self.assertTrue(TimePoint(T.YEAR, 1984, month=2).before(TimePoint(T.YEAR, 1985)))
        self.assertTrue(TimePoint(T.YEAR, 1984, month=2).before(TimePoint(T.MONTH, 1984, month=3)))
        self.assertTrue(TimePoint(T.YEAR, 1984, month=3).before(TimePoint(T.MONTH, 1984, month=3)))
        self.assertTrue(TimePoint(T.YEAR, 1984).before(TimePoint(T.DAY, 1985, month=2, day=3)))
        self.assertTrue(TimePoint(T.YEAR, 1984, month=2, day=3).before(TimePoint(T.MONTH, 1985)))
        self.assertTrue(TimePoint(T.YEAR, 1984, month=2, day=3).before(TimePoint(T.DAY, 1984, month=2, day=4)))
        self.assertTrue(TimePoint(T.YEAR, 1984, month=2, day=3).before(TimePoint(T.DAY, 1984, month=2, day=3)))
        self.assertTrue(TimePoint(T.YEAR, 1984, month=2, day=3).before(TimePoint(T.DAY, 1984, month=2, day=2)))
        self.assertFalse(TimePoint(T.YEAR, 1984, month=2, day=3).before(TimePoint(T.DAY, 1983, month=2, day=3)))

    def test_after(self):
        self.assertFalse(TimePoint(T.YEAR, 1984).after(TimePoint(T.YEAR, 1985)))
        self.assertTrue(TimePoint(T.YEAR, 1984).after(TimePoint(T.YEAR, 1984)))
        self.assertFalse(TimePoint(T.YEAR, 1984).after(TimePoint(T.MONTH, 1985, month=2)))
        self.assertFalse(TimePoint(T.MONTH, 1984, month=2).after(TimePoint(T.YEAR, 1985)))
        self.assertFalse(TimePoint(T.MONTH, 1984, month=2).after(TimePoint(T.MONTH, 1984, month=3)))
        self.assertTrue(TimePoint(T.MONTH, 1984, month=3).after(TimePoint(T.MONTH, 1984, month=3)))
        self.assertFalse(TimePoint(T.YEAR, 1984).after(TimePoint(T.DAY, 1985, month=2, day=3)))
        self.assertFalse(TimePoint(T.DAY, 1984, month=2, day=3).after(TimePoint(T.YEAR, 1985)))
        self.assertFalse(TimePoint(T.DAY, 1984, month=2, day=3).after(TimePoint(T.DAY, 1984, month=2, day=4)))
        self.assertTrue(TimePoint(T.DAY, 1984, month=2, day=3).after(TimePoint(T.DAY, 1984, month=2, day=3)))
        self.assertTrue(TimePoint(T.DAY, 1984, month=2, day=3).after(TimePoint(T.DAY, 1984, month=2, day=2)))
        self.assertTrue(TimePoint(T.DAY, 1984, month=2, day=3).after(TimePoint(T.DAY, 1984, month=1, day=3)))
        self.assertTrue(TimePoint(T.DAY, 1984, month=2, day=3).after(TimePoint(T.DAY, 1983, month=2, day=3)))

    def test_vectorize(self):
        interval    = TimeInterval.parse('M1890_10', 'd2010_4_5')
        levels1 = T.CENTURY|T.DECADE|T.YEAR
        levels2 = T.MONTH|T.DAY|T.YEAR
        levels3 = T.CENTURY|T.YEAR|T.DAY
        point1 = TimePoint(T.YEAR, 1984)
        point2 = TimePoint(T.DAY, 1984, 2, 3)
        scope = Scope(interval, levels1)
        npt.assert_array_equal(point1.vectorize(scope), np.array([ 1 if i in {1, 11, 17} else 0 for i in range(scope.size())]))
        npt.assert_array_equal(point2.vectorize(scope), np.array([ 1 if i in {1, 11, 17} else 0 for i in range(scope.size())]))
        scope = Scope(interval, levels2)
        npt.assert_array_equal(point2.vectorize(scope), np.array([ 1 if i in {94, 122, 135} else 0 for i in range(scope.size())]))
        scope = Scope(interval, levels3)
        npt.assert_array_equal(point2.vectorize(scope), np.array([ 1 if i in {1, 87, 135} else 0 for i in range(scope.size())]))

    def test_truncate(self):
        self.assertEqual(TimePoint(T.YEAR, 1984).truncate(TimePoint(T.YEAR, 1990)), TimePoint(T.YEAR, 1984))
        self.assertEqual(TimePoint(T.YEAR, 1990).truncate(TimePoint(T.YEAR, 1984)), TimePoint(T.YEAR, 1984))
        self.assertEqual(TimePoint(T.MONTH, 1984, 2).truncate(TimePoint(T.YEAR, 1990)), TimePoint(T.MONTH, 1984, 2))
        self.assertEqual(TimePoint(T.MONTH, 1990, 2).truncate(TimePoint(T.YEAR, 1984)), TimePoint(T.MONTH, 1984, 2))
        self.assertEqual(TimePoint(T.YEAR, 1984).truncate(TimePoint(T.MONTH, 1990, 2)), TimePoint(T.YEAR, 1984))
        self.assertEqual(TimePoint(T.YEAR, 1990).truncate(TimePoint(T.MONTH, 1984, 2)), TimePoint(T.YEAR, 1984))
        self.assertEqual(TimePoint(T.MONTH, 1984, 2).truncate(TimePoint(T.MONTH, 1990, 1)), TimePoint(T.MONTH, 1984, 2))
        self.assertEqual(TimePoint(T.MONTH, 1990, 2).truncate(TimePoint(T.MONTH, 1984, 1)), TimePoint(T.MONTH, 1984, 1))
        self.assertEqual(TimePoint(T.MONTH, 1990, 2).truncate(TimePoint(T.MONTH, 1984, 2)), TimePoint(T.MONTH, 1984, 2))
        self.assertEqual(TimePoint(T.MONTH, 1990, 2).truncate(TimePoint(T.MONTH, 1984, 3)), TimePoint(T.MONTH, 1984, 3))
        self.assertEqual(TimePoint(T.DAY, 1984, 2, 3).truncate(TimePoint(T.MONTH, 1990, 2)), TimePoint(T.DAY, 1984, 2, 3))
        self.assertEqual(TimePoint(T.DAY, 1990, 2, 3).truncate(TimePoint(T.MONTH, 1984, 2)), TimePoint(T.DAY, 1984, 2, 3))
        self.assertEqual(TimePoint(T.YEAR, 1984).truncate(TimePoint(T.DAY, 1990, 2, 3)), TimePoint(T.YEAR, 1984))
        self.assertEqual(TimePoint(T.YEAR, 1990).truncate(TimePoint(T.DAY, 1984, 2, 3)), TimePoint(T.YEAR, 1984))
        self.assertEqual(TimePoint(T.DAY, 1984, 2, 3).truncate(TimePoint(T.MONTH, 1990, 1)), TimePoint(T.DAY, 1984, 2, 3))
        self.assertEqual(TimePoint(T.DAY, 1984, 2, 3).truncate(TimePoint(T.MONTH, 1990, 1), reverse=True), TimePoint(T.DAY, 1990, 1, 3))
        self.assertEqual(TimePoint(T.DAY, 1990, 2, 3).truncate(TimePoint(T.DAY, 1984, 1, 3)), TimePoint(T.DAY, 1984, 1, 3))
        self.assertEqual(TimePoint(T.DAY, 1990, 2, 3).truncate(TimePoint(T.DAY, 1984, 2, 3)), TimePoint(T.DAY, 1984, 2, 3))
        self.assertEqual(TimePoint(T.DAY, 1990, 2, 3).truncate(TimePoint(T.DAY, 1984, 3, 3)), TimePoint(T.DAY, 1984, 3, 3))
        self.assertEqual(TimePoint(T.DAY, 1990, 2, 3).truncate(TimePoint(T.DAY, 1984, 2, 2)), TimePoint(T.DAY, 1984, 2, 2))
        self.assertEqual(TimePoint(T.DAY, 1990, 2, 3).truncate(TimePoint(T.DAY, 1984, 2, 3)), TimePoint(T.DAY, 1984, 2, 3))
        self.assertEqual(TimePoint(T.DAY, 1990, 2, 3).truncate(TimePoint(T.DAY, 1984, 2, 4)), TimePoint(T.DAY, 1984, 2, 3))
        self.assertEqual(TimePoint(T.DAY, 1990, 2, 3).truncate(TimePoint(T.DAY, 1984, 2, 4), reverse=True), TimePoint(T.DAY, 1990, 2, 3))

    def test_within(self):
        scope    = TimeInterval.parse('M1890_10', 'd2010_4_5')
        self.assertTrue(TimePoint.parse('Y1984').within(scope))
        self.assertTrue(TimePoint.parse('M1984_2').within(scope))
        self.assertTrue(TimePoint.parse('d1984_2_3').within(scope))
        self.assertTrue(TimePoint.parse('M1890_10').within(scope))
        self.assertTrue(TimePoint.parse('d1890_10_30').within(scope))
        self.assertTrue(TimePoint.parse('Y1890').within(scope))
        self.assertFalse(TimePoint.parse('M1890_9').within(scope))
        self.assertFalse(TimePoint.parse('d1890_9_30').within(scope))
        self.assertTrue(TimePoint.parse('d2010_4_5').within(scope))
        self.assertTrue(TimePoint.parse('M2010_4').within(scope))
        self.assertTrue(TimePoint.parse('Y2010').within(scope))
        self.assertFalse(TimePoint.parse('M2010_5').within(scope))
        self.assertFalse(TimePoint.parse('d2010_4_6').within(scope))

class TestTimeInterval(unittest.TestCase):

    def test_parse(self):
        self.assertEqual(TimePoint.parse('Y1984'), TimePoint(T.YEAR, 1984))
        self.assertEqual(TimePoint.parse('Y-1984'), TimePoint(T.YEAR, -1984))
        self.assertEqual(TimePoint.parse('Y0'), TimePoint(T.YEAR, 0))
        self.assertEqual(TimePoint.parse('M1984_2'), TimePoint(T.MONTH, 1984, month=2))
        self.assertEqual(TimePoint.parse('d1984_2_3'), TimePoint(T.DAY, 1984, month=2, day=3))

    def test_value(self):
        self.assertEqual(TimePoint.parse('Y1984').value(T.CENTURY), 19)
        self.assertEqual(TimePoint.parse('Y1984').value(T.DECADE), 198)
        self.assertEqual(TimePoint.parse('Y1984').value(T.YEAR), 1984)
        self.assertEqual(TimePoint.parse('Y1984').value(T.MONTH), None)
        self.assertEqual(TimePoint.parse('Y1984').value(T.DAY), None)
        self.assertEqual(TimePoint.parse('d1984_2_3').value(T.CENTURY), 19)
        self.assertEqual(TimePoint.parse('d1984_2_3').value(T.DECADE), 198)
        self.assertEqual(TimePoint.parse('d1984_2_3').value(T.YEAR), 1984)
        self.assertEqual(TimePoint.parse('d1984_2_3').value(T.MONTH), 2)
        self.assertEqual(TimePoint.parse('d1984_2_3').value(T.DAY), 3)

    def test_str(self):
        self.assertEqual(str(TimeInterval(TimePoint(T.YEAR, 1948), TimePoint(T.YEAR, 1984))), '[Y1948~Y1984]')
        self.assertEqual(str(TimeInterval(TimePoint(T.MONTH, 1948, month=10), TimePoint(T.MONTH, 1984, month=2))), '[M1948_10~M1984_2]')
        self.assertEqual(str(TimeInterval(TimePoint(T.MONTH, 1948, month=10), TimePoint(T.DAY, 1984, month=2, day=3))), '[M1948_10~d1984_2_3]')

    def test_centuries(self):
        interval = TimeInterval.parse('M1890_10', 'd2010_4_5')
        self.assertEqual(interval.centuries(), 2)

    def test_decades(self):
        interval = TimeInterval.parse('M1890_10', 'd2010_4_5')
        self.assertEqual(interval.decades(), 12)

    def test_years(self):
        interval = TimeInterval.parse('M1890_10', 'd2010_4_5')
        self.assertEqual(interval.years(), 120)

    def test_months(self):
        interval = TimeInterval.parse('M1890_10', 'd2010_4_5')
        self.assertEqual(interval.months(), 1434)

    def test_days(self):
        self.assertEqual(TimeInterval.parse('d2010_4_5', 'd2010_4_5') .days(), 0)
        self.assertEqual(TimeInterval.parse('d2010_4_5', 'd2010_4_7') .days(), 2)
        self.assertEqual(TimeInterval.parse('d2010_4_5', 'd2010_4_6') .days(), 1)
        self.assertEqual(TimeInterval.parse('d2010_4_5', 'd2010_5_5') .days(), 30)
        self.assertEqual(TimeInterval.parse('d2010_4_5', 'd2011_4_5') .days(), 365)
        interval = TimeInterval.parse('M1890_10', 'd2010_4_5')
        self.assertEqual(interval.days(), 43650)

    def test_mean(self):
        interval = TimeInterval(TimePoint(T.YEAR, 1984), TimePoint(T.YEAR, 2003))
        self.assertEqual(interval.mean(), TimePoint(T.YEAR, 1993))

        interval = TimeInterval(TimePoint(T.MONTH, 1984, 8), TimePoint(T.MONTH, 2003, 2))
        self.assertEqual(interval.mean(), TimePoint(T.MONTH, 1993, month=11))

        interval = TimeInterval(TimePoint(T.DAY, 1984, 8, 23), TimePoint(T.DAY, 2003, 2, 12))
        self.assertEqual(interval.mean(), TimePoint(T.DAY, 1993, month=11, day=2))

        interval = TimeInterval(TimePoint(T.DAY, 1984, 2, 23), TimePoint(T.DAY, 1984, 8, 12))
        self.assertEqual(interval.mean(), TimePoint(T.DAY, 1984, month=5, day=2))

        interval = TimeInterval(TimePoint(T.DAY, 1984, 8, 12), TimePoint(T.DAY, 2003, 2, 23))
        self.assertEqual(interval.mean(), TimePoint(T.DAY, 1993, month=11, day=17))

        interval = TimeInterval(TimePoint(T.DAY, 1984, 2, 12), TimePoint(T.DAY, 1984, 8, 23))
        self.assertEqual(interval.mean(), TimePoint(T.DAY, 1984, month=5, day=17))

        interval = TimeInterval(TimePoint(T.DAY, 1984, 2, 3), TimePoint(T.DAY, 1984, 2, 3))
        self.assertEqual(interval.mean(), TimePoint(T.DAY, 1984, 2, 3))

        interval = TimeInterval(TimePoint(T.DAY, 1984, 2, 12), TimePoint(T.MONTH, 1984, 8))
        self.assertEqual(interval.mean(), TimePoint(T.MONTH, 1984, month=5))

        interval = TimeInterval(TimePoint(T.YEAR, 1984), TimePoint(T.DAY, 2003, 3, 2))
        self.assertEqual(interval.mean(), TimePoint(T.YEAR, 1993))

        interval = TimeInterval(TimePoint(T.YEAR, 1984), TimePoint(T.DAY, 2003, 3, 2))
        self.assertEqual(interval.mean(), TimePoint(T.YEAR, 1993))

        interval = TimeInterval(TimePoint(T.DAY, 2003, 3, 2), TimePoint(T.DAY, 2003, 3, 2))
        self.assertEqual(interval.mean(), TimePoint(T.DAY, 2003, 3, 2))

        interval = TimeInterval(TimePoint(T.MONTH, 2003, 3, 2), TimePoint(T.DAY, 2003, 3, 2))
        self.assertEqual(interval.mean(), TimePoint(T.MONTH, 2003, 3))

    def test_vectorize(self):
        scope_interval = TimeInterval.parse('M1890_10', 'd2010_4_5')
        interval1 = TimeInterval(TimePoint(T.YEAR, 1984), TimePoint(T.YEAR, 2003))
        interval2 = TimeInterval(TimePoint(T.MONTH, 1984, 8), TimePoint(T.MONTH, 2003, 2))
        interval3 = TimeInterval(TimePoint(T.DAY, 1984, 8, 23), TimePoint(T.DAY, 2003, 2, 12))
        interval4 = TimeInterval(TimePoint(T.DAY, 1984, 2, 23), TimePoint(T.DAY, 1984, 8, 12))
        interval5 = TimeInterval(TimePoint(T.DAY, 1984, 2, 3), TimePoint(T.DAY, 1984, 8, 3))
        interval6 = TimeInterval(TimePoint(T.DAY, 1984, 2, 3), TimePoint(T.DAY, 1984, 2, 4))
        interval7 = TimeInterval(TimePoint(T.DAY, 1984, 2, 3), TimePoint(T.DAY, 1984, 3, 2))
        interval8 = TimeInterval(TimePoint(T.DAY, 1950, 4, 2), TimePoint(T.YEAR, 2008))
        interval9 = TimeInterval(TimePoint(T.YEAR, 1950), TimePoint(T.MONTH, 2008, 1, 2))


        
        scope = Scope(scope_interval, T.CENTURY|T.DECADE|T.YEAR)
        npt.assert_array_equal(interval1.vectorize(scope, T.YEAR), np.stack((
                [ 1. if i in {1, 11} or i >= 17 else 0. for i in range(23)],
                [ 1. if i in {2, 3} or i in range (13, 17) else 0. for i in range(23)]), axis=0))
        npt.assert_array_equal(interval2.vectorize(scope, T.YEAR), np.stack((
                [ 1. if i in {1, 11} or i >= 17 else 0. for i in range(23)],
                [ 1. if i in {2, 3} or i in range (13, 17) else 0. for i in range(23)]), axis=0))
        npt.assert_array_equal(interval3.vectorize(scope, T.YEAR), np.stack((
                [ 1. if i in {1, 11} or i >= 17 else 0. for i in range(23)],
                [ 1. if i in {2, 3} or i in range (13, 17) else 0. for i in range(23)]), axis=0))
        npt.assert_array_equal(interval4.vectorize(scope, T.YEAR), np.stack((
                [ 1. if i in {1, 11, 17} else 0. for i in range(23)]), axis=0).reshape(1, -1))
        npt.assert_array_equal(interval5.vectorize(scope, T.YEAR), np.stack((
                [ 1. if i in {1, 11, 17} else 0. for i in range(23)]), axis=0).reshape(1, -1))
        npt.assert_array_equal(interval6.vectorize(scope, T.YEAR), np.stack((
                [ 1. if i in {1, 11, 17} else 0. for i in range(23)]), axis=0).reshape(1, -1))
        npt.assert_array_equal(interval7.vectorize(scope, T.YEAR), np.stack((
                [ 1. if i in {1, 11, 17} else 0. for i in range(23)]), axis=0).reshape(1, -1))
        npt.assert_array_equal(interval8.vectorize(scope, T.YEAR), np.stack((
                [ 1. if i in {1, 8} or i >= 13 else 0. for i in range(23)],
                [ 1. if i in {2, 3} or i in range(13, 22) else 0. for i in range(23)]), axis=0))
        npt.assert_array_equal(interval9.vectorize(scope, T.YEAR), np.stack((
                [ 1. if i in {1, 8} or i >= 13 else 0. for i in range(23)],
                [ 1. if i in {2, 3} or i in range(13, 22) else 0. for i in range(23)]), axis=0))

        npt.assert_array_equal(interval1.vectorize(scope, T.DECADE), np.stack((
                [ 1. if i in {1, 11, 12} else 0. for i in range(13)],
                [ 1. if i in {2, 3} else 0. for i in range(13)]), axis=0))
        npt.assert_array_equal(interval2.vectorize(scope, T.DECADE), np.stack((
                [ 1. if i in {1, 11, 12} else 0. for i in range(13)],
                [ 1. if i in {2, 3} else 0. for i in range(13)]), axis=0))
        npt.assert_array_equal(interval3.vectorize(scope, T.DECADE), np.stack((
                [ 1. if i in {1, 11, 12} else 0. for i in range(13)],
                [ 1. if i in {2, 3} else 0. for i in range(13)]), axis=0))
        npt.assert_array_equal(interval4.vectorize(scope, T.DECADE), np.stack((
                [ 1. if i in {1, 11} else 0. for i in range(13)]), axis=0).reshape(1, -1))
        npt.assert_array_equal(interval5.vectorize(scope, T.DECADE), np.stack((
                [ 1. if i in {1, 11} else 0. for i in range(13)]), axis=0).reshape(1, -1))
        npt.assert_array_equal(interval6.vectorize(scope, T.DECADE), np.stack((
                [ 1. if i in {1, 11} else 0. for i in range(13)]), axis=0).reshape(1, -1))
        npt.assert_array_equal(interval7.vectorize(scope, T.DECADE), np.stack((
                [ 1. if i in {1, 11} else 0. for i in range(13)]), axis=0).reshape(1, -1))
        npt.assert_array_equal(interval8.vectorize(scope, T.DECADE), np.stack((
                [ 1. if i == 1 or i >= 8 else 0. for i in range(13)],
                [ 1. if i in {2, 3} else 0. for i in range(13)]), axis=0))
        npt.assert_array_equal(interval9.vectorize(scope, T.DECADE), np.stack((
                [ 1. if i == 1 or i >= 8 else 0. for i in range(13)],
                [ 1. if i in {2, 3} else 0. for i in range(13)]), axis=0))

        npt.assert_array_equal(interval1.vectorize(scope, T.CENTURY), np.array(
                [ 1. if i > 0 else 0. for i in range(3)]).reshape(1, -1))
        npt.assert_array_equal(interval2.vectorize(scope, T.CENTURY), np.array(
                [ 1. if i > 0 else 0. for i in range(3)]).reshape(1, -1))
        npt.assert_array_equal(interval3.vectorize(scope, T.CENTURY), np.array(
                [ 1. if i > 0 else 0. for i in range(3)]).reshape(1, -1))
        npt.assert_array_equal(interval4.vectorize(scope, T.CENTURY), np.array(
                [ 1. if i == 1 else 0. for i in range(3)]).reshape(1, -1))
        npt.assert_array_equal(interval5.vectorize(scope, T.CENTURY), np.array(
                [ 1. if i == 1 else 0. for i in range(3)]).reshape(1, -1))
        npt.assert_array_equal(interval6.vectorize(scope, T.CENTURY), np.array(
                [ 1. if i == 1 else 0. for i in range(3)]).reshape(1, -1))
        npt.assert_array_equal(interval7.vectorize(scope, T.CENTURY), np.array(
                [ 1. if i == 1 else 0. for i in range(3)]).reshape(1, -1))
        npt.assert_array_equal(interval8.vectorize(scope, T.CENTURY), np.array(
                [ 1. if i in {1, 2} else 0. for i in range(3)]).reshape(1, -1))
        npt.assert_array_equal(interval9.vectorize(scope, T.CENTURY), np.array(
                [ 1. if i in {1, 2} else 0. for i in range(3)]).reshape(1, -1))

        scope = Scope(scope_interval, T.MONTH|T.DAY|T.YEAR)
        npt.assert_array_equal(interval1.vectorize(scope, T.DAY), np.stack((
                [ 1. if i in {94, 121}  or i >= 133 else 0. for i in range(163)],
                [ 1. if i in {113, 132}  or i >= 133 else 0. for i in range(163)]), axis=0))
        npt.assert_array_equal(interval2.vectorize(scope, T.DAY), np.stack((
                [ 1. if i in {94, 128}  or i > 132 else 0. for i in range(163)],
                [ 1. if i in {113, 122} or i > 132 else 0. for i in range(163)]), axis=0))
        npt.assert_array_equal(interval3.vectorize(scope, T.DAY), np.stack((
                [ 1. if i in {94, 128}  or i >= 155 else 0. for i in range(163)],
                [ 1. if i in {113, 122} or i in range(133, 145) else 0. for i in range(163)]), axis=0))
        npt.assert_array_equal(interval4.vectorize(scope, T.DAY), np.stack((
                [ 1. if i in {94, 122} or i >= 155 else 0. for i in range(163)],
                [ 1. if i in {94, 128} or i in range(133, 145) else 0. for i in range(163)]), axis=0))
        npt.assert_array_equal(interval5.vectorize(scope, T.DAY), np.stack((
                [ 1. if i in {94, 122} or i >= 135 else 0. for i in range(163)],
                [ 1. if i in {94, 128} or i in range(133, 136) else 0. for i in range(163)]), axis=0))
        npt.assert_array_equal(interval6.vectorize(scope, T.DAY), np.stack((
                [ 1. if i in {94, 122, 135, 136} else 0. for i in range(163)]), axis=0).reshape(1, -1))
        npt.assert_array_equal(interval7.vectorize(scope, T.DAY), np.stack((
                [ 1. if i in {94, 122} or i >= 135 else 0. for i in range(163)],
                [ 1. if i in {94, 123} or i in range(133, 135) else 0. for i in range(163)]), axis=0))
        npt.assert_array_equal(interval8.vectorize(scope, T.DAY), np.stack((
                [ 1. if i in {60, 124} or i > 133 else 0. for i in range(163)],
                [ 1. if i in {118, 132}  or i >= 133 else 0. for i in range(163)]), axis=0))
        npt.assert_array_equal(interval9.vectorize(scope, T.DAY), np.stack((
                [ 1. if i in {60, 121} or i >= 133 else 0. for i in range(163)],
                [ 1. if i in {118, 121, 133, 134} else 0. for i in range(163)]), axis=0))

        npt.assert_array_equal(interval1.vectorize(scope, T.MONTH)-np.stack((
                [ 1. if i == 94 or i >= 121 else 0. for i in range(133)],
                [ 1. if i == 113 or i >= 121 else 0. for i in range(133)]), axis=0), np.zeros((2, 133)))
        npt.assert_array_equal(interval2.vectorize(scope, T.MONTH)-np.stack((
                [ 1. if i == 94 or i in range(128, 133) else 0. for i in range(133)],
                [ 1. if i == 113 or i in {121, 122} else 0. for i in range(133)]), axis=0), np.zeros((2, 133)))
        npt.assert_array_equal(interval3.vectorize(scope, T.MONTH)-np.stack((
                [ 1. if i == 94 or i in range(128, 133) else 0. for i in range(133)],
                [ 1. if i == 113 or i in {121, 122} else 0. for i in range(133)]), axis=0), np.zeros((2, 133)))
        npt.assert_array_equal(interval4.vectorize(scope, T.MONTH), np.array(
                [ 1. if i == 94 or i in range(122, 129) else 0. for i in range(133)]).reshape(1, -1))
        npt.assert_array_equal(interval5.vectorize(scope, T.MONTH), np.array(
                [ 1. if i == 94 or i in range(122, 129) else 0. for i in range(133)]).reshape(1, -1))
        npt.assert_array_equal(interval6.vectorize(scope, T.MONTH), np.stack((
                [ 1. if i in {94, 122} else 0. for i in range(133)]), axis=0).reshape(1, -1))
        npt.assert_array_equal(interval7.vectorize(scope, T.MONTH), np.stack((
                [ 1. if i in {94, 122, 123} else 0. for i in range(133)]), axis=0).reshape(1, -1))
        npt.assert_array_equal(interval8.vectorize(scope, T.MONTH), np.stack((
                [ 1. if i in {60} or i >= 124 else 0. for i in range(133)],
                [ 1. if i in {118} or i >= 121 else 0. for i in range(133)]), axis=0))
        npt.assert_array_equal(interval9.vectorize(scope, T.MONTH), np.stack((
                [ 1. if i in {60} or i >= 121 else 0. for i in range(133)],
                [ 1. if i in {118, 121} else 0. for i in range(133)]), axis=0))

        npt.assert_array_equal(interval1.vectorize(scope, T.YEAR), np.array(
                [ 1. if i in range(94, 114) else 0. for i in range(121)]).reshape(1, -1))
        npt.assert_array_equal(interval2.vectorize(scope, T.YEAR), np.array(
                [ 1. if i in range(94, 114) else 0. for i in range(121)]).reshape(1, -1))
        npt.assert_array_equal(interval3.vectorize(scope, T.YEAR), np.array(
                [ 1. if i in range(94, 114) else 0. for i in range(121)]).reshape(1, -1))
        npt.assert_array_equal(interval4.vectorize(scope, T.YEAR), np.array(
                [ 1. if i == 94 else 0. for i in range(121)]).reshape(1, -1))
        npt.assert_array_equal(interval5.vectorize(scope, T.YEAR), np.array(
                [ 1. if i == 94 else 0. for i in range(121)]).reshape(1, -1))
        npt.assert_array_equal(interval6.vectorize(scope, T.YEAR), np.array(
                [ 1. if i == 94 else 0. for i in range(121)]).reshape(1, -1))
        npt.assert_array_equal(interval7.vectorize(scope, T.YEAR), np.array(
                [ 1. if i == 94 else 0. for i in range(121)]).reshape(1, -1))
        npt.assert_array_equal(interval8.vectorize(scope, T.YEAR), np.array(
                [ 1. if i in range(60, 119) else 0. for i in range(121)]).reshape(1, -1))
        npt.assert_array_equal(interval9.vectorize(scope, T.YEAR), np.array(
                [ 1. if i in range(60, 119) else 0. for i in range(121)]).reshape(1, -1))

        scope = Scope(scope_interval, T.CENTURY|T.YEAR|T.DAY)
        npt.assert_array_equal(interval1.vectorize(scope, T.DAY), np.stack((
                [ 1. if i in {1, 87}  or i >= 103 else 0. for i in range(463)],
                [ 1. if i in {2, 6}  or i >= 103 else 0. for i in range(463)]), axis=0))
        npt.assert_array_equal(interval2.vectorize(scope, T.DAY), np.stack((
                [ 1. if i in {1, 87}  or i >= 103 else 0. for i in range(463)],
                [ 1. if i in {2, 6}  or i >= 103 else 0. for i in range(463)]), axis=0))
        npt.assert_array_equal(interval3.vectorize(scope, T.DAY), np.stack((
                [ 1. if i in {1, 87}  or i >= 335 else 0. for i in range(463)],
                [ 1. if i in {2, 6}  or i in range(103, 145) else 0. for i in range(463)]), axis=0))
        npt.assert_array_equal(interval4.vectorize(scope, T.DAY), np.stack((
                [ 1. if i in {1, 87} or i in range(155, 325) else 0. for i in range(463)]), axis=0).reshape(1, -1))
        npt.assert_array_equal(interval5.vectorize(scope, T.DAY), np.stack((
                [ 1. if i in {1, 87} or i in range(135, 316) else 0. for i in range(463)]), axis=0).reshape(1, -1))
        npt.assert_array_equal(interval6.vectorize(scope, T.DAY), np.stack((
                [ 1. if i in {1, 87, 135, 136} else 0. for i in range(463)]), axis=0).reshape(1, -1))
        npt.assert_array_equal(interval7.vectorize(scope, T.DAY), np.stack((
                [ 1. if i in {1, 87}  or i in range(135, 165) else 0. for i in range(463)]), axis=0).reshape(1, -1))

        npt.assert_array_equal(interval1.vectorize(scope, T.YEAR), np.stack((
                [ 1. if i == 1 or i >= 87 else 0. for i in range(103)],
                [ 1. if i in {2, 3, 4, 5, 6} else 0. for i in range(103)]), axis=0))
        npt.assert_array_equal(interval2.vectorize(scope, T.YEAR), np.stack((
                [ 1. if i == 1 or i >= 87 else 0. for i in range(103)],
                [ 1. if i in {2, 3, 4, 5, 6} else 0. for i in range(103)]), axis=0))
        npt.assert_array_equal(interval3.vectorize(scope, T.YEAR), np.stack((
                [ 1. if i == 1 or i >= 87 else 0. for i in range(103)],
                [ 1. if i in {2, 3, 4, 5, 6} else 0. for i in range(103)]), axis=0))
        npt.assert_array_equal(interval4.vectorize(scope, T.YEAR), np.stack((
                [ 1. if i in {1, 87} else 0. for i in range(103)]), axis=0).reshape(1, -1))
        npt.assert_array_equal(interval5.vectorize(scope, T.YEAR), np.stack((
                [ 1. if i in {1, 87} else 0. for i in range(103)]), axis=0).reshape(1, -1))
        npt.assert_array_equal(interval6.vectorize(scope, T.YEAR), np.stack((
                [ 1. if i in {1, 87} else 0. for i in range(103)]), axis=0).reshape(1, -1))
        npt.assert_array_equal(interval7.vectorize(scope, T.YEAR), np.stack((
                [ 1. if i in {1, 87} else 0. for i in range(103)]), axis=0).reshape(1, -1))

        npt.assert_array_equal(interval1.vectorize(scope, T.CENTURY), np.array(
                [ 1. if i > 0 else 0. for i in range(3)]).reshape(1, -1))
        npt.assert_array_equal(interval2.vectorize(scope, T.CENTURY), np.array(
                [ 1. if i > 0 else 0. for i in range(3)]).reshape(1, -1))
        npt.assert_array_equal(interval3.vectorize(scope, T.CENTURY), np.array(
                [ 1. if i > 0 else 0. for i in range(3)]).reshape(1, -1))
        npt.assert_array_equal(interval4.vectorize(scope, T.CENTURY), np.array(
                [ 1. if i == 1 else 0. for i in range(3)]).reshape(1, -1))
        npt.assert_array_equal(interval5.vectorize(scope, T.CENTURY), np.array(
                [ 1. if i == 1 else 0. for i in range(3)]).reshape(1, -1))
        npt.assert_array_equal(interval6.vectorize(scope, T.CENTURY), np.array(
                [ 1. if i == 1 else 0. for i in range(3)]).reshape(1, -1))
        npt.assert_array_equal(interval7.vectorize(scope, T.CENTURY), np.array(
                [ 1. if i == 1 else 0. for i in range(3)]).reshape(1, -1))

        scope_interval = TimeInterval.parse('d2014_1_1','d2014_12_30')
        scope = Scope(scope_interval, T.MONTH|T.DAY)
        npt.assert_array_equal(TimeInterval.parse('d2014_3_4','d2014_3_4').vectorize(scope, T.DAY), 
                                [[ 1. if i in {2, 15} else 0. for i in range(42)]])
        
class TestScope(unittest.TestCase):

    def test_truncate(self):
        scope = Scope(TimeInterval.parse('M1890_10', 'd2010_4_5'), T.YEAR|T.MONTH|T.DAY)
        included = TimeInterval.parse('M1980_10', 'd2000_4_5')
        including = TimeInterval.parse('M1780_10', 'd2100_4_5')
        overleft = TimeInterval.parse('M1780_10', 'd2000_4_5')
        overright = TimeInterval.parse('M1980_10', 'd2100_4_5')
        touchleft = TimeInterval.parse('M1890_10', 'd2000_4_5')
        touchright = TimeInterval.parse('M1980_10', 'd2010_4_5')
        outleft = TimeInterval.parse('M1780_10', 'd1880_4_5')
        outright = TimeInterval.parse('M2011_10', 'd2110_4_5')

        openleftoutleft = TimeInterval.parse(None, 'd1780_4_5')
        openleftin = TimeInterval.parse(None, 'd1980_4_5')
        openleftoutright = TimeInterval.parse(None, 'd2180_4_5')
        openrightoutleft = TimeInterval.parse('d1780_4_5', None)
        openrightin = TimeInterval.parse('d1980_4_5', None)
        openrightoutright = TimeInterval.parse('d2180_4_5', None)

        self.assertEqual(scope.truncate(included), included)
        self.assertEqual(scope.truncate(including), scope.interval())
        self.assertEqual(scope.truncate(overleft), TimeInterval.parse('M1890_10', 'd2000_4_5'))
        self.assertEqual(scope.truncate(overright), TimeInterval.parse('M1980_10', 'd2010_4_5'))
        self.assertEqual(scope.truncate(touchleft), TimeInterval.parse('M1890_10', 'd2000_4_5'))
        self.assertEqual(scope.truncate(touchright), TimeInterval.parse('M1980_10', 'd2010_4_5'))
        self.assertEqual(scope.truncate(outleft), TimeInterval.parse('M1890_10', 'M1890_10'))
        self.assertEqual(scope.truncate(outright), TimeInterval.parse('d2010_4_5', 'd2010_4_5'))

        self.assertEqual(scope.truncate(openleftoutleft), TimeInterval.parse('M1890_10', 'M1890_10'))
        self.assertEqual(scope.truncate(openleftin), TimeInterval.parse('M1890_10', 'd1980_4_5'))
        self.assertEqual(scope.truncate(openleftoutright), TimeInterval.parse('M1890_10', 'd2010_4_5'))
        self.assertEqual(scope.truncate(openrightoutleft), TimeInterval.parse('d1890_10_5', 'd2010_4_5'))
        self.assertEqual(scope.truncate(openrightin), TimeInterval.parse('d1980_4_5', 'd2010_4_5'))
        self.assertEqual(scope.truncate(openrightoutright), TimeInterval.parse('d2010_4_5', 'd2010_4_5'))

        scope = Scope(TimeInterval.parse('d1100_1_1','d2019_6_30'), T.YEAR|T.DECADE|T.CENTURY)
        self.assertEqual(scope.truncate(TimeInterval.parse('C2000', 'C2000')), TimeInterval.parse('C2000', 'C2000'))
        
    def test_offset(self):
        interval = TimeInterval.parse('M1890_10', 's2010_4_5')
        point1 = TimePoint.parse('Y1984')
        point2 = TimePoint.parse('d1984_2_3')
        c   = T.CENTURY
        y   = T.YEAR
        cy  = T.CENTURY|T.YEAR
        cdy = T.CENTURY|T.DECADE|T.YEAR
        cyd = T.CENTURY|T.YEAR|T.DAY
        dm  = T.DECADE|T.MONTH
        ymd  = T.YEAR|T.MONTH|T.DAY
        md  = T.DAY|T.MONTH
        self.assertEqual(Scope(interval, c).offset(T.CENTURY, point1), 1)
        self.assertEqual(Scope(interval, c).offset(T.CENTURY, point2), 1)
        self.assertEqual(Scope(interval, cdy).offset(T.CENTURY, point1), 1)
        self.assertEqual(Scope(interval, cy).offset(T.CENTURY, point1), 1)
        self.assertEqual(Scope(interval, cyd).offset(T.CENTURY, point1), 1)
        self.assertEqual(Scope(interval, cdy).offset(T.CENTURY, point2), 1)
        self.assertEqual(Scope(interval, cy).offset(T.CENTURY, point2), 1)
        self.assertEqual(Scope(interval, cyd).offset(T.CENTURY, point2), 1)
        self.assertEqual(Scope(interval, cdy).offset(T.DECADE, point1), 8)
        self.assertEqual(Scope(interval, cdy).offset(T.DECADE, point2), 8)
        self.assertEqual(Scope(interval, cdy).offset(T.YEAR, point1), 4)
        self.assertEqual(Scope(interval, cy).offset(T.YEAR, point1), 84)
        self.assertEqual(Scope(interval, cyd).offset(T.YEAR, point1), 84)
        self.assertEqual(Scope(interval, cdy).offset(T.YEAR, point2), 4)
        self.assertEqual(Scope(interval, cy).offset(T.YEAR, point2), 84)
        self.assertEqual(Scope(interval, cyd).offset(T.YEAR, point2), 84)
        self.assertEqual(Scope(interval, y).offset(T.YEAR, point1), 94)
        self.assertEqual(Scope(interval, cyd).offset(T.DAY, point1), 0)
        self.assertEqual(Scope(interval, cyd).offset(T.DAY, point2), 32)
        self.assertEqual(Scope(interval, dm).offset(T.DECADE, point1), 9)
        self.assertEqual(Scope(interval, dm).offset(T.DECADE, point2), 9)
        self.assertEqual(Scope(interval, dm).offset(T.MONTH, point1), 48)
        self.assertEqual(Scope(interval, dm).offset(T.MONTH, point2), 49)
        self.assertEqual(Scope(interval, dm).offset(T.MONTH, point2), 49)
        self.assertEqual(Scope(interval, ymd).offset(T.YEAR, point2), 94)
        self.assertEqual(Scope(interval, ymd).offset(T.MONTH, point2), 1)
        self.assertEqual(Scope(interval, ymd).offset(T.DAY, point2), 2)
        self.assertEqual(Scope(interval, md).offset(T.YEAR, point2), None)
        self.assertEqual(Scope(interval, md).offset(T.MONTH, point2), 1120)
        self.assertEqual(Scope(interval, md).offset(T.DAY, point2), 2)

    def test_size(self):
        interval = TimeInterval.parse('M1890_10', 'd2010_4_5')

        self.assertEqual(Scope(interval, T.CENTURY).size(T.CENTURY), 3)
        self.assertEqual(Scope(interval, T.CENTURY).size(T.DECADE), 0)
        self.assertEqual(Scope(interval, T.CENTURY).size(T.YEAR), 0)
        self.assertEqual(Scope(interval, T.CENTURY).size(T.MONTH), 0)
        self.assertEqual(Scope(interval, T.CENTURY).size(), 3)

        self.assertEqual(Scope(interval, T.CENTURY|T.DECADE).size(T.CENTURY), 3)
        self.assertEqual(Scope(interval, T.CENTURY|T.DECADE).size(T.DECADE), 10)
        self.assertEqual(Scope(interval, T.CENTURY|T.DECADE).size(T.YEAR), 0)
        self.assertEqual(Scope(interval, T.CENTURY|T.DECADE).size(T.MONTH), 0)
        self.assertEqual(Scope(interval, T.CENTURY|T.DECADE).size(), 13)
        
        self.assertEqual(Scope(interval, T.CENTURY|T.YEAR).size(T.CENTURY), 3)
        self.assertEqual(Scope(interval, T.CENTURY|T.YEAR).size(T.DECADE), 0)
        self.assertEqual(Scope(interval, T.CENTURY|T.YEAR).size(T.YEAR), 100)
        self.assertEqual(Scope(interval, T.CENTURY|T.YEAR).size(T.MONTH), 0)
        self.assertEqual(Scope(interval, T.CENTURY|T.YEAR).size(), 103)
        
        self.assertEqual(Scope(interval, T.CENTURY|T.DECADE|T.YEAR).size(T.CENTURY), 3)
        self.assertEqual(Scope(interval, T.CENTURY|T.DECADE|T.YEAR).size(T.CENTURY|T.DECADE), 13)
        self.assertEqual(Scope(interval, T.CENTURY|T.DECADE|T.YEAR).size(T.DECADE), 10)
        self.assertEqual(Scope(interval, T.CENTURY|T.DECADE|T.YEAR).size(T.YEAR), 10)
        self.assertEqual(Scope(interval, T.CENTURY|T.DECADE|T.YEAR).size(T.MONTH), 0)
        self.assertEqual(Scope(interval, T.CENTURY|T.DECADE|T.YEAR).size(), 23)
        
        self.assertEqual(Scope(interval, T.CENTURY|T.MONTH|T.YEAR).size(T.CENTURY), 3)
        self.assertEqual(Scope(interval, T.CENTURY|T.MONTH|T.YEAR).size(T.DECADE), 0)
        self.assertEqual(Scope(interval, T.CENTURY|T.MONTH|T.YEAR).size(T.YEAR), 100)
        self.assertEqual(Scope(interval, T.CENTURY|T.MONTH|T.YEAR).size(T.MONTH), 12)
        self.assertEqual(Scope(interval, T.CENTURY|T.MONTH|T.YEAR).size(), 115)
        
        self.assertEqual(Scope(interval, T.DAY|T.MONTH|T.YEAR).size(T.CENTURY), 0)
        self.assertEqual(Scope(interval, T.DAY|T.MONTH|T.YEAR).size(T.DECADE), 0)
        self.assertEqual(Scope(interval, T.DAY|T.MONTH|T.YEAR).size(T.YEAR), 121)
        self.assertEqual(Scope(interval, T.DAY|T.MONTH|T.YEAR).size(T.MONTH), 12)
        self.assertEqual(Scope(interval, T.DAY|T.MONTH|T.YEAR).size(T.DAY), 30)
        self.assertEqual(Scope(interval, T.DAY|T.MONTH|T.YEAR).size(), 163)
        
        self.assertEqual(Scope(interval, T.DAY|T.DECADE).size(T.CENTURY), 0)
        self.assertEqual(Scope(interval, T.DAY|T.DECADE).size(T.DECADE), 13)
        self.assertEqual(Scope(interval, T.DAY|T.DECADE).size(T.YEAR), 0)
        self.assertEqual(Scope(interval, T.DAY|T.DECADE).size(T.MONTH), 0)
        self.assertEqual(Scope(interval, T.DAY|T.DECADE).size(T.DAY), 3600)
        self.assertEqual(Scope(interval, T.DAY|T.DECADE).size(), 3613)
        
        self.assertEqual(Scope(interval, T.DAY).size(T.CENTURY), 0)
        self.assertEqual(Scope(interval, T.DAY).size(T.DECADE), 0)
        self.assertEqual(Scope(interval, T.DAY).size(T.YEAR), 0)
        self.assertEqual(Scope(interval, T.DAY).size(T.MONTH), 0)
        self.assertEqual(Scope(interval, T.DAY).size(T.DAY), 43651)
        self.assertEqual(Scope(interval, T.DAY).size(), 43651)

    def test_vectorize(self):
        scope_interval = TimeInterval.parse('M1890_10', 'd2010_4_5')
        interval1 = TimeInterval(TimePoint(T.DAY, 1974, 9, 3), TimePoint(T.DAY, 1978, 2, 2))
        interval2 = TimeInterval(TimePoint(T.DAY, 1971, 10, 11), TimePoint(T.DAY, 1972, 10, 5))
        interval3 = TimeInterval(TimePoint(T.DAY, 1984, 8, 23), TimePoint(T.DAY, 2003, 2, 12))
        interval4 = TimeInterval(TimePoint(T.DAY, 1984, 2, 23), TimePoint(T.DAY, 1984, 8, 12))
        interval5 = TimeInterval(TimePoint(T.DAY, 1984, 2, 3), TimePoint(T.DAY, 1984, 8, 3))
        interval6 = TimeInterval(TimePoint(T.DAY, 1984, 2, 3), TimePoint(T.DAY, 1984, 2, 4))
        interval7 = TimeInterval(TimePoint(T.DAY, 1984, 2, 3), TimePoint(T.DAY, 1984, 3, 2))
        interval8 = TimeInterval(TimePoint(T.DAY, 1950, 4, 2), TimePoint(T.YEAR, 2008))
        interval9 = TimeInterval(TimePoint(T.YEAR, 1950), TimePoint(T.DAY, 2008, 1, 2))

        scope = Scope(scope_interval, T.CENTURY|T.DECADE|T.YEAR|T.MONTH|T.DAY)
        # print(interval1.vectorize(scope, T.CENTURY))
        # print(interval1.vectorize(scope, T.DECADE))
        # print(interval1.vectorize(scope, T.YEAR))
        # print(interval1.vectorize(scope, T.MONTH))
        # print(interval1.vectorize(scope, T.DAY))
        # print(interval1.vectorize(scope, T.CENTURY, flip=True))
        # print(interval1.vectorize(scope, T.DECADE, flip=True))
        # print(interval1.vectorize(scope, T.YEAR, flip=True))
        # print(interval1.vectorize(scope, T.MONTH, flip=True))
        # print(interval1.vectorize(scope, T.DAY, flip=True))

        # print(interval2.vectorize(scope, T.CENTURY))
        # print(interval2.vectorize(scope, T.DECADE))
        # print(interval2.vectorize(scope, T.YEAR))
        # print(interval2.vectorize(scope, T.MONTH))
        # print(interval2.vectorize(scope, T.DAY))
        # print(interval2.vectorize(scope, T.CENTURY, flip=True))
        # print(interval2.vectorize(scope, T.DECADE, flip=True))
        # print(interval2.vectorize(scope, T.YEAR, flip=True))
        # print(interval2.vectorize(scope, T.MONTH, flip=True))
        # print(interval2.vectorize(scope, T.DAY, flip=True))

        # print(scope.vectorize([interval1, interval2], T.CENTURY))
        # print(scope.vectorize([interval1, interval2], T.DECADE))
        # print(scope.vectorize([interval1, interval2], T.YEAR))
        # print(scope.vectorize([interval1, interval2], T.MONTH))
        # print(scope.vectorize([interval1, interval2], T.DAY))
        # print(scope.vectorize([interval1, interval2], T.CENTURY, flip=True))
        # print(scope.vectorize([interval1, interval2], T.DECADE, flip=True))
        # print(scope.vectorize([interval1, interval2], T.YEAR, flip=True))
        # print(scope.vectorize([interval1, interval2], T.MONTH, flip=True))
        # print(scope.vectorize([interval1, interval2], T.DAY, flip=True))

        # npt.assert_array_equal(scope.vectorize([interval1, interval2], T.YEAR), np.stack((
        #         [ 1. if i in {1, 11} or i >= 17 else 0. for i in range(23)],
        #         [ 1. if i in {2, 3} or i in range (13, 17) else 0. for i in range(23)]), axis=0))
        # npt.assert_array_equal(interval2.vectorize(scope, T.YEAR), np.stack((
        #         [ 1. if i in {1, 11} or i >= 17 else 0. for i in range(23)],
        #         [ 1. if i in {2, 3} or i in range (13, 17) else 0. for i in range(23)]), axis=0))

    def test_merge(self):
        scope_interval = TimeInterval.parse('M1890_10', 'd2010_4_5')
        interval1 = TimeInterval(TimePoint(T.DAY, 1978, 1, 3), TimePoint(T.DAY, 1978, 3, 8))
        interval2 = TimeInterval(TimePoint(T.DAY, 1978, 9, 11), TimePoint(T.DAY, 1978, 11, 15))
        scope = Scope(scope_interval, T.CENTURY|T.DECADE|T.YEAR|T.MONTH|T.DAY)
        npt.assert_array_equal(scope.vectorize([interval1, interval2], T.MONTH), np.array(
                [ 1. if i in {1, 10, 21, 23, 24, 25, 31, 32, 33} else 0. for i in range(35)]).reshape(1, -1))
        interval3 = TimeInterval.parse('d1978_6_3', 'd1978_6_5')
        interval4 = TimeInterval.parse('d1978_6_10', 'd1978_6_14')
        scope = Scope(scope_interval, T.CENTURY|T.DECADE|T.YEAR|T.MONTH|T.DAY)
        npt.assert_array_equal(scope.vectorize([interval3, interval4], T.DAY), np.array(
                [ 1. if i in {1, 10, 21, 28, 37, 38, 39, 44, 45, 46, 47, 48} else 0. for i in range(65)]).reshape(1, -1))

    def test_sample(self):
        interval = TimeInterval.parse('M1890_10', 'd2010_4_5')
        scope = Scope(interval, T.CENTURY|T.DECADE|T.YEAR|T.MONTH|T.DAY)
        # for s in scope.sample(T.MONTH, 100):
        #     print(s)
        # for s in scope.sample(T.DAY, 100):
        #     print(s)

    def test_jaccard(self):
        interval = TimeInterval.parse('M1890_10', 'd2010_4_5')
        scope = Scope(TimeInterval.parse('C1500', 'C2100'), T.CENTURY|T.DECADE|T.YEAR|T.MONTH|T.DAY)
        interval = TimeInterval.parse('M1890_10', 'd2010_4_5')
        self.assertEqual(scope.jaccard(interval, interval, T.CENTURY), 1.0)
        self.assertEqual(scope.jaccard(interval, interval, T.DECADE),  1.0)
        self.assertEqual(scope.jaccard(interval, interval, T.YEAR),    1.0)
        self.assertEqual(scope.jaccard(interval, interval, T.MONTH),   1.0)
        self.assertEqual(scope.jaccard(interval, interval, T.DAY),     1.0)

        interval2 = TimeInterval.parse('M1790_10', 'd1889_4_5')
        self.assertEqual(scope.jaccard(interval, interval2, T.CENTURY), .0)
        self.assertEqual(scope.jaccard(interval, interval2, T.DECADE),  .0)
        self.assertEqual(scope.jaccard(interval, interval2, T.YEAR),    .0)
        self.assertEqual(scope.jaccard(interval, interval2, T.MONTH),   .0)
        self.assertEqual(scope.jaccard(interval, interval2, T.DAY),     .0)

        interval3 = TimeInterval.parse('M1790_10', 'd1953_4_5')
        self.assertAlmostEqual(scope.jaccard(interval, interval3, T.CENTURY), .5,  places=3)
        self.assertAlmostEqual(scope.jaccard(interval, interval3, T.DECADE),  .304, places=3)
        self.assertAlmostEqual(scope.jaccard(interval, interval3, T.YEAR),    .29,  places=3)
        self.assertAlmostEqual(scope.jaccard(interval, interval3, T.MONTH),   .285, places=3)
        self.assertAlmostEqual(scope.jaccard(interval, interval3, T.DAY),     .285, places=3)

        self.assertAlmostEqual(scope.jaccard(interval3, interval, T.CENTURY), .5,   places=3)
        self.assertAlmostEqual(scope.jaccard(interval3, interval, T.DECADE),  .304, places=3)
        self.assertAlmostEqual(scope.jaccard(interval3, interval, T.YEAR),    .29,  places=3)
        self.assertAlmostEqual(scope.jaccard(interval3, interval, T.MONTH),   .285, places=3)
        self.assertAlmostEqual(scope.jaccard(interval3, interval, T.DAY),     .285, places=3)
 
        interval1 = TimeInterval.parse('d2014_1_1', 'd2014_1_1')
        interval2 = TimeInterval.parse('d2014_1_1', 'd2014_2_2')
        interval3 = TimeInterval.parse('d2014_2_2', 'M2014_4_3')
        scope = Scope(TimeInterval.parse('d2014_1_1', 'd2014_12_30'), T.CENTURY|T.DECADE|T.YEAR|T.MONTH|T.DAY)
        self.assertAlmostEqual(scope.jaccard(interval1, interval1, T.MONTH), 1., places=3)
        self.assertAlmostEqual(scope.jaccard(interval1, interval1, T.DAY), 1., places=3)
        self.assertAlmostEqual(scope.jaccard(interval2, interval2, T.MONTH), 1., places=3)
        self.assertAlmostEqual(scope.jaccard(interval2, interval2, T.DAY), 1., places=3)
        self.assertAlmostEqual(scope.jaccard(interval3, interval3, T.MONTH), 1., places=3)
        self.assertAlmostEqual(scope.jaccard(interval3, interval3, T.DAY), 1., places=3)
        self.assertAlmostEqual(scope.jaccard(interval1, interval2, T.MONTH), .5, places=3)
        self.assertAlmostEqual(scope.jaccard(interval2, interval3, T.MONTH), .25, places=3)
        self.assertAlmostEqual(scope.jaccard(interval1, interval2, T.DAY), .030, places=3)
        self.assertAlmostEqual(scope.jaccard(interval2, interval3, T.DAY), .011, places=3)

if __name__ == '__main__':
    unittest.main()
