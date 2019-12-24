import unittest
from skyfield.api import load
from pprint import pprint
import numpy as np
import argparse

# Globals are bad, never use them
planets = load('de421.bsp')  # pretty good precision, small; de430t.bsp provides high precision but is huge
ts = load.timescale()

# https://nssdc.gsfc.nasa.gov/planetary/factsheet/
# these are the bodies contained in most spice kernels, particularly de421 and de430t
orbital_days = {'sun': { 'earth': 365,
                         'mercury': 88,
                         'venus': 224.7,
                         'mars': 687,
                         'jupiter': 4331,
                         'saturn': 10747,
                         'uranus': 30589,
                         'neptune': 59800,
                         'pluto': 90560},
                'earth': {'moon': 27.3}
                }


def build_time_scale(date_start, granularity, daymax=187):
    '''
    builds a timescale for use by Skyfield based on
    :param date_start: day to start the scale
    :param granularity: granularity to use (day, hour, minute, second)
    :param ts:
    :return:
    '''
    # build time scale of given granularity
    date = date_start.tt_calendar()
    # print(f"building {granularity} granularity around {date_start.utc_iso()}")
    if granularity is 'day':
        jd = ts.utc(date[0], date[1], range(date[2], date[2] + round(daymax)))
    elif granularity is 'hour':
        jd = ts.utc(date[0], date[1], date[2], range(date[3] - 24, date[3] + 24))  # bracket at 2 day resolution
    elif granularity is 'minute':
        jd = ts.utc(date[0], date[1], date[2], date[3], range(date[4] - 60, date[4] + 60))  # 2 hour resolution
    elif granularity is 'second':
        jd = ts.utc(date[0], date[1], date[2], date[3], date[4],
                    range(int(date[5] - 60), int(date[5] + 60)))  # 2 minute resultion
        granularity = 'Done'
    else:
        raise NameError('undefined granularity %s' % granularity)
    return jd


def rerun(f, center_date, body1, body2, previous_granularity):
    # call recursively with next level of time granularity
    # print(f" improving {center_date.utc_iso()} to {granularity} ")
    if previous_granularity is 'day':
        result = next_apsis(f, center_date, body1, body2, granularity='hour')
    elif previous_granularity is 'hour':
        result = next_apsis(f, center_date, body1, body2, granularity='minute')
    elif previous_granularity is 'minute':
        result = next_apsis(f, center_date, body1, body2, granularity='second')
    else:
        raise NameError('undefined granularity %s' % previous_granularity)
    return result

def next_apsis(f, date_start, body1, body2, granularity='day', daymax=187):
    jd = build_time_scale(date_start, granularity, daymax=daymax)
    pos = planets[body1].at(jd) - planets[body2].at(jd)
    d = pos.distance().km
    i = f(d)

    if granularity == 'second':
        if f is np.ndarray.argmax:
            desc = 'apo'
        else:
            desc = 'peri'
        if body2.lower()=='earth':
            desc+='gee'
        elif body2.lower()=='sun':
            desc+='helion'
        result = {'date': jd[i].utc_datetime(),
                  'distance_km': round(d[i]),
                  'description': desc.replace('apoh', 'aph'),
                  }
    else:
        result = rerun(f, jd[i], body1, body2, granularity)
    return result


def next_apsides(date_start, body1='earth', body2='sun'):
    '''
    Calculates the date and distance of the nearest and furthest points in the orbit of body1 around body2
    :date_start: date to begin search for next apsis points
    :param body1: first body, defaults to Earth
    :param body2: second body, defaults to the Sun
    :return:
    '''
    # calculate max period in days to evaluate, half
    if body2.lower() in orbital_days.keys():
        if body1.lower() in orbital_days[body2.lower()].keys():
            daymax =orbital_days[body2.lower()][body1.lower()]
    else:
        # default to an Earth solar year
        daymax = 187

    result = {'body1': body1,
              'body2': body2,
              'furthest': next_apsis(np.ndarray.argmax, date_start, body1, body2, daymax=daymax),
              'nearest': next_apsis(np.ndarray.argmin, date_start, body1, body2, daymax=daymax),
              # 'furthest_description': "ap%s" % postfix
              }
    result['delta_km'] = result['furthest']['distance_km'] - result['nearest']['distance_km']
    return result


class Test(unittest.TestCase):

    def testEarthSun(self):
        ts = load.timescale()
        uut = next_apsides(ts.utc(2019, 5, 1), body1='Earth', body2='Sun')
        # matching up values with the US Naval Observatory
        self.assertEqual(7, uut['furthest']['date'].month)
        self.assertEqual(4, uut['furthest']['date'].day)
        self.assertEqual(1, uut['nearest']['date'].month)
        self.assertEqual(5, uut['nearest']['date'].day)

    def testEarthMoon(self):
        ts = load.timescale()
        uut = next_apsides(ts.utc(2019, 12, 25), body1='Moon', body2='Earth')
        pprint(uut)
        self.assertEqual(1, uut['furthest']['date'].month)
        self.assertEqual(2, uut['furthest']['date'].day)
        self.assertEqual(1, uut['nearest']['date'].month)
        self.assertEqual(13, uut['nearest']['date'].day)



if __name__ == '__main__':
    unittest.main()
    uut = next_apsides(ts.now(), body1='Earth', body2='Sun')
