#!/usr/bin/python3
from enum import IntFlag
import random
import logging
import datetime
import numpy as np
import torch 

logger = logging.getLogger('gregorian')

class Granularities(IntFlag):
    '''
    This enum represents granularity levels for an idealized gregorian division of time,
    from centuries down to seconds.
    '''
    CENTURY = 1
    DECADE  = 2
    YEAR    = 4
    MONTH   = 8
    DAY     = 16
    HOUR    = 32
    MINUTE  = 64
    SECOND  = 128

    @staticmethod
    def for_abbrv(char):
        '''
        Returns granularity level corresponding to the given charactet
        '''
        return {
            "C" : Granularities.CENTURY,
            "D" : Granularities.DECADE,
            "Y" : Granularities.YEAR,
            "M" : Granularities.MONTH,
            "d" : Granularities.DAY,
            "h" : Granularities.HOUR,
            "m" : Granularities.MINUTE,
            "s" : Granularities.SECOND
        }[char]

    @staticmethod
    def to_abbrv(levels):
        '''
        Returns an abbreviation of a union of granularity levels
        '''
        result = ''
        for l in Granularities.each(levels):
            result += l.shortname()
        return result

    def shortname(self):
        '''
        Returns granularity level corresponding to the given charactet
        '''
        return {
            Granularities.CENTURY : "C",
            Granularities.DECADE  : "D",
            Granularities.YEAR    : "Y",
            Granularities.MONTH   : "M",
            Granularities.DAY     : "d",
            Granularities.HOUR    : "h",
            Granularities.MINUTE  : "m",
            Granularities.SECOND  : "s"
        }[self]

    def division(self):
        '''
        Returns the number of division of this level wrt to its direct parent
        '''
        return {
            Granularities.CENTURY : 1,
            Granularities.DECADE  : 10,
            Granularities.YEAR    : 10,
            Granularities.MONTH   : 12,
            Granularities.DAY     : 30, # We assume a perfect division of time
            Granularities.HOUR    : 24,
            Granularities.MINUTE  : 60,
            Granularities.SECOND  : 60
        }[self]

    def natural_min(self):
        if self in Granularities.MONTH|Granularities.DAY:
            return 1
        return 0

    @staticmethod
    def each(n):
        '''
        Generator for disjunctions of levels.
        '''
        while n:
            b = n & (~n+1)
            yield b
            n ^= b

    @staticmethod
    def highest(n):
        '''
        Returns the highest levels of a disjunction thereof (i.e. of lowest rank).
        '''
        for l in Granularities:
            if n&l:
                return l
        return None

    @staticmethod
    def lowest(n):
        '''
        Returns the lowest levels of a disjunction thereof (i.e. of lowest rank).
        '''
        for l in reversed(Granularities):
            if n&l:
                return l
        return None

    def next(self, levels=None):
        '''
        Returns the next level (ie of higher rank) within a disjunction, directly follows the current level.
        If the level is the lowest, returns None.
        '''
        found = False
        for l in (Granularities if not levels else Granularities.each(levels)):
            if found:
                return l
            if self&l:
                found = True
        return None

    def previous(self, levels=None):
        '''
        Returns the previous level (ie of lower rank) within a disjunction, directly preceeds the current level.
        If the level is the highest, returns None.
        '''
        last = None
        for l in (Granularities if not levels else Granularities.each(levels)):
            if self&l:
                return last
            else:
                last = l
        return None

    def prefix(self, levels=None):
        '''
        Returns the previous level (ie of lower rank) within a disjunction, directly preceeds the current level.
        If the level is the highest, returns None.
        '''
        result = 0
        for l in (Granularities if not levels else Granularities.each(levels)):
            if self&l:
                break 
            result |= l
        return result if result != 0 else None

T = Granularities

class TimePoint:
    '''
    Objects of this class represent a single point in time which may be of any granularity
    '''
    def __init__(self, granularity, year, month=None, day=None, 
                hour=None, minute=None, second=None):
        '''
        Initializes a time point of arbitrary precision.
        '''
        assert granularity in Granularities
        self.granularity = granularity
        self.century=int(year/100)
        self.decade=int(year/10)
        self.year = year
        self.month = max(1, min(month, 12)) if month else None
        self.day = max(1, min(day, 30)) if day else None
        self.hour = hour
        self.minute = minute
        self.second = second

    def __lt__(self, other):
        if not other:
            return False
        return self.before(other)

    def __gt__(self, other):
        if not other:
            return False
        return other.before(self)

    def __eq__(self, other):
        '''
        Two TimePoint are equals if they coincide on all values at all granularities
        '''
        result = self.century == other.century
        if self.granularity >= T.DECADE:
            result &= self.decade == other.decade
        if self.granularity >= T.YEAR:
            result &= self.year == other.year
        if self.granularity >= T.MONTH:
            result &= self.month == other.month
        if self.granularity >= T.DAY:
            result &= self.day == other.day
        if self.granularity >= T.HOUR:
            result &= self.hour == other.hour
        if self.granularity >= T.MINUTE:
            result &= self.minute == other.minute
        if self.granularity >= T.SECOND:
            result &= self.second == other.second
        return result
    
    def __hash__(self):
        return hash((self.granularity,
                     self.century,
                     self.decade if self.granularity >= T.DECADE else None,
                     self.year   if self.granularity >= T.YEAR   else None,
                     self.month  if self.granularity >= T.MONTH  else None,
                     self.day    if self.granularity >= T.DAY    else None,
                     self.hour   if self.granularity >= T.HOUR   else None,
                     self.minute if self.granularity >= T.MINUTE else None,
                     self.second if self.granularity >= T.SECOND else None))
    
    def __str__(self):
        if self.granularity == T.CENTURY:
            result = str(self.century) + "00"
        elif self.granularity == T.DECADE:
            result = str(self.decade) + "0"
        else:
            result = str(self.year)
        for i in [T.MONTH, T.DAY, T.HOUR, T.MINUTE, T.SECOND]:
            if self.granularity < i or not self.value(i):
                break
            result += '_' + str(self.value(i))
        return T.to_abbrv(self.granularity) + result

    def value(self, granularity):
        '''
        Returns the value of this TimePoint at the given granularity
        '''
        return {
            T.CENTURY : self.century,
            T.DECADE  : self.decade,
            T.YEAR    : self.year,
            T.MONTH   : self.month,
            T.DAY     : self.day,
            T.HOUR    : self.hour,
            T.MINUTE  : self.minute,
            T.SECOND  : self.second
        }[granularity]

    def todate(self):
        '''
        Converts this TimePoint to a date object. Any information of granularity finer than DAY is lost.
        '''
        assert self.month and self.day, "Month and day undefined for %s" % str(self)
        return datetime.date(self.year, self.month, self.day) 

    def todatetime(self):
        '''
        Converts this TimePoint to a datetime object.
        '''
        assert self.month and self.day and self.hour and self.minute and self.second, "Unable to convert to datetime (undefined fields):  %s" % str(self)
        return datetime.datetime(self.year, self.month, self.day, self.hour, self.minute, self.second)

    def within(self, interval, down_to=None):
        '''
        Returns True iff this TimePoint is within the given interval (the operation is inclusive on both interval bounds)
        '''
        return (interval.start.before(self, down_to=down_to) if interval.start else True
             ) and (self.before(interval.end, down_to=down_to) if interval.end else True)

    def before(self, other, down_to=None):
        '''
        Returns True if this TimePoint is before the given TimePoint or equal down to the given or lowest common granularity levels.
        '''
        if not other:
            return False
        if self.century > other.century:
            return False 
        if self.century < other.century or T.CENTURY in {down_to, self.granularity, other.granularity}:
            return True
        if self.decade > other.decade:
            return False
        if self.decade < other.decade or T.DECADE in {down_to, self.granularity, other.granularity}:
            return True
        if self.year > other.year:
            return False
        if self.year < other.year or T.YEAR in {down_to, self.granularity, other.granularity}:
            return True
        if self.month > other.month:
            return False
        if self.month < other.month or T.MONTH in {down_to, self.granularity, other.granularity}:
            return True
        if self.day > other.day:
            return False
        return True

    def after(self, other):
        '''
        Returns True if this TimePoint is after the given TimePoint or equal down to their lowest common granularity levels.
        '''
        return other == self or not self.before(other)

    def truncate(self, other, reverse=False):
        '''
        Returns a timepoint adjusted to be no later (resp. no earlier) than the other
        '''
        g = self.granularity
        x = self
        z = other
        if reverse:
            x = other
            z = self
        if x.century < z.century:
            return self
        y = int(other.year/100)*100
        if x.decade < z.decade:
            return self
        y = int(other.year/10)*10
        if x.year < z.year:
            return self
        y = other.year
        if not x.month or not z.month or (x.year == z.year and x.month <= z.month):
            return TimePoint(g, y, month=self.month, day=self.day, hour=self.hour, minute=self.minute, second=self.second)
        m = other.month
        if not x.day or not z.day or (x.month == z.month and x.day <= z.day):
            return TimePoint(g, y, month=m, day=self.day, hour=self.hour, minute=self.minute, second=self.second)
        d = other.day
        if not x.hour or not z.hour or (x.day == z.day and x.hour <= z.hour):
            return TimePoint(g, y, month=m, day=d, hour=self.hour, minute=self.minute, second=self.second)
        h = other.hour
        if not x.minute or not z.minute or (x.hour == z.hour and x.minute <= z.minute):
            return TimePoint(g, y, month=m, day=d, hour=h, minute=self.minute, second=self.second)
        n = other.minute
        if not x.second or not z.second or (x.minute == z.minute and x.second <= z.second):
            return TimePoint(g, y, month=m, day=d, hour=h, minute=n, second=self.second)
        s = other.second
        return TimePoint(g, y, month=m, day=d, hour=h, minute=n, second=s)

    def vectorize(self, scope, granularity=None, concat=True):
        '''
        The vector representation of the time point
        '''
        assert self.within(scope, down_to=granularity), "%s not within scope %s (%s)" % (self, scope, granularity)

        if granularity:
            value = self.value(granularity)
            if not value:
                return np.ones(scope.size(granularities=granularity))
            result = np.zeros(scope.size(granularities=granularity))
            result[scope.offset(granularity, self)]=1
            return result

        result = []
        for p in T.each(scope.levels):
            result.append(self.vectorize(scope, granularity=p))
        if concat:
            result = np.concatenate(result, axis=0)
        return result

    @staticmethod
    def extend(other, level, value):
        if not other:
            return TimePoint(level, value)
        result = TimePoint(level, other.year, month=other.month, day=other.day, hour=other.hour, minute=other.minute, second=other.second)
        if level == T.CENTURY:
            result.century = value
            result.decade = result.century*10
            result.year = result.century*100
        if level == T.DECADE:
            result.century = other.century
            result.decade = other.century*10+value
            result.year = result.decade*10
        if level == T.YEAR:
            result.century = other.century
            result.decade = other.decade
            result.year = other.decade*10+value
        if level == T.MONTH:
            result.month = value
        if level == T.DAY:
            result.day = value
        if level == T.HOUR:
            result.hour = value
        if level == T.MINUTE:
            result.minute = value
        if level == T.SECOND:
            result.second = value
        return result

    @staticmethod
    def parse(input_string, mode=None):
        '''
        Initializes a time point from its string representation.
        '''
        if str(mode).lower() == "yago":
            return parse_yago(input_string)
        if not input_string or input_string == "None":
            return None
        granularity = Granularities.for_abbrv(input_string[0:1])
        bits = input_string[1:].split("_")
        month, day, hour, minute, second = None, None, None, None, None
        try:
            year = int(bits[0])
        except ValueError:
            return None
        if len(bits) >= 2:
            try: month = int(bits[1])
            except ValueError: logger.warn("Unable to parse month ", bits[1])
        if len(bits) >= 3:
            try: day = int(bits[2])
            except ValueError: logger.warn("Unable to parse day ", bits[2])
        if len(bits) >= 4:
            try: hour = int(bits[3])
            except ValueError: logger.warn("Unable to parse hour ", bits[3])
        if len(bits) >= 5:
            try: minute = int(bits[4])
            except ValueError: logger.warn("Unable to parse minute ", bits[4])
        if len(bits) >= 6:
            try: second = int(bits[5])
            except ValueError: logger.warn("Unable to parse second ", bits[5])
        return TimePoint(granularity, year, month=month, day=day, hour=hour, minute=minute, second=second)

    @staticmethod
    def parse_yago(input_string):
        '''
        Initializes a time point from its string representation.
        '''
        if not input_string or input_string == "None" or input_string == "####-##-##":
            return None
        bits = input_string.split("-")
        month, day, hour, minute, second = None, None, None, None, None
        if bits[0].contains("##"):
            year = int(bits[0][:2])*100
            granularity = T.CENTURY
        elif bits[0].contains("#"):
            year = int(bits[0][:2])*10
            granularity = T.DECADE
        else:
            year = int(bits[0])
            granularity = T.YEAR
        if len(bits) >= 2:
            try:
                month = int(bits[1])
                granularity = T.MONTH
            except ValueError:
                return TimePoint(granularity, year)
        if len(bits) >= 3:
            try:
                day = int(bits[2])
                granularity = T.DAY
            except ValueError:
                return TimePoint(granularity, year, month=month)
        if len(bits) >= 4:
            try:
                hour = int(bits[3])
                granularity = T.HOUR
            except ValueError:
                return TimePoint(granularity, year, month=month, day=day)
        if len(bits) >= 5:
            try:
                minute = int(bits[4])
                granularity = T.MINUTE
            except ValueError:
                return TimePoint(granularity, year, month=month, day=day, hour=hour)
        if len(bits) >= 6:
            try:
                second = int(bits[5])
                granularity = T.SECOND
            except ValueError:
                pass
        return TimePoint(granularity, year, month=month, day=day, hour=hour, minute=minute, second=second)

class TimeInterval:
    '''
    Objects of this class represent an interval between twos points, or before/after a single time point
    '''
    def __init__(self, start:TimePoint=None, end:TimePoint=None):
        '''
        Initializes a time interval.
        '''
        if not (start or end):
            raise ValueError("Either the start or end of interval must be defined")
        if not start or not end or start.before(end):
            self.start = start
            self.end = end
        else:
            self.start = end
            self.end = start

    def __eq__(self, other):
        '''
        Two TimeIntervals are equal of their start and end point are equal
        '''
        return self.start == other.start and self.end == other.end

    def __hash__(self):
        return hash((self.start, self.end))

    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        return '[' + str(self.start) + '~' + str(self.end) + ']'

    def __lt__(self, other):
        if not self.start:
            return True
        if not other.start:
            return False
        return self.start.before(other.start)

    def contains(self, timepoint):
        result = True
        if self.start:
            result &= timepoint.after(self.start)
        if self.end:
            result &= timepoint.before(self.end)
        return result

    def overlaps(self, interval):
        if self.end and interval.start and self.end.before(interval.start):
            return False
        if self.start and interval.end and self.start.after(interval.end):
            return False
        return True
    
    def delta(self, granularity):
        '''
        Returns the time delta of this interval for the given granularity
        '''
        return {
            T.CENTURY : self.centuries(),
            T.DECADE  : self.decades(),
            T.YEAR    : self.years(),
            T.MONTH   : self.months(),
            T.DAY     : self.days()
            # ,
            # T.HOUR    : self.hours(),
            # T.MINUTE  : self.minutes(),
            # T.SECOND  : self.seconds()
        }[granularity]

    def centuries(self):
        '''
        The interval's delta in centuries
        '''
        return self.end.century - self.start.century

    def decades(self):
        '''
        The interval's delta in decades
        '''
        return self.end.decade - self.start.decade

    def years(self):
        '''
        The interval's delta in years
        '''
        return self.end.year - self.start.year

    def months(self):
        '''
        The interval's delta in months
        '''
        delta = self.end.year-self.start.year
        if delta < 0:
            # print(str(self.end.year) + " < " + str(self.start.year))
            self.start.year = self.end.year
            # print(str(self.end.year) + " < " + str(self.start.year))
            delta = 0
        assert delta >= 0, str(self.end.year) + " < " + str(self.start.year)
        sm = self.start.month if self.start.month else 1
        em = self.end.month   if self.end.month   else 12
        if delta == 0:
            return em - sm
        return (delta - 1) * 12 + em + 12 - sm

    def days(self):
        '''
        The interval's delta in months
        '''
        s = self.start
        e = self.end
        if not self.start.month:
            s = TimePoint.extend(s, T.MONTH, 1)
        if not self.start.day:
            s = TimePoint.extend(s, T.DAY, 1)
        if not self.end.month:
            e = TimePoint.extend(e, T.MONTH, 12)
        if not self.end.day:
            e = TimePoint.extend(e, T.DAY, 30)
        if s.month == 2 and s.day > 28:
            s = TimePoint.extend(s, T.DAY, min(28, s.day))
        if e.month == 2 and e.day > 28:
            e = TimePoint.extend(e, T.DAY, min(28, e.day))
        return (e.todate() - s.todate()).days

    def mean(self):
        '''
        Returns the mean timepoint of the current interval
        '''
        if not self.start:
            return self.end
        if not self.end :
            return self.start

        year = int((self.end.year + self.start.year)/2)

        month = None
        if self.start.month and self.end.month:
            s, e  = self.start.month, self.end.month
            if e < s:
                e += 12
            month = int(((e + s) / 2))

        day = None
        if self.start.day and self.end.day:
            s, e  = self.start.day, self.end.day
            if e < s:
                e += 31
            day = int(((e + s) / 2)%31)
            
        return TimePoint(min(self.start.granularity, self.end.granularity), year, month=month, day=day)

    def vectorize(self, scope, granularity, concat=True, flip=None):
        '''
        Returns the vector representation of the given interval, whose scope is the current instance
        '''
        clipped = scope.truncate(self)
        s=clipped.start.vectorize(scope, concat=False)
        e=clipped.end.vectorize(scope, concat=False)
        i, diff = 0, 0
        start, mid, end = [], [], []
        split = False
        for l in T.each(scope.levels):
            fr = np.where(s[i] == 1)[0][0]
            to = np.where(e[i] == 1)[0][0]
            diff = to-fr
            if l < granularity:
                a = s[i]
                if not clipped.start.value(l):
                    a = np.zeros_like(a)
                    a[0]=1
                start.append(a)
                a = e[i]
                if not clipped.end.value(l):
                    a = np.zeros_like(a)
                    a[-1]=1
                end.append(a)
            elif l == granularity:
                if not split and diff >= 0:
                    s[i][fr:to+1]=1
                    start.append(s[i] if not flip else (s[i]==0).astype(int))
                else:
                    s[i][fr:]=1
                    e[i][:to]=1
                    start.append(s[i] if not flip else (s[i]==0).astype(int))
                    end.append(e[i]   if not flip else (e[i]==0).astype(int))
                break
            split |= diff != 0
            i += 1
        if split:
            if concat:
                start = np.concatenate(start, axis=0)
                end = np.concatenate(end, axis=0)
                return np.stack((start, end), axis=0)
            return start, end
        if concat:
            return np.concatenate(start, axis=0).reshape(1, -1)
        return start, None

    @staticmethod
    def parse(start:str, end:str):
        '''
        Initializes a time input from the string representation of its start and end.
        '''
        s = TimePoint.parse(start) if start else None
        e = TimePoint.parse(end)   if end   else None
        return TimeInterval(s, e)

class Scope(TimeInterval):
    '''
    Objects of this class represent an overall time interval within which the model is build.
    '''
    def __init__(self, interval, levels):
        '''
        Initializes a scope.
        '''
        super(Scope, self).__init__(interval.start, interval.end)
        if not (interval.start and interval.end):
            raise ValueError("Scope interval must be closed")
        self.levels = levels

    def __eq__(self, other):
        '''
        Two TimeIntervals are equal of their start and end point are equal
        '''
        return self.start == other.start and self.end == other.end and self.levels == other.levels

    def __str__(self):
        return str(self.start) + ',' + str(self.end) + ',' + Granularities.to_abbrv(self.levels)

    def pretty(self):
        return '[' + str(self.start) + '~' + str(self.end) + ']@' + Granularities.to_abbrv(self.levels)

    def interval(self):
        return TimeInterval(self.start, self.end)

    def jaccard(self, interval1, interval2, level):
        i1 = self.truncate(interval1)
        i2 = self.truncate(interval2)
        if i2.start.before(i1.start):
            tmp = i1
            i1 = i2
            i2 = tmp
        if i1.end.before(i2.start) and i1.end != i2.start:
            return 0.0
        intersection = TimeInterval(i2.start, min(i1.end, i2.end)).delta(level) + 1
        union = TimeInterval(min(i1.start, i2.start), max(i1.end, i2.end)).delta(level) + 1
        assert intersection/max(1, union) <= 1.0
        return intersection/max(1, union)

    def vectorize(self, intervals, granularity, flip=None):
        '''
        Returns the vector representation of the given interval, whose scope is the current instance.
        '''
        merged = []
        for interval in intervals:
            s, e = interval.vectorize(self, granularity, concat=False)
            merged = self.merge(merged, s)
            if e:
                merged = self.merge(merged, e)
        results = []
        for result in merged:
            if flip:
                result[-1] = (result[-1] == 0).astype(int)
            results.append(np.concatenate(result, axis=0))
        return np.stack(results, axis=0)

    def merge(self, existing, new):
        '''
        Merges the "new" interval to the collection of "existing" intervals.
        '''
        for k in range(len(existing)):
            interval = existing[k]
            assert len(interval) == len(new), "Size mismatch " + str(interval) + " vs. " + str(new)
            for i, old in enumerate(interval):
                if i == len(new)-1:
                    existing[k][i] = np.maximum(interval[i], new[i])
                    return existing
                if (old != new[i]).any():
                    break
        existing.append(new)
        return existing

    def interpret(self, vector):
        "Returns an interval or time point as interpreted from the given vector"
        if vector.sum() == 0:
            return None
        offset1, offset2 = 0, 0
        prefix = self.start
        #prefix = None
        valueoffset = self.start.value(Granularities.highest(self.levels))
        for l in T.each(self.levels):
            offset2 += self.size(l)
            result = (vector[offset1:offset2] > .5).int()
            hits = int(result.sum().item())
            if hits == 0:
                break
            elif hits == 1:
                prefix = TimePoint.extend(prefix, l, valueoffset + result.argsort(descending=True)[0].item())
            else:
                spread = result.argsort(descending=True)[:hits].sort()[0]
                return TimeInterval(TimePoint.extend(prefix, l, int(valueoffset + spread[0].item())),
                                    TimePoint.extend(prefix, l, int(valueoffset + spread[-1].item())))
            offset1 = offset2
            nxt = l.next(levels=self.levels)
            if nxt:
                valueoffset = nxt.natural_min()
        return prefix

    def clip(self, start, end):
        '''
        Returns an interval whose bounds are restricted to those of the current scope
        '''
        s = start if start else self.start
        e = end if end else self.end
        if e.before(self.start):
            return TimeInterval(start=self.start, end=self.start)
        e = e.truncate(self.end)
        if s.after(self.end):
            return TimeInterval(start=self.end, end=self.end)
        s = s.truncate(self.start, reverse=True)
        return TimeInterval(start=s, end=e)

    def truncate(self, interval):
        '''
        Returns an interval whose bounds are restricted to those of the current scope
        '''
        return self.clip(interval.start, interval.end)

    def offset(self, granularity, timepoint):
        '''
        Returns the offset to which the timepoint value at the given granularity should be set 
        within a vector representation of that timepoint, with this TimeInterval as the scope.
        '''
        if not granularity&self.levels:
            return None

        prefix = granularity.prefix(self.levels)
        if not prefix:
            return TimeInterval(self.start, timepoint).delta(granularity)

        if prefix&T.MONTH and granularity == T.DAY:
            return min(29, timepoint.day-1)
        
        y = timepoint.year%100
        if T.DECADE == granularity:
            y = int(y / 10)
        elif T.DECADE&prefix:
            y = y%10
        if granularity <= T.YEAR:
            return y
        if T.YEAR&prefix:
            y = 0
            
        y *= T.MONTH.division()
        m = timepoint.month-1 if timepoint.month else 0
        if granularity == T.MONTH:
            return y+m
        d = timepoint.day-1   if timepoint.day else 0
        if not T.MONTH&prefix:
            d += m*T.DAY.division()
        y *= T.YEAR.division()
        if granularity == T.DAY:
            return min(y+d, self.size(T.DAY)-1)
        raise ValueError("Undefined offset")


    def size(self, granularities=None):
        '''
        Returns the vector size of levels, wrt to this TimeInterval as a scope.
        If granularity is specified, returns the size of the sub-vector for that granularity level.
        '''
        lv = self.levels
        if granularities:
            lv = self.levels&granularities
        if not granularities or bin(lv).count('1') > 1:
            result = 0
            for l in T.each(lv):
                result += self.size(granularities=l)
            return result
        
        if bin(lv).count('1') == 0:
            return 0

        if not granularities.previous(self.levels):
            return self.delta(granularities) + 1

        result, stage = granularities.division(), granularities
        while True:
            stage = stage.previous()
            if not stage or stage&self.levels:
                break
            result *= Granularities(stage).division()
        return result


    def prefix_size(self, granularity):
        '''
        Returns the vector size of levels, wrt to this TimeInterval as a scope.
        If granularity is specified, returns the size of the sub-vector for that granularity level.
        '''
        g = granularity&self.levels
        assert bin(g).count('1') > 0, "Invalid granularity " + str(granularity)
        prefix = granularity.prefix(self.levels)
        result = 0
        for l in T.each(prefix):
            result += self.size(granularities=l)
        return result

    def midpoints(self, phase, n):
        result = set()
        start = self.start
        for i in range(n):
            inter = TimeInterval(start=start, end=self.end)
            start = inter.mean()
            result.add(TimePoint(phase, start.year, month=start.month, day=start.day,
                    hour=start.hour, minute=start.minute, second=start.second))
        return result

    def sample(self, phase, n):
        result = set()
        for i in range(n*2):
            pick = self.pick(phase)
            if pick:
                result.add(pick)
            if len(result) == n:
                break
        return result

    def pick(self, phase, max_tries=10, raw=False):
        for n in range(max_tries):
            result = []
            for l in T.each(self.levels):
                sz = self.size(granularities=l)
                vec = np.zeros(sz)
                vec[random.randint(0, sz-1)] = 1.
                result.append(vec)
                if l == phase:
                    candidate = self.interpret(torch.from_numpy(np.concatenate(result, axis=0)))
                    if self.contains(candidate):
                        if raw:
                            return torch.from_numpy(np.concatenate(result, axis=0))
                        else:
                            return candidate
                    break
        return None
