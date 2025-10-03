import datetime
import calendar as cal


class Timestamp:

    def __init__(self, custom_epoch_str: str = None):
        self.DATETIME_FORMAT: str = '%d-%m-%Y %H:%M:%S'
        self._unix_offset_minutes = 0
        if custom_epoch_str:
            try:
                # Calculate the offset of the custom epoch from the true Unix epoch (in minutes)
                custom_epoch_dt = datetime.datetime.strptime(custom_epoch_str, self.DATETIME_FORMAT)
                # Use timegm for a UTC-based timestamp
                self._unix_offset_minutes = cal.timegm(custom_epoch_dt.timetuple()) / 60.0
            except ValueError:
                raise ValueError(f"Custom epoch format is incorrect, should be {self.DATETIME_FORMAT}")


    @property
    def as_unix(self):
        return self._unix - self._unix_offset_minutes

    @as_unix.setter
    def as_unix(self, timestamp):
        self._unix = timestamp
        try:
            self._datetime = [datetime.datetime.fromtimestamp(x*60,datetime.UTC) for x in self._unix]
            self._str = [x.strftime(self.DATETIME_FORMAT) for x in self._datetime]
        except:
            self._datetime = datetime.datetime.fromtimestamp(self._unix*60,datetime.UTC)
            self._str = self._datetime.strftime(self.DATETIME_FORMAT)

    @property
    def as_str(self):
        return self._str

    @as_str.setter
    def as_str(self, timestamp):
        self._str = timestamp
        try:
            self._datetime = [datetime.datetime.strptime(x, self.DATETIME_FORMAT) for x in timestamp]
            self._unix = [cal.timegm(datetime.datetime.strptime(x, self.DATETIME_FORMAT).timetuple()) / 60.0 for x in timestamp]
        except ValueError:
            raise ValueError("Incorrect date format, should be %d-%m-%Y %H:%M:%S or int [minutes]")
        except TypeError:
            self._datetime = datetime.datetime.strptime(timestamp, self.DATETIME_FORMAT)
            self._unix = cal.timegm(datetime.datetime.strptime(timestamp, self.DATETIME_FORMAT).timetuple()) / 60.0

    @property
    def as_datetime(self):
        return self._datetime

    @as_datetime.setter
    def as_datetime(self, timestamp):
        try:
            self._datetime = timestamp
            self._str = [x.strftime(self.DATETIME_FORMAT) for x in self._datetime]
            self._unix = [cal.timegm(datetime.datetime.strptime(x, self.DATETIME_FORMAT).timetuple()) / 60.0 for x in self._str]
        except ValueError:
            raise ValueError("Incorrect date format, should be %d-%m-%Y %H:%M:%S or int [minutes]")
        except TypeError:
            self._str = self._datetime.strftime(self.DATETIME_FORMAT)
            self._unix = cal.timegm(datetime.datetime.strptime(self._str, self.DATETIME_FORMAT).timetuple()) / 60.0