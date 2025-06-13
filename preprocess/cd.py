import datetime
import time

from utils import *
from config import args

# convert datetime to time vector
def convert_date(date_str):
    time_array = time.strptime(date_str, "%Y/%m/%d %H:%M:%S")
    return [time_array.tm_hour, time_array.tm_min, time_array.tm_sec, time_array.tm_year, time_array.tm_mon, time_array.tm_mday]


# Calculate timestamp gap
def timestamp_gap(str1, str2):
    timestamp1 = datetime.datetime.strptime(str1, "%Y/%m/%d %H:%M:%S")
    timestamp2 = datetime.datetime.strptime(str2, "%Y/%m/%d %H:%M:%S")
    return (timestamp2 - timestamp1).total_seconds()

def main():
    logger = get_logger(f"../logs/{args.dataset}.log")
    boundary = {'min_lat': 30.6, 'max_lat': 30.75, 'min_lon': 104, 'max_lon': 104.16}
    columns = ['id', 'lat', 'lon', 'state', 'timestamp']
    grid_size = create_grid(boundary, logger)
    shortest = 30
    longest = 100

    multiprocess(logger, shortest, longest, boundary, convert_date, timestamp_gap, columns, grid_size)
