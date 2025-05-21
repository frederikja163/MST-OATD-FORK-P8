import os

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from geopy import distance
from pandas.errors import EmptyDataError
import logging
import json
from config import args
from sklearn.model_selection import train_test_split
from functools import partial
from multiprocessing import Pool, Manager

# Determine whether a point is in boundary
def in_boundary(lat, lng, b):
    return b['min_lon'] < lng < b['max_lon'] and b['min_lat'] < lat < b['max_lat']


def ensure_size(trajectory, longest, shortest, grid_size, boundary, convert_date, invalid_count, traj_count):
    # Unpack
    lat_size, lon_size, _, lon_grid_num = grid_size

    trajectories = []
    i = 0
    while len(trajectory) > longest:
        # Cut long trajectories
        random_length = np.random.randint(shortest, longest)
        for point in trajectory[:random_length]:
            grid_i = int((point[1] - boundary['min_lat']) / lat_size)
            grid_j = int((point[2] - boundary['min_lon']) / lon_size)
            # 2^6 = 64, assuming there will never be a need to cut the trajectory more than 64 times
            trajectories.append([(point[0] << 6) + i, grid_i * lon_grid_num + grid_j, convert_date(point[3])])
        trajectory = trajectory[random_length:]
        i += 1
    traj_count += i
    if len(trajectory) >= shortest:
        for point in trajectory:
            grid_i = int((point[1] - boundary['min_lat']) / lat_size)
            grid_j = int((point[2] - boundary['min_lon']) / lon_size)
            trajectories.append([(point[0] << 6) + i, grid_i * lon_grid_num + grid_j, convert_date(point[3])])
        traj_count += 1
    else:
        invalid_count += len(trajectory)
    return trajectories, invalid_count


# grid map based lat, lon boundaries and a grid_size in km
def grid_mapping(boundary, grid_size):
    lat_dist = distance.distance((boundary['min_lat'], boundary['min_lon']),
                                 (boundary['max_lat'], boundary['min_lon'])).km
    lat_size = (boundary['max_lat'] - boundary['min_lat']) / lat_dist * grid_size

    lng_dist = distance.distance((boundary['min_lat'], boundary['min_lon']),
                                 (boundary['min_lat'], boundary['max_lon'])).km
    lng_size = (boundary['max_lon'] - boundary['min_lon']) / lng_dist * grid_size

    lat_grid_num = int(lat_dist / grid_size) + 1
    lng_grid_num = int(lng_dist / grid_size) + 1
    return lat_size, lng_size, lat_grid_num, lng_grid_num


# Generate adjacency matrix and normalized degree matrix
def generate_matrix(lat_grid_num, lng_grid_num):
    g = nx.grid_2d_graph(lat_grid_num, lng_grid_num, periodic=False)
    a = nx.adjacency_matrix(g)
    i = sparse.identity(lat_grid_num * lng_grid_num)
    d = np.diag(np.sum(a + i, axis=1))
    d = 1 / (np.sqrt(d) + 1e-10)
    d[d == 1e10] = 0.
    d = sparse.csr_matrix(d)
    return a + i, d

def create_grid(boundary, logger):
    lat_size, lon_size, lat_grid_num, lon_grid_num = grid_mapping(boundary, args.grid_size)
    logger.info(f'Grid count: {(lat_grid_num, lon_grid_num)}')

    a, d = generate_matrix(lat_grid_num, lon_grid_num)
    sparse.save_npz(f'../data/{args.dataset}/adj.npz', a)
    sparse.save_npz(f'../data/{args.dataset}/d_norm.npz', d)

    return lat_size, lon_size, lat_grid_num, lon_grid_num

def preprocess(file, shortest, longest, boundary, convert_date,
               timestamp_gap, grid_size, traj_nums, point_nums, invalid_points, columns):

    try:
        data = pd.read_csv(f"../datasets/{args.dataset}/{file}", header=None, iterator=True, names=columns, chunksize=args.chunk_size)
    except EmptyDataError:
        return
    filename = os.path.splitext(file)[0]

    invalid_count = 0
    traj_count = 0

    trajectory = []
    valid = True
    pre_point = None

    preprocessed_points = []
    for chunk in data:
        for point in chunk.itertuples():
            if not pre_point or (point.id == pre_point.id and timestamp_gap(pre_point.timestamp, point.timestamp) <= args.max_traj_time_delta):
                if in_boundary(point.lat, point.lon, boundary):
                    trajectory.append([point.id, point.lat, point.lon, point.timestamp])
                else:
                    valid = False
                pre_point = point
            else:
                if valid:
                    (trajs, traj_count) = ensure_size(trajectory, longest, shortest, grid_size, boundary, convert_date,
                                                      invalid_count, traj_count)
                    preprocessed_points += trajs
                else:
                    invalid_count += len(trajectory)
                trajectory = []
                valid = True
                pre_point = None

    if valid:
        (trajs, traj_count) = ensure_size(trajectory, longest, shortest, grid_size, boundary, convert_date, invalid_count, traj_count)
        preprocessed_points += trajs
    else:
        invalid_count += len(trajectory)

    # Skip empty files
    if len(preprocessed_points) <= 0:
        return

    # At the end of your processing loop, before saving:
    trajectories_array = np.array(preprocessed_points, dtype=list)

    traj_nums.append(traj_count)
    invalid_points.append(invalid_count)
    point_nums.append(len(preprocessed_points))

    np.save(f"../data/{args.dataset}/data_{filename}.npy", trajectories_array)

def merge(files, outfile):
    trajectories = []

    for file in files:
        try:
            filename = os.path.splitext(file)[0]
            file_trajectories = np.load(f"../data/{args.dataset}/data_{filename}.npy", allow_pickle=True)
            trajectories.append(file_trajectories)
        except FileNotFoundError: # empty files are skipped
            continue

    merged_trajectories = np.concatenate(trajectories, axis=0)
    np.save(f"../data/{args.dataset}/{outfile}", merged_trajectories)

# merges 80% of the files into train_init.npy and 20% into test_init.npy
def split_and_merge_files(files, logger):
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
    logger.debug('Merging train trajectories')
    merge(test_files, "train_init")
    logging.debug('Merging test trajectories')
    merge(train_files, "test_init")

    logging.info('Finished!')



def multiprocess(logger, shortest, longest, boundary, convert_date, timestamp_gap, columns, grid_size):
    logger.info(f'Preprocessing {args.dataset}')
    files = os.listdir(f"../datasets/{args.dataset}")
    manager = Manager()
    traj_nums = manager.list()
    point_nums = manager.list()
    invalid_points = manager.list()

    pool = Pool(args.processes)
    pool.map(partial(preprocess, shortest=shortest, longest=longest, boundary=boundary, convert_date=convert_date,
                     timestamp_gap=timestamp_gap, traj_nums=traj_nums, point_nums=point_nums,
                     invalid_points=invalid_points, columns=columns, grid_size=grid_size), files)

    pool.close()
    pool.join()

    traj_sum = sum(traj_nums)

    logger.info(f"Total valid trajectories: {traj_sum}")
    logger.info(f"Total valid points: {sum(point_nums)}")
    logger.info(f"Total invalid points: {sum(invalid_points)}")

    lat_size, lon_size, lat_grid_num, lon_grid_num = grid_size
    with open(f'../data/{args.dataset}/metadata.json', 'w') as f:
        json.dump(list((lat_grid_num, lon_grid_num, traj_sum)), f)

    split_and_merge_files(files, logger)

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def main():
    traj_path = f"../datasets/{args.dataset}"
    min_lat = [float("inf")]
    max_lat = [-float("inf")]

    min_lon = [float("inf")]
    max_lon = [-float("inf")]

    boundary = {'min_lat': 0.1, 'max_lat': 100, 'min_lon': 0.1, 'max_lon': 250}

    path_list = os.listdir(traj_path)
    a = len(path_list)
    x = 0
    lat_maxima_min = []
    lat_maxima_max = []
    lon_maxima_min = []
    lon_maxima_max = []
    for path in path_list:
        print(f"{x}/{a}")
        x = x+1
        try:
            data = pd.read_csv("{}/{}".format(traj_path, path), header=None)
        except EmptyDataError:
            continue
        data.columns = ['id', 'timestamp', 'lon', 'lat']

        for point in data.itertuples():
            if not in_boundary(point.lat, point.lon, boundary):
                break
        else:
            if min_lat > data['lat'].min():
                min_lat = data['lat'].min()
                lat_maxima_min.append(f"{path}: {data['lat'].min()}")
            if max_lat < data['lat'].max():
                max_lat = data['lat'].max()
                lat_maxima_max.append(f"{path}: {data['lat'].max()}")

            if min_lon > data['lon'].min():
                min_lon = data['lon'].min()
                lon_maxima_min.append(f"{path}: {data['lon'].min()}")

            if max_lon < data['lon'].max():
                max_lon = data['lon'].max()
                lon_maxima_max.append(f"{path}: {data['lon'].max()}")

    lat_maxima_min.reverse()
    lat_maxima_max.reverse()
    lon_maxima_min.reverse()
    lon_maxima_max.reverse()
    print(f"lat min: {min_lat}, {lat_maxima_min}")
    print(f"lat max: {max_lat}, {lat_maxima_max}")
    print(f"lon min: {min_lon}, {lon_maxima_min}")
    print(f"lon max: {max_lon}, {lon_maxima_max}")
