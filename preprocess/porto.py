import datetime
import json
import random

from utils import *
from config import args

def preprocess(trajectories, shortest, longest, grid_size, boundary, invalid_points):
    processed_trajectories = []

    traj_num, point_num = 0, 0

    (lat_size, lon_size, lon_grid_num) = grid_size

    for traj in trajectories.itertuples():

        traj_seq = []
        valid = True  # Flag to determine whether a trajectory is in boundary

        polyline = json.loads(traj.POLYLINE)
        timestamp = traj.TIMESTAMP

        if len(polyline) < shortest:
            invalid_points += len(polyline)
            continue

        for lng, lat in polyline:

            if in_boundary(lat, lng, boundary):
                grid_i = int((lat - boundary['min_lat']) / lat_size)
                grid_j = int((lng - boundary['min_lon']) / lon_size)

                t = datetime.datetime.fromtimestamp(timestamp)
                t = [t.hour, t.minute, t.second, t.year, t.month, t.day]  # Time vector

                traj_seq.append([int(grid_i * lon_grid_num + grid_j), t])
                timestamp += 15  # In porto dataset, the sampling rate is 15

            else:
                invalid_points += len(polyline)
                valid = False
                break

        # Randomly delete 30% trajectory points to make the sampling rate not fixed
        to_delete = set(random.sample(range(len(traj_seq)), int(len(traj_seq) * 0.3)))
        traj_seq = [item for index, item in enumerate(traj_seq) if index not in to_delete]

        # Lengths are limited from 20 to 50
        if valid:
            if len(traj_seq) <= longest:
                processed_trajectories.append(traj_seq)
            else:
                processed_trajectories += ensure_size(traj_seq, longest, shortest)
        else:
            invalid_points += len(polyline)

    traj_num += len(processed_trajectories)

    for traj in processed_trajectories:
        point_num += len(traj)

    return processed_trajectories, traj_num, point_num

def time_convert(timestamp):
    return datetime.datetime.fromtimestamp(timestamp)

def main():
    boundary = {'min_lat': 41.140092, 'max_lat': 41.185969, 'min_lon': -8.690261, 'max_lon': -8.549155}
    shortest, longest = 20, 50

    logger = get_logger(f"../logs/{args.dataset}.log")
    logger.info(f'Preprocessing {args.dataset}')

    grid_size = create_grid(boundary, logger)

    invalid_points = 0

    # Read csv file
    file = "porto.csv"
    trajectories = pd.read_csv(f"../datasets/{args.dataset}/{file}", header=0, usecols=['POLYLINE', 'TIMESTAMP'])
    trajectories['datetime'] = trajectories['TIMESTAMP'].apply(time_convert)

    # Initial dataset
    preprocessed_trajectories, traj_num, point_num = preprocess(trajectories, shortest, longest, grid_size, boundary)

    np.save(f"../data/{args.dataset}/preprocessed_data.npy", np.array(preprocessed_trajectories, dtype=object))

    # Dataset statistics
    logger.info(f"Total trajectory num: {traj_num:n}")
    logger.info(f"Total point num: {point_num:n}")
    logger.info(f"Invalid points: {invalid_points:n}")

    split_files_for_evolving(logger, f"../data/{args.dataset}/preprocessed_data.npy")
    logger.info('Finished!')
