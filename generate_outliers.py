import datetime
import math
from datetime import timedelta

import numpy as np

from config import args
import json
import os


# Trajectory location offset
def perturb_point(point, level, offset=None):
    point_id, grid_loc, timestamp = point
    x, y = convert(grid_loc)

    if offset is None:
        offset = [[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [-1, -1], [-1, 1], [1, -1]]
        x_offset, y_offset = offset[np.random.randint(0, len(offset))]

    else:
        x_offset, y_offset = offset

    if 0 <= x + x_offset * level < lat_grid_num and 0 <= y + y_offset * level < lon_grid_num:
        x += x_offset * level
        y += y_offset * level
    return [point_id, int(x * lon_grid_num + y), timestamp]


def convert(grid_loc):
    x, y = int(grid_loc // lon_grid_num), int(grid_loc % lon_grid_num)
    return [x, y]


def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def time_calcuate(vec, s):
    a = datetime.datetime(vec[3], vec[4], vec[5], vec[0], vec[1], vec[2])
    t = a + timedelta(seconds=s)
    return [t.hour, t.minute, t.second, t.year, t.month, t.day]


# Trajectory time offset
def perturb_time(traj, st_loc, end_loc, time_offset, interval):
    for i in range(st_loc, end_loc):
        traj[i][2] = time_calcuate(traj[i][2], int((i - st_loc + 1) * time_offset * interval))

    for i in range(end_loc, len(traj)):
        traj[i][2] = time_calcuate(traj[i][2], int((end_loc - st_loc) * time_offset * interval))
    return traj


def perturb_batch(batch_x, level, prob, selected_idx):
    noisy_batch_x = []
    iterator = enumerate(batch_x)
    i = 0
    traj_acc = [next(iterator)[1]]

    for _, point in enumerate(batch_x):
        # point = [id, grid_num, [time]]
        if point[0] == traj_acc[0][0]:
            # replaces the traj id with a dense index rather than sparse
            traj_acc.append([i, point[1], point[2]])
            continue
        noisy_batch_x += create_anomaly(traj_acc, level, prob, selected_idx, i)

        traj_acc = [point]
        i += 1

    noisy_batch_x += create_anomaly(traj_acc, level, prob, selected_idx, i)

    return noisy_batch_x

def create_anomaly(traj, level, prob, selected_idx, idx):
    if idx in selected_idx:
        anomaly_len = int(len(traj) * prob)
        anomaly_start_location = np.random.randint(1, len(traj) - anomaly_len - 1)

        anomaly_end_location = anomaly_start_location + anomaly_len

        p_traj = (traj[:anomaly_start_location]
                  + [perturb_point(p, level) for p in traj[anomaly_start_location:anomaly_end_location]]
                  + traj[anomaly_end_location:])

        dist = max(distance(convert(traj[anomaly_start_location][0]), convert(traj[anomaly_end_location][0])), 1)
        time_offset = (level * 2) / dist

        p_traj = perturb_time(p_traj, anomaly_start_location, anomaly_end_location, time_offset, interval)
    else:
        p_traj = traj

    return p_traj[:int(len(p_traj) * args.obeserved_ratio)]


def generate_outliers(trajs, ratio=args.ratio, level=args.distance, point_prob=args.fraction):
    selected_idx = np.random.randint(0, traj_num, size=int(traj_num * ratio))
    new_trajs = perturb_batch(trajs, level, point_prob, selected_idx)
    return new_trajs, selected_idx


if __name__ == '__main__':
    np.random.seed(1234)
    print("=========================")
    print(f"Dataset: {args.dataset}")
    print(f"d = {args.distance}, {chr(945)} = {args.fraction}, {chr(961)} = {args.obeserved_ratio}")

    with open(f'./data/{args.dataset}/metadata.json', 'r') as f:
        (lat_grid_num, lon_grid_num, traj_num) = tuple(json.load(f))

    interval = 10
    if args.dataset == 'porto':
        interval = 15

    data = np.load(f"./data/{args.dataset}/test_init.npy", allow_pickle=True)
    outliers_trajs, outliers_idx = generate_outliers(data)
    outliers_trajs = np.array(outliers_trajs, dtype=object)
    outliers_idx = np.array(outliers_idx)

    np.save(f"./data/{args.dataset}/outliers_data_init_{args.distance}_{args.fraction}_{args.obeserved_ratio}.npy", outliers_trajs)
    np.save(f"./data/{args.dataset}/outliers_idx_init_{args.distance}_{args.fraction}_{args.obeserved_ratio}.npy", outliers_idx)


    files = os.listdir(f"./data/{args.dataset}/test/")
    i=0
    for file in files:
        data = np.load(f"./data/{args.dataset}/test/{i}.npy", allow_pickle=True)
        outliers_trajs, outliers_idx = generate_outliers(data)
        outliers_trajs = np.array(outliers_trajs, dtype=object)
        outliers_idx = np.array(outliers_idx)

        np.save(
            f"./data/{args.dataset}/outliers_data_{i}_{args.distance}_{args.fraction}_{args.obeserved_ratio}.npy", outliers_trajs)
        np.save(
            f"./data/{args.dataset}/outliers_idx_{i}_{args.distance}_{args.fraction}_{args.obeserved_ratio}.npy", outliers_idx)
        i+=1
