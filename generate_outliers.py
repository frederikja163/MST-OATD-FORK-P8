import datetime
import math
import os
from datetime import timedelta

import numpy as np

from config import args


# Trajectory location offset
def perturb_point(point, level, offset=None):
    point, times = point[0], point[1]
    x, y = int(point // map_size[1]), int(point % map_size[1])

    if offset is None:
        offset = [[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [-1, -1], [-1, 1], [1, -1]]
        x_offset, y_offset = offset[np.random.randint(0, len(offset))]

    else:
        x_offset, y_offset = offset

    if 0 <= x + x_offset * level < map_size[0] and 0 <= y + y_offset * level < map_size[1]:
        x += x_offset * level
        y += y_offset * level

    return [int(x * map_size[1] + y), times]


def convert(point):
    x, y = int(point // map_size[1]), int(point % map_size[1])
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
        traj[i][1] = time_calcuate(traj[i][1], int((i - st_loc + 1) * time_offset * interval))

    for i in range(end_loc, len(traj)):
        traj[i][1] = time_calcuate(traj[i][1], int((end_loc - st_loc) * time_offset * interval))
    return traj


def perturb_batch(batch_x, level, prob, selected_idx):
    noisy_batch_x = []

    if args.dataset == 'porto':
        interval = 15
    else:
        interval = 10

    for idx, traj in enumerate(batch_x):

        anomaly_len = int(len(traj) * prob)
        anomaly_st_loc = np.random.randint(1, len(traj) - anomaly_len - 1)

        if idx in selected_idx:
            anomaly_ed_loc = anomaly_st_loc + anomaly_len

            p_traj = traj[:anomaly_st_loc] + [perturb_point(p, level) for p in
                                              traj[anomaly_st_loc:anomaly_ed_loc]] + traj[anomaly_ed_loc:]

            dis = max(distance(convert(traj[anomaly_st_loc][0]), convert(traj[anomaly_ed_loc][0])), 1)
            time_offset = (level * 2) / dis

            p_traj = perturb_time(p_traj, anomaly_st_loc, anomaly_ed_loc, time_offset, interval)

        else:
            p_traj = traj

        p_traj = p_traj[:int(len(p_traj) * args.obeserved_ratio)]
        noisy_batch_x.append(p_traj)

    return noisy_batch_x


def generate_outliers(trajs, ratio=args.ratio, level=args.distance, point_prob=args.fraction):
    traj_num = len(trajs)
    selected_idx = np.random.randint(0, traj_num, size=int(traj_num * ratio))
    new_trajs = perturb_batch(trajs, level, point_prob, selected_idx)
    return new_trajs, selected_idx


if __name__ == '__main__':
    np.random.seed(1234)
    print("=========================")
    print(f"Dataset: {args.dataset}")
    print(f"d = {args.distance}, {chr(945)} = {args.fraction}, {chr(961)} = {args.obeserved_ratio}")

    if args.dataset == 'porto':
        map_size = (51, 119)
    elif args.dataset == 'cd':
        map_size = (167, 154)

    data = np.load(f"./data/{args.dataset}/test_init.npy", allow_pickle=True)
    outliers_trajs, outliers_idx = generate_outliers(data)
    outliers_trajs = np.array(outliers_trajs, dtype=object)
    outliers_idx = np.array(outliers_idx)

    np.save(f"./data/{args.dataset}/outliers_data_init_{args.distance}_{args.fraction}_{args.obeserved_ratio}.npy", outliers_trajs)
    np.save(f"./data/{args.dataset}/outliers_idx_init_{args.distance}_{args.fraction}_{args.obeserved_ratio}.npy", outliers_idx)
    ## data saved here, so ought to be simple to just split then save seperately, though it is important that the outlier idx is split correctly so the order is preserved



    if args.dataset == 'porto':
        for i in range(1, 11):
            data = np.load(f"./data/{args.dataset}/test_data_{i}.npy", allow_pickle=True)
            outliers_trajs, outliers_idx = generate_outliers(data)
            outliers_trajs = np.array(outliers_trajs, dtype=object)
            outliers_idx = np.array(outliers_idx)

            np.save(
                f"./data/{args.dataset}/outliers_data_{i}_{args.distance}_{args.fraction}_{args.obeserved_ratio}.npy", outliers_trajs)
            np.save(
                f"./data/{args.dataset}/outliers_idx_{i}_{args.distance}_{args.fraction}_{args.obeserved_ratio}.npy", outliers_idx)
