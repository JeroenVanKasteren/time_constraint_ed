import argparse
import numpy as np
from datetime import datetime
from time import perf_counter as clock, sleep, strptime
import os
import pandas as pd

FILEPATH = 'insights/results.csv'
MAX_TARGET_PROB = 0.9

def load_args(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id')  # SULRM_JOBID
    parser.add_argument('--array_id')  # SLURM_ARRAY_TASK_ID
    parser.add_argument('--time')  # User input
    parser.add_argument('--method', type=str, default='VI')  # VI or OSPI
    args, unknown = parser.parse_args(raw_args)
    return args

def main(raw_args=None):
    args = load_args(raw_args)
    print('Something')
    print(args.time)
    df = pd.DataFrame({'C1': [1], 'C2': args.time})
    df.to_csv('results/results.csv', mode='w')

    sleep(5)
    print('Woke up')
    df = pd.DataFrame({'C1': [2], 'C2': args.time})
    df.to_csv('results/results.csv', mode='a', headers=False)

if __name__ == '__main__':
    main()
