import datetime
import multiprocessing
from typing import List
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import tempfile
import os
from abc import abstractmethod
import time
from joblib import load, dump, _memmapping_reducer

# from .features.td_features import TdFeatures
# from ..stats.common import get_stats
# from ..util import processing

SYSTEM_SHARED_MEM_FS = './cache_folder'
SYSTEM_SHARED_MEM_FS_MIN_SIZE = int(2e9)
def memmap_auto(data, callable, **args):
    memmapped_data, memmap_filename = memmap_data(data)
    del data
    result = callable(memmapped_data, **args)
    memmap_unlink(memmap_filename)
    return result
def memmap_data(data, read_only: bool = True):
    new_folder_name = ("flirt_memmap_%d" % os.getpid())
    temp_dir, _ = __get_temp_dir(new_folder_name)
    filename = os.path.join(temp_dir, 'memmap_%s.mmap' % next(tempfile._RandomNameSequence()))
    if os.path.exists(filename):
        os.unlink(filename)
    _ = dump(data, filename)
    return load(filename, mmap_mode='r+' if read_only else 'w+'), filename
def memmap_unlink(filename):
    if os.path.exists(filename):
        NUM_RETRIES = 10
        for retry_no in range(1, NUM_RETRIES + 1):
            try:
                os.unlink(filename)
                break
            except PermissionError:
                if retry_no == NUM_RETRIES:
                    print("Unable to remove memmapped file")
                    break
                else:
                    time.sleep(.2)
def __get_temp_dir(pool_folder_name, temp_folder=None):
    use_shared_mem = False
    if temp_folder is None:
        if os.path.exists(SYSTEM_SHARED_MEM_FS):
            try:
                shm_stats = os.statvfs(SYSTEM_SHARED_MEM_FS)
                available_nbytes = shm_stats.f_bsize * shm_stats.f_bavail
                if available_nbytes > SYSTEM_SHARED_MEM_FS_MIN_SIZE:
                    temp_folder = SYSTEM_SHARED_MEM_FS
                    pool_folder = os.path.join(temp_folder, pool_folder_name)
                    if not os.path.exists(pool_folder):
                        os.makedirs(pool_folder)
                    use_shared_mem = True
            except (IOError, OSError):
                temp_folder = None
    if temp_folder is None:
        temp_folder = tempfile.gettempdir()
    temp_folder = os.path.abspath(os.path.expanduser(temp_folder))
    pool_folder = os.path.join(temp_folder, pool_folder_name)
    if not os.path.exists(pool_folder):
        os.makedirs(pool_folder)
    return pool_folder, use_shared_mem
class DomainFeatures(object):
    def __get_type__(self) -> str:
        return type(self)
    @abstractmethod
    def __generate__(self, data: np.array) -> dict:
        raise NotImplementedError
class TdFeatures(DomainFeatures):
    def __get_type__(self) -> str:
        return "Time Domain"
    def __generate__(self, data: np.array) -> dict:
        nn_intervals = np.asarray(data)
        out = {}  # Initialize empty container for results
        diff_nni = np.diff(nn_intervals)
        length_int = len(nn_intervals)
        hr = np.divide(60000, nn_intervals)
        nni_50 = sum(np.abs(diff_nni) > 50)
        nni_20 = sum(np.abs(diff_nni) > 20)
        mean_nni = np.mean(nn_intervals)
        median_nni = np.median(nn_intervals)
        rmssd = np.sqrt(np.mean(diff_nni ** 2))
        sdnn = np.std(nn_intervals, ddof=1)  # ddof = 1 : unbiased estimator => divide std by n-1
        out['hrv_mean_nni'] = mean_nni
        out['hrv_median_nni'] = median_nni
        out['hrv_range_nni'] = max(nn_intervals) - min(nn_intervals)
        out['hrv_sdsd'] = np.std(diff_nni)
        out['hrv_rmssd'] = rmssd
        out['hrv_nni_50'] = nni_50
        out['hrv_pnni_50'] = 100 * nni_50 / length_int
        out['hrv_nni_20'] = nni_20
        out['hrv_pnni_20'] = 100 * nni_20 / length_int
        out['hrv_cvsd'] = rmssd / mean_nni
        out['hrv_sdnn'] = sdnn
        out['hrv_cvnni'] = sdnn / mean_nni
        out['hrv_mean_hr'] = np.mean(hr)
        out['hrv_min_hr'] = min(hr)
        out['hrv_max_hr'] = max(hr)
        out['hrv_std_hr'] = np.std(hr)
        return out

class StatFeatures(DomainFeatures):
    def __get_type__(self) -> str:
        return "Statistical"
    def __generate__(self, data: np.array) -> dict:
        return get_stats(data, 'hrv')

FEATURE_FUNCTIONS = {
    'td': TdFeatures()
}

def get_hrv_features(data: pd.Series, window_length: int = 180, window_step_size: int = 1,
                     domains: List[str] = ['td', 'fd', 'stat'], threshold: float = 0.2,
                     clean_data: bool = True, num_cores: int = 0):
    if not num_cores >= 1:
        num_cores = multiprocessing.cpu_count()
    if clean_data:
        # print("Cleaning data...")
        clean_data = __clean_artifacts(data.copy())
    else:
        # print("Not cleaning data")
        clean_data = data.copy()
    # ensure we have a DatetimeIndex, needed for interpolation
    if not isinstance(clean_data.index, pd.DatetimeIndex):
        clean_data.index = pd.DatetimeIndex(clean_data.index)
    clean_data = clean_data[~clean_data.index.duplicated()]
    # before starting calculations, make sure that there actually is some data left
    if clean_data.empty:
        warnings.warn(f'Empty dataset after cleaning: 0 of {len(data)} rows left), returning empty features dataframe',
                      stacklevel=3)
        return pd.DataFrame.empty
    window_length_timedelta = pd.to_timedelta(window_length, unit='s')
    window_step_size_timedelta = pd.to_timedelta(window_step_size, unit='s')
    first_index = clean_data.index[0].floor('s')
    last_index = clean_data.index[-1].ceil('s')
    target_index = pd.date_range(start=first_index,
                                 end=max(first_index, last_index - window_length_timedelta),
                                 freq=window_step_size_timedelta)
    def process(memmap_data) -> pd.DataFrame:
        with Parallel(n_jobs=num_cores, max_nbytes=None) as parallel:
            for domain in domains:
                if domain not in FEATURE_FUNCTIONS.keys():
                    raise ValueError("invalid feature domain: " + domain)
            return __generate_features_for_domain(memmap_data, target_index, window_length_timedelta, threshold,
                                                  feature_functions=[FEATURE_FUNCTIONS.get(x) for x in domains],
                                                  parallel=parallel)
    features = memmap_auto(clean_data, process)
    # only interpolate if there are overlapping time windows
    if window_length_timedelta > window_step_size_timedelta:
        limit = max(1, int((1 - threshold) * (window_length / window_step_size)))
        features.interpolate(method='time', limit=limit, inplace=True)
    return features
def __clean_artifacts(data: pd.Series, threshold=0.2) -> pd.Series:
    diff = data.diff().abs()
    drop_indices = diff > threshold * data
    if drop_indices.any():
        data.drop(data[drop_indices].index, inplace=True)
    drop_indices = (data < 250) | (data > 2000)
    if drop_indices.any():
        data.drop(data[drop_indices].index, inplace=True)  # drop by bpm > 240 or bpm < 30
    data.dropna(inplace=True)  # just to be sure
    return data
def __generate_features_for_domain(clean_data: pd.Series, target_index: pd.DatetimeIndex,
                                   window_length: datetime.timedelta, threshold: float,
                                   feature_functions: List[DomainFeatures], parallel: Parallel) -> pd.DataFrame:
    inputs = tqdm(target_index, desc="HRV features")
    features = parallel(delayed(__calculate_hrv_features)
                        (clean_data, start_datetime=k, window_length=window_length,
                         threshold=threshold,
                         feature_functions=feature_functions)
                        for k in inputs)
    features = pd.DataFrame(list(filter(None, features)))
    if not features.empty:
        features.set_index('datetime', inplace=True)
        features.sort_index(inplace=True)
    return features
def __calculate_hrv_features(data: pd.Series, window_length: datetime.timedelta, start_datetime: datetime.datetime,
                             threshold: float, feature_functions: List[DomainFeatures]):
    relevant_data = data.loc[(data.index >= start_datetime) & (data.index < start_datetime + window_length)]
    return_val = {'datetime': start_datetime + window_length, "num_ibis": len(relevant_data)}
    # first check if there is at least one IBI in the epoch
    if len(relevant_data) > 0:
        expected_length = (window_length.total_seconds() / (relevant_data.mean() / 1000))
        actual_length = len(relevant_data)
        if actual_length >= (expected_length * threshold):
            for feature_function in feature_functions:
                return_val.update(feature_function.__generate__(relevant_data))
    return return_val
