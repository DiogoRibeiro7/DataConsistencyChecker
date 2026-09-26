"""
This file is executed first and will save a pickle of each dc object, allowing the
other tests to execute faster. It only does anything when TEST_REAL is enabled in utils.py.
"""

import os

import dill
from list_real_files import real_files
from utils import cache_folder, load_openml_file, requires_real_data

from data_consistency_checker import DataConsistencyChecker


@requires_real_data
def test_init_cache():
    # If not present, create the folder for the cache
    os.makedirs(cache_folder, exist_ok=True)

    for dataset_name in real_files:
        file_name = os.path.join(cache_folder, dataset_name + "_dc.pkl")
        if os.path.exists(file_name):
            print(f"{dataset_name} already in cache")
        else:
            print(f"Initializing and caching {dataset_name}")
            data_df = load_openml_file(dataset_name)
            dc = DataConsistencyChecker(verbose=-1)
            dc.init_data(data_df)
            with open(file_name, "wb") as filehandler:
                dill.dump(dc, filehandler)
