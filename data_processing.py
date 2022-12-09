import os
import csv
import time
import numpy as np
import multiprocessing

from hdf5_getters import *

# Reference: https://github.com/tbertinmahieux/MSongsDB/blob/master/PythonSrc/hdf5_getters.py

"""A Complete list of features I am interested in
'artist_familiarity',
'artist_hotttnesss',
'artist_id',
'artist_latitude',
'artist_location',
'artist_longitude',
'artist_name',
'title',
"artist_terms",
"artist_terms_freq",
"artist_terms_weight",
'danceability',
'duration',
'end_of_fade_in',
'energy',
'key',
'key_confidence',
'loudness',
'mode',
'mode_confidence',
'start_of_fade_out',
'tempo',
'time_signature',
'time_signature_confidence'
'year',
"""


def process_h5_file(h5_file):
    """
    Process a single h5 file to extract features listed above.
    Each h5 file contains information for only one song.
    """

    # return the row as a list of values
    row = []

    # If song_hotness is NaN, set it as -1
    hotness = get_song_hotttnesss(h5_file)
    if np.isnan(hotness):
        row.append(-1)
    else:
        row.append(hotness)

    # Extract meta data
    title = get_title(h5_file).decode('utf-8').lower()
    album_name = get_release(h5_file).decode('utf-8').lower()
    artist_familiarity = get_artist_familiarity(h5_file)
    artist_hotttnesss = get_artist_hotttnesss(h5_file)

    # since the prediction is for future, year seems have no meaning
    # so I will not ues it
    year = get_year(h5_file)

    # To simplify, only take the first(most important term)
    # And to avoid too many binary features, only take the first word of each term
    artist_terms = get_artist_terms(h5_file)[0].decode('utf-8').lower()
    artist_terms = artist_terms.split(" ")[0]

    # artist_terms_freq = get_artist_terms_freq(h5_file)
    # artist_terms_weight = get_artist_terms_weight(h5_file)

    # Extract analysis data
    dance_ability = get_danceability(h5_file)
    duration = get_duration(h5_file)
    end_of_fade_in = get_end_of_fade_in(h5_file)
    energy = get_energy(h5_file)
    key = get_key(h5_file)
    key_confidence = get_key_confidence(h5_file)
    loudness = get_loudness(h5_file)
    mode = get_mode(h5_file)
    mode_confidence = get_mode_confidence(h5_file)

    start_of_fade_out = get_start_of_fade_out(h5_file)
    tempo = get_tempo(h5_file)
    time_signature = get_time_signature(h5_file)
    time_signature_confidence = get_time_signature_confidence(h5_file)

    row.extend([title, album_name, artist_familiarity, artist_hotttnesss, artist_terms, dance_ability, duration,
                end_of_fade_in, energy, key, key_confidence, loudness, mode, mode_confidence,
                start_of_fade_out, tempo, time_signature, time_signature_confidence])

    return row


def save_row(process_id, row):
    """
    Save a row into a local CSV
    - process_id: id of current process, also the name of the csv file
    - rows: A list of rows which are results of `transform_local`
    """

    save_path = f'processed/{process_id}.csv'

    with open(save_path, 'a', encoding='utf-8', newline="") as file:
        writer = csv.writer(file)
        writer.writerow(row)


def process_h5_file_wrapper(h5_file):
    """
    Wrapper function that processes a local h5 file and save result locally
    """
    try:
        with tables.open_file(h5_file) as h5:
            row = process_h5_file(h5)
            if len(row) > 0:
                save_row(os.getpid(), row)
    except IndexError or IOError or Exception as e:
        # print(e)
        return


if __name__ == "__main__":
    n_cpu = multiprocessing.cpu_count()
    print("CPU Number on this machine:", n_cpu)

    if not os.path.exists("./processed"):
        os.mkdir("processed")

    data_path = "millionsongsubset"

    # 1 - 138.30 s
    # 2 - 145.41 s
    # 6 - 98.06 s
    # 12 - 100.59 s
    my_pool = multiprocessing.Pool(processes=1)
    # my_process = multiprocessing.Process()
    processed_count = 0
    start = time.perf_counter()
    dirs_1 = ['A', 'B']
    dirs_2 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    dirs_3 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    for d_1 in dirs_1:
        for d_2 in dirs_2:
            for d_3 in dirs_3:
                cur_path = os.path.join(data_path, d_1, d_2, d_3)
                for _, _, filenames in os.walk(cur_path):
                    for f_name in filenames:
                        file_path = cur_path + "\\" + f_name
                        my_pool.apply(process_h5_file_wrapper, args=(file_path, ))
                        # process_h5_file_wrapper(file_path)
                        processed_count += 1
                        if processed_count % 100 == 0:
                            print("\rProcessing %d/%d..." %
                                  (processed_count, 10000), end="...")
    wall_clock_time = time.perf_counter() - start
    print("\nUsing time:", wall_clock_time, "seconds")
    my_pool.close()
