import time
import re
import os
from mpi4py import MPI
from collections import defaultdict

# record time
start_time = time.time()

communicator = MPI.COMM_WORLD
rank = communicator.Get_rank()
size = communicator.Get_size()

# declare temp variables
happiest_hour_dict = defaultdict(float)
happiest_day_dict = defaultdict(float)
most_active_hour_dict = defaultdict(float)
most_active_day_dict = defaultdict(float)

# declare final dict variables
ans_happiest_hour_dict = defaultdict(float)
ans_happiest_day_dict = defaultdict(float)
ans_most_active_hour_dict = defaultdict(float)
ans_most_active_day_dict = defaultdict(float)


def merge_dict(dict1, dict2):
    for key, value in dict2.items():
        dict1[key] += value
    return dict1


def utf8len(s):
    return len(s.encode('utf-8')) + 1


# split file into equal size of chunk based on rank
FILE = 'twitter-100gb.json'
pattern_created_at = r'created_at":\s*"(.*?)"'
pattern_sentiment = r'"sentiment":\s*(-?\d+(\.\d+)?)'
file_size = os.path.getsize(FILE)
bt_per_core = file_size // size
chunk_start = rank * bt_per_core

# open file
with open(FILE, 'r', encoding='utf-8') as tweet_file:
    bt_read = 0
    sentiment = 0
    tweet_file.seek(chunk_start)
    tweet_file.readline()
    tweet_file.readline()
    tweet_file.readline()

    while (tweet_str := tweet_file.readline()) != '{}]}\n':

        # break if reach this rank's limit
        if bt_read >= bt_per_core:
            break

        # data extracting
        created_match = re.findall(pattern_created_at, tweet_str)
        sentiment_match = re.findall(pattern_sentiment, tweet_str)
        if not len(created_match):
            bt_read += utf8len(tweet_str)
            continue
        if len(sentiment_match):
            sentiment = float(sentiment_match[0][0])

        created_at = created_match[0]
        day = created_at.split('T')[0]
        hour = created_at.split(':')[0]

        happiest_hour_dict[hour] += sentiment
        happiest_day_dict[day] += sentiment
        most_active_hour_dict[hour] += 1
        most_active_day_dict[day] += 1

        # add bytes already read
        bt_read += utf8len(tweet_str)

# Gather results from all processes
gathered_dict_lists = communicator.gather([happiest_hour_dict, happiest_day_dict, most_active_hour_dict,
                                           most_active_day_dict], root=0)

# merge each dicts and print results
if rank == 0:
    for item in gathered_dict_lists:
        ans_happiest_hour_dict = merge_dict(ans_happiest_hour_dict, item[0])
        ans_happiest_day_dict = merge_dict(ans_happiest_day_dict, item[1])
        ans_most_active_hour_dict = merge_dict(ans_most_active_hour_dict, item[2])
        ans_most_active_day_dict = merge_dict(ans_most_active_day_dict, item[3])

    # sort dicts by items and print output
    sorted_happiest_hour = sorted(ans_happiest_hour_dict.items(), key=lambda x: x[1])
    sorted_happiest_day = sorted(ans_happiest_day_dict.items(), key=lambda x: x[1])
    sorted_active_hour = sorted(ans_most_active_hour_dict.items(), key=lambda x: x[1])
    sorted_active_day = sorted(ans_most_active_day_dict.items(), key=lambda x: x[1])

    hourHappy_str = str(sorted_happiest_hour[-1][0][-2]) + str(sorted_happiest_hour[-1][0][-1])
    dayHappy_str = str(sorted_happiest_hour[-1][0][:-3])
    hourActive_str = str(sorted_active_hour[-1][0][-2]) + str(sorted_active_hour[-1][0][-1])
    dayActivate_str = str(sorted_active_hour[-1][0][:-3])

    print("Happiest Hour: " + hourHappy_str + " in " + dayHappy_str + " with "
          + str(sorted_happiest_hour[-1][1]) + " score")
    print("Happiest Day: " + str(sorted_happiest_day[-1][0]) + " with " + str(sorted_happiest_day[-1][1]) + " score")
    print("Most active Hour: " + hourActive_str + " in " + dayActivate_str
          + " with " + str(sorted_active_hour[-1][1]) + " tweets")
    print("Most active Day: " + str(sorted_active_day[-1][0]) + " with " + str(sorted_active_day[-1][1]) + " tweets")
    print(time.time() - start_time)
