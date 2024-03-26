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


FILE = 'twitter-50mb.json'
# pattern = r'created_at":\s*"(.*?)".*?"sentiment":\s*(-?\d+(\.\d+)?)'
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

        if bt_read >= bt_per_core:
            print(f"Rank {rank}: Reached byte limit ({bt_read} >= {bt_per_core}), breaking out of loop")
            break

        # matches = re.findall(pattern, tweet_str)
        created_match = re.findall(pattern_created_at, tweet_str)
        sentiment_match = re.findall(pattern_sentiment, tweet_str)
        if not len(created_match):
            bt_read += utf8len(tweet_str)
            continue
        if len(sentiment_match):
            # with open("output.txt", 'w') as f:
            #     f.write(str(sentiment_match))
            sentiment = float(sentiment_match[0][0])

        # if not len(matches):
        #     # bt_read += utf8len(tweet_str)
        #     continue

        # created_at = matches[0][0]
        created_at = created_match[0]
        # with open("output.txt", 'w') as f:
        #     f.write(str(created_match))
        # sentiment = float(matches[0][1])
        day = created_at.split('T')[0]
        hour = created_at.split(':')[0]

        happiest_hour_dict[hour] += sentiment
        happiest_day_dict[day] += sentiment
        most_active_hour_dict[hour] += 1
        most_active_day_dict[day] += 1

        bt_read += utf8len(tweet_str)

# Gather results from all processes
gathered_dict_lists = communicator.gather([happiest_hour_dict, happiest_day_dict, most_active_hour_dict,
                                           most_active_day_dict], root=0)

# merge each dicts
if rank == 0:
    for item in gathered_dict_lists:
        # print(item[0])
        ans_happiest_hour_dict = merge_dict(ans_happiest_hour_dict, item[0])
        ans_happiest_day_dict = merge_dict(ans_happiest_day_dict, item[1])
        ans_most_active_hour_dict = merge_dict(ans_most_active_hour_dict, item[2])
        ans_most_active_day_dict = merge_dict(ans_most_active_day_dict, item[3])
#
    # sort dicts by items and print output
    sorted_happiest_hour = sorted(ans_happiest_hour_dict.items(), key=lambda x: x[1])
    sorted_happiest_day = sorted(ans_happiest_day_dict.items(), key=lambda x: x[1])
    sorted_active_hour = sorted(ans_most_active_hour_dict.items(), key=lambda x: x[1])
    sorted_activate_day = sorted(ans_most_active_day_dict.items(), key=lambda x: x[1])

    print("Happiest Hour: " + str(sorted_happiest_hour[-1]))
    print("Happiest Day: " + str(sorted_happiest_day[-1]))
    print("Most active Hour: " + str(sorted_active_hour[-1]))
    print("Most active Day: " + str(sorted_activate_day[-1]))

    print(time.time() - start_time)
