import time
import json
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


# open file
with open('./twitter-50mb.json', 'r', encoding='utf-8') as tweet_file:
    jsonData = json.load(tweet_file)
    tweets = jsonData.get("rows")

    # processing data
    term = 0
    for tweet in tweets:
        # print(term)
        if term % size != rank or (len(tweet) < 1):
            term += 1
            continue
        tweet_data = tweet.get('doc').get('data')
        # if tweet_data is not None:

        # get date and hour, doc -> data -> created_at, get sentiment, doc -> data -> sentiment
        created_at = tweet_data.get('created_at').split('T')
        day = created_at[0]
        hour = created_at[1].split(':')[0]
        sentiment_score = tweet_data.get('sentiment')
        if not isinstance(sentiment_score, float) and not (isinstance(sentiment_score, int)):
            sentiment_score = 0

        # add data into dict
        most_active_hour_dict[hour] += 1
        most_active_day_dict[day] += 1
        happiest_hour_dict[hour] += sentiment_score
        happiest_day_dict[day] += sentiment_score

        # with open('output.txt', 'w', encoding='utf-8') as f:
        #     f.write('hour: ' + hour + ' day: ' + day + ' sentiment_score: ' + str(sentiment_score))
        term += 1

# gather result from children
gathered_dict_lists = communicator.gather([happiest_hour_dict, happiest_day_dict, most_active_hour_dict,
                                           most_active_day_dict], root=0)

# todo: merge each dicts
if rank == 0:
    for item in gathered_dict_lists:
        # print(item[0])
        ans_happiest_hour_dict = merge_dict(ans_happiest_hour_dict, item[0])
        ans_happiest_day_dict = merge_dict(ans_happiest_day_dict, item[1])
        ans_most_active_hour_dict = merge_dict(ans_most_active_hour_dict, item[2])
        ans_most_active_day_dict = merge_dict(ans_most_active_day_dict, item[3])

    # todo: sort dicts by items and print output
    sorted_happiest_hour = sorted(ans_happiest_hour_dict.items(), key=lambda x: x[1])
    sorted_happiest_day = sorted(ans_happiest_day_dict.items(), key=lambda x: x[1])
    sorted_active_hour = sorted(ans_most_active_hour_dict.items(), key=lambda x: x[1])
    sorted_activate_day = sorted(ans_most_active_day_dict.items(), key=lambda x: x[1])

    print("Happiest Hour: " + str(sorted_happiest_hour[-1]))
    print("Happiest Day: " + str(sorted_happiest_day[-1]))
    print("Most active Hour: " + str(sorted_active_hour[-1]))
    print("Most active Day: " + str(sorted_activate_day[-1]))

    # todo: print time spend
    print(time.time() - start_time)

# check output of file
# with open('output.txt', 'w', encoding='utf-8') as f:
#     f.write(str(tweets[0]))
# print(tweets)
