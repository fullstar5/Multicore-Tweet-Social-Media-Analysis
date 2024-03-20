import time
import json
from mpi4py import MPI
from collections import defaultdict

# record time
start_time = time.time()

communicator = MPI.COMM_WORLD
rank = communicator.Get_rank()
size = communicator.Get_size()

happiest_hour_dict = defaultdict(int)
happiest_day_dict = defaultdict(int)
most_active_hour_dict = defaultdict(int)
most_active_day_dict = defaultdict(int)

# open file
with open('./twitter-1mb.json', 'r', encoding='utf-8') as tweet_file:
    jsonData = json.load(tweet_file)
    tweets = jsonData.get("rows")

    # processing data
    term = 0
    for tweet in tweets:
        print(term)
        if term % size == rank and (len(tweet) > 0):
            tweet_data = tweet.get('doc').get('data')
            if tweet_data is not None:

                # get date and hour, doc -> data -> created_at, get sentiment, doc -> data -> sentiment
                created_at = tweet_data.get('created_at', None)
                hour = created_at.split(':')[0]
                day = created_at.split('T')[0]
                sentiment_score = tweet_data.get('sentiment', float(0))

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


# todo: print output and time spend


# print time spend
print(time.time() - start_time)

# check output of file
# with open('output.txt', 'w', encoding='utf-8') as f:
#     f.write(str(tweets[0]))
# print(tweets)



