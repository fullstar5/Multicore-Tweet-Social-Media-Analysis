import time
import json
from mpi4py import MPI

# record time
start_time = time.time()

communicator = MPI.COMM_WORLD
rank = communicator.Get_rank()
size = communicator.Get_size()

happiest_hour_dict = {}
happiest_day_dict = {}
most_active_hour_dict = {}
most_active_day_dict = {}

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
                # get date and hour, doc -> data -> created_at
                created_at = tweet_data.get('created_at', None)
                hour = created_at.split(':')[0]
                day = created_at.split('T')[0]
                # get sentiment, doc -> data -> sentiment
                sentiment_score = tweet_data.get('sentiment', 0)


                # with open('output.txt', 'w', encoding='utf-8') as f:
                #     f.write('hour: ' + hour + ' day: ' + day + ' sentiment_score: ' + str(sentiment_score))


        term += 1

# print(time.time() - start_time)
# check output of file
# with open('output.txt', 'w', encoding='utf-8') as f:
#     f.write(str(tweets[0]))
# print(tweets)



