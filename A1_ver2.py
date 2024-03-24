import time
import numpy as np
import os
from mpi4py import MPI
import re

# set starting timer
start_time = time.time()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#read the file row by row
def read_file(rank, size, path):
    file_size = os.path.getsize(path)
    bt_per_node = file_size //size
    chunk_start = rank * bt_per_node
    bt_read = 0

    f = open(path, 'rb')
    for line in f:
        if rank == 0 or bt_read >0:
            yield line.decode()

        bt_read += len(line)
        if bt_read >= bt_per_node:
            break
    f.close()

#find the tweet's created time
def get_created_at(tweet):
    start_index = tweet.find('"created_at":"')
    #if tweet does not contain created time, return None
    if start_index == -1:
        return None
    
    #find the createde time string
    created_at_str = tweet[start_index+len('"created_at":"'):start_index+len('"created_at":"')+13]
    
    date_time_parts = created_at_str.split("T")
    date_parts = date_time_parts[0].split("-")
    time_parts = date_time_parts[1].split(":")
    
    #save created time
    month = int(date_parts[1])
    day = int(date_parts[2])
    hour = int(time_parts[0])    
    return (month, day, hour)

#find sentiment score or the tweet
def get_sentiment(tweet):
    sentiment_index = tweet.find('"sentiment":')
    # if tweet does not contain sentiment score, return none
    if sentiment_index == -1:
        return None

    #find sentiment score by regex
    sentiment_string = tweet[sentiment_index:]
    sentiment_value = re.findall(r"[-+]?\d*\.\d+|\d+", sentiment_string)[0]

    return float(sentiment_value)

#file path
FILE = 'twitter-50mb.json'

#create 3D array, first dimension represents month, second dimension represents day, third dimension represents hour
SHAPE = (12,31,24)
hour_sentiment = np.zeros(shape=SHAPE, dtype=float)
hour_count = np.zeros(shape=SHAPE, dtype=int)

for tweet in read_file(rank, size, FILE):
    #get created time
    datetime = get_created_at(tweet)
    #get sentiment score
    sentiment = get_sentiment(tweet)
    
    #count number of tweets created in each hour
    if (datetime is not None):
        hour_count[datetime[0]-1, datetime[1]-1, datetime[2]] += 1
    #count sentiment score of each hour
    if (sentiment is not None):
        hour_sentiment[datetime[0]-1, datetime[1]-1, datetime[2]] += sentiment

hour_sentiment_gathered = comm.reduce(hour_sentiment, op=MPI.SUM, root=0)
hour_count_gathered = comm.reduce(hour_count, op=MPI.SUM, root=0)


happiest_hour = np.unravel_index(np.argmax(hour_sentiment_gathered), hour_sentiment_gathered.shape)

day_sentiment = np.sum(hour_sentiment_gathered, axis=2)
happiest_day = np.unravel_index(np.argmax(day_sentiment), day_sentiment.shape)

most_active_hour = np.unravel_index(np.argmax(hour_count_gathered), hour_count_gathered.shape)

day_count = np.sum(hour_count_gathered, axis=2)
most_active_day = np.unravel_index(np.argmax(day_count), day_count.shape)

print(happiest_hour, hour_sentiment[happiest_hour])
print(happiest_day, day_sentiment[happiest_day])
print(most_active_hour, hour_count[most_active_hour])
print(most_active_day, day_count[most_active_day])

print(time.time() - start_time)
