import time
import numpy as np
from mpi4py import MPI
import re

start_time = time.time()

def read_file(path):
    f = open(path, 'rb')
    for line in f:
        yield line.decode()
    f.close()

def get_created_at(tweet):
    start_index = tweet.find('"created_at":"')
    if start_index == -1:
        return None
    created_at_str = tweet[start_index+len('"created_at":"'):start_index+len('"created_at":"')+13]
    
    date_time_parts = created_at_str.split("T")
    date_parts = date_time_parts[0].split("-")
    time_parts = date_time_parts[1].split(":")
    
    month = int(date_parts[1])
    day = int(date_parts[2])
    hour = int(time_parts[0])    
    return (month, day, hour)

def get_sentiment(tweet):
    sentiment_index = tweet.find("sentiment")

    if sentiment_index == -1:
        return None

    sentiment_string = tweet[sentiment_index:]
    sentiment_value = re.findall(r"[-+]?\d*\.\d+|\d+", sentiment_string)[0]

    return float(sentiment_value)

FILE = 'twitter-50mb.json'

SHAPE = (12,31,24)
hour_sentiment = np.zeros(shape=SHAPE, dtype=float)
hour_count = np.zeros(shape=SHAPE, dtype=int)

for tweet in read_file(FILE):
    datetime = get_created_at(tweet)
    sentiment = get_sentiment(tweet)
    
    if (datetime is not None):
        hour_count[datetime[0]-1, datetime[1]-1, datetime[2]] += 1

    if (sentiment is not None):
        hour_sentiment[datetime[0]-1, datetime[1]-1, datetime[2]] += sentiment

happiest_hour = np.unravel_index(np.argmax(hour_sentiment), hour_sentiment.shape)[2]+1

day_sentiment = np.sum(hour_sentiment, axis=0)
happiest_day = np.unravel_index(np.argmax(day_sentiment), day_sentiment.shape)[1]+1

most_active_hour = np.unravel_index(np.argmax(hour_count), hour_sentiment.shape)[2]+1

day_count = np.sum(hour_count, axis=0)
most_active_day = np.unravel_index(np.argmax(hour_count), hour_sentiment.shape)[1]+1

print(happiest_hour)
print(happiest_day)
print(most_active_hour)
print(most_active_day)

print(time.time() - start_time)
