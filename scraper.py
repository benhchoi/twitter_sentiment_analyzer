import tweepy
import csv
import numpy as np
import pandas as pd
import sys
####input your credentials here
consumer_key = 'YcLDM9cSFJFXWRhRSnL1ZZHEm'
consumer_secret = 'kN2QIZJ526TD0s0yzmmPjA0PsNRi05h1cruzcX7vbY2SWYpXTB'
access_token = '854059662857187334-LxA0RdttIzwJZTPiVA3ajIJZYZI9GE1'
access_token_secret = '5K2QID2WDksSBFcU5RCWXBPfHjpJsaY7dj4QDcpjjdVpl'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)
#####United Airlines
# Open/Create a file to append data
csvFile = open('data.csv', 'a')
#Use csv Writer
csvWriter = csv.writer(csvFile)

for tweet in tweepy.Cursor(api.search,q="#NipseyHussle",count=2000,
                           lang="en",
                           since="2019-04-01").items():
    csvWriter.writerow([tweet.created_at, tweet.text.encode('utf-8')])
