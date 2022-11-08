from datetime import datetime
import tweepy
import pandas as pd
import time
import csv
import preprocessor as p



# Twitter Authorization
# Read keys.csv for keys
with open('keys.csv', mode='r') as infile:
    reader = csv.reader(infile)
    KEYS = {rows[0]:rows[1] for rows in reader}  

consumer_key = KEYS['consumer_key']
consumer_secret = KEYS['consumer_secret']
bearer_token = KEYS['bearer_token']
access_key = KEYS['access_key']
access_secret = KEYS['access_secret']
# Pass twitter credentials to tweepy
auth = tweepy.OAuth2BearerHandler(bearer_token)
api = tweepy.API(auth)



# Helper function to extract more information from twweettext
p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.RESERVED, p.OPT.EMOJI, p.OPT.SMILEY)
def processTweetText(tweettext):
    parsed_tweet = p.parse(tweettext)
    tweeturls = None
    tweethashtags = None
    tweetmentions = None
    
    if(parsed_tweet.urls):
        tweeturls = [url.match for url in parsed_tweet.urls]
    if(parsed_tweet.hashtags):
        tweethashtags = [ht.match for ht in parsed_tweet.hashtags]
    if(parsed_tweet.mentions):
        tweetmentions = [m.match for m in parsed_tweet.mentions]
    
    tweettextcleaned = p.clean(tweettext).replace("#", "").replace(":", "")
    return [tweeturls, tweethashtags, tweetmentions, tweettextcleaned]



def scrapeTwitter(q, df, count, numTweets, seen):
    searchResults = tweepy.Cursor(
        api.search_tweets,
        q,
        lang='en',
        result_type='mixed',
        tweet_mode='extended',
        count=100,
    ).items(count)

    statusList = [status for status in searchResults]
    for status in statusList:
        _, _, _, tweetTextCleaned = processTweetText(status.full_text)
        
        if tweetTextCleaned in seen: # prevent duplicates
            continue

        seen.add(tweetTextCleaned)
        df.loc[len(df)] = [tweetTextCleaned]
        numTweets += 1

    print(f"Query: {q}\t\tTotal scraped: {numTweets}")
    return numTweets



df = pd.DataFrame(columns=['Cleaned Tweet'])
numTweets = 0
# count = 1000
count = 100
queries = [
    # top 25 in S&P by market cap
    '$SPY', '$AAPL', '$MSFT', '$AMZN', '$TSLA', '$GOOG', '$BRK', '$META', '$UNH', '$JNJ', 
    '$XOM', '$JPM', '$V', '$PG', '$NVDA', '$HD', '$CVX', '$LLY', '$MA', '$ABBV', 
    '$PFE', '$MRK', '$PEP', '$BAC', '$KO', 

    # top 10 crypto on coinmarketcap (a lot of ads)
    # 'BTC', 'ETH', 'USDT', 'USDC', 'BNB', 'XRP', 'BUSD', 'ADA', 'SOL', 'DOGE',

    # other relevant keywords
    'bullish', 'bearish', 'stocks', 'nasdaq', 'dow jones'
]
seen = set()

for q in queries:
    try:
        numTweets = scrapeTwitter(q, df, count, numTweets, seen)
    except Exception as e:
        print(e)
        print("Time to take a 15 minutes break...")
        time.sleep(15*60)
        print("Resume scraping...")
        numTweets = scrapeTwitter(q, df, count, numTweets, seen)



timestamp = datetime.today().strftime('%Y%m%d_%H%M%S')
# filename_csv = 'tweets_' + str(numTweets) + '_' + timestamp + '.csv'
filename_csv = 'latest_tweets.csv'
df.to_csv(filename_csv, index=False)
print("File saved as: ", filename_csv)

