import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

import twokenize  # Twitter tokenizer: pip3 install twokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split


class DataManager(object):
    def __init__(self):
        self.all_games = pd.read_csv("data/games.csv")
        self.teams = pd.read_csv("data/teams.csv")
        self.tweets = pd.read_csv("data/Tweets_for_games_with2_team_tweets.csv")
    
        self.nick_to_id = {team.NICKNAME: team.TEAM_ID for _, team in self.teams.iterrows()}
        self.id_to_nick = {team.TEAM_ID: team.NICKNAME for _, team in self.teams.iterrows()}
        self.id_to_handle = {team.TEAM_ID: team.TWITTER_HANDLE.lower() for _, team in self.teams.iterrows()}
        self.game_ids = [int(ID) for ID in self.tweets.game_id.unique() if np.isnan(ID)==False]
        games = self.all_games[self.all_games['GAME_ID'].isin(self.game_ids)]
        self.games = games[['GAME_ID','HOME_TEAM_ID','VISITOR_TEAM_ID','HOME_TEAM_WINS']]
        #self.DATA = pd.read_csv(dir/"games_with_processed_tweets.csv")
    
    def get_team_tweets(self, game_id, team_id):
        nick, handle = self.id_to_nick[team_id], self.id_to_handle[team_id]
        team_tweets = self.tweets[(self.tweets.game_id==game_id) & (self.tweets.team_nick==nick)]["tweet"]
        remove_handle = lambda tweet: tweet.lower().replace(handle,'')
        tweet_string = ' '.join(team_tweets.map(remove_handle))
        return tweet_string

    def combine_tweets(self):
        home_tweets, away_tweets = [],[]
        count = 0
        for _, game in self.games.iterrows():
            home_tweets.append(self.get_team_tweets(game.GAME_ID, game.HOME_TEAM_ID))
            away_tweets.append(self.get_team_tweets(game.GAME_ID, game.VISITOR_TEAM_ID))
            count +=1
            if count % 500 == 0:
                print(f"Completed top {count}")
                
        self.games['HOME_TWEETS'] = home_tweets
        self.games['AWAY_TWEETS'] = away_tweets
        self.games['ALL_TWEETS'] = self.games[['HOME_TWEETS','AWAY_TWEETS']].agg(' '.join, axis=1)
        self.games.to_csv("data/games_with_preprocessed_tweets_all.csv", sep=",", index=False)

    def _pad_sequences(self, sequences, pad_tok, max_length):
        sequence_padded = []
        for seq in sequences:
            seq = list(seq)
            seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
            sequence_padded.append(seq_)
        return sequence_padded
        
    def pad_sequences(self, sequences, pad_tok, nlevels=1):
        """
        Args:
            sequences: a generator of list or tuple
            pad_tok: the char to pad with
        Returns:
            a list of list where each sublist has same length
        """
        if nlevels == 1:
            max_length = max(map(lambda x : len(x), sequences))
            sequence_padded = self._pad_sequences(sequences, pad_tok, max_length)
        elif nlevels==2:
            max_length_tweet = max([max(map(lambda x: len(x), seq)) for seq in sequences])
            sequence_padded = []
            for seq in sequences:
                # all tweets are same length now
                sp = self._pad_sequences(seq, pad_tok, max_length_tweet)
                sequence_padded.append(sp)
        return sequence_padded
    
    def preprocess(self, data_path):
        data = pd.read_csv(data_path)
        print(f"{datetime.now()}: Tokenizing tweets")
        all_twokens = [' '.join(twokenize.tokenizeRawTweetText(tweet))
                        for tweet in data['ALL_TWEETS']]
        #away_twokens = [' '.join(twokenize.tokenizeRawTweetText(tweet))
        #                for tweet in data['AWAY_TWEETS']]
        #all_twokens = home_twokens
        #all_twokens.extend(away_twokens)
        tokenizer = Tokenizer()
        print(f"{datetime.now()}: Fitting tokenizer")
        tokenizer.fit_on_texts(all_twokens)
        print(f"{datetime.now()}: Texts to sequences")
        seqs = tokenizer.texts_to_sequences(all_twokens)# for t in home_twokens]
        #away_seqs = tokenizer.texts_to_sequences(away_twokens)# for t in away_twokens]
        print(f"{datetime.now()}: Padding sequences")
        #X = self.pad_sequences(X, 0)
        X = self.pad_sequences(seqs, 0)
        y = data['HOME_TEAM_WINS'].values.tolist()
        print(f"{datetime.now()}: Done")
        return X, y

    

    
