from pathlib import Path
import pandas as pd


dir = Path("./data/")
list(dir.glob("*"))
tweets_dir = dir/"tweets"

all_games = pd.read_csv(dir/"games.csv")
all_games['GAME_ID'] = all_games["GAME_ID"].astype(str)

tweets_for_games = pd.read_csv(dir/"Tweets_for_games_with2_team_tweets.csv")
tweet_counts = tweets_for_games.groupby(["game_id", "team_nick"]).count().reset_index()[["game_id", "team_nick", "id"]]