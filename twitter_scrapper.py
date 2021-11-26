import twint
import time
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

games = pd.read_csv("./games.csv")
teams = pd.read_csv("./teams.csv")
games["GAME_DATE_EST"] = pd.to_datetime(games["GAME_DATE_EST"])
config = twint.Config
config.Store_csv = True
config.Stats=True
for index, game in tqdm(games.iterrows(), total=games.shape[0]):
    try:
        game_id = game.GAME_ID
        home_team_id = game['HOME_TEAM_ID']
        print(game_id, home_team_id)
        home_team_details = teams[teams["TEAM_ID"] == game["TEAM_ID_home"]]
        away_team_details = teams[teams["TEAM_ID"] == game["TEAM_ID_away"]]
        home_nickname = home_team_details.NICKNAME.values[0]
        visitor_nickname = away_team_details.NICKNAME.values[0]
        home_twitter_handle = home_team_details.TWITTER_HANDLE.values[0]
        visitor_twitter_handle = away_team_details.TWITTER_HANDLE.values[0]
        game_date = game.GAME_DATE_EST
        week_before = game_date - pd.Timedelta(weeks=1)
        game_date_as_string = game_date.strftime('%Y-%m-%d %H:%M:%S')
        week_before_as_string = week_before.strftime('%Y-%m-%d %H:%M:%S')
        config.Since = week_before_as_string
        config.Until = game_date_as_string
        config.Hide_output = True
        config.To = home_twitter_handle
        config.Output = f"tweets/{game_id}_{home_nickname}.csv"
        twint.run.Search(config)

        config.To = visitor_twitter_handle
        config.Output = f"tweets/{game_id}_{visitor_nickname}.csv"
        twint.run.Search(config)
        time.sleep(1)
    except Exception as e:
        time.sleep(30)
    