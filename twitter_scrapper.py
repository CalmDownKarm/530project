import twint
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

games_details = pd.read_csv("./games_details.csv")
games = pd.read_csv("./games.csv")
teams = pd.read_csv("./teams.csv")
one_game = games.sample(1)

home_team_details = teams[teams["TEAM_ID"]== one_game.HOME_TEAM_ID.values[0]]
visitor_team_details = teams[teams["TEAM_ID"]==one_game.VISITOR_TEAM_ID.values[0]]
print(home_team_details.info())
ht = home_team_details.NICKNAME.values[0]
vt = visitor_team_details.NICKNAME.values[0]
search_term  = f"({ht} OR {vt})"
print(visitor_team_details)
print(search_term)

game_date = pd.to_datetime(one_game.GAME_DATE_EST)
week_before = game_date - pd.Timedelta(weeks=1)
week_before = week_before.dt.strftime('%Y-%m-%d %H:%M:%S').values[0]
game_date = game_date.dt.strftime('%Y-%m-%d %H:%M:%S').values[0]
print(week_before, game_date)

config = twint.Config
# config.Search = "to:MiamiHEAT OR to:HoustonRockets"
config.To = "@HoustonRockets"
config.Since = "2019-10-11"
config.Until = "2019-10-18"
config
twint.run.Search(config)
