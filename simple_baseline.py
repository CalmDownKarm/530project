from pathlib import Path
import pandas as pd
import numpy as np

# HOME_TEAM_WINS or TEXT_BASELINE
baseline = "HOME_TEAM_WINS"

dir = Path("./data/")
list(dir.glob("*"))
tweets_dir = dir / "tweets"

all_games = pd.read_csv(dir / "games.csv")
all_games["GAME_ID"] = all_games["GAME_ID"].astype(str)

tweets_for_games = pd.read_csv(dir / "Tweets_for_games_with2_team_tweets.csv")
tweet_counts = (
    tweets_for_games.groupby(["game_id", "team_nick"])
    .count()
    .reset_index()[["game_id", "team_nick", "id"]]
)
relevant_game_ids = tweets_for_games.game_id.unique()
relevant_game_ids = [str(int(ID)) for ID in relevant_game_ids if np.isnan(ID)==False]
dataset_games = all_games[all_games["GAME_ID"].isin(relevant_game_ids)]

teams = pd.read_csv(dir / "teams.csv")
nicks_to_id = {team.NICKNAME: team.TEAM_ID for _, team in teams.iterrows()}

def get_higher_number_of_tweets(row):
    home_team_counts = tweet_counts[
        (tweet_counts.game_id == row["GAME_ID"])
        & (tweet_counts.team_id == row["HOME_TEAM_ID"])
    ]["id"].values[0]
    away_team_counts = tweet_counts[
        (tweet_counts.game_id == row["GAME_ID"])
        & (tweet_counts.team_id == row["TEAM_ID_away"])
    ]["id"].values[0]
    if home_team_counts >= away_team_counts:
        return 1
    return 0

if baseline == "HOME_TEAM_WINS":
    home_team_wins_baseline_preds = pd.DataFrame(
        {"y": [1] * dataset_games.shape[0], "id": dataset_games.GAME_ID}
    )
    home_team_wins_baseline_preds.to_csv(dir / "home_team_wins_baseline_preds.csv")

if baseline == "TEXT_BASELINE":
    tweet_counts["team_id"] = tweet_counts["team_nick"].map(nicks_to_id)
    weak_baseline_preds = dataset_games.apply(get_higher_number_of_tweets, axis=1)
    pd.concat([weak_baseline_preds, dataset_games["GAME_ID"]], axis=1).reset_index(
        drop=True
    ).rename({0: "y", "GAME_ID": "id"}, axis=1).to_csv(
        dir / "text_baseline.csv", index=None
    )

