import pandas as pd

dir = Path("./data/")

# based on published results
games = pd.read_csv("data/games.csv")
train_games = games[games.GAME_ID.isin(train_data.GAME_ID)]
test_games = games[games.GAME_ID.isin(test_data.GAME_ID)]
features = ['FG_PCT_home','FG_PCT_away','FG3_PCT_home','FG3_PCT_away'] 

train_feats = train_games[features]
train_labels = train_games['HOME_TEAM_WINS']
test_feats = test_games[features]
test_labels = test_games['HOME_TEAM_WINS']

model = DecisionTreeClassifier()
model.fit(train_feats, train_labels)
test_pred = model.predict(test_feats)
fpr, tpr, _ = metrics.roc_curve(test_labels, test_pred)
auc = metrics.auc(fpr,tpr)
acc = metrics.accuracy_score(test_labels, test_pred)
f1 = metrics.f1_score(test_labels, test_pred, average="weighted")
print(f"AUC={round(auc,2)}, ACC={round(acc,2)}, F1={round(f1,2)}")


# other strong baselines created
game_results = pd.read_csv(dir / "games_with2_team_tweets.csv", index_col=0)
teams_home = pd.read_csv(dir / "teams.csv")
teams_away = pd.read_csv(dir / "teams.csv")
all_games = pd.read_csv(dir / "games.csv")


all_games['GAME_DATE_EST'] = pd.to_datetime(all_games['GAME_DATE_EST'])

game_results = game_results[['GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'HOME_TEAM_WINS']]
game_results['GAME_DATE_EST'] = pd.to_datetime(game_results['GAME_DATE_EST'])


game_results = game_results.reset_index()

all_games = all_games.dropna()


def get_strong_baseline_home(row):
    home_team_games_playing_home = all_games[(all_games["HOME_TEAM_ID"] == row["HOME_TEAM_ID"])
                                     &(all_games["GAME_DATE_EST"] < row["GAME_DATE_EST"])]
    home_team_games_playing_away = all_games[(all_games["VISITOR_TEAM_ID"] == row["HOME_TEAM_ID"])
                                     &(all_games["GAME_DATE_EST"] < row["GAME_DATE_EST"])]
    total_home_team_games = home_team_games_playing_home.shape[0] + home_team_games_playing_away.shape[0]
 
    home_team_wins_playing_home = home_team_games_playing_home[home_team_games_playing_home["HOME_TEAM_WINS"]==1].shape[0]
    home_team_wins_playing_away = home_team_games_playing_away[home_team_games_playing_away["HOME_TEAM_WINS"]==0].shape[0]
    home_team_wins = home_team_wins_playing_home + home_team_wins_playing_away
 
    home_team_losses_playing_home = home_team_games_playing_home[home_team_games_playing_home["HOME_TEAM_WINS"]==0].shape[0]
    home_team_losses_playing_away = home_team_games_playing_away[home_team_games_playing_away["HOME_TEAM_WINS"]==1].shape[0]
    home_team_losses = home_team_losses_playing_home + home_team_losses_playing_away
 
    return home_team_wins/home_team_losses


 
home_win_loss = game_results.apply(get_strong_baseline_home, axis=1)



def get_strong_baseline_away(row):
    away_team_games_playing_away = all_games[(all_games["VISITOR_TEAM_ID"] == row["VISITOR_TEAM_ID"])
                                     &(all_games["GAME_DATE_EST"] < row["GAME_DATE_EST"])]
    away_team_games_playing_home = all_games[(all_games["HOME_TEAM_ID"] == row["VISITOR_TEAM_ID"])
                                     &(all_games["GAME_DATE_EST"] < row["GAME_DATE_EST"])]
    total_away_team_games = away_team_games_playing_away.shape[0] + away_team_games_playing_home.shape[0]
 
    away_team_wins_playing_away = away_team_games_playing_away[away_team_games_playing_away["HOME_TEAM_WINS"]==0].shape[0]
    away_team_wins_playing_home = away_team_games_playing_home[away_team_games_playing_home["HOME_TEAM_WINS"]==1].shape[0]
    away_team_wins = away_team_wins_playing_away + away_team_wins_playing_home
 
    away_team_losses_playing_away = away_team_games_playing_away[away_team_games_playing_away["HOME_TEAM_WINS"]==1].shape[0]
    away_team_losses_playing_home = away_team_games_playing_home[away_team_games_playing_home["HOME_TEAM_WINS"]==0].shape[0]
    away_team_losses = away_team_losses_playing_away + away_team_losses_playing_home
 
    return away_team_wins/away_team_losses


 
away_win_loss = game_results.apply(get_strong_baseline_away, axis=1)



game_results['HOME_W_L'] = home_win_loss
game_results['AWAY_W_L'] = away_win_loss


game_results['PRED_HOME_TEAM_WINS'] = game_results.apply(lambda row: 1 if row.HOME_W_L >= row.AWAY_W_L else 0, axis=1)

actual = game_results['HOME_TEAM_WINS']
predicted = game_results['PRED_HOME_TEAM_WINS']

strong_baseline_preds = pd.DataFrame(
        {"y": predicted, "id": dataset_games.GAME_ID}
    )
strong_baseline_preds.to_csv(dir / "strong_baseline_preds.csv")


