# Win/Loss Ratio Baseline
The idea we had here with the strong baseline is: for a specific game, we predict a team will win if it has the better record (thus the win/loss ratio) from previously played games. We don't discriminate per season, so we look at records of teams since 2004.

This baseline has an F1 of 0.58 and Accuracy of 0.54

# Generating a baseline
Look at simple_baseline.py, change the string baseline on line 5  to either HOME_TEAM_WINS or TEXT_BASELINE to create a predictions file that the scorer.py will accept. Default storage location is ./data/
