# Home Team Always Wins
A common idea in sports is that of `Home Team Advantage` aka, the idea that the home team is more likely to win, thus for a simple baseline we evaluate this idea.
The nba games dataset records whether or not the Home Team wins for every game, thus for this baseline we predict that the home team always wins. 

This baseline is not great, with an F1 of 0.42 and Accuracy of 0.58

# Number of Tweets Baseline
The second possible simple baseline is to just count the number of tweets in our dataset to each team before each game. This performs about the same, with an F1 score of 0.52 and Accuracy of 0.53

# Generating a baseline
Look at simple_baseline.py, change the string baseline on line 5  to either HOME_TEAM_WINS or TEXT_BASELINE to create a predictions file that the scorer.py will accept. Default storage location is ./data/
