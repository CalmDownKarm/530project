# Extension 1
Extension 1 is a simple AWD LSTM trained on concatenated tweets from each game. This is unideal, because it assumes the model is able to assume whom the home and away teams are, but it serves a simple extension and proof of concept.

The model achieves an accuracy of .56 and an f1 score of .67 which is better in f1 than the simple baselines.

This extension depends on the Fastai library - and assumes the data is already preprocessed, ie the tweets for all the games are concatenated together in the dataframe. 

to run the code, create a folder called data in the same directory as the script with the preprocessed data file, then run ```python extension1fastai.py```