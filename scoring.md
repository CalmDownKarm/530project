# Evaluation script

## Metrics

### F1 score ([reference](https://en.wikipedia.org/wiki/F-score))
The F1 score is the harmonic mean of precision and recall and represents a measure of the test's accuracy. Values range from 0-1. Precision is the proportion of correct positive predictions among the total number of positive predictions. Recall is the number of correct positive predictions among the total number of actual positive observations. The following equations are used to compute F1 score:
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1score = 2 * (precision * recall) / (precision + recall)
```


### Accuracy ([reference](https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification))
Accuracy is the number of correct predictions over the total number of predictions. Values range from 0-1. To compute accuracy, we simply use the following equation: 

```
Accuracy = # of correct predictions / # of total predictions
         = (TP + TN) / (TP + TN + FP + FN)
```

## Usage
Use the following command at the command line. Each argument is explained below.

```
score.py [-h] -p PREDICTED_PATH -a ACTUAL_PATH

  -p PREDICTED_PATH, --predicted PREDICTED_PATH
                        path to model's prediction labels file
  -a ACTUAL_PATH, --actual ACTUAL_PATH
                        path to true/actual y labels file
```

### Example
Consider the following files:

*predicted.csv*
```
id,y
1,0
2,1
3,1
4,1
```
*actual.csv*
```
id,y
1,0
2,0
3,1
4,1
```

We run the following command using these files:

`$ python3 score.py -p predicted.csv -a actual.csv`

Here is the output:
```
F1 score: 0.7333333333333334
Accuracy: 0.75
```