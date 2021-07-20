# Inference Evaluation

## Seqeval Running Result 

### Variable Meaning
1. `precision` = `tp / (tp + fp)` where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.

2. `recall` = `tp / (tp + fn)` where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.

3. `F-1 score` can be interpreted as a weighted harmonic mean of the precision and recall, where an F-1 score reaches its best value at 1 and worst score at 0.
The F-1 score weights recall more than precision by a factor of beta. beta == 1.0 means recall and precision are equally important.

4. `support` is the number of occurrences of each class in y_true

### Avg values
`micro`:
Calculate metrics globally by counting the total true positives, false negatives and false positives.

`macro`:
Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.

`weighted`:
Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
