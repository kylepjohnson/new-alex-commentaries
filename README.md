# About
Project for scraping and parsing ancient Greek literature citations from public domain Classics books  

1) Find public domain books related to Illiad and Odyssey
2) Find each citation (book & line, section number) in the public domain books 
3) Write citations URL to database

http://homer.beta.newalexandria.info/

# Data sources

- Archive.org Books UI: https://archive.org/details/books
- Archive.org API: https://archive.org/services/docs/api/


# Key words

When searching Archive.org, you will need to find the resources with key words included.

- Illiad and Odyssey: https://archive.org/search.php?query=iliad%20OR%20odyssey%20AND%20mediatype%3Atexts

# Examples

- https://archive.org/details/firstthreebooks03homegoog/page/n224/mode/2up
- https://archive.org/details/sprachwissenscha00herm/page/52/mode/2up


# Example map

Homer, Iliad, Book 1, Line 1: https://archive.org/details/firstthreebooks03homegoog/page/n122/mode/2up

# Pipeline

## Step 1: Data Scraping

Query user-defined documents, such as `Homer Iliad OR Odyssey` for citations and label the data during the process of scraping.

The text data is scraped by lines and an example of the labeled data is shown as follows:\
*{'content': 'Megarians for Salamis, they quoted Iliad 2. 558, where ', 'annotations': [{'start': 35, 'end': 47, 'label': 'Citation'}]}*\
Here the `start` and `end` variables record the indices of the begining and end of the citation included in the `content`.

The procedure of data scraping is illustrated in `1_do_scraping.ipynb`
 

## Step 2: Train BERT Model

Based on the labeled dataset, train a Token Classification BERT model for Named Entity Recognition (NER) that automatically finds citations (For more information: [Link](https://huggingface.co/transformers/model_doc/bert.html#bertfortokenclassification)).

The procedure of model training using a sample dataset is shown in `2_do_ner_training.ipynb`.

## Step 3: Inference Evaluation

By adopting trained model, we can predict citations for input texts. The performace of the BERT model is evaluated based on the precision and recall.

The procedure of inference evaluation is shown in `3_do_ner_inference.ipynb`.



### To Understand Evaluation Results

The inference evaluation result of a BERT model trained on 5000 positive & 5000 negative citation instances is shown below.

![inference_eval](https://user-images.githubusercontent.com/43858610/133504138-1f5a9908-eb0f-4770-bb80-b6de33d8549a.png)

#### Variable Meaning
1. `precision` = `tp / (tp + fp)` where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.

2. `recall` = `tp / (tp + fn)` where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.

3. `F-1 score` can be interpreted as a weighted harmonic mean of the precision and recall, where an F-1 score reaches its best value at 1 and worst score at 0.
The F-1 score weights recall more than precision by a factor of beta. beta == 1.0 means recall and precision are equally important.

4. `support` is the number of occurrences of each class in y_true

#### Avg values
`micro`:
Calculate metrics globally by counting the total true positives, false negatives and false positives.

`macro`:
Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.

`weighted`:
Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
