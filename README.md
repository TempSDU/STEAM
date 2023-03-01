# A Self-Correcting Sequential Recommender

> Codes for WWW'23: A Self-Correcting Sequential Recommender

## Introduction

Sequential recommendations aim to capture users’ preferences from their historical interactions so as to predict the next item that they will interact with. Sequential recommendation methods usually assume that all items in a user’s historical interactions reflect her/his preference and transition patterns between items. However, real-world interaction data is imperfect in that (i) users might erroneously click on items, i.e., so-called misclicks on irrelevant items, and (ii) users might miss items, i.e., unexposed relevant items due to inaccurate recommendations.

To tackle the two issues listed above, we propose STEAM, a Self-correcTing sEquentiAl recoMmender. STEAM first corrects an input item sequence by adjusting the misclicked and/or missed items. It then uses the corrected item sequence to train a recommender and make the next item prediction. We design an item-wise corrector that can adaptively select one type of operation for each item in the sequence. The operation types are ‘keep’, ‘delete’ and ‘insert.’ In order to train the item-wise corrector without requiring additional labeling, we design two self-supervised learning mechanisms: (i) deletion correction (i.e., deleting randomly inserted items), and (ii) insertion correction (i.e., predicting randomly deleted items). We integrate the corrector with the recommender by sharing the encoder and by training them jointly.

## Requirements

To install the required packages, please run:

```python
pip install -r requirements.txt
```

## Datasets

We use [Beauty](http://jmcauley.ucsd.edu/data/amazon/links.html), [Sports_and_Outdoors ](http://jmcauley.ucsd.edu/data/amazon/links.html)and [Yelp](https://www.yelp.com/dataset) datasets for experiments. We have uploaded the processed datasets here. However, if you download raw datasets from official websites, please refer to *./dataprocessing/readme.md* for the details about dataset processing.

## Experiments

For training, validating and testing model on *Beauty* dataset, please run:

```python
python main.py -m=train
python main.py -m=valid
python main.py -m=test
```

For other datasets, please revise the path of dataset and item_num in *main.py*.

If you want to set the probabilities of keep, delete and insert for generating randomly modified sequences when training, please revise the plist in *main.py*.

If you want to get the performances on the changed sequence group and the unchanged sequence group under a certain epoch, for example, evaluate the 100th epoch, please run:

```python
python evaluate_on_mod.py -e 100
```

Tips:
We run on the Tesla A100, which contains 40g of video memory, so we can set the maximum batch size to 256. If there is no such large resource, you need to set a smaller batch size in *main.py*. We also upload the trained models to *./checkpoint/*.
