# Dataset Processing

Note: All data sort by timestamp. Index of user and item start at 1.

## Introduction

For each dataset, we filter out users and items less than five interactions. We only keep the latest fifty items for a sequence. For each sequence, the last item is for testing, the second last item is for validation, and the rest items are for training.

## Dataset

We use [Beauty](http://jmcauley.ucsd.edu/data/amazon/links.html), [Sports_and_Outdoors ](http://jmcauley.ucsd.edu/data/amazon/links.html)and [Yelp](https://www.yelp.com/dataset) datasets for experiments. Please download them from official websites and put them in the root directory. Then run the following commands to process the datasets.

```python
python DataProcessing-beauty.py
python DataProcessing-sports.py
python DataProcessing-yelp.py
```

By running the above scripts, we can get the file named *train.txt*, *valid.txt*, *test.txt*. Next, run the following command to further process the data (for different datasets, please revise the path in *further_process.py*). 

```python
python further_process.py
```

We can get the file *train.dat*, *valid.dat*, *test.dat*, *valid_neg.dat* and *test_neg.dat*. As for the file *train.dat*, *valid.dat* and *test.dat*, each row contains a unique user interaction sequence. Files with **_neg.dat* are negative sample files. 

Finally, we can run following command to create simulated datasets.

```python
python create_noise_data.py
```

Function *raw_data_rand_modify* randomly modifies test sequences. For each item in a sequence (excluding the last two items, since they are used for validation and testing),  we keep it with 80% probability, delete it with 10% probability, or add an item before it with 10% probability. The probability of continuing to add an item before it is 10%*10%, and so on.

Statistics:
Beauty: user-22362, item-12101
Sports: user-35597, item-18357
Yelp: user-22844, item-16552