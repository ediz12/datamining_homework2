# datamining_homework2
You are asked to implement naive bayes classifier on abalone dataset. 
Detailed information about abalone dataset can be found at
http://archive.ics.uci.edu/ml/datasets/Abalone

The aim of the dataset is to predict the age of abalone from physical measurements.
Originally it is a regression problem in which the output is age in years. However,
we will use it as a classification problem. The age value is discretized as young,
middle-aged, and old. The dataset (input features and class labels of the samples) is
provided as a seperate text file (abalone_dataset.txt):

input: Sex,Length,Diameter,Height,Whole weight,Shucked weight,Viscera weight,Shell
weight

output: class label which is the last column of the dataset
(less than 8 in age belongs to class 1 (young), between 8 and 12 to class 2 (middle-
aged), greater than 12 to class 3 (old))   

Optimization is not required in Naive Bayes classification. So, the dataset will be
divided into training and validation sets only (there will not be a test set). Assume
gaussian distribution for continuous features. 

You are asked to implement naive bayes classifier for the following six cases:

1) Using only the first 3 features (sex, length, and diameter) as input, and

  1.1) 100 samples for training, and rest for validation set
  1.2) 1000 samples for training, and rest for validation set

2) Using all features as input, and

  2.1) 100 samples for training, and rest for validation set
  2.2) 1000 samples for training, and rest for validation set

For each of the above cases,
- Report how many total misclassification errors are there on the
training and validation sets, together with the confusion matrices.
(Note: A confusion matrix is a 3x3 matrix (if # of classes is 3) where entry (i,j)
contains the number of instances belonging to i but are assigned to j; ideally it
should be a diagonal matrix.)

- Report the case in which highest accuracy is obtained. Write your comments about
the results. 
