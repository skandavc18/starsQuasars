#**Separating Stars from Quasars: Machine Learning Investigation Using Photometric Data**

A problem that lends itself to the application of machine learning is classifying matched sources in the Galex (Galaxy
Evolution Explorer) and SDSS (Sloan Digital Sky Survey) catalogs into stars and quasars based on color-color
plots. The problem is daunting because stars and quasars are still inextricably mixed elsewhere in the color-color
plots and no clear linear/non-linear boundary separates the two entities. Diversity and volume of samples add
to the complexity of the problem.

We explored the efficacy of LightGBM Algorithm in indiscriminating between stars and quasars using GALEX and SDSS photometric data.

What is LightGBM ?

Light GBM is a gradient boosting framework that uses tree based learning algorithm.

How it differs from other tree based algorithm?

>Light GBM grows tree vertically while other algorithm grows trees horizontally meaning that Light GBM grows tree leaf-wise while other >algorithm grows level-wise. It will choose the leaf with max delta loss to grow. When growing the same leaf, Leaf-wise algorithm can >reduce more loss than a level-wise algorithm.

Source : https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc

To run train the data, install the below python modules

1. lightgbm
2. sklearn
3. numpy
4. pandas

There are 2 datasets. 

data.csv is taken from https://drive.google.com/drive/folders/1uoqTK81QoqPkxJ7-zHO03zpciECWpvp7 . The v2_predicted.csv is renamed as data.csv

cat4.csv is taken from https://drive.google.com/drive/folders/1aAVtdO_b6AsC6Agww0OgpdKYCDOoZ__p (category4 folder)

The datasets are split in 80:20 ratio before training. 80% for training data and 20% for test data.

 
To train using data.csv dataset, type 
python train.py

To train the cat4.csv dataset, type
python train2.py

