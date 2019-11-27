# **Separating Stars from Quasars: Machine Learning Investigation Using Photometric Data**

A problem that lends itself to the application of machine learning is classifying matched sources in the Galex (Galaxy
Evolution Explorer) and SDSS (Sloan Digital Sky Survey) catalogs into stars and quasars based on color-color
plots. The problem is daunting because stars and quasars are still inextricably mixed elsewhere in the color-color
plots and no clear linear/non-linear boundary separates the two entities. Diversity and volume of samples add
to the complexity of the problem.

We explored the efficacy of GBT in indiscriminating between stars and quasars using GALEX and SDSS photometric data.

To run train the data, install the below python modules

1. sklearn
2. numpy
3. pandas

There are 4 datasets. 

The datasets are split in 85:15 ratio before training. 85% for training data and 15% for test data.

To train the model on a particular catelogue, run the respective new.py file

For example : To train catelogue 4 type,

python3 new4.py


