**Separating Stars from Quasars: Machine Learning Investigation Using Photometric Data**

A problem that lends itself to the application of machine learning is classifying matched sources in the Galex (Galaxy
Evolution Explorer) and SDSS (Sloan Digital Sky Survey) catalogs into stars and quasars based on color-color
plots. The problem is daunting because stars and quasars are still inextricably mixed elsewhere in the color-color
plots and no clear linear/non-linear boundary separates the two entities. Diversity and volume of samples add
to the complexity of the problem.

We explored the efficacy of LightGBM Algorithm in indiscriminating between stars and quasars using GALEX and SDSS photometric data.

To run train the data, install the below python module requirements

1. lightgbm
2. sklearn
3. numpy
4. pandas

To train the dataset, type 

python train.py 
