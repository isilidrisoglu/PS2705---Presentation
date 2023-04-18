# Titanic Survival Prediction with SVM

## Introduction

Titanic dataset in Kaggle is a very popular dataset to create toy examples using SVMs. It includes features that gave information of passengers of Titanic such as passanger name, class, cabin, ticket, sex, age,  the number of parents/siblings/spouses/children who were on the ship with the passenger etc. 
Using this dataset we will try to build an SVM model to predict survival in Titanic.  

## How to run 

Prerequisities:
* Conda or miniconda

Installation Steps:
* Create a new environment. `conda create -n titanic python=3.8`
* Activate the env. `conda activate titanic`
* Install jupyter notebook.
    * `conda install -c conda-forge notebook`
    * `conda install -c conda-forge nb_conda_kernels`

* Install requirements. `pip3 install -r requirements.txt`


Run Jupyter Notebook (recommended):
* Open a terminal activate environment. `conda activate titanic`
* Run jupyter notebook. `jupyter notebook`
* Change kernel to `Python [conda env:titanic]` under Kernel tab.

Run .py:
* Open a terminal activate environment. `conda activate titanic`
* Run python file. `python titanic-svm.py`

