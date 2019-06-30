Fake Review Detection and Analysis
===================================

data ---> This is where all the CSV files to be imported are stored

final_results ---> Empty folder. This where the classification report for all classification models, their precision-recall curve and confusion matrix will be saved. 

mallet-2.0.8 ---> The LDA model from UMass used for topic modelling.

lda.py ---> This is the file from where different functions will be imported in the main pipeline jupyter notebook SecurityAnalytics_Project_Final.ipnyb for topic modelling of reviews.

models_final.py ---> This file is used to train and test the different classification models used for the project. This file generates the classification report, precision-recall curve and confusion matrix.

requirements.txt ---> This file contains all the dependencies required for the project to compile.

SecurityAnalytics_Project_Final.ipnyb ---> This is the main pipeline of the project. This is the only file other than StackNet.py that has to be run for the entire project to compile.

Stacknet.py ---> This trains and tests the Stacknet boosting model used in the project and also generates the classification report, precision-recall curve and confusion matrix for the same. This was not included in the main pipeline as it did not run from jupyter notebook and was therefore run from the GPU cluster provided for the course. It requires the file models_final.py for its compilation.

text_feat.py ---> This file is used to generate all the textual features as mentioned in the report.


======================================
Instructions
======================================

1. Unzip the submitted folder
2. cd into the folder
3. create a virtual environment and activate it using the commands:
    virtualenv venv --python=python3.6
    source venv/bin/activate (to activate the venv)
4. install the dependencies of the project using the command:
    pip install requirements.txt
5. create a new folder new_mallet in the C: drive and copy the mallet-2.0.8 folder inside the new_mallet folder.
6. from the project folder open jupyter notebook using the command:
    jupyter notebook
7. open the SecurityAnalytics_Project_Final.ipnyb notebook and run it.
8. all the results will be saved in the final_results folder.
9. 4 new csv files will be saved in the data folder which will be used as data sources for the StackNet.py:
    X.csv
    X_text.csv
    y.csv
    y_text.csv
10. on the gpu cluster, create a project folder.
11. repeat steps 2 and 3
12. copy the following files to the folder:
    StackNet.py
    models_final.py
    requirements.txt
    X.csv (from data folder)
    X_text.csv (from data folder)
    y.csv (from data folder)
    y_text.csv (from data folder)
13. in the project folder download pystacknet using the following steps:
    git clone https://github.com/h2oai/pystacknet
    cd pystacknet
    python setup.py install
14. in the project folder create a folder named final_results where the results will be saved
15. now run the stacknet using the command:
    python StackNet.py
16. the final results will be saved in the folder final_results.