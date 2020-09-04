#!/usr/bin/bash
jupyter nbconvert --to script Ensemble_73.ipynb
kaggle competitions submit -c home-data-for-ml-course -f submission.csv -m "Ensemble_73" 
#kaggle competitions submit -c house-prices-advanced-regression-techniques -f submission.csv -m "Ensemble_73"
kaggle competitions submissions -c home-data-for-ml-course > scores_Ensemble_73.txt
cp submission.csv  submission_73.csv
head -10 scores_Ensemble_73.txt
#kaggle competitions leaderboard -s -c home-data-for-ml-course 
