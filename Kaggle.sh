#!/usr/bin/bash
jupyter nbconvert --to script StackEnsemble_86e.ipynb
kaggle competitions submit -c home-data-for-ml-course -f submission.csv -m "StackEnsemble_86e" 
#kaggle competitions submit -c house-prices-advanced-regression-techniques -f submission.csv -m "StackEnsemble_86e"
kaggle competitions submissions -c home-data-for-ml-course > scores_StackEnsemble_86e.txt
cp submission.csv  submission_Ensemble_86e.csv
head -10 scores_StackEnsemble_86e.txt
#kaggle competitions leaderboard -s -c home-data-for-ml-course 
