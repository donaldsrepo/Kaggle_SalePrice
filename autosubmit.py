import subprocess
i = 0
message = "autosubmit_" + str(i)
subprocess.run(["kaggle", "competitions", "submit", "-c", "home-data-for-ml-course", "-f", "submission.csv", "-m", "'auto_submit'"])


#kaggle competitions submit -c home-data-for-ml-course -f submission.csv -m "auto_submit"