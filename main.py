from utils import *

print("========= Stimulates Start =========")
print("Algorithms use: ")
print("Chi-Squared")
team_file = getNewDataset()
subteam2(team_file.train, team_file.resultColName,
         team_file.listFileTest, 1, 0.1, 25)
