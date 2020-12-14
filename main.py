from utils import *

print("========= Stimulates Start =========")
print("Algorithms use: ")
print("Chi-Squared")
team_file = getNewDataset()
for x in range(len(team_file.listFileTest) + 1):
    if(x!=0):
        tmp_name = team_file.train
        team_file.train = team_file.listFileTest[x-1]
        team_file.listFileTest[x - 1] = tmp_name
        subteam2(team_file.train, team_file.resultColName,
                 team_file.listFileTest, 1, 0.1, 25)
    else:
        subteam2(team_file.train, team_file.resultColName,
                 team_file.listFileTest, 1, 0.1, 25)
