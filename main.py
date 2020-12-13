from utils import *
from pathlib import Path

while True:
    print("========= Stimulates Start =========")
    print(textcolor_display("Algorithms you want to use: ", color.WARNING))
    print("1. Pearson Correlation")
    print("2. Chi-Squared")
    print("3. Recursive Feature Elimination")
    print("4. Lasso: SelectFromModel")
    print("5. Tree-based: SelectFromModel")
    print("6. Run only dataset summary!")
    algo_choice = input("Input : ")
    algo_choice = int(algo_choice)
    team_file = getNewDataset()
    if algo_choice == 1:
        print("1. Pearson Correlation")
        subteam2(team_file.train, team_file.resultColName,
                          team_file.listFileTest, 1, 0.1, Select_Method.Pearson, 25,10)
    if algo_choice == 2:
        print("2. Chi-Squared")
        subteam2(team_file.train, team_file.resultColName,
                team_file.listFileTest, 1, 0.1, Select_Method.Chi, 25,10)
    if algo_choice == 3:
        print("3. Recursive Feature Elimination")
        subteam2(team_file.train, team_file.resultColName,
                team_file.listFileTest, 1, 0.1, Select_Method.Recursive, 25, 10)
    if algo_choice == 4:
        print("4. Lasso: SelectFromModel")
        subteam2(team_file.train, team_file.resultColName,
                team_file.listFileTest, 1, 0.1, Select_Method.Lasso, 25, 10)
    if algo_choice == 5:
        print("5. Tree-based: SelectFromModel")
        subteam2(team_file.train, team_file.resultColName,
                team_file.listFileTest, 1, 0.1, Select_Method.Tree, 25, 10)
    if algo_choice == 6:
        exec(Path("datasets_summary.py").read_text())
    print(textcolor_display("Run stimulations again? ", color.WARNING))
    again_no = str(input(textcolor_display("Enter no for stop, otherwise : ", color.OKBLUE)))
    if 'no' in again_no:
        print(textcolor_display("Terminated the stimulations", color.FAIL))
        break
    else:
        print(textcolor_display("========= Welcome back =========", color.OKGREEN))
