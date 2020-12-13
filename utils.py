import os

import pandas as pd
import xlsxwriter
from mpmath import *
from numpy.random import default_rng
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler


# creating the DataFrame
def exportResults(file_name, file_train_name, list_test_file_name, result_list_in_orders):
    number_of_rows = len(result_list_in_orders)
    number_of_test = len(list_test_file_name)
    if (len(list_test_file_name) != 5):
        print("Sorry! We are not implement auto numbber of tex")
    workbook = xlsxwriter.Workbook(file_name)
    worksheet = workbook.add_worksheet(name="Chi-Squared")
    # Increase the cell size of the merged cells to highlight the formatting.
    worksheet.set_column('A:C')
    worksheet.set_column('A:B', 60)
    worksheet.set_column('C:C',40)
    # Create a format to use in the merged range.
    merge_format = workbook.add_format({
        'bold': 1,
        'border': 1,
        'align': 'center',
        'valign': 'vcenter',
        'fg_color': 'yellow'})
    merge_format_2 = workbook.add_format({
        'bold': 1,
        'border': 1,
        'align': 'center',
        'font_color': 'white',
        'valign': 'vcenter',
        'fg_color': 'green'})
    merge_format_3 = workbook.add_format({
        'font_color': 'green'
    })
    merge_format_4 = workbook.add_format({
        'bottom': 1,
        'left': 1
    })
    merge_format_2.set_text_wrap()
    merge_format.set_text_wrap()
    # Merge 3 cells.
    worksheet.merge_range('A1:A' + str(number_of_rows), "Train by " + file_train_name, merge_format_2)
    # Merge 3 cells over two rows.
    count_test = 0
    for x in range(number_of_test):
        first_row = x * 6 + 1
        last_row = x * 6 + 6
        worksheet.conditional_format('C' + str(last_row) + ':C' + str(last_row), {'type': 'no_blanks', 'format': merge_format_4})
        if (count_test <= (len(list_test_file_name) - 1)):
            worksheet.merge_range('B' + str(first_row) + ':B' + str(last_row),
                                  "Test on " + list_test_file_name[count_test], merge_format)
            count_test += 1

    for x in range(number_of_rows):
        if "Random" in result_list_in_orders[x]:
            worksheet.write('C'+str(x+1),result_list_in_orders[x])
        else:
            worksheet.write('C'+str(x+1),result_list_in_orders[x],merge_format_3)
    workbook.close()
    print('Data is written to excel File.')


class TeamFile:
    # instance attribute
    def __init__(self, train, listFileTest, resultColName):
        self.train = train
        self.listFileTest = listFileTest
        self.resultColName = resultColName


dirname = os.path.dirname(__file__)


def getNewDataset():
    train = os.path.join(dirname, 'data/ibdfullHS_iCDf_x.csv')  # iCDr & UCf &iCDf &CDr&CDf ibdfullHS_iCDf_x
    fileListTest = []
    fileListTest.append(os.path.join(dirname, 'data/ibdfullHS_iCDr_x.csv'))
    fileListTest.append(os.path.join(dirname, 'data/ibdfullHS_CDr_x.csv'))
    fileListTest.append(os.path.join(dirname, 'data/ibdfullHS_CDf_x.csv'))
    fileListTest.append(os.path.join(dirname, 'data/ibdfullHS_UCr_x.csv'))
    fileListTest.append(os.path.join(dirname, 'data/ibdfullHS_UCf_x.csv'))
    return TeamFile(train, fileListTest, "RS")


def findImportancesFeatures(resultColName, filenameTrain, coef_percent, nlargestFeatures, num_feats):
    print("Feature selection method to be used : " + "Chi-Squared")
    print(str("Train by file ") + str(filenameTrain))
    data = pd.read_csv(filenameTrain)
    colName = data.columns
    df = pd.DataFrame(data, columns=colName)
    df.head()
    X = df[colName]
    y = df[resultColName]
    X_No_First = df.drop(df.columns[0], axis=1)
    X_norm = MinMaxScaler().fit_transform(X_No_First)
    print("num_feats = " + str(num_feats))
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X_norm, y)
    chi_support = chi_selector.get_support()
    chi_feature = X_No_First.loc[:, chi_support].columns.tolist()
    importanceFeature = chi_feature
    print("Number feature selected : " + str(len(importanceFeature)))
    X_Train_ImportFeature = df[importanceFeature]
    y_Train_ImportFeature = y
    return importanceFeature, X_Train_ImportFeature, y_Train_ImportFeature


def findRandomeFeaturesSets(resultColName, filenameTrain, sizeIF):
    data = pd.read_csv(filenameTrain)
    colName = data.columns
    df = pd.DataFrame(data, columns=colName)
    df.head()
    y = df[resultColName]
    rng = default_rng()
    # In colName has n columns, position of RS is n - 1. Because of a noname rows of V1,V2,V3,...
    numbers = rng.choice(len(colName) - 2, size=sizeIF, replace=False)
    randomeFeatureSameSize = colName.delete(0).take(numbers)
    X_Train_Random = df[randomeFeatureSameSize]
    y_Train_Random = y
    return randomeFeatureSameSize, X_Train_Random, y_Train_Random


def printResult(acc_random, mcc_random, auc_random, acc_if, mcc_if, auc_if, nTimes):
    print("When do Random ")
    print("ACC = " + str(acc_random / nTimes))
    print("MCC = " + str(mcc_random / nTimes))
    print("AUC = " + str(auc_random / nTimes))
    print("+++++ ")
    print("When we got Importance Features")
    print("ACC = " + str(acc_if / nTimes))
    print("MCC = " + str(mcc_if / nTimes))
    print("AUC = " + str(auc_if / nTimes))
    print("--------------------------------- ")


def sumThenAveragePercisely(accuracy_model_acc):
    return fdiv(fsum(accuracy_model_acc), len(accuracy_model_acc), prec=5)


# Train on one dataset, then test of another dataset
def subteam2(filenameTrain, resultColName, fileListTest, nTimes, coef_percent, nlargestFeatures):
    importanceFeature, X_Train_ImportFeature, y_Train_ImportFeature = findImportancesFeatures(resultColName,
                                                                                              filenameTrain,
                                                                                              coef_percent,
                                                                                              nlargestFeatures, 20)
    randomeFeatureSameSize, X_Train_Random, y_Train_Random = findRandomeFeaturesSets(resultColName, filenameTrain,
                                                                                     len(importanceFeature))
    # Just assign new name for variables.
    X_train_IF_Div = X_Train_ImportFeature
    y_train_IF_Div = y_Train_ImportFeature
    X_train_Random_Div = X_Train_Random
    y_Train_Random_Div = y_Train_Random
    acc_random = 0.0
    mcc_random = 0.0
    auc_random = 0.0
    acc_if = 0.0
    mcc_if = 0.0
    auc_if = 0.0
    print("Results begin ----------------- ")
    resultSetForExcel = []
    for x in range(len(fileListTest)):
        print("Run test on the file name " + fileListTest[x])
        for n in range(nTimes):
            if nTimes == 0:
                break
            # Get data from file to test
            data_yu = pd.read_csv(fileListTest[x])

            # Get the test data same column name with Random for compare
            df_Test = pd.DataFrame(data_yu, columns=randomeFeatureSameSize).fillna(0)
            X_Test_Random = df_Test[randomeFeatureSameSize]
            y_Test_Random = data_yu[resultColName]
            # Train with method Random
            clfRandom = RandomForestClassifier(n_estimators=1000, max_features='auto')
            clfRandom.fit(X_train_Random_Div, y_Train_Random_Div)
            y_Pred_Random = clfRandom.predict(X_Test_Random)
            acc_random += metrics.accuracy_score(y_Test_Random, y_Pred_Random)
            mcc_random += metrics.matthews_corrcoef(y_Test_Random, y_Pred_Random)
            auc_random += metrics.roc_auc_score(y_Test_Random, y_Pred_Random)

            # Get the test data same column name with IF
            df_IF = pd.DataFrame(data_yu, columns=importanceFeature).fillna(0)
            X_Test_IF = df_IF[importanceFeature]
            y_Test_IF = data_yu[resultColName]
            # Train with method FS
            clf = RandomForestClassifier(n_estimators=1000, max_features='auto')
            clf.fit(X_train_IF_Div, y_train_IF_Div)  # Build a forest of trees from the training set (X, y).
            y_Predict_IF = clf.predict(X_Test_IF)

            acc_if += metrics.accuracy_score(y_Test_IF, y_Predict_IF)
            mcc_if += metrics.matthews_corrcoef(y_Test_IF, y_Predict_IF)
            auc_if += metrics.roc_auc_score(y_Test_IF, y_Predict_IF)
            if nTimes == 0:
                break
        printResult(acc_random, mcc_random, auc_random, acc_if, mcc_if, auc_if, nTimes)
        resultSetForExcel.append("Random ACC = " + str(acc_random));
        resultSetForExcel.append("Random MCC = " + str(mcc_random));
        resultSetForExcel.append("Random AUC = " + str(auc_random));
        resultSetForExcel.append("IF ACC= "+ str(acc_if));
        resultSetForExcel.append("IF MCC= "+ str(mcc_if));
        resultSetForExcel.append("IF AUC= "+ str(auc_if));
        acc_if = 0.0
        mcc_if = 0.0
        auc_if = 0.0
        acc_random = 0.0
        mcc_random = 0.0
        auc_random = 0.0
    import time
    file_excel_name = "Result-" + time.strftime("%d%B%Y-%Hh%Mm%Ss") + ".xlsx"
    exportResults(file_excel_name, filenameTrain, fileListTest, resultSetForExcel)
