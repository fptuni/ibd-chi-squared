import pandas as pd
from numpy.random import default_rng
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn import linear_model
from sklearn import feature_selection
import statsmodels.api as sm
from sklearn.feature_selection import chi2
from mpmath import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import seaborn as sns
import networkx as nx
from textwrap import wrap
from sklearn.feature_selection import SelectFromModel
import os
from sklearn.preprocessing import MinMaxScaler
def printGraph(columns, data):
    # plt.style.use('ggplot')
    columns = columns.delete(len(columns) - 1)
    data = data[columns]
    corData = data.corr()
    # links = corData.stack().reset_index()
    # links.columns = ['var1', 'var2', 'value']
    # links_filtered = links.loc[(links['value'] > 0.8) & (links['var1'] != links['var2'])]
    # G = nx.from_pandas_edgelist(links_filtered, 'var1', 'var2')
    # nx.draw(G, with_labels=True, node_color='orange', node_size=400, edge_color='black', linewidths=1, font_size=15)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(corData, annot=False, cmap=plt.cm.Reds,ax=ax)
    plt.show()


class TeamFile:
    # instance attribute
    def __init__(self, train, listFileTest, resultColName):
        self.train = train
        self.listFileTest = listFileTest
        self.resultColName = resultColName
dirname = os.path.dirname(__file__)

def getOldDataset():
    train = os.path.join(dirname, 'data/feng_x.csv')
    fileListTest = []
    fileListTest.append(os.path.join(dirname, 'data/yu_x.csv'))
    fileListTest.append(os.path.join(dirname, 'data/zeller_x.csv'))
    fileListTest.append(os.path.join(dirname, 'data/vogtmann_x.csv'))
    return TeamFile(train, fileListTest, "RS")


def getNewDataset():
    train = os.path.join(dirname, 'data/ibdfullHS_UCr_x.csv') #iCDr & UCf &iCDf &CDr&CDf
    fileListTest = []
    fileListTest.append(os.path.join(dirname, 'data/ibdfullHS_iCDr_x.csv'))
    fileListTest.append(os.path.join(dirname, 'data/ibdfullHS_UCf_x.csv'))
    fileListTest.append(os.path.join(dirname, 'data/ibdfullHS_iCDf_x.csv'))
    fileListTest.append(os.path.join(dirname, 'data/ibdfullHS_CDr_x.csv'))
    fileListTest.append(os.path.join(dirname, 'data/ibdfullHS_CDf_x.csv'))
    return TeamFile(train, fileListTest, "RS")


class color:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def textcolor_display(text, values):
    return f"{values}" + text + f"{color.ENDC}"


class Select_Method:
    Pearson = 1
    Chi = 2
    Lasso = 3
    Recursive = 4
    Tree = 5


def findImportancesFeatures(resultColName, filenameTrain, coef_percent, flag, nlargestFeatures,num_feats):
    ways_to_if = "NONE"
    if (flag == Select_Method.Chi):
        ways_to_if = "Chi-Squared"
    if (flag == Select_Method.Lasso):
        ways_to_if = "Lasso: SelectFromModel"
    if(flag == Select_Method.Pearson):
        ways_to_if = "Pearson Correlation"
    if (flag == Select_Method.Recursive):
        ways_to_if = "Recursive Feature Elimination"
    if (flag == Select_Method.Tree):
        ways_to_if = "Tree-based: SelectFromModel"
    print("Cách để chọn features : " + ways_to_if)
    print(str("Train bằng file ") + str(filenameTrain))
    data = pd.read_csv(filenameTrain)
    colName = data.columns
    df = pd.DataFrame(data, columns=colName)
    df.head()
    X = df[colName]
    y = df[resultColName]
    X_No_First = df.drop(df.columns[0], axis=1)
    X_norm = MinMaxScaler().fit_transform(X_No_First)
    if (flag == Select_Method.Pearson):
        print("Hệ số tương quan > " + str(coef_percent))
        if float(coef_percent) != 0.0:
            cor = X.corr()
            cor_target = abs(cor[resultColName])
            relevant_features = cor_target[cor_target > float(coef_percent)]
        else:
            relevant_features = X_train_new
        importanceFeature = relevant_features.index
        print("Number feature selected : " + str(len(importanceFeature)))
    if (flag == Select_Method.Chi):
        print("num_feats = " + str(num_feats))
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import chi2
        chi_selector = SelectKBest(chi2, k=num_feats)
        chi_selector.fit(X_norm, y)
        chi_support = chi_selector.get_support()
        chi_feature = X_No_First.loc[:, chi_support].columns.tolist()
        importanceFeature = chi_feature
        print("Number feature selected : " + str(len(importanceFeature)))
    if (flag == Select_Method.Lasso):
        from sklearn.feature_selection import SelectFromModel
        from sklearn.linear_model import LogisticRegression
        print("num_feats = " + str(num_feats))
        embeded_lr_selector = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear'), max_features=num_feats)
        embeded_lr_selector.fit(X_norm, y)
        embeded_lr_support = embeded_lr_selector.get_support()
        embeded_lr_feature = X_No_First.loc[:, embeded_lr_support].columns.tolist()
        importanceFeature = embeded_lr_feature
        print("Number feature selected : " + str(len(importanceFeature)))
    if (flag == Select_Method.Recursive):
        print("num_feats = " + str(num_feats))
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression
        rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats, step=10, verbose=5)
        rfe_selector.fit(X_norm, y)
        rfe_support = rfe_selector.get_support()
        rfe_feature = X_No_First.loc[:, rfe_support].columns.tolist()
        importanceFeature = rfe_feature
        print("Number feature selected : " + str(len(importanceFeature)))
    if (flag == Select_Method.Tree):
        from sklearn.feature_selection import SelectFromModel
        from sklearn.ensemble import RandomForestClassifier
        print("num_feats = " + str(num_feats))
        embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
        embeded_rf_selector.fit(X_norm, y)

        embeded_rf_support = embeded_rf_selector.get_support()
        embeded_rf_feature = X_No_First.loc[:, embeded_rf_support].columns.tolist()
        importanceFeature = embeded_rf_feature
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
    print("Khi Random ")
    print("ACC = " + str(acc_random / nTimes))
    print("MCC = " + str(mcc_random / nTimes))
    print("AUC = " + str(auc_random / nTimes))
    print("+++++ ")
    print("Khi xét Importance Features")
    print("ACC = " + str(acc_if / nTimes))
    print("MCC = " + str(mcc_if / nTimes))
    print("AUC = " + str(auc_if / nTimes))
    print("--------------------------------- ")
def sumThenAveragePercisely(accuracy_model_acc):
    return fdiv(fsum(accuracy_model_acc), len(accuracy_model_acc), prec=5)
# Train on one dataset, then test of another dataset
def subteam2(filenameTrain, resultColName, fileListTest, nTimes, coef_percent, flag, nlargestFeatures,num_feats):
    importanceFeature, X_Train_ImportFeature, y_Train_ImportFeature = findImportancesFeatures(resultColName,filenameTrain,
        coef_percent, flag, nlargestFeatures,10)
    randomeFeatureSameSize, X_Train_Random, y_Train_Random = findRandomeFeaturesSets(resultColName,filenameTrain,len(importanceFeature))
    #Just assign new name for variables.
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
    print("Bắt đầu kết quả ----------------- ")
    for x in range(len(fileListTest)):
        print("Chạy test trên " + fileListTest[x] + ". Chạy lặp " + str(nTimes) + " lần.")
        for n in range(nTimes):
            if nTimes == 0:
                break
            data_yu = pd.read_csv(fileListTest[x])
            # df_IF = data_yu.iloc[1:, : len(importanceFeature)]
            df_IF = pd.DataFrame(data_yu, columns=importanceFeature).fillna(0)
            X_Test_IF = df_IF[importanceFeature]
            y_Test_IF = data_yu[resultColName]
            # Labels

            # Remove feature and Test with method 2 - random
            df_Test = pd.DataFrame(data_yu, columns=randomeFeatureSameSize).fillna(0)
            X_Test_Random = df_Test[randomeFeatureSameSize]
            y_Test_Random = data_yu[resultColName]  # Labels

            # Train with method 2
            clfRandom = RandomForestClassifier(n_estimators=1000, max_features='auto')
            clfRandom.fit(X_train_Random_Div, y_Train_Random_Div)
            y_Pred_Random = clfRandom.predict(X_Test_Random)
            acc_random += metrics.accuracy_score(y_Test_Random, y_Pred_Random.round())
            mcc_random += metrics.matthews_corrcoef(y_Test_Random, y_Pred_Random.round())
            auc_random += metrics.roc_auc_score(y_Test_Random, y_Pred_Random.round())

            # Train with method 1
            clf = RandomForestClassifier(n_estimators=1000, max_features='auto')
            clf.fit(X_train_IF_Div, y_train_IF_Div)  # Build a forest of trees from the training set (X, y).
            y_Predict_IF = clf.predict(X_Test_IF)

            acc_if += metrics.accuracy_score(y_Test_IF, y_Predict_IF.round())
            mcc_if += metrics.matthews_corrcoef(y_Test_IF, y_Predict_IF.round())
            auc_if += metrics.roc_auc_score(y_Test_IF, y_Predict_IF.round())
            if nTimes == 0:
                break
        printResult(acc_random, mcc_random, auc_random, acc_if, mcc_if, auc_if, nTimes)
        acc_if = 0.0
        mcc_if = 0.0
        auc_if = 0.0
        acc_random = 0.0
        mcc_random = 0.0
        auc_random = 0.0

