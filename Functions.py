import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from numpy import mean
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier#random Forests implementation
from sklearn.metrics import confusion_matrix#confusion matrix
from sklearn.externals import joblib #will be used to save scaler

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit

def CreateTeamData(data,TeamName):
    TeamData = (data[(data.loc[:,'HomeTeam']==TeamName) | (data.loc[:,'AwayTeam']==TeamName)])
    IsHome=(TeamData.loc[:,'HomeTeam']==TeamName)
    TeamData=TeamData.assign(IsHome=IsHome)
    Target=TeamData.loc[:,'FTR']
    opNames=[]#initilize list of oponent names
    for Match in TeamData.iterrows():#pick every match and find oponent of TeamName
        if Match[1]['IsHome']==True:#if team is Home then return AwayTeam name as oponent
            opNames.append(Match[1]['AwayTeam'])
        else:
            opNames.append(Match[1]['HomeTeam'])
    #print(opNames)
    TeamData=TeamData.drop(['Div','Date','HomeTeam','AwayTeam','HTHG','HTAG','HTR','FTR','Referee','BbAv>2.5','BbMx<2.5','BbMx>2.5','BbAv<2.5'],1)#remove those features
    return([TeamData,Target,opNames])


#Function that Calculates Home Wins
def CalcHomeWins(data,TeamName):
    TeamData=data[data.loc[:,'HomeTeam']== TeamName]
    #print(TeamData[TeamData.loc[:,'FTR'] == 'H'])
    return len(TeamData[TeamData.loc[:,'FTR'] == 'H'])/len(TeamData)

#Function that Calculates Away Wins
def CalcAwayWins(data,TeamName):
    TeamData=data[data.loc[:,'AwayTeam'] == TeamName]
    #print(TeamData[TeamData.loc[:,'FTR'] == 'A'])
    return len(TeamData[TeamData.loc[:,'FTR'] == 'A'])/len(TeamData)


def CalculateWinStreak(x,y,TeamName):
    TeamData = x[TeamName]
    FTR=y[TeamName]
    TeamWonLost=[]

    i=0
    for Match in TeamData.iterrows():
        #print(FTR.iloc[i])
        if (Match[1]['IsHome']==True)& (FTR.iloc[i]=='H'):
            TeamWonLost.append(1)
        elif (Match[1]['IsHome']==True)& (FTR.iloc[i]=='A'):
            TeamWonLost.append(-1)
        elif (Match[1]['IsHome']== False) & (FTR.iloc[i]=='H'):
            TeamWonLost.append(-1)
        elif (Match[1]['IsHome']== False) & (FTR.iloc[i]=='A'):
            TeamWonLost.append(1)
        elif FTR.iloc[i]=='D':
            TeamWonLost.append(0)
        i = i + 1

    WinStreak=[]
    WinStreak.append(0)#for first match set streak to zero
    for i in range(1,len(TeamData)):#for each match
        if TeamWonLost[i-1]==1 & WinStreak[i-1]>0:
            WinStreak.append(WinStreak[i-1]+1)#if previous game was a win add 1 to streak
        elif TeamWonLost[i-1]==1 :
            WinStreak.append(1)
        else:
            WinStreak.append(0)
        #print(WinStreak)
    return(WinStreak)

def ScaleTeamData(TeamData):
    min_max_scaler = preprocessing.MinMaxScaler()
    Feature_pos=0 #Feature position,initialize
    for Feature in TeamData.iloc[1]:#check datatype of first element of feature vector
        #print(type(Feature))
        if isinstance(Feature,np.bool_) == 0: #check if  feature is float, if it is, normalize the vector
            Feature_temp=TeamData.iloc[:,Feature_pos].values.reshape(-1,1)
            np_scaled = min_max_scaler.fit_transform(Feature_temp)
            df_normalized = pd.DataFrame(np_scaled)
            TeamData.iloc[:,Feature_pos]=df_normalized.values
        Feature_pos = Feature_pos+1
        scaler_filename = "scaler.save"
        joblib.dump(min_max_scaler, scaler_filename)#save scaler to file 
    return(TeamData)

def kTeamStats(TeamData,k):
    TeamStatistics = TeamData
    FeatureNames = (list(TeamStatistics))
    MatchTeamStatistics = []
    #if TeamData.empty:
     #   return (0 for i in range(7))#if the TeamData is empty return zeros
    Goals,Shoots,ShootsT,Fouls,Corners,Yellow,Red=([] for i in range(7))#initialize lists

    for (idx, Match) in TeamStatistics.iloc[-k:].iterrows():#Gather Data from K last games
        TeamFeatures = []
        if Match.IsHome == True:
            HomeAway = 'H'
        else:
            HomeAway = 'A'
        # Team given is home or away?
        # keep only old Data for given Team
        for FeatureName in FeatureNames[0:14]:  #
            if HomeAway in FeatureName:
                TeamFeatures.append(FeatureName)

        MatchTeamStatistics=(Match.loc[TeamFeatures])  # select K last games
        Goals.append(MatchTeamStatistics.iloc[0])
        Shoots.append(MatchTeamStatistics.iloc[1])
        ShootsT.append(MatchTeamStatistics.iloc[2])
        Fouls.append(MatchTeamStatistics.iloc[3])
        Corners.append(MatchTeamStatistics.iloc[4])
        Yellow.append(MatchTeamStatistics.iloc[5])
        Red.append(MatchTeamStatistics.iloc[6])
    #print(len(Goals))
    return([mean(Goals),mean(Shoots),mean(ShootsT),mean(Fouls),mean(Corners),mean(Yellow),mean(Red)])

def CreateDatasetK(TeamData,data,oponentNames,k):#Returns Dataset with Statistics of the last game for Home/Away team
    #k=5 #Gather Data on k=5 last games(more recent)

    Goals, Shoots, ShootsT, Fouls, Corners, Yellow, Red = ([] for i in range(7))  # initialize lists
    GoalsOp, ShootsOp, ShootsTOp, FoulsOp, CornersOp, YellowOp, RedOp = ([] for i in range(7))  # initialize lists of oponent kStats

    for MatchNum in range(1,len(TeamData)):
        StatsTemp=kTeamStats(TeamData.iloc[0:MatchNum],k)#Stats in last k games(store in temp variable)
        Goals.append(StatsTemp[0])
        Shoots.append(StatsTemp[1])
        ShootsT.append(StatsTemp[2])
        Fouls.append(StatsTemp[3])
        Corners.append(StatsTemp[4])
        Yellow.append(StatsTemp[5])
        Red.append(StatsTemp[6])

        lastmatchInd=TeamData.iloc[0:MatchNum].index.values[-1]#index of last match(most recent)
        data_upto_now=data.iloc[0:lastmatchInd]#gather Data of oponent team up-to this date
        enemydata,_,_ = CreateTeamData(data_upto_now,oponentNames[MatchNum])#create Team Data for oponent

        #print(enemydata)
        StatsTemp = kTeamStats(enemydata, k) # Stats in last k games of oponent(store in temp variable)
        GoalsOp.append(StatsTemp[0])
        ShootsOp.append(StatsTemp[1])
        ShootsTOp.append(StatsTemp[2])
        FoulsOp.append(StatsTemp[3])
        CornersOp.append(StatsTemp[4])
        YellowOp.append(StatsTemp[5])
        RedOp.append(StatsTemp[6])
    TeamData=TeamData.drop(TeamData.index[0])#drop first match
    #print(len(TeamData))
    TeamData.loc[:,'GoalsK'] = Goals
    TeamData.loc[:,'ShootsK'] = Shoots
    TeamData.loc[:,'ShootsTK'] = ShootsT
    TeamData.loc[:,'FoulsK'] = Fouls
    TeamData.loc[:,'CornersK'] = Corners
    TeamData.loc[:,'YellowK'] = Yellow
    TeamData.loc[:,'RedK'] = Red
    TeamData=TeamData.loc[:,'B365H':]

    TeamData.loc[:, 'OpGoalsK'] = GoalsOp #Enemy Data
    TeamData.loc[:, 'OpShootsK'] = ShootsOp
    TeamData.loc[:, 'OpShootsTK'] = ShootsTOp
    TeamData.loc[:, 'OpFoulsK'] = FoulsOp
    TeamData.loc[:, 'OpCornersK'] = CornersOp
    TeamData.loc[:, 'OpYellowK'] = YellowOp
    TeamData.loc[:,'OpRedK'] = RedOp
    return(TeamData.fillna(0))#turn NaN to 0

def classScore(Target):
    class2number = {"A" : 0,"D" : 0.5,"H" : 1,}
    Target_num=[]
    for y_i in Target:
        Target_num.append(class2number[y_i])
    return( pd.Series(Target_num))
            
def LearnTeamModel(HomeData,Target,k):
    #HomeData=CreateDatasetK(TeamData,'Home',data,oponent[TeamName],k)
    HomeData=ScaleTeamData(HomeData)#apply scaling
    #Target=y[TeamName][1:]#dont count first match because no previous data are there to consider
  
    #print(H_wins/len(data))#Trivial classifier performance(Distribution of Classes)
    #temp=chi2(HomeData,Target)
    FeatureNames=list(HomeData)
    
    #Independance test : it is included as comment only because the small amount of samples makes the test unreliable
    #print(FeatureNames[any(temp[1]<1)])#find features that are independent
   # pvalue_pos=0
    #for pvalue in temp[1]:
     #   if pvalue<0.05 :
      #      print(FeatureNames[pvalue_pos])
       #     pvalue_pos=pvalue_pos+1
    #rf = RandomForestRegressor(n_estimators=50)
    #rf = RandomForestClassifier(n_estimators=50)
    #rf.fit(HomeData,Target)
    #print (sorted(zip(map(lambda x: round(x, 3), rf.feature_importances_), FeatureNames), reverse=True))#Feature importance according to Random Forests
    #clf = rf
    #scores = cross_val_score(clf, HomeData, Target, cv=rskf)
    
    
    perform=True#whether we will perform feature selection or not
    RFparameters={'n_estimators':np.linspace(50,1000, num=20).astype(int)}
    clf_RF_best=TuneHyperPars(RandomForestClassifier(),RFparameters,HomeData,Target)
    featSelected=PerformRFE(perform,clf_RF_best,HomeData,Target,FeatureNames)
    
    SVMparameters={'kernel':('linear', 'rbf','poly'), 'C':np.linspace(0.1, 1, num=20)}
    clf_SVM_best=TuneHyperPars(SVC(),SVMparameters,HomeData,Target)
    #featSelected=PerformRFE(perform,clf_SVM_best,HomeData,Target,FeatureNames)
    
    knnparameters={'n_neighbors':np.linspace(1,5, num=5).astype(int)}#try 
    clf_knn_best=TuneHyperPars(KNeighborsClassifier(),knnparameters,HomeData,Target)
  
    final_RF_model=clf_RF_best.fit(HomeData.loc[:,featSelected],Target)
    final_SVM_model=clf_SVM_best.fit(HomeData.loc[:,featSelected],Target)
    final_knn_model=clf_knn_best.fit(HomeData.loc[:,featSelected],Target)
    return(final_RF_model,final_SVM_model,final_knn_model,featSelected,final_SVM_model.kernel)#return model trained only on selected Features and the names of those features

def CreateSeasonData(data,TeamName,k):
    UniqueTeams=set(data.loc[:,'HomeTeam'])
    UniqueTeams = [x for x in UniqueTeams if str(x) != 'nan']# clean nan's
    x={}
    y={}
    oponent={}
    for Team in UniqueTeams:
        [xtemp,ytemp,op_temp]=CreateTeamData(data,Team)
        x[Team]=xtemp
        y[Team]=ytemp
        oponent[Team]=op_temp
    if TeamName in x:#if the team appears in this season gather data
        x[TeamName] = x[TeamName].assign(WinStreak=CalculateWinStreak(x, y,TeamName))#no time depedencies because each row
        TeamData=x[TeamName]
        #print('Away Wins ',CalcAwayWins(data,TeamName))#Away Win Rate
        #print('Home WIns ',CalcHomeWins(data,TeamName))#Home Win Rate
        
        HomeData=CreateDatasetK(TeamData,data,oponent[TeamName],k)
        Target=y[TeamName][1:]#dont count first match because no previous data are there to consider
    else :#else don't gather data
        print('Team',TeamName,'does not appear in this Season')
        HomeData=[]
        Target=[]
    return(HomeData,Target)
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(title+'.eps',format='eps', dpi=100)    #enable if needed to save matrix
    
def TuneHyperPars(alg,parameters,HomeData,Target): #tunes hyperparameters of model using Grid Search on learning algorithm 'alg'
    tscv = TimeSeriesSplit(n_splits=5)#partition to three "folds"
    #SVMparameters={'kernel':('linear', 'rbf','poly'), 'C':np.linspace(0.1, 1, num=10)}
    clf = GridSearchCV(alg,parameters,cv=tscv)
    clf.fit(HomeData,Target)
    return(clf.best_estimator_)#model that achieved higher score
    
def PerformRFE(perform,clf,HomeData,Target,FeatureNames):
    if perform:#if parameter is True,peform Feature selection
        selector = RFE(clf,n_features_to_select=25)#keep only 25 features
        selector = selector.fit(HomeData, Target)
        index=np.asarray(selector.support_)
        featSelected=(np.asarray(FeatureNames)[index])
    else:#return the whole list otherwise
        featSelected=FeatureNames
    return(featSelected)
    
def train_validate_team(TeamName,SeasonData,k):#trains-validates on specific team for algorithms RandomForests(RF),SVM and KNN
    #AllSeasonData=[]#initialize
    HomeData=[]#initialize 
    Target=[]
    scoresSVM,scoresRF,scoresKNN=([] for i in range(3))
    conf_ar_listRF,conf_ar_listSVM,conf_ar_listKNN=([] for i in range(3))#initialize lists
    for seas_ind,data in SeasonData.items():
        HomeData_temp,Target_temp=CreateSeasonData(data,TeamName,k)
        if len(HomeData_temp)!=0: #if TeamName appears in season then keep the data
            HomeData.append(HomeData_temp)
            Target.append(Target_temp)
    
    common_cols = list(set.intersection(*(set(df.columns) for df in HomeData)))#create list of common columns/features (some of the previous versions of Dataset have more columns)
    AllSeasonTeamData=pd.concat([df[common_cols] for df in HomeData], ignore_index=True)#merge all Seasons but with common columns only
    #print(AllSeasonTeamData)
    Target=pd.concat(Target, ignore_index=True)#ignore index, which means that the matches are numbered in order 0:...
    #Target=classScore(Target)#map each class to a number(score) 0='A',0.5='D' and 1='H'
   
    tscv = TimeSeriesSplit(n_splits=5)#partition to 5"folds"
    conf_ar_listSVM=[]#list with confusion arrays
    scoresTrivial=[]
    kernel_returned=[]
    for train_index, test_index in tscv.split(AllSeasonTeamData):#train indices must always be smaller than testing because of time depedencies
        #print('Test index is',test_index)
        #print("TRAIN:", train_index, "TEST:", test_index)
        
        X_train, X_test = AllSeasonTeamData.loc[train_index], AllSeasonTeamData.loc[test_index]
        y_train, y_test = Target.loc[train_index], Target.loc[test_index]
        #TeamModel,featSelected=LearnTeamModel(X_train,y_train,k)#train on train data
        TeamModelRF,TeamModelSVM,TeamModelKNN,featSelected,kernel=LearnTeamModel(X_train,y_train,k)
        kernel_returned.append(kernel)
        
        #print('The Features selected are: ',featSelected)
        EvaluateScores_ConfMatrix(X_test,y_test,TeamModelRF,featSelected,scoresRF,conf_ar_listRF)
        EvaluateScores_ConfMatrix(X_test,y_test,TeamModelSVM,featSelected,scoresSVM,conf_ar_listSVM)
        EvaluateScores_ConfMatrix(X_test,y_test,TeamModelKNN,featSelected,scoresKNN,conf_ar_listKNN)
        #scoresTrivial.append(y_test.count('H')/len(y_test))
        scoresTrivial.append(np.sum('H'==y_test)/len(y_test))
    print('Mean Score of RF is:',mean(scoresRF))
    print('Mean Score of SVM is:',mean(scoresSVM))
    print('Mean Score of KNN is:',mean(scoresKNN))
    print('Mean Score of Trivial Classifier is:',mean(scoresTrivial))
    print('Kernels chosen for each fold respectively is:',kernel_returned)
    return(mean(scoresRF),sum(conf_ar_listRF),mean(scoresSVM),sum(conf_ar_listSVM),mean(scoresKNN),sum(conf_ar_listKNN),mean(scoresTrivial))
    #scores.append(mean_squared_error(y_pred,y_test))
 
def EvaluateScores_ConfMatrix(X_test,y_test,TeamModel,featSelected,scores,conf_ar_list):#ecaluate scores and confusion matrix
            #apply scaling in test set using the scaler that was performed on train set
            Feature_pos=0 #Feature position,initialize
            min_max_scaler = joblib.load("scaler.save")#load scaler
            for Feature in X_test.iloc[1]:#check datatype of first element of feature vector
                #print(type(Feature))
                if isinstance(Feature,np.bool_) == 0: #check if  feature is float, if it is, normalize the vector
                    Feature_temp=X_test.iloc[:,Feature_pos].values.reshape(-1,1)
                    np_scaled = min_max_scaler.fit_transform(Feature_temp)
                    df_normalized = pd.DataFrame(np_scaled)
                    X_test.iloc[:,Feature_pos]=df_normalized.values
                Feature_pos = Feature_pos+1
            y_pred=TeamModel.predict(X_test.loc[:,featSelected])#train only on selected features
            #print('Predictions are: ',y_pred)
            #print('Prediction accuracy is: ',np.sum(y_pred==y_test)/len(y_test))
            conf_ar_list.append(confusion_matrix(y_test,y_pred,labels=['A','D','H']))
            scores.append(np.sum(y_pred==y_test)/len(y_test))
            #classProbs.append(TeamModel.predict_proba(X_test.loc[:,featSelected]))
            #return(scores,conf_ar_list)       
    
    #print(conf_arr)
