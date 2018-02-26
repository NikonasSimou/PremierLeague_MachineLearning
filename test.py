import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import Functions

from Functions import plot_confusion_matrix
from Functions import train_validate_team
   
#MAIN
SeasonData={}#each element of dictionary is data from a Season
SeasonData[0]  = pd.read_csv('Dataset/E0_2007.csv')
SeasonData[1]  = pd.read_csv('Dataset/E0_2008.csv')
SeasonData[2]  = pd.read_csv('Dataset/E0_2009.csv')
SeasonData[3]  = pd.read_csv('Dataset/E0_2010.csv')
SeasonData[4]  = pd.read_csv('Dataset/E0_2011.csv')
SeasonData[5]  = pd.read_csv('Dataset/E0_2012.csv')
SeasonData[6]  = pd.read_csv('Dataset/E0_2013.csv')
SeasonData[7]  = pd.read_csv('Dataset/E0_2014.csv')
SeasonData[8]  = pd.read_csv('Dataset/E0_2015.csv')
SeasonData[9]  = pd.read_csv('Dataset/E0_2016.csv')
SeasonData[10]  = pd.read_csv('Dataset/E0_2017.csv')
k=5 #how many past matches will be taken into account

#UniqueTeams = list(set.intersection(*(set(df.loc[:,'HomeTeam']) for key,df in SeasonData.items())))

#set(data['HomeTeam'])
meanscoresRF,conf_ar_list_RF,meanscoresSVM,conf_ar_list_SVM,meanscoresKNN,conf_ar_list_KNN,meanscoresTrivial=([] for i in range(7))#initialize lists
#TeamNames=list(set(df.loc[:,'HomeTeam']) for key,df in SeasonData.items())
TeamNameList=[]
#TeamNames=['QPR','Arsenal']
TeamNames=UniqueTeams = list(set.union(*(set(df.loc[:,'HomeTeam']) for key,df in SeasonData.items())))
TeamNames = [x for x in TeamNames if str(x) != 'nan']
for TeamName in TeamNames:
    TeamMatches=0#How many matches exist with this team
    for key,df in SeasonData.items():#check how many matches this Team has on record
        TeamMatches=TeamMatches+sum((df.loc[:,'HomeTeam']==TeamName) | (df.loc[:,'AwayTeam']==TeamName))
    if TeamMatches>=200: #at least 100 matches should be recorded, if not go to next team
        TeamNameList.append(TeamName)
        HomeData=[]#initialize 
        Target=[]
        scoresSVM,scoresRF,scoresKNN=([] for i in range(3))
        conf_ar_listRF,conf_ar_listSVM,conf_ar_listKNN=([] for i in range(3))#initialize lists
             
        meanscoresRF_temp,conf_ar_listRF_temp,meanscoresSVM_temp,conf_ar_listSVM_temp,meanscoresKNN_temp,conf_ar_listKNN_temp,meanscoresTrivial_temp = train_validate_team(TeamName,SeasonData,k)
        
        meanscoresRF.append(meanscoresRF_temp)
        conf_ar_list_RF.append(conf_ar_listRF_temp)
        
        meanscoresSVM.append(meanscoresSVM_temp)
        conf_ar_list_SVM.append(conf_ar_listSVM_temp)
        
        meanscoresKNN.append(meanscoresKNN_temp)
        conf_ar_list_KNN.append(conf_ar_listKNN_temp)
        
        meanscoresTrivial.append(meanscoresTrivial_temp)
    else:
        print('Team ',TeamName,'has less than 200 matches and therefore should not be considered')
plt.figure()
plot_confusion_matrix(sum(conf_ar_list_RF),classes=['A','D','H'],normalize=False,title='Confusion matrix-RF')
plt.figure()
plot_confusion_matrix(sum(conf_ar_list_SVM),classes=['A','D','H'],normalize=False,title='Confusion matrix-SVM')
plt.figure()
plot_confusion_matrix(sum(conf_ar_list_KNN),classes=['A','D','H'],normalize=False,title='Confusion matrix-KNN')



#x_ticks_labels = ['Everton', 'Man City', 'West Brom', 'West Ham', 'Man United', 'Chelsea', 'Southampton', 'Liverpool', 'Tottenham', 'Wigan', 'Swansea', 'Aston Villa', 'Sunderland', 'Arsenal', 'Stoke', 'Fulham', 'Hull', 'Newcastle']
x_ticks_labels = TeamNameList
plt.figure()
plt.tight_layout()
x=np.linspace(0,len(TeamNameList)-1,len(TeamNameList))
#plt.axis([1,18,0.3,0.7])
plt.plot(x,meanscoresSVM,'-^',label='SVM')
plt.plot(x,meanscoresRF,'-v',label='RF')
plt.plot(x,meanscoresKNN,'-o',label='KNN')
plt.plot(x,meanscoresTrivial,'-x',label='Trivial')
#plt.xticks(x_ticks_labels, rotation='vertical', fontsize=18)
plt.xticks(range(len(x_ticks_labels)), x_ticks_labels, rotation=90)
plt.legend(loc='upper left')
plt.xlabel('Team Name')
plt.ylabel('Accuracy %')
plt.savefig('Accuracies.eps', format='eps')   
plt.show()
 