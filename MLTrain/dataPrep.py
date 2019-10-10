import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



csvData = []

rootDir = '.\\MLTrain\\DataSets'
for dirName, subdirList, fileList in os.walk(rootDir):
    #print('Found directory: %s' % dirName)
    for fname in fileList:
        
        filePath = dirName + "\\" + fname
        
        imgclass = filePath.split("\\")[-3]
        print('\t%s' % dirName + "/" + fname)
        print (imgclass)
        print ("")
        csvData.append([filePath, imgclass])

dfObj = pd.DataFrame(csvData, columns = ['ImagePath' , 'Class']) 
dfObj.to_csv(r'.\MLTrain\DataSets\FullFoodDataSet.csv', index=False, header=True)


y = dfObj.Class
X = dfObj.drop('Class', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

train = pd.concat([X_train, y_train], axis=1)
train.to_csv(r'.\MLTrain\DataSets\TrainFoodDataSet.csv', index=False, header=True)

test = pd.concat([X_test, y_test], axis=1)
test.to_csv(r'.\MLTrain\DataSets\TestFoodDataSet.csv', index=False, header=True)