import os
import pandas as pd
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
import uuid



def copy_rename(old_file_name, new_file_name):
    shutil.copy(old_file_name,new_file_name)        

TrainImagePath = "./MLTrain/DataSets/TrainImages/"
if not os.path.exists("./MLTrain/DataSets/TrainImages/"):
    os.mkdir(TrainImagePath)


csvData = []

rootDir = './MLTrain/DataSets/FooDD'
for dirName, subdirList, fileList in os.walk(rootDir):
    #print('Found directory: %s' % dirName)
    for fname in fileList:
        
        filePath = dirName + "/" + fname
        filePath = filePath.replace("\\", "/")
        imgclass = filePath.split("/")[-3]
        print('\t%s' % dirName + "/" + fname)
        print (imgclass)
        print ("")

        newFileName = TrainImagePath + imgclass + "_" + str(uuid.uuid4()) + ".jpg"
        copy_rename(filePath, newFileName)

        csvData.append([newFileName, imgclass])

dfObj = pd.DataFrame(csvData, columns = ['ImagePath' , 'Class']) 
dfObj.to_csv(r'./MLTrain/DataSets/FullFoodDataSet.csv', index=False, header=True)


y = dfObj.Class
X = dfObj.drop('Class', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

train = pd.concat([X_train, y_train], axis=1)
train.to_csv(r'./MLTrain/DataSets/TrainFoodDataSet.csv', index=False, header=True)

test = pd.concat([X_test, y_test], axis=1)
test.to_csv(r'./MLTrain/DataSets/TestFoodDataSet.csv', index=False, header=True)