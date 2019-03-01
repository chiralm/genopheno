from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Imputer
import common
import pandas as pd



def xgboosting():
    df = pd.read_csv('exported.csv')

    print(df)

    #dataset = loadtxt('test.csv',delimiter =',')

    X = df.drop(labels=['phenotype'], axis = 1)
    Y = df['phenotype']

    seed = 7
    test_size = 0.1
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = test_size,random_state = seed,stratify=Y)
    imputer , X_train, X_test = common._impute_data(X_train, X_test)


    model = XGBClassifier()
    model.fit(X_train,Y_train)

    print(model)

    y_pred = model.predict(X_test)
    predictions = [value for value in y_pred]
    print (predictions)
    print (Y_test)
    accuracy = accuracy_score(Y_test,predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))




xgboosting()
