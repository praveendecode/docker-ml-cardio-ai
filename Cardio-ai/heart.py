# Required Libs
import pandas as pd 

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import OrdinalEncoder

from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings("ignore",category=UserWarning)

# DataFrame
df = pd.read_csv("heart_attack.csv")

# Preprocessing

encoder = OrdinalEncoder()

df['class']  = encoder.fit_transform(df[['class']])

# Split data 

x = df.drop('class',axis=1)

y = df['class']


# Train and Test Split 

xtrain , xtest , ytrain , ytest = train_test_split(x,y,test_size=0.2)

# Model Creation 

model = RandomForestClassifier().fit(xtrain,ytrain)

# Model Prediction

yhat = model.predict(xtest)


# Test model 
 
value = list(map(float,input('Enter the values to test the model : ').split()))

test = model.predict([value])

# Accuracy of the model 

acc = accuracy_score(ytest,yhat)

# Result 

if test[0] == 0.0:
  print(f'Provided Data : {value} , Result : "Negative" found with the accuracy of {acc}' )
elif test[0] == 1.0:
  print(f'Provided Data : {value} , Result : "Positive" found with the accuracy of {acc}' )