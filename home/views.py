from django.shortcuts import render


# for training data and making predictions
import numpy as np   # for arrays
import pandas as pd  # for datasets
from sklearn.model_selection import train_test_split , cross_val_score # for spliting data and checking accuracy more accurately
from sklearn.preprocessing import OneHotEncoder  # for encoding categorical columns
from sklearn.ensemble import RandomForestClassifier   # for prediction
from sklearn.metrics import accuracy_score   # for checking accuracy
from sklearn.compose import ColumnTransformer   # for ColumnTransformer
from sklearn.pipeline import Pipeline       # for making pipeline


# Create your views here.

def index(request):
    return render(request, 'index.html')

def submit(request):
    if request.method=="POST":
        pclass=request.POST.get('pclass')
        sex=request.POST.get('sex')
        age=request.POST.get('age')
        embarked=request.POST.get('embarked')
        family=request.POST.get('family')

    # training data
    # calling dataset
    df=pd.read_csv("data/train.csv", usecols=["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"])

    df["family"]= df["SibSp"]+df["Parch"]+1   # combining two columns into one for less input
    df=df.drop(columns=["SibSp", "Parch"])    # droping extra columns

    # filling milling values
    df["Age"].fillna(df["Age"].mean(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode().iloc[0], inplace=True)

    # training dataset
    # input and output   
    x=df.iloc[:, 1:]
    y=df.iloc[:, 0]

    cat_cols=["Pclass", "Sex", "Embarked"]  # category columns

    #spliting data
    x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2, random_state=42)


    # making ColumnTransformer
    processor=ColumnTransformer([("ohe", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols)], remainder="passthrough")


    # making pipeline
    pipe=Pipeline([("processor", processor),
               ("clf", RandomForestClassifier(n_estimators=80, random_state=42))])
    pipe.fit(x_train,y_train)
    
    # checking accuracy
    y_pred=pipe.predict(x_test)
    cross_val=np.mean(cross_val_score(pipe, x_train, y_train, scoring= "accuracy", cv=5))
    accuracy= accuracy_score(y_test,y_pred)

    # predicting user data
    new_data= pd.DataFrame([[pclass, sex, age, embarked, family]], columns=x_train.columns)
    processor.transform(new_data)
    prediction= pipe.predict(new_data)
    if prediction==1:
        output="Person survived"
    else:
        output ="Person does not survived"


    if embarked=="S":
        embarked="Southampton"
    elif embarked=="C":
        embarked="Cherbourg"
    else:
        embarked="Queenstown"
    context={
        "pclass":pclass,
        "sex":sex,
        "age":age,
        "embarked":embarked,
        "family":family,
        "output": output,
        "cvs":cross_val,
        "accuracy":accuracy,
    }
    return render(request, "submit.html", context)