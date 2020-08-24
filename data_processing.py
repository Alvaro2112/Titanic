import pandas as pd


def process_train(file):
    df = pd.read_csv(file)
    categorical = ['Pclass', 'Embarked']

    cols = ["PassengerId", "Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    df = df[cols]

    df['Sex'].replace({'male': 1, 'female': 0}, inplace=True)
    df['Embarked'].fillna('C', inplace=True)

    ### CREATE ONE HOT ENCODINGS

    for var in categorical:
        df = pd.concat([df, pd.get_dummies(df[var], prefix=var)], axis=1)
        del df[var]

    ### STANDARDIZE DATA

    a = df["PassengerId"]
    df = (df - df.min()) / (df.max() - df.min())
    df["PassengerId"] = a

    return df


def process_test(file):
    df = pd.read_csv(file)
    categorical = ['Pclass', 'Embarked']

    cols = ["PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    df = df[cols]

    df['Sex'].replace({'male': 1, 'female': 0}, inplace=True)
    df['Embarked'].fillna('C', inplace=True)

    ### CREATE ONE HOT ENCODINGS

    for var in categorical:
        df = pd.concat([df, pd.get_dummies(df[var], prefix=var)], axis=1)
        del df[var]

    ### STANDARDIZE DATA

    a = df["PassengerId"]
    df = (df - df.min()) / (df.max() - df.min())
    df["PassengerId"] = a

    return df
