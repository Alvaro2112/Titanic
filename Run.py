import copy
import math

import numpy as np
import pandas
import torch
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from torch import nn, optim

from Network import Net
from data_processing import process_train, process_test


def calculate_accuracy(y_true, y_pred):
    predicted = y_pred.ge(.5).view(-1)
    return (y_true == predicted).sum().float() / len(y_true)


def round_tensor(t, decimal_places=3):
    return round(t.item(), decimal_places)


if __name__ == '__main__':

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    df = process_train('train.csv')


    ### GENERATE MISSING DATA WITH LINEAR REGRESSION

    dfk = copy.deepcopy(df)
    dfk.dropna(inplace=True)
    cols = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q',
            'Embarked_S']

    X = dfk[cols]
    y = dfk[['Age']]

    model = Lasso(positive=True).fit(X, y)

    for i, row in df.iterrows():
        if math.isnan(row['Age']):
            df.at[i, 'Age'] = model.predict(df.iloc[[i]][cols])

    ### PREPARE DATASET FOR TRAINING

    X = df[['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q',
            'Embarked_S']]
    y = df[['Survived']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    X_train = torch.from_numpy(X_train.to_numpy()).float()
    y_train = torch.squeeze(torch.from_numpy(y_train.to_numpy()).float())

    X_test = torch.from_numpy(X_test.to_numpy()).float()
    y_test = torch.squeeze(torch.from_numpy(y_test.to_numpy()).float())

    ### CREATE NEURAL NETWORK

    net = Net(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    ### TRAINING

    for epoch in range(15000):  # 15000

        y_pred = net(X_train)

        y_pred = torch.squeeze(y_pred)
        train_loss = criterion(y_pred, y_train)

        if epoch % 100 == 0:
            train_acc = calculate_accuracy(y_train, y_pred)

            y_test_pred = net(X_test)
            y_test_pred = torch.squeeze(y_test_pred)

            test_loss = criterion(y_test_pred, y_test)

            test_acc = calculate_accuracy(y_test, y_test_pred)
            print(
                f'''epoch {epoch}
    Train set - loss: {round_tensor(train_loss)}, accuracy: {round_tensor(train_acc)}
    Test  set - loss: {round_tensor(test_loss)}, accuracy: {round_tensor(test_acc)}
    ''')

        optimizer.zero_grad()

        train_loss.backward()

        optimizer.step()

    ### GET TESTING DATA FOR KAGGLE

    df = process_test("test.csv")
    final = pandas.DataFrame(columns=['PassengerId', 'Survived'])

    ### COMPLETE MISSING DATA

    for i, row in df.iterrows():
        if math.isnan(row['Age']):
            df.at[i, 'Age'] = model.predict(df.iloc[[i]][
                                                ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex', 'SibSp', 'Parch', 'Fare',
                                                 'Embarked_C', 'Embarked_Q', 'Embarked_S']])

    ### PREDICT

    for i, row in df.iterrows():
        next = df.iloc[[i]][
            ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q',
             'Embarked_S']]
        next = next.to_numpy()
        next = torch.from_numpy(next).float()
        final.loc[i] = pandas.Series(
            {'PassengerId': df.at[i, "PassengerId"], 'Survived': int(net(next).ge(0.5).item())})

    final.to_csv('final.csv', index=False)

