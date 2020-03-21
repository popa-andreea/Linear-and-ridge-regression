import numpy as np
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error,mean_squared_error


#exercitiul 1
def normalize(train_data, test_data):
    sc = preprocessing.StandardScaler()
    sc.fit(train_data)
    scaled_train = sc.transform(train_data)

    if isinstance(test_data, np.ndarray) == False and test_data == None:
        return scaled_train

    scaled_test = sc.transform(test_data)
    return scaled_train, scaled_test

#exercitiul 2
training_data = np.load('data/training_data.npy')
prices = np.load('data/prices.npy')

num_samples_fold = len(training_data) // 3

training_data_1, prices_1 = training_data[:num_samples_fold + 1], prices[:num_samples_fold + 1]
training_data_2, prices_2 = training_data[num_samples_fold + 1 : 2 *(num_samples_fold + 1)], prices[num_samples_fold + 1: 2 * (num_samples_fold + 1)]
training_data_3, prices_3 = training_data[2 * num_samples_fold:], prices[2 * num_samples_fold:]

def linear_regression(train_data, train_labels, test_data, test_labels):
    train_data, test_data = normalize(train_data, test_data)
    linear = linear_model.LinearRegression()
    linear.fit(train_data, train_labels)
    pred = linear.predict(test_data)

    mae = mean_absolute_error(test_labels,pred)
    mse = mean_squared_error(test_labels,pred)
    return mae,mse

def linear_mae_mse():
    mae_1, mse_1 = linear_regression(training_data_1 + training_data_3, prices_1 + prices_3, training_data_2, prices_2)
    mae_2, mse_2 = linear_regression(training_data_1 + training_data_2, prices_1 + prices_2, training_data_3, prices_3)
    mae_3, mse_3 = linear_regression(training_data_2 + training_data_3, prices_2 + prices_3, training_data_1, prices_1)

    avrg_mae = mae_1 + mae_2 + mae_3 / 3
    avrg_mse = mse_1 + mse_2 + mse_3 / 3
    return avrg_mae, avrg_mse

#exercitiul 3
def ridge_regression(train_data, train_labels, test_data, test_labels, alpha):
    train_data, test_data = normalize(train_data, test_data)
    ridge = linear_model.Ridge(alpha)
    ridge.fit(train_data, train_labels)
    pred = ridge.predict(test_data)

    mae = mean_absolute_error(test_labels, pred)
    mse = mean_squared_error(test_labels, pred)
    return mae, mse

def ridge_mae_mse():
    best, mae, mse = 0, 0, 0
    for alpha in [1,10,100,1000]:
        mae_1, mse_1 = ridge_regression(training_data_1 + training_data_3, prices_1 + prices_3, training_data_2, prices_2, alpha)
        mae_2, mse_2 = ridge_regression(training_data_1 + training_data_2, prices_1 + prices_2, training_data_3, prices_3, alpha)
        mae_3, mse_3 = ridge_regression(training_data_2 + training_data_3, prices_2 + prices_3, training_data_1, prices_1, alpha)

        avrg_mae = mae_1 + mae_2 + mae_3 / 3
        avrg_mse = mse_1 + mse_2 + mse_3 / 3

        if alpha == 1:
            best = 1
            mae = avrg_mae
            mse = avrg_mse

        elif avrg_mse < mse and avrg_mae < mae:
            best = alpha
            mae = avrg_mae
            mse = avrg_mse

    return best

#exercitiul 4
def ridge_(train_data, train_labels):
    alpha = ridge_mae_mse()
    train_data = normalize(train_data, None)
    ridge = linear_model.Ridge(alpha)
    ridge.fit(train_data, train_labels)

    coef = ridge.coef_
    bias = ridge.intercept_
    print('coeficienti:\n', coef)
    print('bias:', bias)

    print('cel mai semnificativ atribut:', np.argmax(np.abs(coef)) + 1)
    print('al doilea cel mai semnificativ atribut:', np.argpartition(np.abs(coef),-2)[-2] + 1)
    print('cel mai putin semnificativ atribut:', np.argmin(np.abs(coef)) + 1)

ridge_(training_data,prices)















