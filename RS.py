# This file is for ML HW9 Recommend System
# Group member: Yijia Jin, Wenyu Liu
# Created at 30/04/2017

import numpy as np
import pandas as pd
import random

# # This function read data and return
# def readData1():
#     train_rate_path = "./data/training_rating.dat"
#     R_marix = {}
#     for line in open(train_rate_path, 'r'):
#         item = line.rstrip()
#         cand = item.split("::")
#         assert len(cand) == 3
#         if cand[0] not in R_marix.keys():
#             R_marix[cand[0]] = dict()
#         R_marix[cand[0]][cand[1]] = cand[2]
#     return R_marix

def readData1():
    file = open('./data/training_rating.dat')
    training_data = []
    for line in file.readlines():
        tuple = line.strip('\n').split('::')
        ignore = False
        for t in tuple:
            if len(t) == 0:
                ignore = True
                break
        if ignore:
            continue
        training_data.append(map(int, line.strip('\n').split('::')))
    return training_data

def readData():
    file = open('./data/training_rating.dat')
    training_data = []
    user_mean = np.zeros((6040, 1))
    user_count = np.zeros(6040)
    movie_mean = np.zeros((3952, 1))
    movie_count = np.zeros(3952)
    mean = 0
    count = 0
    for line in file.readlines():
        tuple = line.strip('\n').split('::')
        ignore = False
        for t in tuple:
            if len(t) == 0:
                ignore = True
                break
        if ignore:
            continue
        userID, movieID, rating = map(int, line.strip('\n').split('::'))
        training_data.append((userID, movieID, rating))

        user_mean[userID - 1] += rating
        user_count[userID - 1] += 1
        movie_mean[movieID - 1] += rating
        movie_count[movieID - 1] += 1
        mean += rating
        count += 1

    for i in range(6040):
        if user_count[i] == 0:
            continue
        user_mean[i] = float(user_mean[i]) / user_count[i]

    for i in range(3952):
        if movie_count[i] == 0:
            continue
        movie_mean[i] = float(movie_mean[i]) / movie_count[i]

    mean = float(mean) / count

    return training_data, user_mean - mean, movie_mean - mean

# - UserIDs range between 1 and 6040
# - MovieIDs range between 1 and 3952
# - Ratings are made on a 5-star scale (whole-star ratings only)
# - Timestamp is represented in seconds since the epoch as returned by time(2)
# - Each user has at least 20 ratings

# This is the basic function to train U and V matrix, using SGD method
# Input: R contains training_rating.dat information, format as list of tuple(i, j, score)
#        U: user's information, nparray(user, k)
#        V: item's information, nparray(item, k)
#        step: step size
#        lam: parameter for regularization
#        err: when converage, rmse change < err
# Output: trained U, V
def SGD(R, U, V, step, lam, err):
    length = len(R)
    R_train = R[: 9*length/10]
    R_test = R[9*length/10:]
    lenth = len(R_train)
    rmse_pre = 0
    rmse = Error(R_test, U, V)
    # print rmse
    cnt = 0
    while abs(rmse - rmse_pre) > err:
        idx = random.randint(0, len(R_train) - 1)
        i = R_train[idx][0] - 1
        j = R_train[idx][1] - 1
        r = R_train[idx][2]
        e = r - np.dot(U[i], V[j])
        delta_U = -1 * e * V[j] + lam * U[i]
        delta_V = -1 * e * U[i] + lam * V[j]
        U = update(U, step, delta_U, i)
        V = update(V, step, delta_V, j)
        cnt += 1
        if cnt % 10000 == 0:
            rmse_pre = rmse
            rmse = Error(R_test, U, V)
            print rmse
    return U, V


# This function is used in SGD to calculate object function - rmse
def Error(R_test, U, V):
    res = 0
    for sample in R_test:
        res += pow(abs(sample[2] - np.dot(U[sample[0] - 1], V[sample[1] - 1])), 2)
    res /= len(R_test)
    return np.sqrt(res)


# This function is used in SGD to update parameters
def update(matrix, step, delta, idx):
    matrix[idx] = matrix[idx] - step * delta
    return matrix

# This is bias + SGD function
# mu is the global average of movies, bi is bias of items, bu is bias of users
def biasSGD(R, U, V, step, lam, err):
    length = len(R)
    R_train = R[: 9*length/10]
    R_test = R[9*length/10:]
    rmse_pre = 0
    rmse = Error(R_test, U, V)
    print rmse
    cnt = 0
    write90 = True
    while abs(rmse - rmse_pre) > err:
        for idx in range(len(R_train)):
            # idx = random.randint(0, len(R_train) - 1)
            i = R_train[idx][0] - 1
            j = R_train[idx][1] - 1
            r = R_train[idx][2]
            e = r - np.dot(U[i], V[j])
            delta_U = -1 * e * V[j] + lam * U[i]
            delta_V = -1 * e * U[i] + lam * V[j]
            U = update(U, step, delta_U, i)
            V = update(V, step, delta_V, j)
        rmse_pre = rmse
        rmse = biasError(R_test, U, V)
        print rmse    
        if rmse < 0.89 and write90:
            step = 0.0001
            write90 = False
            np.savetxt("./data/U.csv", U, delimiter=",")
            np.savetxt("./data/V.csv", V, delimiter=",")
            print "write 90"
        if rmse < 0.88:
            np.savetxt("./data/U_88.csv", U, delimiter=",")
            np.savetxt("./data/V_88.csv", V, delimiter=",")
            print "write 88"
    return U, V

def biasError(R_test, U, V):
    res = 0
    for sample in R_test:
        res += pow(abs(sample[2] - (np.dot(U[sample[0] - 1], V[sample[1] - 1]))), 2)
    res /= len(R_test)
    return np.sqrt(res)


R, user_bias, movie_bias = readData()
print user_bias.shape, movie_bias.shape
k = 3
step = 0.001
lam = 0.01
err = 1e-15
U = np.full((6040, k), 1.09)
V = np.full((3952, k), 1.09)
movie_1 = np.ones((3952, 1))
user_1 = np.ones((6040, 1))
U1 = np.concatenate((U, user_bias), axis=1)
U1 = np.concatenate((U1, user_1), axis=1)
V1 = np.concatenate((V, movie_1), axis=1)
V1 = np.concatenate((V1, movie_bias), axis=1)
# SGD(R, U, V, 0.01, 0.01, 1e-15)
biasSGD(R, U1, V1, step, lam, err)