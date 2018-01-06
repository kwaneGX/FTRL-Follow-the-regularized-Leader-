#-*- coding utf-8 -*-
import numpy as np;
import math;
import time;
import pandas as pd;
import random;
import matplotlib.pyplot as plt;
import re;



# sigmoid 函数
# X 为列向量 返回 Y 列向量
def sigmoid(X):
	return 1.0 / (1.0 + np.exp(-1.0 * X));

#训练参数 theta 列向量
# train_X 每一行为不同样本每一列为不同特征, train_Y 列向量
# alpha 学习率 theta 需要训练的参数
# arg 为各种参数, dict = {参数名：参数值}
# 损失函数采用最大似然函数
#theta.T*X = theta[0] + theta[1] * x1 +...+theta[n]*xn;
def train_theta(train_X, train_Y, arg = {}):
	#train_X = np.asarray(train_X);
	#train_Y = np.asarray(train_Y);
	#start_time = time.time();
	alpha = 0.01;#
	maxIter = 200;#30
	row = len(train_X[0]);
	theta = np.ones([row, 1]);
	#for i in range(row):
	#	theta[i, 0] = random.uniform(-1, 1);
	if 'alpha' in arg:
		alpha = arg['alpha'];
	if 'maxIter' in arg:
		maxIter = arg['maxIter'];
	if 'theta0' in arg:
		theta = arg['theta0'];
		theta = np.asarray(theta);
	#梯度下降
	for i in range(maxIter):
		A = np.dot(train_X, theta);# X : n*m   theta: m*1	A: n*1
		E = sigmoid(A) - train_Y;# E: n*1
		theta = theta - alpha * np.dot(train_X.T, E);# X_T: m*n
		#print(theta.T)
	#print('time = ',time.time() - start_time);
	return theta;
#逻辑回归
# X 行向量, theta 列向量
def LR(X, theta):
	#X = np.asarray(X);
	#theta = np.asarray(theta);
	return sigmoid(np.dot(X, theta));







if __name__ == '__main__':
	#datat = pd.read_csv('C:\\Users\\kaihua\\Desktop\\ex2data1.csv').values;
	road = "C:\\Users\\kaihua\\Desktop\\data.txt";
	fin = open(road);
	datat = [];
	for line in fin:
		a = re.split('\t| |\r|\n', line.strip());
		a = [item for item in filter(lambda x:x != '', a)]
		a[0] = float(a[0]);
		a[1] = float(a[1]);
		a[2] = int(a[2]);
		datat.append(a);
	datat = np.array(datat);
	data = np.append(np.ones((datat.shape[0], 1)), datat,axis=1);
	save = "C:\\Users\\kaihua\\Desktop\\datat.txt";
	fout = open(save, 'w');
	#for i in range(100):
	#	print(data[i][0],data[i][1],data[i][2],data[i][3], file = fout);
	fout.close();
	train = data[0:80,:];
	text = data[80:100,:];
	#train = data;
	#text = data;
	train_X = train[:, 0:3];
	train_Y = train[:, 3:4];
	text_X = text[:, 0:3];
	text_Y = text[:, 3:4];
	start_time = time.time();
	theta = train_theta(train_X, train_Y);
	train_time = time.time();
	print(theta);
	ans = LR(text_X, theta)
	end_time = time.time();
	cnt = 0;
	cnt = ((ans >= 0.5) == (text_Y >= 0.5)).sum();
	#print(cnt);
	'''for i in range(len(text_X)):
		t = ans[i] >= 0.5;
		p = text_Y[i] >= 0.5;
		if t == p:
			cnt+=1;'''
	#print(text_Y)
	ans = np.where(ans >= 0.5, 1, 0);
	#print(ans);
	#print(text_Y-ans);
	print("train_time = ", train_time - start_time);
	print("predict_time = ", end_time - train_time);
	print(cnt / len(text_X));