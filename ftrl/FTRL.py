#-*- coding = utf-8 -*-
import numpy as np;
import pandas as pd;
import re;
import time;

#交叉熵损失函数(即最大似然估计)
class MLE(object):
	@staticmethod
	def loss(y, y_hat):
		return -y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat);
	@staticmethod
	def grad(y, y_hat, x):
		return (y_hat - y) * x;

#FTRL
class FTRL(object):
	def __init__(self, dim, lambda1, lambda2, alpha, beta = 1, method = MLE, road = './save.npz'):
		self.dim = dim;
		self.lambda1 = lambda1;
		self.lambda2 = lambda2;
		self.alpha = alpha;
		self.beta = beta;
		self.method = method;
		self.w = np.zeros(dim);
		self.z = np.zeros(dim);
		self.n = np.zeros(dim);
		self.road = road;
	#决策函数
	def sigmoid(self, x):
		return 1.0 / (1.0 + np.exp(-1.0 * x));
	#更新权值
	#每个特征的学习率用以下公式计算
	#eta(t,i) = alpha / (beta + sqrt(sigma(grad(s,i)^2)))
	#								 s=1->t
	#保存
	def update(self, x, y, loadandsave = False):
		if loadandsave == True:
			self.load();
		temp = (np.sign(self.z) * self.lambda1 - self.z) / ((self.beta + np.sqrt(self.n)) / self.alpha + self.lambda2);
		self.w = np.where(abs(self.z) <= self.lambda1, 0.0, temp);
		#print(self.w);
		y_hat = self.predict(x);
		grd = self.method.grad(y, y_hat, x);
		sigma = (np.sqrt(self.n + grd * grd) - np.sqrt(self.n)) / self.alpha;#equals 1 / eta(t, i) - 1 / eta(t - 1, i);
		self.z = self.z + grd - sigma * self.w;
		self.n = self.n + grd * grd;
		if loadandsave == True:
			self.save();
		return self.method.loss(y, y_hat);
	#训练样本(批量+单个)
	def train(self, X, Y, max_iter = 1000000, err = 0.01, max_cnt = 100):
		cnt = 0;
		iter = 0;
		for i in range(max_iter):
			'''if i % 100 == 0:
				print("iter = ", i);'''
			#print("iter = ", i);
			for j in range(len(train_X)):
				diff = self.update(train_X[j], train_Y[j]);
				if diff <= err:
					cnt += 1;
				else:
					cnt = 0;
				if cnt >= max_cnt:
					self.save();
					return 1;
			iter += 1;
			if iter >= max_iter:
				self.save();
				return 0;
	#预测
	def predict(self, X):
		return self.sigmoid(np.dot(X, self.w));
	#各种参数的保存
	def save(self):
		np.savez(self.road, w = self.w, z = self.z, n = self.n);
	#各种参数的读取
	def load(self):
		temp = np.load(self.road);
		self.w = temp['w'];
		self.z = temp['z'];
		self.n = temp['n'];

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
	train = data[0:80,:];
	text = data[80:100,:];
	#train = data;
	#text = data;
	train_X = train[:, 0:3];
	train_Y = train[:, 3:4];
	text_X = text[:, 0:3];
	text_Y = text[:, 3:4];
	d = 3;
	ftrl = FTRL(dim = d, lambda1 = 1, lambda2 = 1, alpha = 1, beta = 1, method = MLE, road = 'C:\\Users\\kaihua\\Desktop\\推荐算法实现\\save.npz');
	start_time = time.time();
	flag = ftrl.train(X = train_X, Y = train_Y, max_iter = 1000, err = 0.01, max_cnt = 20);
	train_time = time.time();
	print(flag);
	ans = ftrl.predict(text_X);
	end_time = time.time();
	text_Y = text_Y.T;
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
	print(ans);
	print(text_Y-ans);
	print(cnt / len(text_X));
	print("train_time = ", train_time - start_time);
	print("predict_time = ", end_time - train_time);