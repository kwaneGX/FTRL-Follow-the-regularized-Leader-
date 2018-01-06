#-*- coding = utf-8 -*-
import numpy as np;
import pandas as pd;
import re;
import random;
import time;

#
class SIG(object):
	@staticmethod
	def loss(y, y_hat):
		return np.abs(y - y_hat);
	@staticmethod
	def grad(y, y_hat, x):
		return y_hat * (1 - y_hat) * y;

#交叉熵损失函数(即最大似然估计)
# y 为真实值 y_hat 为预测值 x 为输入
class MLE(object):
	@staticmethod
	def loss(y, y_hat):
		return np.sum(np.nan_to_num(-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)));
		#return -y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat);
	@staticmethod
	def grad(y, y_hat, x):
		return (y_hat - y) * x;

#FTRL
class FTRL(object):
	#初始化
	#dim 参数的维度 lambda1 lambda2 L1L2范数的系数 alpha beta 学习率的两个参数（beta推荐为1） r 取负样本的概率 method 损失函数及其梯度 subSampling 为 TRUE 时使用概率采样 road 参数的保存路径
	def __init__(self, dim, lambda1, lambda2, alpha, beta = 1, r = 1, method = MLE, subSampling = False, road = './save.npz'):
		self.dim = dim;
		self.lambda1 = lambda1;
		self.lambda2 = lambda2;
		self.alpha = alpha;
		self.beta = beta;
		self.r = r;
		self.method = method;
		self.subSampling = subSampling;
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
	#保存历史梯度和以及历史梯度平方和
	# x 当前的样本输入 y 当前样本的实际输出 loadandsave 是否读取参数并保存 FALSE 为不执行
	def update(self, x, y, loadandsave = False):
		if loadandsave == True:
			self.load();
		temp = (np.sign(self.z) * self.lambda1 - self.z) / ((self.beta + np.sqrt(self.n)) / self.alpha + self.lambda2);
		self.w = np.where(abs(self.z) <= self.lambda1, 0.0, temp);
		#若是使用随机分布减少负样本的优化，那么更新的时候梯度和损失函数上乘以1/r(分子根据实际情况调整)
		if not self.subSampling:
			wt = 1.0;
			#self.w = np.where(abs(self.z) <= self.lambda1, 0.0, temp);
		else:
			if y == 0:
				wt = 1.0 / self.r;
				#self.w = (1.0 / self.r) * np.where(abs(self.z) <= self.lambda1, 0.0, temp);
			else:
				wt = 1.0;
				#self.w = np.where(abs(self.z) <= self.lambda1, 0.0, temp);
		#print(self.w);
		y_hat = self.predict(x);
		#print(y_hat);
		grd = wt * self.method.grad(y, y_hat, x);
		sigma = (np.sqrt(self.n + grd * grd) - np.sqrt(self.n)) / self.alpha;#equals 1 / eta(t, i) - 1 / eta(t - 1, i);
		self.z = self.z + grd - sigma * self.w;
		self.n = self.n + grd * grd;
		if loadandsave == True:
			self.save();
		return wt * self.method.loss(y, y_hat);
	#训练样本(批量+单个)
	#返回值为 1 时代表达到最小重复次数，也意味着迭代收敛
	#返回值为 0 时代表达到最大迭代次数，这时迭代不一定收敛
	# train_X 为训练的输入 train_Y 为训练的输出 max_iter 为最大迭代次数 err 为最大误差 min_cnt 为最小重复次数 也就是当出现损失函数小于 err 超过 min_cnt 时，迭代终止
	def train(self, train_X, train_Y, max_iter = 1000000, err = 0.01, min_cnt = 100):
		cnt = 0;
		iter = 0;
		fo = open('./diff.txt','w');
		for i in range(max_iter):
			print("iter = ", i);
			print("iter = ", i, file = fo);
			for j in range(len(train_X)):
				#判断正负样本并判断是否使用随机分布减少负样本
				if self.subSampling & (train_Y[j] == 0):
					#取0,1之间的随机数，大于给定的r就不更新
					if random.uniform(0, 1) > self.r:
						continue;
				diff = self.update(train_X[j], train_Y[j]);
				if train_Y[j] == 1:
					print('True', file = fo);
				print('diff =', diff, file = fo);
				if diff <= err:
					cnt += 1;
				else:
					cnt = 0;
				if cnt >= min_cnt:
					self.save();
					return 1;
			iter += 1;
			if iter >= max_iter:
				self.save();
				return 0;
	#预测
	# X 为输入
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

if __name__ == '__main__:
	datat = pd.read_csv('C:\\Users\\kaihua\\Desktop\\train_feature1.csv').values;
	datat = np.array(datat);
	datat[:,1] = 0;
	datat[:,2] = 0;
	data = np.insert(datat, 1, np.ones([1, datat.shape[0]]),axis=1);
	flag = (datat[:,0] == 1);
	flag = np.array(flag);
	positive = data[flag];
	negative = data[np.where(flag, False, True)];
	train = positive[0:1].copy();
	text = positive[0:1].copy();
	for i in range(data.shape[0]):
		p = random.randint(1, 10);
		if p == 10:
			text = np.append(data[i].reshape(1, positive.shape[1]), text, axis = 0);
		else:
			train = np.append(data[i].reshape(1, positive.shape[1]), train, axis = 0);
	train_X = train[:, 1:];
	train_Y = train[:, 0:1];
	text_X = text[:, 1:];
	text_Y = text[:, 0:1];
	d = data.shape[1] - 1;
	#实例化 FTRL 类，并设置参数 主要调参对象为 lambda1 lambda2 alpha r
	ftrl = FTRL(dim = d, lambda1 = 1, lambda2 = 1, alpha = 1, beta = 1, r = 0.1, subSampling = True, method = MLE, road = 'C:\\Users\\kaihua\\Desktop\\推荐算法实现\\save.npz');
	start_time = time.time();
	#进行参数的训练，这里主要调整最大迭代次数误差限以及最小重复次数
	flag = ftrl.train(train_X = train_X, train_Y = train_Y, max_iter = 1000, err = 0.01, min_cnt = 10000);
	train_time = time.time();
	ans = ftrl.predict(text_X);
	end_time = time.time();
	text_Y = text_Y.T;
	cnt = 0;
	cnt = ((ans >= 0.5) == (text_Y >= 0.5)).sum();
	ttt = (ans >= 0.5).sum();
	ppp = (text_Y >= 0.5).sum();
	ans = np.where(ans >= 0.5, 1, 0);
	fout = open('./result.txt','a');
	print("flag =", flag, file = fout);
	print("predict =", ttt, text.shape[0] - ttt, file = fout);
	print("really =", ppp, text.shape[0] - ppp, file = fout);
	print("cnt =",cnt, "all =", len(text_X), file = fout);
	print("accuracy =", cnt / len(text_X), file = fout);
	print("accuracy =", cnt / len(text_X));
	#print("w =", ftrl.w, file = fout);
	print("train_time = ", train_time - start_time, file = fout);
	print("predict_time = ", end_time - train_time, file = fout);
	print("----------------------------------------------", file = fout);