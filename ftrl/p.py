#-*- coding = utf-8 -*-
import sys;
import numpy as np;
import pandas as pd;
import re;
import time;

sys.path.append("C:\\Users\\kaihua\\Desktop\\推荐算法实现");
from FTRL import *;

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
	ftrl.load();
	train_time = time.time();
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