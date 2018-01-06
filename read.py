#s = fin.readline();
def loadRating(road):#s is string instead of road of the data like C:\\Users\\kaihua\\Desktop\\Data\\ratings.dat
	fin = open(road);
	#fin = open("C:\\Users\\kaihua\\Desktop\\Data\\ratings.dat");
	#f = open("./myRating.txt","w");
	rating = dict();
	for s in fin:
		a = s.split("::");
		for i in range(0,4):
			a[i] = int(a[i]);
		if a[0] not in rating:
			rating[a[0]] = dict();
		if a[1] not in rating[a[0]]:
			rating[a[0]][a[1]] = 0;
		rating[a[0]][a[1]] = rating[a[0]][a[1]] + a[2];
		#print(a, file = f);
	fin.close();
	return rating;

