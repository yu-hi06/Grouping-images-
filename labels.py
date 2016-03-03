from __future__ import division
import numpy as np
from scipy import stats
import community
import networkx as nx
import matplotlib
import math
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')
import Tkinter
import Image
import ImageTk
import os
import pickle, time
DATA_ROOT = "/home/yuhi/data"
colors = ["red", "blue", "yellow", "green", "pink", "gray", "orange", "violet", "navy", "skyblue","springgreen","brown","gold","cyan","magenta","black","darkslategrey","lime","mediumseagreen","peru","seashell","peachpuff","floralwhite","mintcream","mediumspringgreen","mediumblue","aqua","crimson","fuchsia","thistle","plum","cornsilk"]


############################# label ############################################################
################################################################################################
def rmse(w1,w2):
	x=np.sqrt((((w1-w2)**2).sum())/(len(w1)))
	return x

def regularize(w1):
	return (w1-np.average(w1))/np.std(w1)

def thre(w1,ratio):
	for threshold in range(10000):
		threshold = threshold*0.1
		if ratio > len([i for i in w1 if i > threshold])/len(w1):
			return threshold

start = time.time()
ndata = np.loadtxt("input.txt")

n = ndata.shape[0]
net= nx.Graph()
for i in range(n):
	net.add_node(i)

sum=[]
for i in range(n-1):
	for j in range(i,n-1):
		j=j+1
		sum.append(rmse(ndata[i],ndata[j]))

b=regularize(sum)

b=np.abs([j if j < 0 else 0 for j in b])
threshold = 0.0 # thre(b, 0.25) 
count = 0
alpha = 0
for i in range(n-1):
	for j in range(i,n-1):
		if b[alpha] > threshold:
			count=count+1
			net.add_edge(i,j+1, weight=b[alpha])
		alpha=alpha+1

partition = community.best_partition(net)
labels =  partition.values()

elapsed_time=time.time() - start
print ("elapsed_time:{0}".format(elapsed_time))
print count


################################################################################################


point = np.loadtxt("output.txt")
X = []
Y = []
for p in point:
	X.append(p[0])
	Y.append(p[1])


X = np.array(X)
Y = np.array(Y)
dic = pickle.load(open("dic_jpeg.pkl", "rb"))

def search_i(x,y):
		for i in range(len(X)):
			if X[i] == x and Y[i] == y:
				break;
		return i

class PointDrag:
	def __init__(self, fig, ax):
		self.fig = fig
		self.selected,  = ax.plot([], [], 'o', ms=20, alpha=0.4,color='yellow', visible=False)

	def onpick(self, event):
		if not len(event.ind):
			return True
		ind = event.ind[0]
		x = event.artist.get_xdata()[ind]
		y = event.artist.get_ydata()[ind]
		ind = search_i(x, y)
		clicked=(x,y)
		print(ind)
		root = Tkinter.Tk()
		image = Image.open(DATA_ROOT+'/'+dic[ind])
		tkpi = ImageTk.PhotoImage(image)
		label_image = Tkinter.Label(root, image=tkpi)
		label_image.pack()
		root.mainloop()

		self.selected.set_visible(True)
		self.selected.set_data(clicked)
		self.fig.canvas.draw()



dirname = os.listdir(DATA_ROOT)
fig = plt.figure()
ax = fig.add_subplot(111)
vnum=list(set(labels))


for i in range(len(vnum)): 
	ux = [X[j] for j in range(len(X)) if labels[j] == i]
	uy = [Y[j] for j in range(len(X)) if labels[j] == i]
	ax.plot(ux, uy, 'o',ms=8,color=colors[i], picker=True)
	print colors[i] + ":" + str(len(ux)) 

browser = PointDrag(fig,ax)
fig.canvas.mpl_connect('pick_event', browser.onpick)
plt.show()
