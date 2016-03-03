#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os, os.path, numpy, caffe,pickle,time
DATA_ROOT = "/home/yuhi/data" # imagedata root
MEAN_FILE = '/home/yuhi/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy'
PRETRAINED = "/home/yuhi/caffe/examples/imagenet/imagenet_feature.prototxt"
MODEL_FILE = "/home/yuhi/caffe/examples/imagenet/caffe_reference_imagenet_model"
LAYER = 'fc7' #full-convolutional layer 7
INDEX = 4
batchs = 10
def main():
	start = time.time() # processing start time
	net = caffe.Classifier(PRETRAINED, MODEL_FILE, mean = numpy.load(MEAN_FILE), raw_scale=255, image_dims=(256, 256))
	
	####################### Get the name of images ########################

	#if DATA_ROOT has folders
	folderlist = [] # The list of folder name
	dirname = os.listdir(DATA_ROOT)
	for dname1 in dirname:
		dname2 = DATA_ROOT + "/" + dname1
		for filenames in os.listdir(dname2):
			folderlist.append(dname1+"/"+filenames)

	dic = {i:name for i,name in enumerate(folderlist)}
	pickle.dump(dic, open("dic_jpeg.pkl", "wb"))
	image_data = [caffe.io.load_image(os.path.join(DATA_ROOT,imagename)) for imagename in folderlist]

	#if DATA_ROOT has only images
	# dirname = os.listdir(DATA_ROOT)
	#dic = {i:name for i,name in enumerate(dirname)}
	#pickle.dump(dic, open("dic_jpeg.pkl", "wb"))
	#image_data = [caffe.io.load_image(os.path.join(DATA_ROOT101,imagename)) for imagename in dirname]
	
	print len(image_data)
	######################################################################

	#Extract feature vactor
	featurelist = []
	pred = []
	for i in range(len(image_data)):
		pred=numpy.append(pred,net.predict([image_data[i]]))
		blobs = net.blobs[LAYER].data
		features = (blobs.reshape((batchs,4096)))[0]
		featurelist=numpy.append(featurelist,features)

	featurelist2 = featurelist.reshape((len(image_data),len(featurelist)/len(image_data)))
	numpy.savetxt('input.txt',featurelist2)
	
	elapsed_time=time.time() - start
	print ("elapsed_time:{0}".format(elapsed_time))
    
	return 1


if __name__ == "__main__":
    sys.exit(main())
    
