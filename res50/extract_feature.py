import os, sys
import math
import cv2
from tqdm import tqdm
import numpy as np

os.environ['GLOG_minloglevel'] = '2'
sys.path.append(r'/S2/MI/xbw/caffe')
sys.path.append(r'/S2/MI/xbw/caffe/python')
import caffe

dest_dir = '/local/MI/jbr/streetshop/res50/features/'
data_list = '/S2/MI/jbr/streetshop/res50/datalst/list.txt'
data_dir = '/local/MI/chil/tencentmap/cube500/'

# TODO define img_mean
img_mean = np.load('/S2/MI/xbw/models/ilsvrc_2012_mean.npy').mean(1).mean(1)

root_dir = '/S2/MI/xbw/models/resnet'
network_config = {
	#'vgg19': ['deploy.prototxt', 'vgg19_cvgj_iter_300000.caffemodel', 224, 'fc7'],
	'resnet101': ['deploy_resnet101_v2.prototxt', 'resnet101_v2.caffemodel', 224, 'pool5'],
	#'inception': ['deploy_inception_resnet_v2.prototxt', 'inception_resnet_v2.caffemodel', 331, 'pool_8x8_s1']
}


def get_net(prototxt, caffemodel, gpu_id):
	caffe.set_mode_gpu()
	caffe.set_device(gpu_id)

	net = caffe.Net(os.path.join(root_dir, prototxt),
			os.path.join(root_dir, caffemodel),
			caffe.TEST)
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

	transformer.set_mean('data', img_mean)
	transformer.set_transpose('data', (2, 0, 1))
	transformer.set_raw_scale('data', 255.0)
	transformer.set_channel_swap('data', (2, 1, 0))

	return net, transformer


def get_feature(network_name, layer, net, transformer, batch_size, input_size, img_pool, feature_dir):
	features = []
	net.blobs['data'].reshape(batch_size, 3, input_size, input_size)
	
	for i, img_dir in enumerate(img_pool):
		try:
			#img_data = caffe.io.load_image(img_dir)
			img_data = cv2.imread(img_dir).astype(np.float32)
		except Exception as ex:
			print(ex)
			print('When reading image data: %s' % img_dir)
			net.blobs['data'].data[i, ...] = np.zeros((3, input_size, input_size))
		else:
			img_data = cv2.resize(img_data, (224, 224))
			img_data = (img_data - img_mean).transpose(2, 0, 1)
			net.blobs['data'].data[i, ...] = img_data
			#net.blobs['data'].data[i, ...] = transformer.preprocess('data', img_data)
	net.forward()
	features.append(net.blobs[layer].data)
	
	save_to_dir = os.path.join(dest_dir, network_name)
	if not os.path.exists(save_to_dir):
		os.mkdir(save_to_dir)
	try:
		features = np.concatenate(features, axis=0)
		np.save(feature_dir, features)
	except Exception as ex:
		print(ex)
		print(layer)
		print('====================')
		print(img_pool)
		print('====================')

def main():
	gpu_id = 2
	network_name = 'resnet101'
	batch_size = 32
	handler_id = 1
	print('Handler_id = %d.' % handler_id)
	
	if not os.path.exists(dest_dir):
		os.mkdir(dest_dir)

	with open(data_list, 'r') as f:
		lines = f.readlines()
		imgs = [l.strip() for l in lines if l.strip()]
	
	prototxt, caffemodel, input_size, layer = network_config[network_name]
	net, transformer = get_net(prototxt, caffemodel, gpu_id)
	
	feature_cnt = 0
	img_pool = []
	pbar = tqdm(total = len(imgs)/batch_size/3)
	for i, img_dir in enumerate(imgs):
		img_hdid = int(img_dir[-1])
		img_pool.append(img_dir[:-2])
		if len(img_pool) == batch_size or i == len(imgs):
			feature_dir = os.path.join(dest_dir, network_name, 'batch_%s.npy' % feature_cnt)
			if not os.path.exists(feature_dir):
				if img_hdid == handler_id:
					get_feature(network_name, layer, net, transformer, batch_size, input_size, img_pool, feature_dir)
					pbar.update(1)
				#else:
					#print('Feature file %s is skipped by handler %d.' % (feature_dir, handler_id))
			else:
				#print('Feature file %s has already exists.' % feature_dir)
				if img_hdid == handler_id:
					pbar.update(1)
			
			img_pool = []
			feature_cnt = feature_cnt + 1
	pbar.close()

if __name__ == '__main__':
	main()
