import os, sys
import math
import cv2
from tqdm import tqdm
from Queue import Queue
import multiprocessing
import numpy as np

dest_dir = '/local/MI/jbr/streetshop/res50/features'
data_list = '/S2/MI/jbr/streetshop/res50/datalst/list.txt'
data_dir = '/local/MI/chil/tencentmap/cube500/'
root_dir = '/S2/MI/xbw/models/resnet'
network_config = {
	'resnet101': ['deploy_resnet101_v2.prototxt', 'resnet101_v2.caffemodel', 224, 'pool5'],
}
img_mean = np.load('/S2/MI/xbw/models/ilsvrc_2012_mean.npy').mean(1).mean(1)
handler_cnt = 3
provider_gpu_id = 0
dp_queue_procs = 40
network_name = 'resnet101'
batch_size = 32

def provider(data_queue, gpu_id, worker_id, lock):
	os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
	
	with open(data_list, 'r') as f:
		lines = f.readlines()
		imgs = [l.strip() for l in lines if l.strip()]
	print('Process %d: dp_queue_provider started with worker_id %d.' % (os.getpid(), worker_id))

	img_pool = []
	img_hdcnt = 0
	feature_cnt = 0
	for i, img_dir in enumerate(imgs):
		img_hdid = int(img_dir[-1])
		img_pool.append(img_dir[:-2])
		if len(img_pool) != batch_size and i != len(imgs):
			continue

		feature_dir = os.path.join(dest_dir, network_name, 'batch_%s.npy' % feature_cnt)
		if not os.path.exists(feature_dir) and img_hdid == handler_id and img_hdcnt % dp_queue_procs == worker_id:
			value = {}
			img_datas = []
			for i, img_dir in enumerate(img_pool):
				img_data = cv2.imread(img_dir).astype(np.float32)
				img_data = cv2.resize(img_data, (224, 224))
				img_data = (img_data - img_mean).transpose(2, 0, 1)
				img_datas.append(img_data)
			value['img_data'] = img_datas
			value['ign'] = False
			value['feature_dir'] = feature_dir

			lock.acquire()
			data_queue.put(value)
			lock.release()
		else:
			if img_hdid == handler_id and img_hdcnt % dp_queue_procs == worker_id:
				lock.acquire()
				data_queue.put({'ign':True})
				lock.release()

		img_pool = []
		if img_hdid == handler_id:
			img_hdcnt += 1
		feature_cnt = feature_cnt + 1
	print('Process %d(provider): Tasks have already finished.' % os.getpid())
	return

def network(data_queue, total_imgs, network_name):

	os.environ['GLOG_minloglevel'] = '2'
	sys.path.append(r'/S2/MI/xbw/caffe')
	sys.path.append(r'/S2/MI/xbw/caffe/python')
	import caffe	
	
	# get net
	prototxt, caffemodel, input_size, layer = network_config[network_name]
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
	net.blobs['data'].reshape(batch_size, 3, input_size, input_size)
	
	n_batch = 0
	total_batch = total_imgs/batch_size/handler_cnt
	pbar = tqdm(total = total_batch)
	while n_batch < total_batch:
		if not data_queue.empty():
			value = data_queue.get(False)

			if value['ign'] == False:
				if data_queue.qsize() > 1000:
					print('Warning! DP_queue(%d) occupied too many spaces!' % data_queue.qsize())
				img_data = value['img_data']
				feature_dir = value['feature_dir']

				for i in range(batch_size):
					net.blobs['data'].data[i, ...] = img_data[i]
				net.forward()
				features = net.blobs[layer].data	
				
				save_to_dir = os.path.join(dest_dir, network_name)
				if not os.path.exists(save_to_dir):
					os.mkdir(save_to_dir)
				try:
					features = np.concatenate(features, axis=0)
					np.save(feature_dir, features)
				except Exception as ex:
					print(ex)
			pbar.update(1)
			n_batch += 1
	pbar.close()
	print('Process %d(network): Tasks have already finished.' % os.getpid())
	return

if __name__ == '__main__':
	assert(len(sys.argv) == 3)

	gpu_id = int(sys.argv[1])
	handler_id = int(sys.argv[2])
	print('use gpu %d' % gpu_id)
	print('Handler_id = %d.' % handler_id)
	
	total_imgs = 0
	if not os.path.exists(dest_dir):
		os.mkdir(dest_dir)
	with open(data_list, 'r') as f:
		lines = f.readlines()
		total_imgs = len([l.strip() for l in lines if l.strip()])
	
	manager = multiprocessing.Manager()
	data_queue = manager.Queue()
	lock = manager.Lock()

	processes = []
	for i in range(dp_queue_procs):
		p = multiprocessing.Process(target=provider, args=(data_queue, provider_gpu_id, i, lock))
		p.start()
		processes.append(p)
	p = multiprocessing.Process(target=network, args=(data_queue, total_imgs, network_name))
	p.start()
	processes.append(p)
	
	for p in processes:
		p.join()
	
	print('Features extract done.')
