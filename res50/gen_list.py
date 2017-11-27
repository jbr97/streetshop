import os
import random

key_dir = '/local/MI/chil/tencentmap/cube500/'
list_dir = '/S2/MI/jbr/streetshop/res50/datalst/'

def get_img(data_dir):
	print('Enter directory: %s' % data_dir)
	lst = os.listdir(data_dir)
	lst.sort()

	dir_list = []
	for i, idir in enumerate(lst):
		ndir = os.path.join(data_dir, idir)
		if os.path.isfile(ndir):
			if idir.split('_')[-1] in ['r.jpg', 'l.jpg']:
				img_list.append(ndir)
		else:
			dir_list.append(ndir)
	
	for ndir in dir_list:
		get_img(ndir)

if __name__ == '__main__':
	img_list = []
	get_img(key_dir)
	img_list.sort()
	print('total imgs: %d ' % len(img_list))

	if not os.path.exists(list_dir):
		os.mkdir(list_dir)
	
	id_count = 0
	img_count = 0
	with open(os.path.join(list_dir, 'list.txt'), 'w') as f:
		for img in img_list:
			if img_count % 32 == 0:
				id_count += 1
				id_count %= 3
			f.write('%s %d\n' % (img, id_count))
			img_count += 1
	print(img_count)
