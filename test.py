### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
from collections import OrderedDict
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
from torch.autograd import Variable
import numpy as np
import cv2

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
save_results = True

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
model.cuda()
visualizer = Visualizer(opt)
# create website
# web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
# webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
mae = 0.0
mse = 0.0
for i, data in enumerate(dataset):
    # if i >= opt.how_many:
    #     break
    generated = model.inference(Variable(data['label'].cuda()), Variable(data['roi'].cuda()))
    generated = generated.data.cpu().numpy()
    # print generated.shape
    es_count = np.sum(generated)
    gt_data = data['image'].numpy()
    gt_count = np.sum(gt_data)
    print es_count,gt_count
    mae += abs(gt_count-es_count)
    mse += ((gt_count-es_count)*(gt_count-es_count))
    if save_results:
    	result_dir = './results/'
    	# print os.path.join(result_dir,str(i),'.jpg')
    	if not os.path.exists(result_dir):
    		os.makedirs(result_dir)
    	save_image = 255*generated/np.max(generated)
    	save_image = save_image[0][0]
    	cv2.imwrite(os.path.join(result_dir,str(i)+'.jpg'),save_image)


mae = mae/len(dataset)
mse = np.sqrt(mse/len(dataset))
    # visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
    #                        ('synthesized_image', util.tensor2im(generated.data[0]))])
    # img_path = data['path']
    # print('process image... %s' % img_path)
    # visualizer.save_images(webpage, visuals, img_path)
print '\nMAE: %0.2f, MSE: %0.2f' % (mae,mse)
# webpage.save()
