### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import torch.nn as nn
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import cv2

class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none': # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        input_nc = opt.label_nc if opt.label_nc != 0 else 3
        # input_nc = 1

        ##### define networks        
        # Generator network
        netG_input_nc = input_nc        
        # if not opt.no_instance:
        #     netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num                  
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, 
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)        

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            # netD_input_nc = 1
            netD_input_nc = input_nc + opt.output_nc
            # if not opt.no_instance:
            #     netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        ### Encoder network
        if self.gen_features:          
            self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder', 
                                          opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)  
            
        print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)            
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)  
            if self.gen_features:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)              
        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
        
            # Names so we can breakout loss
            self.loss_names = ['G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake']

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():       
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [{'params':[value],'lr':opt.lr}]
                    else:
                        params += [{'params':[value],'lr':0.0}]                            
            else:
                params = list(self.netG.parameters())
            if self.gen_features:              
                params += list(self.netE.parameters())         
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))                            

            # optimizer D                        
            params = list(self.netD.parameters())    
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False):             
        if self.opt.label_nc == 0:
            input_label = label_map.data.cuda()
        else:
            # create one-hot vector for label map 
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)

        # get edges from instance map
        # if not self.opt.no_instance:
        #     inst_map = inst_map.data.cuda()
        #     edge_map = self.get_edges(inst_map)
        #     input_label = torch.cat((input_label, edge_map), dim=1) 
        input_label = Variable(input_label, volatile=infer)

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda())

        # instance map for feature encoding
        if self.use_features:
            # get precomputed feature maps
            if self.opt.load_features:
                feat_map = Variable(feat_map.data.cuda())

        return input_label, inst_map, real_image, feat_map

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)
    def local_slice(self,label_map,real_image=None):
        downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        label_map = downsample(label_map)
        real_image = downsample(real_image)
        height = label_map.size()[2]
        width = label_map.size()[3]
        label_slice = []
        image_slice = []
        label_slice.append(label_map[:,:,0:height/2,0:width/2])
        image_slice.append(real_image[:,:,0:height/2,0:width/2])
        label_slice.append(label_map[:,:,0:height/2,width/2:width])
        image_slice.append(real_image[:,:,0:height/2,width/2:width])
        label_slice.append(label_map[:,:,height/2:height,0:width/2])
        image_slice.append(real_image[:,:,height/2:height,0:width/2])
        label_slice.append(label_map[:,:,height/2:height,width/2:width])
        image_slice.append(real_image[:,:,height/2:height,width/2:width])

        return label_slice,image_slice,label_map,real_image
    
    def global_slice(self,label_map,real_image=None):
        downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        label_map = downsample(label_map)
        if real_image:
            real_image = downsample(real_image)
        height = label_map.size()[2]
        width = label_map.size()[3]
        label_slice = []
        label_slice.append(label_map[:,:,0:height/2,0:width/2])
        label_slice.append(label_map[:,:,0:height/2,width/2:width])
        label_slice.append(label_map[:,:,height/2:height,0:width/2])
        label_slice.append(label_map[:,:,height/2:height,width/2:width])

        return label_slice,label_map,real_image        


    def forward(self, label, roi, image, feat, infer=False):
        # Encode Inputs
        # input_label,real_image = label,image
        # label_slice,image_slice = self.slice(label,image)
        label_slice,image_slice,down_label,down_image = self.local_slice(label,image)
        # height = input_label.size()[2]
        # width = input_label.size()[3]
        if self.opt.netG == "local":
            fake_image,fake_image_slice = self.netG.forward(label,label_slice)

            # Fake Detection and Loss
            pred_fake_pool = self.discriminate(label, fake_image, use_pool=True)
            loss_D_fake = self.criterionGAN(pred_fake_pool, False)        

            # Real Detection and Loss        
            pred_real = self.discriminate(label, image)
            loss_D_real = self.criterionGAN(pred_real, True)

            # GAN loss (Fake Passability Loss)        
            pred_fake = self.netD.forward(torch.cat((label, fake_image), dim=1))        
            loss_G_GAN = self.criterionGAN(pred_fake, True)               
            
            # GAN feature matching loss
            loss_G_GAN_Feat = 0
            if not self.opt.no_ganFeat_loss:
                feat_weights = 4.0 / (self.opt.n_layers_D + 1)
                D_weights = 1.0 / self.opt.num_D
                for i in range(self.opt.num_D):
                    for j in range(len(pred_fake[i])-1):
                        loss_G_GAN_Feat += D_weights * feat_weights * \
                            self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
            for  i in range(4):
                # Fake Detection and Loss
                pred_fake_pool = self.discriminate(label_slice[i], fake_image_slice[i], use_pool=True)
                loss_D_fake += self.criterionGAN(pred_fake_pool, False)        

                # Real Detection and Loss        
                pred_real = self.discriminate(label_slice[i], image_slice[i])
                loss_D_real += self.criterionGAN(pred_real, True)

                # GAN loss (Fake Passability Loss)        
                pred_fake = self.netD.forward(torch.cat((label_slice[i], fake_image_slice[i]), dim=1))        
                loss_G_GAN += self.criterionGAN(pred_fake, True)               
                
                # GAN feature matching loss
                # loss_G_GAN_Feat = 0
                if not self.opt.no_ganFeat_loss:
                    feat_weights = 4.0 / (self.opt.n_layers_D + 1)
                    D_weights = 1.0 / self.opt.num_D
                    for i in range(self.opt.num_D):
                        for j in range(len(pred_fake[i])-1):
                            loss_G_GAN_Feat += D_weights * feat_weights * \
                                self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat       
        # VGG feature matching loss
            loss_G_VGG = 0
        # if not self.opt.no_vgg_loss:
        #     loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat


        ####################### padding for global
        if self.opt.netG == 'global':
            pad = nn.Sequential(nn.ConstantPad2d((0,width/2,0,height/2),0),
            nn.ConstantPad2d((width/2,0,0,height/2),0),
            nn.ConstantPad2d((0,width/2,height/2,0),0),
            nn.ConstantPad2d((width/2,0,height/2,0),0))
            fake_image_slice = []
            for i in range(4):
                fake_image_tmp = self.netG.forward(label_slice[i])
                fake_image_tmp = pad[i](fake_image_tmp)
                fake_image_slice.append(fake_image_tmp)
            fake_image = fake_image_slice[0] + fake_image_slice[1] + fake_image_slice[2] + fake_image_slice[3]

            pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True)
            loss_D_fake = self.criterionGAN(pred_fake_pool, False)
            # Real Detection and Loss        
            pred_real = self.discriminate(input_label, real_image)
            loss_D_real = self.criterionGAN(pred_real, True)

            # GAN loss (Fake Passability Loss)        
            pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))        
            loss_G_GAN = self.criterionGAN(pred_fake, True)               
            
            # GAN feature matching loss
            loss_G_GAN_Feat = 0
            if not self.opt.no_ganFeat_loss:
                feat_weights = 4.0 / (self.opt.n_layers_D + 1)
                D_weights = 1.0 / self.opt.num_D
                for i in range(self.opt.num_D):
                    for j in range(len(pred_fake[i])-1):
                        loss_G_GAN_Feat += D_weights * feat_weights * \
                            self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
                    
            # VGG feature matching loss
            loss_G_VGG = 0
        
        
        ####################### only for global
        # if self.opt.netG == 'global':
        #     fake_image_slice = []
        #     loss_D_fake = 0
        #     loss_D_real = 0
        #     loss_G_GAN = 0
        #     loss_G_GAN_Feat = 0
        #     for i in range(4):
        #         fake_image_tmp = self.netG.forward(label_slice[i])

        #         # Fake Detection and Loss
        #         pred_fake_pool_tmp = self.discriminate(label_slice[i], fake_image_tmp, use_pool=True)
        #         loss_D_fake_tmp = self.criterionGAN(pred_fake_pool_tmp, False)        

        #         # Real Detection and Loss        
        #         pred_real_tmp = self.discriminate(label_slice[i], image_slice[i])
        #         loss_D_real_tmp = self.criterionGAN(pred_real_tmp, True)

        #         # GAN loss (Fake Passability Loss)        
        #         pred_fake_tmp = self.netD.forward(torch.cat((label_slice[i], fake_image_tmp), dim=1))        
        #         loss_G_GAN_tmp = self.criterionGAN(pred_fake_tmp, True)               
                
        #         # GAN feature matching loss
        #         loss_G_GAN_Feat_tmp = 0
        #         if not self.opt.no_ganFeat_loss:
        #             feat_weights = 4.0 / (self.opt.n_layers_D + 1)
        #             D_weights = 1.0 / self.opt.num_D
        #             for i in range(self.opt.num_D):
        #                 for j in range(len(pred_fake_tmp[i])-1):
        #                     loss_G_GAN_Feat_tmp += D_weights * feat_weights * \
        #                         self.criterionFeat(pred_fake_tmp[i][j], pred_real_tmp[i][j].detach()) * self.opt.lambda_feat
        #         fake_image_slice.append(fake_image_tmp)
        #         loss_D_fake += loss_D_fake_tmp
        #         loss_D_real += loss_D_real_tmp
        #         loss_G_GAN += loss_G_GAN_tmp
        #         loss_G_GAN_Feat += loss_G_GAN_Feat_tmp

        #     # VGG feature matching loss
        #     loss_G_VGG = 0
        #     fake_image = torch.zeros(label.size())
        #     # print self.model(coarse_input[:,:,0:height/2,0:width/2]).size()
        #     fake_image[:,:,0:height/2,0:width/2] = fake_image_slice[0].data
        #     fake_image[:,:,0:height/2,width/2:width] = fake_image_slice[1].data
        #     fake_image[:,:,height/2:height,0:width/2] = fake_image_slice[2].data
        #     fake_image[:,:,height/2:height,width/2:width] = fake_image_slice[3].data
        #     fake_image = Variable(fake_image,requires_grad=True).cuda()
        #     if not self.opt.no_vgg_loss:
        #         loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat
        
        # Only return the fake_B image if necessary to save BW
        return [ [ loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake ], None if not infer else fake_image ]

    def inference(self, label, roi):
        # label_slice,input_label,real_image = self.global_slice(label)
        height = label.size()[2]
        width = label.size()[3]
        label_slice = []
        label_slice.append(label[:,:,0:height/2,0:width/2])
        label_slice.append(label[:,:,0:height/2,width/2:width])
        label_slice.append(label[:,:,height/2:height,0:width/2])
        label_slice.append(label[:,:,height/2:height,width/2:width])

        if self.opt.netG == 'global':
            pad = nn.Sequential(nn.ConstantPad2d((0,width/2,0,height/2),0),
            nn.ConstantPad2d((width/2,0,0,height/2),0),
            nn.ConstantPad2d((0,width/2,height/2,0),0),
            nn.ConstantPad2d((width/2,0,height/2,0),0))
            fake_image_slice = []
            for i in range(4):
                fake_image_tmp = self.netG.forward(label_slice[i])
                fake_image_tmp = pad[i](fake_image_tmp)
                fake_image_slice.append(fake_image_tmp)
            fake_image = fake_image_slice[0] + fake_image_slice[1] + fake_image_slice[2] + fake_image_slice[3]
            # label_slice0 = self.netG.forward(label_slice0)
            # label_slice0 = label_slice0.data.cpu().numpy()
            # save_image = 255*label_slice0/np.max(label_slice0)
    	    # save_image = save_image[0][0]
    	    # cv2.imwrite('1.jpg',save_image)
        # fake_image = self.netG.forward(input_label,label_slice)
        
        if self.opt.roi:
            fake_image = torch.mul(fake_image,roi)
        return fake_image

    def sample_features(self, inst): 
        # read precomputed feature clusters 
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)        
        features_clustered = np.load(cluster_path).item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)                                      
        feat_map = torch.cuda.FloatTensor(1, self.opt.feat_num, inst.size()[2], inst.size()[3])                   
        for i in np.unique(inst_np):    
            label = i if i < 1000 else i//1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0]) 
                                            
                idx = (inst == i).nonzero()
                for k in range(self.opt.feat_num):                                    
                    feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k] 
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.cuda(), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda())
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num+1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            idx = (inst == i).nonzero()
            num = idx.size()[0]
            idx = idx[num//2,:]
            val = np.zeros((1, feat_num+1))                        
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]            
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())           
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999)) 
        print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
