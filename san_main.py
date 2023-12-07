from __future__ import division

import os, sys, random, PIL
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # please use Pillow 4.0.0 or it may fail for some images
from os import path as osp
import numpy as np
import torch
import datasets
from san_vision import transforms
from utils import print_log
from utils import time_for_file
import models, options, procedure

opt = options.Options(None)
args = opt.opt
# Prepare options
if args.manualSeed is None: args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)
torch.backends.cudnn.enabled   = True
#torch.backends.cudnn.benchmark = True

def main():
  # Init logger
  if not os.path.isdir(args.save_path): os.makedirs(args.save_path)
  log = open(os.path.join(args.save_path, 'seed-{}-{}.log'.format(args.manualSeed, time_for_file())), 'w')
  # General Data Argumentation
  mean_fill   = tuple( [int(x*255) for x in [0.485, 0.456, 0.406] ] )
  normalize   = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
  assert args.arg_flip == False, 'The flip is : {}, rotate is {}'.format(args.arg_flip, args.rotate_max)
  train_transform = [
                      transforms.PreCrop(args.pre_crop_expand),
                      transforms.TrainScale2WH((args.crop_width, args.crop_height)),
                      transforms.AugScale(args.scale_prob, args.scale_min, args.scale_max),
                      transforms.AugCrop(args.crop_width, args.crop_height, args.crop_perturb_max, mean_fill),
                      transforms.ToTensor(), 
                      normalize
                    ]
  
  train_transform = transforms.Compose( train_transform )

  eval_transform  = transforms.Compose([
                    transforms.PreCrop(args.pre_crop_expand), 
                    transforms.TrainScale2WH((args.crop_width, args.crop_height)),  
                    transforms.ToTensor(), 
                    normalize
                  ])
  assert (args.scale_min+args.scale_max) / 2 == args.scale_eval, 'The scale is not ok : {},{} vs {}'.format(args.scale_min, args.scale_max, args.scale_eval)
  
  args.downsample = 8 # By default
  args.sigma = args.sigma * args.scale_eval

  train_data = datasets.GeneralDataset(train_transform, args.sigma, args.downsample, 
                                       args.heatmap_type, args.dataset_name)
  train_data.load_list(args.train_list, args.num_pts, True)

  train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, 
                                             shuffle=True, num_workers=args.workers, pin_memory=True)

  eval_loaders = []
  if args.eval_lists is not None:
    for eval_list in args.eval_lists:
      eval_data = datasets.GeneralDataset(eval_transform, args.sigma, args.downsample, args.heatmap_type, args.dataset_name)
      eval_data.load_list(eval_list, args.num_pts, True)
      # if args.convert68to49:
      #   eval_data.convert68to49()
      # elif args.convert68to51:
      #   eval_data.convert68to51()
      eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=args.eval_batch, shuffle=False, num_workers=args.workers, pin_memory=True)
      eval_loaders.append(eval_loader)

  # if args.convert68to49 or args.convert68to51:
  #   assert args.num_pts == 68, 'The format of num-pts is not right : {}'.format(args.num_pts)
  #   assert args.convert68to49 + args.convert68to51 == 1, 'Only support one convert'
  #   if args.convert68to49: args.num_pts = 49
  #   else:                  args.num_pts = 51

  args.modelconfig = models.ModelConfig(train_data.NUM_PTS+1, args.cpm_stage, args.pretrain, args.argmax_size)

  if args.cycle_model_path is None:
    # define the network
    itnetwork = models.itn_model(args.modelconfig, args, log)

    cycledata = datasets.CycleDataset(train_transform, args.dataset_name)
    cycledata.set_a(args.cycle_a_lists)
    cycledata.set_b(args.cycle_b_lists)
    print_log('Cycle-data initialize done : {}'.format(cycledata), log)

    args.cycle_model_path = procedure.train_cycle_gan(cycledata, itnetwork, args, log)
  assert osp.isdir(args.cycle_model_path), '{} does not exist or is not dir.'.format(args.cycle_model_path)

  # start train itn-cpm model
  itn_cpm = models.__dict__[args.arch](args.modelconfig, args.cycle_model_path)
  procedure.train_san_epoch(args, itn_cpm, train_loader, eval_loaders, log)

  log.close()

if __name__ == '__main__':
  main()
