from __future__ import print_function

import os
import sys
import argparse
# cur_path = os.path.abspath(os.path.dirname(__file__))
# root_path = os.path.split(cur_path)[0]
# sys.path.append(root_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from PIL import Image as PILImage
import numpy as np


from utils.score import SegmentationMetric
from utils.logger import setup_logger
from utils.distributed import synchronize, get_rank, make_data_sampler, make_batch_data_sampler
from dataset.voc import VOCDataValSet


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Test With Pytorch')
    # model and dataset
    parser.add_argument('--model', type=str, default='LRseg',
                        help='model name')  
    parser.add_argument('--method', type=str, default='Railway-seg',
                        help='method name')  
    parser.add_argument('--backbone', type=str, default='None',
                        help='backbone name')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='dataset name')
    parser.add_argument('--data', type=str, default='./dataset/Railway-seg/',
                        help='dataset directory')
    parser.add_argument('--data-list', type=str, default='./dataset/list/voc_rail/val.txt',
                        help='dataset directory')
    parser.add_argument('--workers', '-j', type=int, default=0,
                        metavar='N', help='dataloader threads')
    
    # training hyper params
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    # cuda setting
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--pretrained', type=str, default='./weight/mobile.pth',
                        help='pretrained seg model')
    parser.add_argument('--save-dir', default='./runs/',
                        help='Directory for saving predictions')
    parser.add_argument('--save-pred', action='store_true', default=True,
                    help='save predictions')

    # validation 
    parser.add_argument('--flip-eval', action='store_true', default=False,
                        help='flip_evaluation')
    parser.add_argument('--scales', default=[1.], type=float, nargs='+', help='multiple scales')
    args = parser.parse_args()
    args.aux = False
    return args


class Evaluator(object):
    def __init__(self, args, num_gpus):
        self.args = args
        self.num_gpus = num_gpus
        self.device = torch.device(args.device)
        print(self.device)
        ignore_label = -1

        self.id_to_trainid = {-1: ignore_label, 0: 0, 1: 255}

        # dataset and dataloader
        self.val_dataset = VOCDataValSet(args.data, args.data_list)

        val_sampler = make_data_sampler(self.val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1)
        self.val_loader = data.DataLoader(dataset=self.val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)

        from model_LRseg import Our_model
        self.model = Our_model(num_classes=2).to('cpu')
        model_path = './weight/LRseg_weight.pth'
        weights_dict = torch.load(model_path, map_location='cpu')
        load_weights_dict = {k: weights_dict[k] for k, v in self.model.state_dict().items()
                                         if weights_dict[k].numel() == v.numel()}
        print(load_weights_dict.keys())
        print(self.model.load_state_dict(load_weights_dict, strict=False))
        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.local_rank], output_device=args.local_rank)
        self.model.to(self.device)

        self.metric = SegmentationMetric(self.val_dataset.num_class)

    def id2trainId(self, label, id_to_trainid, reverse=False):
        label_copy = label.copy()
        if reverse:
            for v, k in id_to_trainid.items():
                label_copy[label == k] = v
        else:
            for k, v in id_to_trainid.items():
                label_copy[label == k] = v
        return label_copy


    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        return rt

    def predict_whole(self, net, image, tile_size):
        interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
        prediction = net(image.to('cpu'))
        # print(prediction[0].shape)
        if isinstance(prediction, tuple) or isinstance(prediction, list):
            prediction = prediction[0]
        prediction = interp(prediction)
        return prediction

    def eval(self):
        self.metric.reset()
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        logger.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        b_list = []
        r_list = []
        for i, (image, target, filename) in enumerate(self.val_loader):
            # print(i)
            image = image.to(self.device)
            target = target.to(self.device)

            N_, C_, H_, W_ = image.size()
            tile_size = (H_, W_)
            full_probs = torch.zeros((1, self.val_dataset.num_class, H_, W_)).to('cpu')

            scales = args.scales

            with torch.no_grad():
                for scale in scales:
                    scale = float(scale)
                    print("Predicting image scaled by %f" % scale)
                    scale_image = F.interpolate(image, scale_factor=scale, mode='bilinear', align_corners=True)
                    scaled_probs = self.predict_whole(model, scale_image, tile_size)

                    if args.flip_eval:
                        print("flip evaluation")
                        flip_scaled_probs = self.predict_whole(model, torch.flip(scale_image, dims=[3]), tile_size)
                        scaled_probs = 0.5 * (scaled_probs + torch.flip(flip_scaled_probs, dims=[3]))
                    full_probs += scaled_probs
                full_probs /= len(scales)
                # print('11111',full_probs)
                self.metric.update(full_probs, target)
                pixAcc, mIoU, Iou = self.metric.get()
                Iou_list = Iou.tolist()
                b_list.append(Iou_list[0]*100)
                r_list.append(Iou_list[1]*100)

                # logger.info("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
                #     i + 1, pixAcc * 100, mIoU * 100))
            if self.args.save_pred:
                pred = torch.argmax(full_probs, 1)
                pred = pred.cpu().data.numpy()
                import numpy as np
                print(np.unique(pred))
                seg_pred = self.id2trainId(pred, self.id_to_trainid, reverse=False)

                predict = seg_pred.squeeze(0)
                # mask = get_color_pallete(predict, self.args.dataset)
                # print(predict.shape, predict)
                mask = PILImage.fromarray(predict.astype('uint8'))
                print(mask, type(mask))
                print(args.outdir, filename[1][0])
                # mask.save('1.png')
                mask.save(os.path.join(args.outdir, filename[1][0] + '.png'))
                print('Save mask to ' + filename[1][0] + '.png' + ' Successfully!')
        print('BIOU', sum(b_list)/len(b_list))
        print('RIOU', sum(r_list)/len(r_list))
        synchronize()


if __name__ == '__main__':
    args = parse_args()
    num_gpus = -1
    args.distributed = -1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    args.device = "cpu"
    # TODO: optim code
    outdir = '{}_{}_{}_{}'.format(args.model, args.backbone, args.dataset, args.method)
    args.outdir = os.path.join(args.save_dir, outdir)
    if args.save_pred:
        if (args.distributed and args.local_rank == 0) or args.distributed is False:
            if not os.path.exists(args.outdir):
                os.makedirs(args.outdir)

    logger = setup_logger("semantic_segmentation", args.save_dir, get_rank(),
                          filename='{}_{}_{}_log.txt'.format(args.model, args.backbone, args.dataset), mode='a+')

    evaluator = Evaluator(args, num_gpus)
    evaluator.eval()
    torch.cuda.empty_cache()
