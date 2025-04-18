import argparse
import logging
import os
import random
import shutil
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss, KLDivLoss, BCELoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, ramps, val_2d


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/Polyp', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='MPDCNet', help='experiment_name')
parser.add_argument('--model', type=str, default='mcnet_kd', help='model_name')
parser.add_argument('--max_iterations', type=int, default=50000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7, help='labeled data')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
parser.add_argument('--lamda', type=float, default=1, help='weight to balance all losses')
parser.add_argument('--beta', type=float, default=0.3, help='balance factor to control regional and sdm loss')
parser.add_argument('--temp', default=1, type=float)

parser.add_argument('--device_cpu', default="cpu", type=str, help='use cpu')
parser.add_argument('--device_gpu', default="cuda", type=str, help='use gpu')
parser.add_argument('--model_save_path', default="/root/autodl-tmp/Model/test", type=str, help='Save Path of Model')
parser.add_argument('--data_path', type=str, default='/root/autodl-tmp/Database/Polyp/h5py/', help='Path of Data')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device(args.device_gpu)

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 34, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate":
        ref_dict = {"2": 47, "4": 111, "7": 191,
                    "11": 306, "14": 391, "18": 478, "35": 940}
    elif "Polyp" in dataset:
        ref_dict = {"2": 47, "3": 51, "7": 102,
                    "11": 306, "14": 203, "28": 405, "35": 940, "70": 1015}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def train(args, snapshot_path):
    base_lr = args.base_lr
    labeled_bs = args.labeled_bs
    num_classes = args.num_classes
    max_iterations = args.max_iterations

    def create_model(ema=False):
        model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes)
        print(torch.cuda.is_available())
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        model = model.to(device)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.data_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([
                                RandomGenerator(args.patch_size)
                            ]))
    db_val = BaseDataSets(base_dir=args.data_path, split="val")
    total_slices = len(db_train)

    labeled_slice = patients_to_slices(args.data_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size,
                                          args.batch_size - args.labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    model.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()

    mse_criterion = losses.mse_loss()
    criterion_att = losses.Attention()

    dice_loss = losses.DiceLoss(n_classes=num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    cur_threshold = 1 / num_classes

    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
            output1, output2, output3, encoder_features, decoder_features1, encoder_features_2, decoder_features2 = model(
                volume_batch)
            loss_seg_dice = 0

            output1_soft = F.softmax(output1, dim=1)
            output2_soft = F.softmax((-1 * output2), dim=1)
            output3_soft = F.softmax(output3, dim=1)

            output1_soft0 = F.softmax(output1 / 0.5, dim=1)
            output2_soft0 = F.softmax((-1 * (output2 / 0.5)), dim=1)

            with torch.no_grad():
                max_values1, _ = torch.max(output1_soft, dim=1)
                max_values2, _ = torch.max(output2_soft, dim=1)
                percent = (iter_num + 1) / max_iterations

                cur_threshold1 = (1 - percent) * cur_threshold + percent * max_values1.mean()
                cur_threshold2 = (1 - percent) * cur_threshold + percent * max_values2.mean()
                mean_max_values = min(max_values1.mean(), max_values2.mean())

                cur_threshold = min(cur_threshold1, cur_threshold2)
                cur_threshold = torch.clip(cur_threshold, 0.25, 0.95)

            mask_high = (output1_soft > cur_threshold) & (output2_soft > cur_threshold)
            mask_non_similarity = (mask_high == False)

            new_output1_soft = torch.mul(mask_non_similarity, output1_soft)
            new_output2_soft = torch.mul(mask_non_similarity, output2_soft)

            high_output1 = torch.mul(mask_high, output1)
            high_output2 = torch.mul(mask_high, -1 * output2)
            high_output1_soft = torch.mul(mask_high, output1_soft)
            high_output2_soft = torch.mul(mask_high, output2_soft)

            output1_soft_pseudo = torch.argmax(output1_soft, dim=1)
            output2_soft_pseudo = torch.argmax(output2_soft, dim=1)
            output3_soft_pseudo = torch.argmax(output3_soft, dim=1)
            pseudo_high_output1 = torch.argmax(high_output1_soft, dim=1)
            pseudo_high_output2 = torch.argmax(high_output2_soft, dim=1)

            max_output1_indices = new_output1_soft > new_output2_soft

            max_output1_value0 = torch.mul(max_output1_indices, output1_soft0)
            min_output2_value0 = torch.mul(max_output1_indices, output2_soft0)

            max_output2_indices = new_output2_soft > new_output1_soft

            max_output2_value0 = torch.mul(max_output2_indices, output2_soft0)
            min_output1_value0 = torch.mul(max_output2_indices, output1_soft0)

            loss_dc0 = 0
            loss_cer = 0
            loss_at_kd = 0.0

            # feature consistency loss
            loss_at_kd += criterion_att(encoder_features, decoder_features1)
            loss_at_kd += criterion_att(encoder_features_2, decoder_features2)

            # direction consistency loss
            loss_dc0 += mse_criterion(max_output1_value0.detach(), min_output2_value0)
            loss_dc0 += mse_criterion(max_output2_value0.detach(), min_output1_value0)

            # supervised Loss
            loss_seg_dice += dice_loss(output1_soft[:labeled_bs, ...], label_batch[:labeled_bs].unsqueeze(1))
            loss_seg_dice += dice_loss(output2_soft[:labeled_bs, ...], label_batch[:labeled_bs].unsqueeze(1))
            loss_seg_dice += dice_loss(output3_soft[:labeled_bs, ...], label_batch[:labeled_bs].unsqueeze(1))

            # cross pseudo supervised Loss
            if mean_max_values >= 0.95:
                loss_cer += ce_loss(output1, output3_soft_pseudo.long().detach())
                loss_cer += ce_loss(output3, output1_soft_pseudo.long().detach())

                loss_cer += ce_loss((-1 * output2), output3_soft_pseudo.long().detach())
                loss_cer += ce_loss(output3, output2_soft_pseudo.long().detach())

                loss_cer += ce_loss((-1 * output2), output1_soft_pseudo.long().detach())
                loss_cer += ce_loss(output1, output2_soft_pseudo.long().detach())

            else:
                loss_cer += ce_loss(high_output1, pseudo_high_output2.long().detach())
                loss_cer += ce_loss(high_output2, pseudo_high_output1.long().detach())

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            supervised_loss = loss_seg_dice

            # total Loss
            loss = supervised_loss + (1 - consistency_weight) * (1000 * loss_at_kd) + consistency_weight * (
                        1000 * loss_dc0) + 0.3 * loss_cer

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            logging.info(
                'iteration %d : loss : %03f,  loss_seg_dice: %03f, loss_at_kd: %03f, loss_dc0: %03f, cur_threshold: %03f, '
                'loss_cer: %03f, consistency_weight: %03f, mean_max_values: %03f' % (
                    iter_num, loss, loss_seg_dice, loss_at_kd, loss_dc0, cur_threshold, loss_cer, consistency_weight,
                    mean_max_values))

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                output = torch.argmax(torch.softmax(output1, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', output[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model,
                                                         classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)

                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    snapshot_path = args.model_save_path + "/Polyp_{}_{}_{}_labeled".format(args.model, args.exp, args.labeled_num)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('../code/', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
