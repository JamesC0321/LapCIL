import pandas
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import json
import os
from torch.utils.tensorboard import SummaryWriter
import random
from utils import ski_eval_metric
from model.LapCIL import Contextual_intences,reshape_tensor,restore_tensor,Bottleneck,ChannelAttentionClassifier
import numpy as np
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 32
set_seed(SEED)





parser = argparse.ArgumentParser(description='')


parser.add_argument('--name', default='fold0', type=str)
parser.add_argument('--EPOCH', default=200, type=int)
parser.add_argument('--epoch_step', default='[100]', type=str)  # 没有之前好
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--log_dir', default=None, type=str)   ## log file path
parser.add_argument('--train_show_freq', default=40, type=int)
parser.add_argument('--lr', default=1e-4, type=float)

parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--seed', default=32, type=int)
parser.add_argument('--num_cls', default=2, type=int)
parser.add_argument('--mDATA0_dir_train0', default=None, type=str)  ## Train Set
parser.add_argument('--mDATA0_dir_val0', default=None, type=str)      ## Validation Set
parser.add_argument('--mDATA_dir_test0', default=None, type=str)         ## Test SetSet
parser.add_argument('--dataset', default="CAMELYON16", type=str)         ## Test SetSet


parser.add_argument('--isSaveModel', action='store_false')




def remove_continuous_mask_data(x, fraction=0.1):
    n = x.shape[0]
    keep_sample = int(fraction * n)

    start_index = np.random.randint(0, n - keep_sample + 1)

    remove_indices = np.arange(start_index, start_index + keep_sample)

    mask = np.ones(n, dtype=bool)
    mask[remove_indices] = False
    xn = x[mask]
    return xn


def main_loop(params):

    epoch_step = json.loads(params.epoch_step)
    writer = SummaryWriter(os.path.join(params.log_dir, 'LOG', params.name))
    # ResNet50 VMamba UNI 1024 CONCH 512 VIT 384
    CoTNet = Contextual_intences(Bottleneck, 1024, 256, 1).to(params.device)  # 堆叠 CotLayer 块，效果提升不明显。

    classifier = ChannelAttentionClassifier(params.num_cls,1024,"cuda")


    set_seed(params.seed)
    print(CoTNet)



    if not os.path.exists(params.log_dir):
        os.makedirs(params.log_dir)
    log_dir = os.path.join(params.log_dir, 'log.txt')
    save_dir = os.path.join(params.log_dir, 'best_model.pth')
    z = vars(params).copy()
    with open(log_dir, 'a') as f:
        f.write(json.dumps(z))
    log_file = open(log_dir, 'a')

    mDATA_train = pandas.read_csv(params.mDATA0_dir_train0)
    mDATA_val = pandas.read_csv(params.mDATA0_dir_val0)
    mDATA_test = pandas.read_csv(params.mDATA_dir_test0)



    print_log(f'training seed: {params.seed}', log_file)

    SlideNames_train, FeatList_train, Label_train = reOrganize_mLIST(mDATA_train,params.feat_dir,params.dataset)
    SlideNames_val, FeatList_val, Label_val = reOrganize_mLIST(mDATA_val,params.feat_dir,params.dataset)
    SlideNames_test, FeatList_test, Label_test = reOrganize_mLIST(mDATA_test,params.feat_dir,params.dataset)


    ce_cri = torch.nn.CrossEntropyLoss(reduction='none').to(params.device)

    print_log(f'training slides: {len(SlideNames_train)}, validation slides: {len(SlideNames_val)}, test slides: {len(SlideNames_test)}', log_file)

    print_log(f'training slides: {len(SlideNames_train)}',log_file)
    trainable_parameters = []
    trainable_parameters += list(CoTNet.parameters())
    trainable_parameters += list(classifier.parameters())


    optimizer_adam0 = torch.optim.Adam(trainable_parameters, lr=params.lr,  weight_decay=params.weight_decay)

    scheduler0 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam0, epoch_step, gamma=params.lr_decay_ratio)


    best_index = 0

    stop_early = 0
    for ii in range(params.EPOCH):
        if ii % 24 == 0:  #
            fraction = 0.10+0.05 * (ii / 24)
            if fraction > 0.5:
                fraction = 0.5
        train_attention_preFeature_DTFD(classifier=classifier,CoTNet=CoTNet,fraction=fraction,mDATA_list=(SlideNames_train, FeatList_train, Label_train), ce_cri=ce_cri,
                                                   optimizer0=optimizer_adam0,  epoch=ii, params=params, f_log=log_file, writer=writer)
        print_log(f'>>>>>>>>>>> Validation Epoch: {ii}', log_file)
        auc_val,acc_val = val_attention_DTFD_preFeat_MultipleMean(mDATA_list=(SlideNames_val, FeatList_val, Label_val),
                                                                  classifier=classifier, CoTNet=CoTNet, is_Test=False,
                                                                  epoch=ii, criterion=ce_cri, params=params,
                                                                  f_log=log_file, writer=writer)
        print_log(' ', log_file)


        if best_index < (auc_val+acc_val):
            stop_early=0
            best_index = auc_val + acc_val
            best_epoch = ii
            if params.isSaveModel:
                tsave_dict = {
                    'classifier': classifier.state_dict(),
                    'COT': CoTNet.state_dict(),
                }
                torch.save(tsave_dict, save_dir)
        else:
            stop_early = stop_early + 1
            print_log(f"no. {stop_early} step stop_early", log_file)

        if stop_early>=20:
            print_log(f"epoch {ii}, auc no up in {stop_early},so stop early", log_file)
            break


        scheduler0.step()

def val_attention_DTFD_preFeat_MultipleMean(mDATA_list, classifier, CoTNet,is_Test, epoch, criterion=None,  params=None, f_log=None, writer=None, ):

    classifier.eval()


    SlideNames, FeatLists, Label = mDATA_list

    test_loss0 = AverageMeter()


    gPred_0 = torch.FloatTensor().to(params.device)
    gt_0 = torch.LongTensor().to(params.device)

    with torch.no_grad():

        numSlides = len(SlideNames)
        numIter = numSlides // params.batch_size
        tIDX = list(range(numSlides))

        for idx in range(numIter):

            tidx_slide = tIDX[idx * params.batch_size:(idx + 1) * params.batch_size]
            slide_names = [SlideNames[sst] for sst in tidx_slide]
            tlabel = [Label[sst] for sst in tidx_slide]
            label_tensor = torch.LongTensor(tlabel).to(params.device)
            batch_feat = [ FeatLists[sst] for sst in tidx_slide ]

            for tidx, tfeat in enumerate(batch_feat):
                tslideName = slide_names[tidx]
                tslideLabel = label_tensor[tidx].unsqueeze(0)

                tfeat = torch.load(tfeat).to("cuda")
                original_shape = tfeat.shape
                tfeat_tensor2d = reshape_tensor(tfeat)

                tAA = CoTNet(tfeat_tensor2d)
                tAA = restore_tensor(tAA, original_shape)

                tPredict = classifier(tAA)  ### 1 x 2

                torch.cuda.empty_cache()
                loss0 = criterion(tPredict, tslideLabel).mean()
                test_loss0.update(loss0.item())
                gPred_0 = torch.cat([gPred_0, tPredict], dim=0)
                gt_0 = torch.cat([gt_0, tslideLabel], dim=0)

    gPred_0 = torch.softmax(gPred_0, dim=1)

    macc_0, mprec_0, mrecal_0, mF1_0, auc_0,conf_mat = ski_eval_metric(gPred_0, gt_0,params.num_cls)

    print_log(
        f'  AUC {auc_0}, recall {mrecal_0}, F1 {mF1_0}, precision {mprec_0}, acc {macc_0},confusion_matrix{conf_mat}',
        f_log)

    if is_Test:
        writer.add_scalar(f'test_auc_0 ', auc_0, epoch)
    else:
        writer.add_scalar(f'val_auc_0 ', auc_0, epoch)
        writer.add_scalar(f'val_acc ', macc_0, epoch)
        writer.add_scalar(f'val_precision',mprec_0, epoch)
        writer.add_scalar(f'val_recall',mrecal_0, epoch)
        writer.add_scalar(f'val_F1',mF1_0, epoch)
    return auc_0,macc_0

def train_attention_preFeature_DTFD(mDATA_list, classifier, CoTNet,fraction,  optimizer0, epoch, ce_cri=None, params=None,
                                          f_log=None, writer=None):
    set_seed(params.seed)
    SlideNames_list, mFeat_list, Label_dict = mDATA_list

    classifier.train()
    CoTNet.train()


    Train_Loss0 = AverageMeter()
    numSlides = len(SlideNames_list)
    numIter = numSlides // params.batch_size

    tIDX = list(range(numSlides))
    random.shuffle(tIDX)

    for idx in range(numIter):

        tidx_slide = tIDX[idx * params.batch_size:(idx + 1) * params.batch_size]

        tslide_name = [SlideNames_list[sst] for sst in tidx_slide]
        tlabel = [Label_dict[sst] for sst in tidx_slide]
        label_tensor = torch.LongTensor(tlabel).to(params.device)

        for tidx, (tslide, slide_idx) in enumerate(zip(tslide_name, tidx_slide)):
            tslideLabel = label_tensor[tidx].unsqueeze(0)

            tfeat_tensor = torch.load(mFeat_list[slide_idx])
            tfeat_tensor = remove_continuous_mask_data(tfeat_tensor, fraction)
            torch.cuda.empty_cache()
            tfeat_tensor = tfeat_tensor.to(params.device)
            original_shape = tfeat_tensor.shape
            tfeat_tensor2d = reshape_tensor(tfeat_tensor)

            tAA = CoTNet(tfeat_tensor2d)
            tAA = restore_tensor(tAA,original_shape)



             ## 1 x fs
            tPredict = classifier(tAA)  ### 1 x 2


            torch.cuda.empty_cache()
            loss0 = ce_cri(tPredict, tslideLabel)
            optimizer0.zero_grad()
            loss0.backward(retain_graph=True)


            optimizer0.step()

            Train_Loss0.update(loss0.item())

        if idx % params.train_show_freq == 0:
            tstr = 'epoch: {} idx: {}'.format(epoch, idx)
            tstr += f'Loss : {Train_Loss0.avg}'
            print_log(tstr, f_log)
    print_log('train_loss_0 {}'.format(Train_Loss0.avg), f_log)
    writer.add_scalar(f'train_loss_0 ', Train_Loss0.avg, epoch)
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_log(tstr, f):
    # with open(dir, 'a') as f:
    f.write('\n')
    f.write(tstr)
    print(tstr)


def reOrganize_mLIST(mDATA,feat_dir,dataset):

    SlideNames = []
    PathList = []
    Label = []
    if dataset == "EC":
        for index, slide in  mDATA.iterrows():
            SlideNames.append(slide[0])
            slide_type = slide[2]
            if slide_type.startswith('MMRd'):
                label = 0
            elif slide_type.startswith('NSMP'):
                label = 1
            elif slide_type.startswith('P53mut'):
                label = 2
            elif slide_type.startswith('POLE'):
                label = 3
            else:
                raise RuntimeError('Undefined slide type')
            Label.append(label)
            slide_h5 = feat_dir + "pt_files/" + slide[1] + ".pt"
            PathList.append(slide_h5)
    elif dataset == "CAMELYON16":
        for index, slide in mDATA.iterrows():
            SlideNames.append(slide[0])
            slide_type = slide[2]
            if slide_type.startswith('normal'):
                label = 0
            elif slide_type.startswith('tumor'):
                label = 1
            else:
                raise RuntimeError('Undefined slide type')
            Label.append(label)
            slide_h5 = feat_dir + "pt_files/" + slide[1] + ".pt"
            PathList.append(slide_h5)
    elif dataset == "BRACS":
        for index, slide in mDATA.iterrows():
            SlideNames.append(slide[0])
            slide_type = slide[2]
            if slide_type.startswith('Benign'):
                label = 0
            elif slide_type.startswith('Atypical'):
                label = 1
            elif slide_type.startswith('Malignant'):
                label = 2
            else:
                raise RuntimeError('Undefined slide type')
            Label.append(label)
            slide_h5 = feat_dir + "pt_files/" + slide[1] + ".pt"
            PathList.append(slide_h5)





    return SlideNames, PathList, Label



def loop_main(path,params):
    paths = os.listdir(path)
    paths.sort()
    feat_dir_list = [

        r'/media/chen/KINGSTON/CAMELYON16/data_feature/resnet50/', # 1024
        # r'/media/chen/KINGSTON/CAMELYON16/data_feature/vit/', # 1024



    ]
    log_dir_list = [

        r'./LapCIL_CAMELYON16/',

    ]
    for i, feat in enumerate(feat_dir_list):
        log_dir = log_dir_list[i]
        params.feat_dir = feat
        for p in paths:
            set_seed(params.seed)
            listdirs = os.listdir(os.path.join(path,p))
            listdirs.sort()
            test,train,val = listdirs
            print(" train:{} \t val:{}\t test : {}".format(train,val,test))
            params.mDATA0_dir_train0 = os.path.join(path,p,train)
            params.mDATA0_dir_val0 = os.path.join(path,p,val)
            params.mDATA_dir_test0 = os.path.join(path,p,test)
            params.log_dir = log_dir + p




            if not os.path.isdir(params.log_dir):

                main_loop(params)


if __name__ == "__main__":

    path = r'/media/chen/新加卷/EC_code/VMCIL/train_labels/camelyon16'  # train labels
    params = parser.parse_args()
    set_seed(params.seed)
    loop_main(path,params)
