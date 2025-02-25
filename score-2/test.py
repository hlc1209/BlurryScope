"""
Author: Hanlong Chen
Version: 1.0
Contact: hanlong@ucla.edu
"""

import logger_config
logger = logger_config.get_logger(__name__)

import pynvml
import psutil
import os
def get_free_gpus(num_gpus=1, max_memory_usage=0.3, max_gpu_usage=0.3):
    def is_training_process(process_name):
        training_keywords = ['python', 'python3', 'tensorflow', 'pytorch', 'train']
        return any(keyword in process_name.lower() for keyword in training_keywords)
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    free_gpus = []
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory_usage = memory_info.used / memory_info.total
        gpu_usage = util_info.gpu / 100
        training_process_found = False
        try:
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            for process in processes:
                pid = process.pid
                process_name = psutil.Process(pid).name()
                if is_training_process(process_name):
                    training_process_found = True
                    break
        except pynvml.NVMLError as err:
            logger.debug(f"Error: {err}")
        if not training_process_found and memory_usage < max_memory_usage and gpu_usage < max_gpu_usage:
            free_gpus.append(i)
        if len(free_gpus) >= num_gpus:
            break
    pynvml.nvmlShutdown()
    return free_gpus
num_gpus =1
available_gpus = get_free_gpus(num_gpus=num_gpus)
if available_gpus:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, available_gpus))
    logger.warning(f"Using GPU(s): {available_gpus}")
elif False:
    logger.critical("overriding GPU check")
    available_gpus = [1]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, available_gpus))
    logger.warning(f"Using GPU(s): {available_gpus}")
else:
    logger.critical("No available GPU found.")
    raise SystemExit("No GPU available.")

from pathlib import Path
script_path = Path(__file__).resolve()
script_dir = script_path.parent
folder_name = script_dir.name
parts = folder_name.split('-')
if len(parts) == 2:
    stage = parts[0]
    version = parts[1]
    logger.warning(f"Stage: {stage}")
    logger.warning(f"Version: {version}")

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from torchvision.utils import save_image
import glob
from skimage.metrics import structural_similarity as ssim

from unet_parts import *

import my_tools
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from timeit import default_timer
import tqdm

from torch.optim import AdamW

torch.manual_seed(42)
np.random.seed(42)

def check_zero_grad(model, model_name="Model"):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if param.grad.abs().sum().item() == 0:
                logger.critical(f"{model_name} Parameter {name} has zero gradient!")
            else:
                logger.debug(f"{model_name} Parameter {name} has non-zero gradient.")
        else:
            logger.critical(f"{model_name} Parameter {name} has no gradient!")

################################################################
# fourier layer
################################################################

class Unet(nn.Module):
    def __init__(self, in_channels):
        super(Unet, self).__init__()

        self.n_channels = in_channels
        self.n_classes = in_channels*2
        self.bilinear = False

        self.inc = DoubleConv(self.n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        factor = 2 if self.bilinear else 1
        self.up2 = Up(128, 64 // factor, self.bilinear)
        self.up3 = Up(64, 32 // factor, self.bilinear)
        self.outc = OutConv(32, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up2(x3, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

class dSPAF(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(dSPAF, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.unet = Unet(out_channels)

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = (self.scale * torch.ones(1, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)).cuda()
        self.weights2 = (self.scale * torch.ones(1, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)).cuda()

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,bioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        factor = x_ft.abs()
        factor = torch.concat((factor[:, :, :self.modes1, :self.modes2],factor[:, :, -self.modes1:, :self.modes2]),dim=2)
        factor = self.unet(factor)
        factor = factor.reshape((batchsize, 2, 1, x_ft.shape[1], self.modes1*2, self.modes2))

        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1.mul(torch.view_as_complex(torch.stack((factor[:, 0][:, :, :, :self.modes1, :self.modes2],factor[:, 1][:, :, :, :self.modes1, :self.modes2]),dim=-1))))
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2.mul(torch.view_as_complex(torch.stack((factor[:, 0][:, :, :, -self.modes1:, :self.modes2],factor[:, 1][:, :, :, -self.modes1:, :self.modes2]),dim=-1))))

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class EFIN(nn.Module):
    def __init__(self, modes, width):
        super(EFIN, self).__init__()

        self.scales_per_block = [1,1,1,2,2,2]
        self.share_block = [False,False,True,False,True,True]
        self.num_per_block = [2,2,2,2,2,2]
        assert len(self.scales_per_block) == len(self.share_block)
        assert len(self.scales_per_block) == len(self.num_per_block)

        self.modes = modes
        self.width = width
        self.padding = 2
        self.conv_begin_0 = nn.Conv2d(T_in * T_in_comp + 2, 12, 1)
        self.prelu_begin = nn.PReLU(12)
        self.conv_begin_1 = nn.Conv2d(12, self.width, 1)

        self.SConv2d_list = []
        self.w_list = []
        self.prelu_list = []
        self.ssc_list = []
        self.conv_list = []

        current_width = self.width
        total_width = 0
        for i in range(len(self.scales_per_block)):
            logger.debug(f"building scales {i}")
            if self.share_block[i]:
                logger.debug(f"\tshared params {self.scales_per_block[i]}")
                self.conv_list.append(nn.Conv2d(current_width, current_width, 1))
                self.SConv2d_list.append(dSPAF(current_width, current_width, self.modes//self.scales_per_block[i], self.modes//self.scales_per_block[i]))
                self.w_list.append(nn.Conv2d(current_width, current_width, 1))
                self.prelu_list.append(nn.PReLU(current_width))
                total_width += current_width * self.num_per_block[i]
            else:
                logger.debug("not shared params %s", ' '.join([str(self.scales_per_block[i])] * self.num_per_block[i]))
                for _ in range(self.num_per_block[i]):
                    self.conv_list.append(nn.Conv2d(current_width, current_width, 1))
                    self.SConv2d_list.append(dSPAF(current_width, current_width, self.modes//self.scales_per_block[i], self.modes//self.scales_per_block[i]))
                    self.w_list.append(nn.Conv2d(current_width, current_width, 1))
                    self.prelu_list.append(nn.PReLU(current_width))
                    total_width += current_width
            self.ssc_list.append(nn.Conv2d(current_width, current_width, 1))
            current_width += current_width
        
        self.conv_list = nn.ModuleList(self.conv_list)
        self.SConv2d_list = nn.ModuleList(self.SConv2d_list)
        self.w_list = nn.ModuleList(self.w_list)
        self.prelu_list = nn.ModuleList(self.prelu_list)
        self.ssc_list = nn.ModuleList(self.ssc_list)

        self.conv_end1 = nn.Conv2d(current_width, current_width, 1)
        self.conv_end2 = nn.Conv2d(current_width, T, 1)
        self.prelu_end = nn.PReLU(current_width)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)
        x = self.conv_begin_0(x)
        x = self.prelu_begin(x)
        x = self.conv_begin_1(x)

        features = [x]

        pointer = 0
        for i in range(len(self.scales_per_block)):
            x = torch.cat(features, 1)
            x_s = x
            if self.share_block[i]:
                for _ in range(self.num_per_block[i]):
                    x_t = x
                    x = self.conv_list[pointer](x)
                    result = self.SConv2d_list[pointer](x)
                    x = result + self.w_list[pointer](x)
                    x = self.prelu_list[pointer](x)
                    x = x + x_t
                pointer += 1
            else:
                for _ in range(self.num_per_block[i]):
                    x_t = x
                    x = self.conv_list[pointer](x)
                    result = self.SConv2d_list[pointer](x)
                    x = result + self.w_list[pointer](x)
                    x = self.prelu_list[pointer](x)
                    x = x + x_t
                    pointer += 1
            x = self.ssc_list[i](x)
            x = x + x_s
            features.append(x)


        x = torch.cat(features, 1)

        x = self.conv_end1(x)
        x = self.prelu_end(x)
        x = self.conv_end2(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)

################################################################
# configs
################################################################


DATASET_PATH = "../BlurScoreDataScanSplitTIF/"

num_workers = max(1, int(os.cpu_count() * 0.8))
logger.info("Number of workers: %d", num_workers)

ntrain_files = 500
nvalid_files = 99999

batch_size = 2
batch_size_valid = 2

epochs = 50000
learning_rate = 0.001

logger.info("Epochs: %d", epochs)
logger.info("Learning rate: %f", learning_rate)

sub = 1
S = 512 # final resolution
T_in = 15 # number of input chennels
T = 2 # number of output chennels
step = 1
sample_size = 1
pick_files = 400
is_fsp = False
T_in_comp = 1 # for DPM
SR_factor = 1

modes = S//2
width = 7

################################################################
# load data
################################################################
test_dataset = my_tools.TIFDataset(root_dir=os.path.join(DATASET_PATH, 'testing_data'))

logger.debug("Valid files: %d", len(test_dataset))

test_subset_indices = list(range(min(nvalid_files, len(test_dataset))))
test_subset = my_tools.Subset(test_dataset, test_subset_indices)
test_subset_loader = my_tools.DataLoader(test_subset, batch_size=batch_size_valid, shuffle=False, num_workers=0)

def get_train_sampler(dataset, num_samples):
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    selected_indices = indices[:num_samples]
    return my_tools.SubsetRandomSampler(selected_indices)

logger.info("Valid files after limit: %d", len(test_subset_loader)*batch_size_valid)

sample_image, sample_class = test_dataset[0]
logger.debug("Sample image shape: %s", sample_image.shape)
logger.debug("Sample class: %s", sample_class)

path = 'eFIN'+"_{}_".format(stage) + str(version) + '_ep'+str(epochs)+'_m' + str(modes) + '_w' + str(width)
path_model = '../Models/'+path

writer = SummaryWriter(os.path.join("../runs", path))

################################################################
# training and evaluation
################################################################

model = EFIN(modes, width).cuda()

optimizer = AdamW(list(model.parameters()), lr=learning_rate, weight_decay=1e-4, betas=(0.5, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)

celoss = nn.CrossEntropyLoss()

max_valid_accuracy=0

if not os.path.exists(path_model):
    os.makedirs(path_model)

print_target = False
start_ep = -1
ep_relative = start_ep+1
if os.path.isfile(ckpt_path:=os.path.join(path_model,"ckpt.pth")):
    logger.critical(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
else:
    logger.critical("No checkpoint found")
    raise RuntimeError("No checkpoint found")

###############valid##################
model.eval()
yy_list = []
zj_list = []
im_list = []
xx_list = [] 
pred_list = []
confidence_list = []
correct_predictions_valid = 0
total_predictions_valid = 0
with torch.no_grad():
    for xx, yy in (pbar := tqdm.tqdm(test_subset_loader, dynamic_ncols=True)):
        loss = 0
        xx = xx.cuda()
        yy = yy.cuda()

        im = model(xx)

        softmax_outputs = nn.Softmax(dim=1)(im)
        confidences, predicted = torch.max(softmax_outputs, 1)
        correct_predictions_valid += (predicted == yy).sum().item()
        total_predictions_valid += yy.size(0)

        yy_list.append(yy.detach().cpu().numpy())
        im_list.append(im.detach().cpu().numpy())
        pred_list.append(predicted.detach().cpu().numpy())
        confidence_list.append(confidences.detach().cpu().numpy())

yy = np.vstack(yy_list).reshape((-1,)+yy.shape[1:])
im = np.vstack(im_list).reshape((-1,)+im.shape[1:])
pred = np.vstack(pred_list).reshape((-1,))
confidence_list = np.vstack(confidence_list).reshape((-1,))

logger.critical(f"Valid accuracy: {correct_predictions_valid/total_predictions_valid*100:.2f}%")
