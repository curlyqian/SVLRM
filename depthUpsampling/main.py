from __future__ import print_function
import argparse
from math import log10, sqrt
import time
import os
import os.path as osp
import errno
import matplotlib.pyplot as plt
from os.path import join
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_training_set, get_validation_set, get_test_set
from model import SVLRM
from loss import CharnonnierLoss
from torchvision.transforms import ToPILImage



parser = argparse.ArgumentParser(description='PyTorch SVLRM')
parser.add_argument('--dataset', type=str, default='Images',
                    required=True, help="dataset directory name")
parser.add_argument('--upscale_factor', type=int, default=4,
                    required=True, help="super resolution upscale factor")
parser.add_argument('--batch_size', type=int, default=20,
                    help="training batch size")
parser.add_argument('--test_batch_size', type=int,
                    default=1, help="testing batch size")
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='Learning Rate. Default=0.0001')
parser.add_argument("--step", type=int, default=20,
                    help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=20")
parser.add_argument("--clip", type=float, default=0.4,
                    help="Clipping Gradients. Default=0.4")
parser.add_argument("--weight-decay", "--wd", default=1e-5,
                    type=float, help="Weight decay, Default: 1e-4")
parser.add_argument("--eps", default=1e-4,type=float,
                    help="eps, Default: 1e-4")
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=16,
                    help='number of threads for data loader to use')
parser.add_argument('--gpuids', default=[0], nargs='+',
                    help='GPU ID for using')
parser.add_argument('--crop', action='store_true',
                    help='whether to crop?')
parser.add_argument('--test', action='store_true', help='test mode')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to test or resume model')


def main():
    global opt
    opt = parser.parse_args()
    opt.gpuids = list(map(int, opt.gpuids))

    print(opt)



    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    cudnn.benchmark = True

    if not opt.test:
        train_set = get_training_set(opt.dataset, opt.upscale_factor, opt.crop)
        validation_set = get_validation_set(opt.dataset, opt.upscale_factor)

    test_set = get_test_set(opt.dataset, opt.upscale_factor)

    if not opt.test:
        training_data_loader = DataLoader(
            dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
        validating_data_loader = DataLoader(
            dataset=validation_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)

    testing_data_loader = DataLoader(
        dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)

    model = SVLRM()
    criterion1 = CharnonnierLoss()
    criterion2 = nn.MSELoss()
    Loss = []
    PSNR = []
    RMSE = []


    if opt.cuda:
        torch.cuda.set_device(opt.gpuids[0])
        with torch.cuda.device(opt.gpuids[0]):
            model = model.cuda()
            criterion1 = criterion1.cuda()
            criterion2 = criterion2.cuda()
        model = nn.DataParallel(model, device_ids=opt.gpuids,output_device=opt.gpuids[0])

    optimizer = optim.Adam(model.parameters(),eps=opt.eps,weight_decay=opt.weight_decay)

    if opt.test:
        model_name = join("model", opt.model)
        model = torch.load(model_name)
        model.eval()
        start_time = time.time()
        test(model, criterion2, testing_data_loader)
        elapsed_time = time.time() - start_time
        print("===> average {:.2f} image/sec for test".format(
            100.0/elapsed_time))
        return

    train_time = 0.0
    validate_time = 0.0
    for epoch in range(1, opt.epochs + 1):
        start_time = time.time()
        train(model, criterion1, epoch, optimizer, training_data_loader,Loss)
        elapsed_time = time.time() - start_time
        train_time += elapsed_time
        print("===> {:.2f} seconds to train this epoch".format(
            elapsed_time))
        if epoch%50==0:
            start_time = time.time()
            validate(model, criterion2, validating_data_loader,PSNR,RMSE)
            elapsed_time = time.time() - start_time
            validate_time += elapsed_time
            print("===> {:.2f} seconds to validate this epoch".format(
                elapsed_time))
            checkpoint(model, epoch)

    print("===> average training time per epoch: {:.2f} seconds".format(train_time/opt.epochs))
    print("===> average validation time per epoch: {:.2f} seconds".format(validate_time/opt.epochs))
    print("===> training time: {:.2f} seconds".format(train_time))
    print("===> validation time: {:.2f} seconds".format(validate_time))
    print("===> total training time: {:.2f} seconds".format(train_time+validate_time))
    plt.figure(figsize=(15, 5))
    plt.subplot(131)  # 1行3列,第一个图
    plt.plot(Loss)
    plt.ylabel('Loss')
    plt.xlabel('epochs')
    plt.subplot(132)  # 1行3列.第二个图
    plt.plot(PSNR)
    plt.ylabel('PSNR')
    plt.xlabel('epochsX50')
    plt.subplot(133)  # 1行3列.第3个图
    plt.plot(RMSE)
    plt.ylabel('RMSE')
    plt.xlabel('epochsX50')
    plt.savefig("Loss_PSNR_RMSE.jpg")
    '''
    file1 = open('Loss.txt','w')
    for item in Loss:
        file1.write(str(item)+"\n")
    file1.close()
    '''
    file2 = open('PSNR.txt', 'w')
    for item in PSNR:
        file2.write(str(item) + "\n")
    file2.close()




def adjust_learning_rate(epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 10 epochs"""
    lr = opt.lr * (0.5 ** (epoch // opt.step))
    return lr




def train(model, criterion, epoch, optimizer, training_data_loader,Loss):
    lr = adjust_learning_rate(epoch-1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))

    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1], requires_grad=False)
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()


        optimizer.zero_grad()  #清空所有被优化过的Variable的梯度.
        model_out = model(input)
        prediction = torch.zeros(target.shape[0], target.shape[2], target.shape[3]).cuda()
        prediction = torch.addcmul(prediction,1,model_out[:,0,:,:],input[:,0,:,:])+model_out[:,1,:,:]
        loss = criterion(prediction, target[:,0,:,:])
        epoch_loss += loss.item()
        loss.backward()
        #nn.utils.clip_grad_norm_(model.parameters(), opt.clip/lr)
        optimizer.step()  #进行单次优化参数更新

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(
            epoch, iteration, len(training_data_loader), loss.item()))

    Loss.append(epoch_loss / len(training_data_loader))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(
        epoch, epoch_loss / len(training_data_loader)))


def validate(model, criterion, validating_data_loader,PSNR,RMSE):
    avg_psnr = 0
    avg_rmse = 0
    for batch in validating_data_loader:
        input, target = Variable(batch[0]), Variable(batch[1])
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        model_out = model(input)
        prediction = torch.zeros(target.shape[0], target.shape[2], target.shape[3]).cuda()
        prediction = torch.addcmul(prediction, 1, model_out[:, 0, :, :], input[:, 0, :, :]) + model_out[:, 1, :, :]
        mse = criterion(prediction*255.0, target[:,0,:,:]*255.0)
        rmse = sqrt(mse.item())
        psnr = 10 * log10(255.0**2 / mse.item())
        avg_rmse += rmse
        avg_psnr += psnr

    PSNR.append(avg_psnr / len(validating_data_loader))
    RMSE.append(avg_rmse / len(validating_data_loader))
    print("===> Avg. PSNR: {:.4f} dB".format(
        avg_psnr / len(validating_data_loader)))
    print("===> Avg. RMSE: {:.4f}".format(
        avg_rmse / len(validating_data_loader)))

def test(model, criterion, testing_data_loader,PSNR,RMSE):
    avg_psnr = 0
    avg_rmse = 0
    avg_psnr_lr = 0
    avg_rmse_lr = 0
    for batch in testing_data_loader:
        input, target = Variable(batch[0]), Variable(batch[1])
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        depth_lr = input[:, 0, :, :]
        mse_lr = criterion(depth_lr*255.0, target[:, 0, :, :]*255.0)
        rmse_lr = sqrt(mse_lr.item())
        psnr_lr = 10 * log10(255.0**2 / mse_lr.item())
        avg_rmse_lr += rmse_lr
        avg_psnr_lr += psnr_lr
        model_out = model(input)
        prediction = torch.zeros(target.shape[0], target.shape[2], target.shape[3]).cuda()
        prediction = torch.addcmul(prediction, 1, model_out[:, 0, :, :], input[:, 0, :, :]) + model_out[:, 1, :, :]
        mse = criterion(prediction*255.0, target[:, 0, :, :]*255.0)
        rmse = sqrt(mse.item())
        psnr = 10 * log10(255.0**2 / mse.item())
        avg_rmse += rmse
        avg_psnr += psnr

    PSNR.append(avg_psnr / len(testing_data_loader))
    RMSE.append(avg_rmse / len(testing_data_loader))
    print("===> Avg. PSNR_lr: {:.4f} dB".format(
        avg_psnr_lr / len(testing_data_loader)))
    print("===> Avg. RMSE_lr: {:.4f}".format(
        avg_rmse_lr / len(testing_data_loader)))
    print("===> Avg. PSNR: {:.4f} dB".format(
        avg_psnr / len(testing_data_loader)))
    print("===> Avg. RMSE: {:.4f}".format(
        avg_rmse / len(testing_data_loader)))


def save_image(tensor, num, dir):
  image = tensor.cpu().clone() # we clone the tensor to not do changes on it
  image = image.squeeze(0) # remove the fake batch dimension
  image = ToPILImage()(image)
  if not osp.exists(dir):
    os.makedirs(dir)
  image.save(dir+'/{}.png'.format(num))

def checkpoint(model, epoch):
    try:
        if not(os.path.isdir('model')):
            os.makedirs(os.path.join('model'))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise

    model_out_path = "model/model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == '__main__':
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print("===> total time: {:.2f} seconds".format(elapsed_time))
