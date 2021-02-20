import torch
import torch.nn as nn
from model import l3_dense
from model_combine import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
from IPython import embed
import argparse
from TAU_Urban_Dataset import TAU_Urban
import pandas
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import sys

'''
This script is to train audio-visual networks.
Examples to run this code,
python train.py --model_audio_path '../train/audio_model/model.pt'
--model_video_path '../train/video_model/model.pt'
--features_path '../create_data/features_data/' --output_dir './audio_video_model/' --n_epoch 100

'''
#####set the seed to reproduce the results#####
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(description='audio-visual networks')
parser.add_argument('--model_audio_path', type=str,required=True,
                    help='give the audio model path.')
parser.add_argument('--model_video_path', type=str,required=True,
                    help='give the video model path')
parser.add_argument('--features_path', type=str,required=True,
                    help='give the features path')
parser.add_argument('--output_dir', type=str,required=True,
                    help='give the path where you want to save the model')

parser.add_argument('--n_epoch', type=int, default=100,required=False,
                    help='number of epochs you wish to run')
parser.add_argument('--batch_size', type=int, default=64,required=False,
                    help='set the batch size')
parser.add_argument('--num_classes', type=int, default=10,required=False,
                    help='set the batch size')
args, _ = parser.parse_known_args()


tr_Dataset =TAU_Urban('tr',args.features_path)
training_generator = DataLoader(tr_Dataset,batch_size = args.batch_size,
                                        shuffle = True,
                                        num_workers = 16,
                                        drop_last = True)

cv_Dataset =TAU_Urban('val',args.features_path)
validation_loader = DataLoader(cv_Dataset,batch_size = args.batch_size,
                                        shuffle = True,
                                        num_workers = 16,
                                        drop_last = True)

model_combine = l3_combine(256,args.num_classes)

model_audio = l3_dense(512,args.num_classes)
model_video = l3_dense(512,args.num_classes)
print(model_combine)

output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print("Directory " , output_dir ,  " Created ")
else:
    print("Directory " , output_dir ,  " already exists")


use_cuda = torch.cuda.is_available()
print('use_cude',use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")
model_audio.load_state_dict(torch.load(args.model_audio_path))
model_audio.cuda()
model_video.load_state_dict(torch.load(args.model_video_path))
model_video.cuda()

loss_fn=torch.nn.CrossEntropyLoss()

optimizer =optim.Adam(model_combine.parameters(), lr=0.0001,weight_decay=0.0001)
#optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.01)
# scheduler  = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
# scheduler.step()

print('-----------start training')


def train(epoch,writer_tr):
    model_combine.train()
    train_loss = 0.
    #embed()
    start_time = time.time()
    count  = training_generator.__len__()*(epoch-1)
    loader = tqdm(training_generator)
    for batch_idx, data in enumerate(loader):
        #embed()
        count = count + 1
        batch_embed_a = Variable(data[0]).contiguous().to(device)
        batch_embed_v = Variable(data[1]).contiguous().to(device)

        batch_label = Variable(data[2]).contiguous()


        batch_label = batch_label.to(device)

        # training
        optimizer.zero_grad()
        model_audio.eval()
        model_video.eval()
        # embed()
        with torch.no_grad():
            modulelist_audio = list(model_audio.model)
            modulelist_video = list(model_video.model)
            for l in modulelist_audio[:5]:
                batch_embed_a = l(batch_embed_a)
            for l in modulelist_video[:5]:
                batch_embed_v = l(batch_embed_v)

       # embed()
        batch_embed = torch.cat((batch_embed_a,batch_embed_v),1).to(device)
        esti_label = model_combine(batch_embed)
        loss = loss_fn(esti_label,batch_label)
        loss.backward()

        train_loss += loss.data.item()
        optimizer.step()
        #writer_tr.add_scalar('Loss/train', loss.data.item(),batch_idx*epoch)

        if (batch_idx+1) % 100 == 0:
            elapsed = time.time() - start_time

            writer_tr.add_scalar('Loss/train', loss.data.item(),count)
            writer_tr.add_scalar('Loss/train_avg', train_loss/(batch_idx+1),count)
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.2f} |'.format(
                epoch, batch_idx+1, len(training_generator),
                elapsed * 1000 / (batch_idx+1), loss ))


    train_loss /= (batch_idx+1)
    print('-' * 99)
    print('    | end of training epoch {:3d} | time: {:5.2f}s | training loss {:5.2f} |'.format(
            epoch, (time.time() - start_time), train_loss))

    return train_loss
def validate(epoch,writer_val):
    model_combine.eval()
    validation_loss = 0.
    start_time = time.time()
    # data loading
    for batch_idx, data in enumerate(validation_loader):

        batch_embed_a = Variable(data[0]).contiguous().to(device)
        batch_embed_v = Variable(data[1]).contiguous().to(device)

        batch_label = Variable(data[2]).contiguous()

        batch_label = batch_label.to(device)

        model_audio.eval()
        model_video.eval()
        # embed()
        with torch.no_grad():
            modulelist_audio = list(model_audio.model)
            modulelist_video = list(model_video.model)
            for l in modulelist_audio[:5]:
                batch_embed_a = l(batch_embed_a)
            for l in modulelist_video[:5]:
                batch_embed_v = l(batch_embed_v)
       # embed()
        batch_embed = torch.cat((batch_embed_a,batch_embed_v),1).to(device)
        with torch.no_grad():
             esti_label = model_combine(batch_embed)
             loss = loss_fn(esti_label,batch_label)
             validation_loss += loss.data.item()

    #print('the ',batch_idx,'iteration val_loss is ', validation_loss)
    validation_loss /= (batch_idx+1)
   # embed()
    writer_val.add_scalar('Loss/val', loss.data.item(),batch_idx*epoch)
    writer_val.add_scalar('Loss/val_avg', validation_loss,batch_idx*epoch)
    print('    | end of validation epoch {:3d} | time: {:5.2f}s | validation loss {:5.2f} |'.format(
            epoch, (time.time() - start_time), validation_loss))
    print('-' * 99)

    return validation_loss


training_loss = []
validation_loss = []
decay_cnt = 0
writer_tr = SummaryWriter(os.path.join(output_dir,'train'))
writer_val = SummaryWriter(os.path.join(output_dir,'val'))
for epoch in range(1, args.n_epoch):
    model_combine.cuda()
    print('this is epoch', epoch)
    training_loss.append(train(epoch,writer_tr)) # Call training
    validation_loss.append(validate(epoch,writer_val)) # call validation
    print('-' * 99)
    print('after epoch', epoch, 'training loss is ', training_loss, 'validation loss is ', validation_loss)
    if training_loss[-1] == np.min(training_loss):
        print(' Best training model found.')

    if validation_loss[-1] == np.min(validation_loss):
        # save current best model
        with open(args.output_dir+'model.pt', 'wb') as f:
            torch.save(model_combine.cpu().state_dict(), f)

            print(' Best validation model found and saved.')
            print('-' * 99)

        #torch.save(model, os.path.join(os.path.dirname(args.val_save), 'alt_' + os.path.basename(args.val_save))) # save in alternative format

    # decay_cnt += 1
    # #lr decay
    # #if np.min(validation_loss) not in validation_loss[-3:] and decay_cnt >= 3:
    # if np.min(training_loss) not in training_loss[-3:] and decay_cnt >= 3:
    #     scheduler.step()
    #     decay_cnt = 0
    #     print('  Learning rate decreased.')
    #     print('-' * 99)


minmum_val_index=np.argmin(validation_loss)
minmum_val=np.min(validation_loss)
plt.plot(training_loss,'r')
#plt.hold(True)
plt.plot(validation_loss,'b')
plt.axvline(x=minmum_val_index,color='k',linestyle='--')
plt.plot(minmum_val_index,minmum_val,'r*')

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss', 'val_loss'], loc='upper left')
plt.savefig(args.output_dir+'loss.png')
