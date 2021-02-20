import numpy as np
import os
import torch
import torch.nn as nn
from model import l3_dense
from IPython import embed
import pandas
from sklearn.metrics import accuracy_score
import h5py
from tqdm import tqdm
import random
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sn
import argparse
'''
This script is to test audio, video and audio-visual networks.
Examples to run this code,
python test.py --features_path '../create_data/features_data/' --model_type 'audio'
python test.py --features_path '../create_data/features_data/' --model_type 'video'
python test.py --features_path '../create_data/features_data/' --model_type 'audio_video'
'''
parser = argparse.ArgumentParser(description='test audio, video and audio-visual networks')
parser.add_argument('--features_path', type=str,required=True,
                    help='give the features path.')
parser.add_argument('--model_type', type=str,required=True,
                    help='it can be audio, video, audio_video')
args, _ = parser.parse_known_args()

if args.model_type =='audio':

    path_input = args.features_path + 'audio_features_data/tt.hdf5'

    global_mean_std_path = args.features_path +'audio_features_data/global_mean_std.npz'
    mean_std = np.load(global_mean_std_path)
    mean = mean_std['global_mean']
    std = mean_std['global_std']
    model = l3_dense(512,10)
    model_path = './audio_model/model.pt'


if args.model_type =='video':
    path_input = args.features_path +'video_features_data/tt.hdf5'
    global_mean_std_path = args.features_path +'video_features_data/global_mean_std.npz'
    mean_std = np.load(global_mean_std_path)
    mean = mean_std['global_mean']
    std = mean_std['global_std']
    model = l3_dense(512,10)

    model_path = './video_model/model.pt'


if args.model_type =='audio_video':

    path_input = args.features_path +'audio_features_data/tt.hdf5'

    global_mean_std_path_audio = args.features_path +'audio_features_data/global_mean_std.npz'

    mean_std_audio = np.load(global_mean_std_path_audio)
    mean_audio = mean_std_audio['global_mean']
    std_audio = mean_std_audio['global_std']

    global_mean_std_path_video = args.features_path +'video_features_data/global_mean_std.npz'

    mean_std_video = np.load(global_mean_std_path_video)
    mean_video = mean_std_video['global_mean']
    std_video = mean_std_video['global_std']

    model = l3_dense(512*2,10)
    model_path = './audio_video_model/model.pt'



def func(name, obj):     # function to recursively store all the keys
    if isinstance(obj, h5py.Dataset):
        all_files.append(name)
    elif isinstance(obj, h5py.Group):
        group.append(name)

##### load the testing file######
all_files = []
group = []
hf = h5py.File(path_input, 'r')
hf.visititems(func)

model.load_state_dict(torch.load(model_path))
print(model)
#embed()
fn = torch.nn.LogSoftmax(dim=1)
fn_softmax = torch.nn.Softmax(dim=1)

ground_tr_list = []
esti_list=[]
y_hat=[]
y = []
for i in tqdm(range(len(all_files))):

    if args.model_type == 'audio_video':
        path_input_audio = path_input
        path_input_video = path_input.replace('audio','video')
        #embed()
        hf_audio = h5py.File(path_input_audio, 'r')
        #print(path_input_video)
        hf_video = h5py.File(path_input_video, 'r')

        emb_audio= np.array(hf_audio[all_files[i]])
        emb_video = np.array(hf_video[all_files[i].replace('audio','video')])

        ground_tr = np.array(int(all_files[i].split('/')[0]))
        ground_tr_list.append(ground_tr)

        es_label = torch.zeros(1,10)
        es_label_ = torch.zeros(1,10)
        for j in range(emb_audio.shape[0]):

            each_emb_audio = emb_audio[j,:]
            normed_audio_embed =  (each_emb_audio-mean_audio)/std_audio

            each_emb_video = emb_video[j,:]
            normed_video_embed =  (each_emb_video-mean_video)/std_video

            normed_embed = np.concatenate((normed_audio_embed,normed_video_embed))


            normed_embed_tensor = torch.from_numpy(normed_embed).float()
            normed_embed_tensor = normed_embed_tensor.view(1,-1)

            model.eval()
           # embed()
            with torch.no_grad():
                es_label_each_frame = model(normed_embed_tensor)
                es_label_each_frame = fn(es_label_each_frame.view(1,-1))
                es_prob = fn_softmax(es_label_each_frame).flatten()
            es_label += es_label_each_frame
            es_label_ +=  es_prob
        es_label_ = es_label_/emb_audio.shape[0]
        es_label_=es_label_.flatten()
        #embed()
        es_label_ = es_label_.tolist()

        es_class = torch.argmax(es_label)
        esti_list.append(es_class)
        y_hat.append(es_label_)
        y.append(int(all_files[i].split('/')[0]))
       # embed()


    if args.model_type == 'audio' or args.model_type == 'video':

        hf = h5py.File(path_input, 'r')
        emb = np.array(hf[all_files[i]])
        ground_tr = np.array(int(all_files[i].split('/')[0]))
        ground_tr_list.append(ground_tr)

        es_label = torch.zeros(1,10)
        es_label_ = torch.zeros(1,10)
        for j in range(emb.shape[0]):

            each_emb = emb[j,:]

            normed_embed =  (each_emb-mean)/std

            normed_embed_tensor = torch.from_numpy(normed_embed).float()
            normed_embed_tensor = normed_embed_tensor.view(1,-1)

            model.eval()
            with torch.no_grad():
                es_label_each_frame = model(normed_embed_tensor)
                es_label_each_frame = fn(es_label_each_frame.view(1,-1))

                es_prob = fn_softmax(es_label_each_frame).flatten()
                #es_prob = es_prob.flatten().tolist()

            es_label += es_label_each_frame
            es_label_ +=  es_prob
            #embed()
        es_label_ = es_label_/emb.shape[0]
        es_label_=es_label_.flatten()
        #embed()
        es_label_ = es_label_.tolist()

        es_class = torch.argmax(es_label)
        esti_list.append(es_class)

        y_hat.append(es_label_)
        y.append(int(all_files[i].split('/')[0]))
       # embed()
#embed()
##########evaluation##########
y_true = np.array(ground_tr_list)
y_pred = np.array(esti_list)
cm = confusion_matrix(y_true, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm.diagonal()
keys = ['airport',
                'bus',
                'metro',
                'metro_station',
                'park',
                'public_square',
                'shopping_mall',
                'street_pedestrian',
                'street_traffic',
                'tram']
#values = [np.round(i,decimals=3) for i in list(cm.diagonal())]


df_cm = pandas.DataFrame(cm, index = [i for i in keys],
                  columns = [i for i in keys])
plt.figure(figsize = (15,12))
sn.heatmap(df_cm, annot=True)
plt.savefig('cm_'+args.model_type+'.png')
values = [np.round(i,decimals=3) for i in list(cm.diagonal())]

acc = accuracy_score(y_true, y_pred)
acc = np.round(acc,decimals = 3)

#embed()
l = []
for i in range(len(keys)):
    l.append([keys[i],values[i]])

print(tabulate(l, headers=['Class', 'Accuracy']))
print('  ')
print('The overall accuracy using accuracy_score  is', acc)
print('The mean accuracy of all class is', np.mean(values))
from sklearn.metrics import log_loss
logloss_overall = log_loss(y_true=y, y_pred=y_hat)
print('overall log loss is', logloss_overall)
