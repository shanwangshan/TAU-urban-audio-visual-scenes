import numpy as np
import os
import torch
import torch.nn as nn
from model import l3_dense
from  model_combine import *
from IPython import embed
import pandas
from sklearn.metrics import accuracy_score
import h5py
from tqdm import tqdm
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sn
import argparse

'''
This script is to test the proposed audio-visual networks.
Examples to run this code,
python test.py --features_path '../create_data/features_data/' --model_audio_path '../train/audio_model/model.pt' --model_video_path '../train/video_model/model.pt'

'''
parser = argparse.ArgumentParser(description='test audio-visual networks')

parser.add_argument('--features_path', type=str,required=True,
                    help='give the features path')
parser.add_argument('--model_audio_path', type=str,required=True,
                    help='give the audio model path.')
parser.add_argument('--model_video_path', type=str,required=True,
                    help='give the video model path')

args, _ = parser.parse_known_args()


path_input = args.features_path + 'audio_features_data/tt.hdf5'

global_mean_std_path_audio =args.features_path + 'audio_features_data/global_mean_std.npz'
mean_std_audio = np.load(global_mean_std_path_audio)
mean_audio = mean_std_audio['global_mean']
std_audio = mean_std_audio['global_std']

global_mean_std_path_video = args.features_path+ 'video_features_data/global_mean_std.npz'

mean_std_video = np.load(global_mean_std_path_video)
mean_video = mean_std_video['global_mean']
std_video = mean_std_video['global_std']
#embed()
model = l3_combine(256, 10)


#model_path = './audio_video_model/model_soum.pt'
#model_path = './audio_video_model/model.pt'
model_path = './audio_video_model/model.pt'

model.load_state_dict(torch.load(model_path))
print(model)

model_audio = l3_dense(512,10)
model_audio.load_state_dict(torch.load(args.model_audio_path))

model_video = l3_dense(512,10)
model_video.load_state_dict(torch.load(args.model_video_path))

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

#embed()
fn = torch.nn.LogSoftmax(dim=1)
fn_softmax = torch.nn.Softmax(dim=1)
ground_tr_list = []
esti_list=[]
y_hat=[]
y = []
#all_files = random.sample(all_files, 3000)
for i in tqdm(range(len(all_files))):
    path_input_audio = path_input
    path_input_video = path_input.replace('audio','video')

    hf_audio = h5py.File(path_input_audio, 'r')
    hf_video = h5py.File(path_input_video, 'r')

    emb_audio = np.array(hf_audio[all_files[i]])
    normed_embed_audio = (emb_audio-mean_audio)/std_audio

    video_name = all_files[i].replace('audio', 'video')

    emb_video = np.array(hf_video[video_name])
    normed_embed_video = (emb_video-mean_video)/std_video

    ground_tr = np.array(int(all_files[i].split('/')[0]))
    ground_tr_list.append(ground_tr)

    es_label = torch.zeros(1, 10)
    es_label_ = torch.zeros(1, 10)

    for j in range(normed_embed_audio.shape[0]):
        each_normed_embed_audio = normed_embed_audio[j, :]
        normed_audio_embed_tensor = torch.from_numpy(each_normed_embed_audio).float()
        normed_audio_embed_tensor = normed_audio_embed_tensor.view(1, -1)

        each_normed_embed_video = normed_embed_video[j, :]
        normed_video_embed_tensor = torch.from_numpy(each_normed_embed_video).float()
        normed_video_embed_tensor = normed_video_embed_tensor.view(1, -1)

        model_audio.eval()
        model_video.eval()
        model.eval()

        with torch.no_grad():
            modulelist_audio = list(model_audio.model)
            modulelist_video = list(model_video.model)
            for m in modulelist_audio[:5]:
                normed_audio_embed_tensor = m(normed_audio_embed_tensor)
            for k in modulelist_video[:5]:
                normed_video_embed_tensor = k(normed_video_embed_tensor)

        batch_embed = torch.cat((normed_audio_embed_tensor, normed_video_embed_tensor), 1)
        with torch.no_grad():
            es_label_each_frame = model(batch_embed)
            es_label_each_frame = fn(es_label_each_frame.view(1,-1))
            es_prob = fn_softmax(es_label_each_frame).flatten()

        es_label += es_label_each_frame
        es_label_ +=  es_prob
    es_label_ = es_label_/(normed_embed_audio.shape[0])

    es_label_ = es_label_.flatten()
        #embed()
    es_label_ = es_label_.tolist()

    es_class = torch.argmax(es_label)
    esti_list.append(es_class)
    y_hat.append(es_label_)
    y.append(int(all_files[i].split('/')[0]))

#embed()
############evaluation##########
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

#print(classification_report(y_true, y_pred, target_names=target_names))
values = [np.round(i, decimals=3) for i in list(cm.diagonal())]
df_cm = pandas.DataFrame(cm, index=[i for i in keys],columns=[i for i in keys])
plt.figure(figsize=(15, 12))
sn.heatmap(df_cm, annot=True)
plt.savefig('cm.png')
# dictionary = dict(zip(keys, values))
# print(dictionary)
acc = accuracy_score(y_true, y_pred)
acc = np.round(acc, decimals=3)
l = []
for i in range(len(keys)):
    l.append([keys[i], values[i]])

print(tabulate(l, headers=['Class', 'Accuracy']))
print('  ')

print('The overall accuracy using accuracy_score  is', acc)
print('The mean accuracy of all class is', np.mean(values))
from sklearn.metrics import log_loss
logloss_overall = log_loss(y_true=y, y_pred=y_hat)
print('overall log loss is', logloss_overall)
#embed()
