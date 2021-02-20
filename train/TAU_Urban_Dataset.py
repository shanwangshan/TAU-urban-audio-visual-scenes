
import torch
from torch.utils import data
import numpy as np
import os
import pandas
import h5py
from IPython import embed
'''
This script is part of the train.py, it defines the dataset as a class
'''
class TAU_Urban(data.Dataset):

    'Characterizes a dataset for PyTorch'
    def __init__(self,data, path_features,model_type):
        super(TAU_Urban, self).__init__()
        self.data = data 
        self.path_features = path_features 
        self.model_type = model_type

        if self.model_type == 'audio':
            if self.data == 'tr':
                self.path_input = self.path_features +'audio_features_data/tr.hdf5'
                
            if self.data == 'val':
                self.path_input =  self.path_features+'audio_features_data/val.hdf5'
               
            global_mean_std_path = self.path_features+'audio_features_data/global_mean_std.npz'
            mean_std = np.load(global_mean_std_path)
            self.mean = mean_std['global_mean']
            self.std = mean_std['global_std']
      
        if self.model_type == 'video':
            if self.data == 'tr':
                self.path_input = self.path_features+'video_features_data/tr.hdf5'
            if self.data == 'val':
                self.path_input = self.path_features+'video_features_data/val.hdf5'

            global_mean_std_path = self.path_features+'video_features_data/global_mean_std.npz'
            mean_std = np.load(global_mean_std_path)
            self.mean = mean_std['global_mean']
            self.std = mean_std['global_std']

        if self.model_type == 'audio_video':
            if self.data == 'tr':
                self.path_input = self.path_features+'audio_features_data/tr.hdf5'
                
            if self.data == 'val':
                self.path_input = self.path_features+'audio_features_data/val.hdf5'

            global_mean_std_path_audio =self.path_features+ 'audio_features_data/global_mean_std.npz'
            mean_std_audio = np.load(global_mean_std_path_audio)
            self.mean_audio = mean_std_audio['global_mean']
            self.std_audio = mean_std_audio['global_std']

            global_mean_std_path_video = self.path_features+'video_features_data/global_mean_std.npz'
            mean_std_video = np.load(global_mean_std_path_video)
            self.mean_video = mean_std_video['global_mean']
            self.std_video = mean_std_video['global_std']
            
        self.all_files = []
        self.group = []
        def func(name, obj):     
            if isinstance(obj, h5py.Dataset):
                self.all_files.append(name)
            elif isinstance(obj, h5py.Group):
                self.group.append(name)
        self.hf = h5py.File(self.path_input, 'r')
        self.hf.visititems(func)
        self.hf.close()

    print('loading is done')
    def __len__(self):
            'Denotes the total number of samples'
            return len(self.all_files)
    
    def __getitem__(self,index):
        if self.model_type == 'audio_video':
            #print('using', self.model_type)

            path_input_audio = self.path_input
            path_input_video = self.path_input.replace('audio','video')

            hf_audio = h5py.File(path_input_audio, 'r')
            hf_video = h5py.File(path_input_video, 'r')

            emb_audio = np.array(hf_audio[self.all_files[index]])
            normed_embed_audio=  (emb_audio-self.mean_audio)/self.std_audio
            
            video_name = self.all_files[index].replace('audio','video')
           
            emb_video= np.array(hf_video[video_name])

            normed_embed_video =  (emb_video-self.mean_video)/self.std_video
            normed_embed = np.concatenate((normed_embed_audio,normed_embed_video))

            #normed_embed = (normed_embed-np.mean(normed_embed))/np.std(normed_embed)
            ground_tr = np.array(int(self.all_files[index].split('/')[0]))
            normed_embed_tensor = torch.from_numpy(normed_embed).float()
            ground_tr_tensor=torch.from_numpy(ground_tr).long()
                
            return normed_embed_tensor,ground_tr_tensor

        if self.model_type == 'audio' or self.model_type == 'video':
            #print('using', self.model_type)
            #print(self.mean,self.std)
            hf = h5py.File(self.path_input, 'r')

            emb = np.array(hf[self.all_files[index]])
            
            normed_embed =  (emb-self.mean)/self.std
            
            ground_tr = np.array(int(self.all_files[index].split('/')[0]))
            #print(self.all_files[index],ground_tr)
            normed_embed_tensor = torch.from_numpy(normed_embed).float()
            ground_tr_tensor=torch.from_numpy(ground_tr).long()         
            return normed_embed_tensor,ground_tr_tensor

      
