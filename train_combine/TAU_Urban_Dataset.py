
import torch
from torch.utils import data
import numpy as np
import os
import pandas
import h5py
from IPython import embed
class TAU_Urban(data.Dataset):

    'Characterizes a dataset for PyTorch'
    def __init__(self, data, features_path):
        super(TAU_Urban, self).__init__()
        self.data = data 
        self.features_path = features_path
        
        if self.data == 'tr':
            self.path_input = self.features_path+'audio_features_data/tr.hdf5'
            
        if self.data == 'val':
            self.path_input = self.features_path+'audio_features_data/val.hdf5'
            #self.path_input = '/lustre/wang9/all_features_data/audio_features_data/cv.hdf5'

        global_mean_std_path_audio = self.features_path+'audio_features_data/global_mean_std.npz'
        mean_std_audio = np.load(global_mean_std_path_audio)
        self.mean_audio = mean_std_audio['global_mean']
        self.std_audio = mean_std_audio['global_std']

        global_mean_std_path_video = self.features_path+'video_features_data/global_mean_std.npz'
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
        
            #print('using', self.model_type)

        path_input_audio = self.path_input
        path_input_video = self.path_input.replace('audio','video')

        hf_audio = h5py.File(path_input_audio, 'r')
        hf_video = h5py.File(path_input_video, 'r')

        emb_audio = np.array(hf_audio[self.all_files[index]])
        normed_embed_audio =  (emb_audio-self.mean_audio)/self.std_audio

        video_name = self.all_files[index].replace('audio','video')
        
        #print(video_name)
        emb_video= np.array(hf_video[video_name])
        normed_embed_video =  (emb_video-self.mean_video)/self.std_video
        
        ground_tr = np.array(int(self.all_files[index].split('/')[0]))
        normed_embed_audio_tensor = torch.from_numpy(normed_embed_audio).float()
        normed_embed_video_tensor = torch.from_numpy(normed_embed_video).float()

        ground_tr_tensor=torch.from_numpy(ground_tr).long()
            
        return normed_embed_audio_tensor,normed_embed_video_tensor,ground_tr_tensor
    