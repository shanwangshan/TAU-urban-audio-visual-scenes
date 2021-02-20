import openl3
import soundfile as sf
import os
import pandas
import numpy as np
import h5py
from IPython import embed
from tqdm import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip
import argparse
'''
This script is to generate audio and video features from L3 pretrained network for validation data.
For details about the L3 net, please refer to https://openl3.readthedocs.io/en/latest/index.html

To run this script,
python create_tr.py --input_path '/path/to/file' --dataset_path '/path/of/TAU-urban-audio-visual-scenes-2021-development/'
--output_path '/path/to/save/' --hop_size 0.1 --content_type 'env' --input_repr 'mel256' --embedding_size 512

'''
parser = argparse.ArgumentParser(description='Generating audio and video features from L3 network for validation data')
parser.add_argument('--input_path', type=str,
                    help='give the file path of val.csv file you generated from split_data.py')
parser.add_argument('--dataset_path', type=str,
                    help='give the path of TAU-urban-audio-visual-scenes-2021-development data')
parser.add_argument('--output_path', type=str,
                    help='give the folder path of where you want to save audio and video features')
parser.add_argument('--hop_size', type=float,default = 0.1,
                    help='required by L3 net, hop size needs to be defined. e.g. 0.1 or 0.5')
parser.add_argument('--content_type', type=str,default = 'env',
                    help='required by L3 net, e.g. env or music')
parser.add_argument('--input_repr', type=str,default = 'mel256',
                    help='required by L3 net, e.g. linear, mel128, or mel256')
parser.add_argument('--embedding_size', type=int,default = 512,
                    help='required by L3 net')
args, _ = parser.parse_known_args()

#### digitalize the target according to the alphabet order#####
classes_dic = {'airport': 0,
                'bus': 1,
                'metro': 2,
                'metro_station': 3,
                'park': 4,
                'public_square': 5,
                'shopping_mall': 6,
                'street_pedestrian': 7,
                'street_traffic': 8,
                'tram': 9}

#### set the data path#######
path_csv_val =args.input_path

#### load the validation data using pandas######
df_val = pandas.read_csv(path_csv_val, sep = ",")

input_dir_audio_val = df_val['filename_audio'].values
input_dir_audio_val = list(input_dir_audio_val)
input_dir_audio_val.sort()

#embed()
####### load the model to extract the audio and video embeddings#####
model_audio = openl3.models.load_audio_embedding_model(content_type=args.content_type,
input_repr=args.input_repr, embedding_size=args.embedding_size)

model_video = openl3.models.load_image_embedding_model(content_type=args.content_type,
input_repr=args.input_repr, embedding_size=args.embedding_size)


save_data_audio =args.output_path+'audio_features_data/'
if not os.path.exists(save_data_audio):
    os.makedirs(save_data_audio)
    print("Directory " , save_data_audio ,  " Created ")
else:
    print("Directory " , save_data_audio ,  " already exists")

save_data_video =args.output_path+'video_features_data/'
if not os.path.exists(save_data_video):
    os.makedirs(save_data_video)
    print("Directory " , save_data_video ,  " Created ")
else:
    print("Directory " , save_data_video ,  " already exists")

################create validation features #########################

hf_val_audio = h5py.File(save_data_audio+'val.hdf5', 'a')
hf_val_video = h5py.File(save_data_video+'val.hdf5', 'a')

print('generating validation features data ...')
for i in tqdm(range(len(input_dir_audio_val))):

    audio_name = args.dataset_path+input_dir_audio_val[i]
    #embed()
    label = audio_name.split('/')[-1].split('-')[0]
    label = classes_dic[label]

    audio, sr = sf.read(audio_name)
    emb_audio, ts = openl3.get_audio_embedding(audio, sr,hop_size=args.hop_size,model=model_audio)

    grp_audio = hf_val_audio.create_group(str(label)+'/'+input_dir_audio_val[i].replace('.wav',''))
    #print(grp)
    for j in range(emb_audio.shape[0]):
        each_emb_audio = emb_audio[j,:]
        dset_audio = grp_audio.create_dataset(str(j), data=each_emb_audio)

    #embed()

    video_name = input_dir_audio_val[i].replace('audio','video')
    video_name = video_name.replace('.wav','.mp4')
    video_name = args.dataset_path + video_name

    #embed()
    clip = VideoFileClip(video_name,audio=False)
    images = []
    for t,frame in clip.iter_frames(with_times= True):
        images.append(frame)

    index = np.linspace(0,len(images)-1,len(ts))
    index = index.astype(int)

    images = [images[i] for i in list(index)]
    images = np.array(images)
    #embed()
    emb_video = openl3.get_image_embedding(images, model=model_video)

    grp_video = hf_val_video.create_group(str(label)+'/'+input_dir_audio_val[i].replace('.wav','').replace('audio','video'))

    for j in range(emb_video.shape[0]):
        each_emb_video = emb_video[j,:]
        dset_video = grp_video.create_dataset(str(j), data=each_emb_video)
    #embed()
hf_val_audio.close()
hf_val_video.close()
################create validation features#########################
