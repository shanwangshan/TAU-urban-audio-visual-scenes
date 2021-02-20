import os
import random
import pandas
from IPython import embed
import argparse
'''
This script is to split the fold1_train.csv (8646 files) into training and validation csv files with the target ratio.
More importantly, it promises that the training files have different location ID from the validation files.
This is because the files with same location ID are actually the same recordings (~5 minutes).
We split the 5 mimutes recording into several 10s segemants.
To avoid overfitting, files with same location ID can only be seen in either training or validation.
To run this script,

python split_data.py --input_path '/path/to/file' --output_path '/path/to/save/' --ratio 0.1

'''
parser = argparse.ArgumentParser(description='Generating training and validation csv files')
parser.add_argument('--input_path', type=str,
                    help='give the file path of fold1_train.csv file')
parser.add_argument('--output_path', type=str,
                    help='give the folder path of where you want to save training csv and validation csv files')
parser.add_argument('--ratio', type=float, default = 0.1,
                    help='the ratio of the validation data e.g. 0.1 or 0.3')
args, _ = parser.parse_known_args()

#### fix the seed so everytime the files you generate for the training and validation are the same####
random.seed(30)

#### set the data path#######
path_csv = args.input_path

#### read the input csv file######
df = pandas.read_csv(path_csv, sep="\t")
#### insert one extra column indicating the location_id ######
location_id = []
for row in df.iterrows():
    location_id.append(row[1][1].split('-')[2])
df['location_id'] = location_id

########## 10 lists of dataframes #
l_scene_label = [x for _,x in df.groupby('scene_label')]

#####create two empty dataframes with the same head as df#####
column_names = ["filename_audio", "filename_video", "scene_label","location_id"]
df_train = pandas.DataFrame(columns = column_names)
df_val = pandas.DataFrame(columns = column_names)

for i in range(len(l_scene_label)):
    l_location_id= [x for _,x in l_scene_label[i].groupby('location_id')]
    count = 0
    index_ids = []
    a = list(range(len(l_location_id)))
    while True:
        b = random.choice(a)
        index_ids.append(b)
        a.remove(b)
        count+=len(l_location_id[b])
        df_train=df_train.append(l_location_id[b])
        if count >= int((1-args.ratio)* len(l_scene_label[i])):
            print('enough files for training')
            break
    for j in range(len(a)):
        df_val = df_val.append(l_location_id[a[j]])

save_data = args.output_path
if not os.path.exists(save_data):
    os.makedirs(save_data)
    print("Directory " , save_data ,  " Created ")
else:
    print("Directory " , save_data ,  " already exists")

df_train.to_csv(save_data+"train.csv",index=False)
df_val.to_csv(save_data+"val.csv",index=False)
print('training csv files is save',save_data+ "train.csv")
print('validation csv files is save',save_data+ "val.csv")
#embed()
