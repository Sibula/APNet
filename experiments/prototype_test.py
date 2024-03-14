import sys
import os
import argparse
import numpy as np
import librosa
import json
import soundfile as sf

# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots

from dcase_models.util.gui import encode_audio
from dcase_models.data.features import MelSpectrogram
from dcase_models.data.datasets import UrbanSound8k, TAUUrbanAcousticScenes2019

from dcase_models.util.files import load_json, mkdir_if_not_exists, save_pickle, load_pickle
from dcase_models.util.data import evaluation_setup
from dcase_models.data.data_generator import DataGenerator

sys.path.append('../')
from apnet.gui import generate_figure2D 
from apnet.gui import generate_figure_weights
from apnet.gui import generate_figure_mel
from apnet.model import APNet
from apnet.layers import PrototypeLayer, WeightedSum
from apnet.datasets import MedleySolosDb, GoogleSpeechCommands

from tensorflow import get_default_graph

available_models = {
    'APNet' :  APNet,
}

available_features = {
    'MelSpectrogram' :  MelSpectrogram,
}

available_datasets = {
    'UrbanSound8k' :  UrbanSound8k,
    'MedleySolosDb' : MedleySolosDb,
    'GoogleSpeechCommands' : GoogleSpeechCommands,
    'TAUUrbanAcousticScenes2019' : TAUUrbanAcousticScenes2019
}

# Define app
graph = get_default_graph()

# Generate layout

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    '-d', '--dataset', type=str,
    help='dataset name (e.g. UrbanSound8k, ESC50, URBAN_SED, SONYC_UST)',
    default='TAUUrbanAcousticScenes2019'
)
parser.add_argument(
    '-f', '--features', type=str,
    help='features name (e.g. Spectrogram, MelSpectrogram, Openl3)',
    default='MelSpectrogram'
)
parser.add_argument(
    '-mp', '--models_path', type=str,
    help='path to load the trained model',
    default='./'
)
parser.add_argument(
    '-dp', '--dataset_path', type=str,
    help='path to load the dataset',
    default='./'
)
parser.add_argument(
    '-m', '--model', type=str,
    help='model name (e.g. MLP, SB_CNN, SB_CNN_SED, A_CRNN, VGGish)')

parser.add_argument('-fold', '--fold_name', type=str, help='fold name',
                    default='train')

parser.add_argument('--o', dest='overwrite', action='store_true')
parser.set_defaults(overwrite=False)

parser.add_argument('-gpu', '--gpu_visible', type=str, help='gpu_visible',
                    default='0')

parser.add_argument('--force', dest='force_prototpye_calculation', action='store_true')
parser.set_defaults(force_prototpye_calculation=False)

parser.add_argument('--wo_audio', dest='get_audio_prototypes', action='store_false')
parser.set_defaults(get_audio_prototypes=True)


parser.add_argument('--points', dest='num_points', type=int, default=10)
parser.add_argument('--PCA1', dest='PCA1', type=int, default=0)
parser.add_argument('--PCA2', dest='PCA2', type=int, default=1)


args = parser.parse_args()

# only use one GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_visible

dataset_name = args.dataset
model_name = args.model
features_name = args.features

# Model paths
model_input_folder = os.path.join(args.models_path, args.dataset, args.model)

# Get parameters
parameters_file = os.path.join(model_input_folder, 'config.json')
params = load_json(parameters_file)

params_dataset = params['datasets'][dataset_name]
params_features = params['features']


model_containers = {}
fold_name = args.fold_name

exp_folder_fold = os.path.join(model_input_folder, fold_name)

if args.overwrite:
    exp_folder_output = model_input_folder
else:
    exp_folder_output = os.path.join(model_input_folder, 'refine_manual')

exp_folder_output_fold = os.path.join(exp_folder_output, fold_name)
mkdir_if_not_exists(exp_folder_output_fold, parents=True)
print(args.overwrite, exp_folder_output_fold)

dataset_class = available_datasets[dataset_name]
dataset_path = os.path.join(args.dataset_path, params_dataset['dataset_path'])
dataset = dataset_class(dataset_path)

# Get and init feature class
features_class = available_features[features_name]
features = features_class(**params_features[features_name])

features.extract(dataset)

kwargs = {'custom_objects': {'PrototypeLayer': PrototypeLayer, 'WeightedSum': WeightedSum}}
with graph.as_default():
    model_containers[fold_name] = APNet(
        model=None, model_path=exp_folder_fold, metrics=['classification'],
        **kwargs, **params['models']['APNet']['model_arguments']
    )
    model_containers[fold_name].load_model_weights(exp_folder_fold)

scaler_path = os.path.join(exp_folder_fold, 'scaler.pickle') 
scaler = load_pickle(scaler_path)

scaler_path = os.path.join(exp_folder_output_fold, 'scaler.pickle') 
save_pickle(scaler, scaler_path)

folds_train, folds_val, _ = evaluation_setup(
    fold_name, dataset.fold_list,
    params_dataset['evaluation_mode'],
    use_validate_set=True
)

data_gen_train = DataGenerator(
    dataset, features, folds=folds_train,
    batch_size=32,
    shuffle=True, train=True, scaler=scaler
)

if args.dataset == 'MedleySolosDb':
    data_gen_train.audio_file_list = data_gen_train.audio_file_list[:int(len( data_gen_train.audio_file_list)/3)]
if args.dataset ==  'GoogleSpeechCommands':
    data_gen_train.audio_file_list = data_gen_train.audio_file_list[:int(len( data_gen_train.audio_file_list)/10)]
#for j in range(len(data_gen_train)):
X_train, Y_train = data_gen_train.get_data()

file_list = []
for file_dict in data_gen_train.audio_file_list:
    file_list.append(file_dict['file_original'])

# Take first sequence of each file
#for j in range(len(X_train)):
#    X_train[j] = X_train[j][0]
#    Y_train[j] = Y_train[j][0]

#X_train = np.asarray(X_train)
#Y_train = np.asarray(Y_train)

model_input_to_embeddings = model_containers[fold_name].model_input_to_embeddings()

X_feat = model_input_to_embeddings.predict(X_train)[0]

print(X_train.shape, Y_train.shape, X_feat.shape, len(file_list))

data_instances_path = os.path.join(exp_folder_fold, 'data_instances.pickle')
prototypes_path = os.path.join(exp_folder_fold, 'prototypes.pickle')

projection2D = None
if os.path.exists(prototypes_path):
    model_containers[fold_name].prototypes = load_pickle(prototypes_path)
    projection2D = model_containers[fold_name].prototypes.projection2D

model_containers[fold_name].get_data_instances(X_feat, X_train, Y_train, file_list, projection2D=projection2D)

    # model_containers[fold_name].data_instances = load_pickle(data_instances_path)

if (not os.path.exists(prototypes_path)) or (args.force_prototpye_calculation):
    convert_audio_params = None
    if args.get_audio_prototypes:
        convert_audio_params = {
            'sr': params_features[args.features]['sr'],
            'scaler': scaler,
            'mel_basis': features.mel_basis,
            'audio_hop': params_features[args.features]['audio_hop'],
            'audio_win': params_features[args.features]['audio_win']
        }

    model_containers[fold_name].get_prototypes(
        X_train,
        projection2D=model_containers[fold_name].data_instances.projection2D,
        convert_audio_params=convert_audio_params
    )
    save_pickle(model_containers[fold_name].prototypes, prototypes_path)
    # save_pickle(model_containers[fold_name].data_instances, data_instances_path)

label_list = dataset.label_list.copy()
for j in range(len(label_list)):
    label_list[j] = label_list[j].replace('_', ' ')

print(label_list)

figure2D = generate_figure2D(
    model_containers[fold_name], x_select=args.PCA1, y_select=args.PCA2,
    samples_per_class=args.num_points, label_list=label_list
)

fig_weights = generate_figure_weights(model_containers[fold_name], label_list=label_list)

# _,center_mel_blank,_,_,_,_  = model_containers[fold_name].data_instances.get_instance_by_index(0)
# fig_mel = generate_figure_mel(center_mel_blank)

figure2D.write_html('./prototypes/figure2D.html')
fig_weights.write_html('./prototypes/fig_weights.html')
# fig_mel.write_html('fig_mel.html')

prototypes_feat, prototypes_mel, protoypes2D, prototypes_classes, prototypes_audios = model_containers[fold_name].prototypes.get_all_instances()

'''
melspec = prototypes_mel[0,:,:]
generate_figure_mel(melspec).write_image('melspec.pdf')
audio = librosa.feature.inverse.mel_to_audio(
    melspec, sr=22050, n_fft=2048, hop_length=2048, win_length=2048, 
    pad_mode='constant', power=2.0, htk=True, fmax=None
)
with open('audio.txt', 'a') as f:
    for val in audio:
        f.write(f'{repr(val)}\n')
sf.write('audio.wav', audio/np.amax(audio), 22050)
'''

for i in range(len(prototypes_classes)):
    fpath = f'./prototypes/{i:02d}_{label_list[prototypes_classes[i]]}'
    image = generate_figure_mel(prototypes_mel[i])
    image.write_image(f'{fpath}.pdf')
    audio = prototypes_audios[i]['data']
    sr = prototypes_audios[i]['sr']
    sf.write(f'{fpath}.wav', audio/np.amax(audio), sr)
