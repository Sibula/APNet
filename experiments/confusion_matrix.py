import os
import argparse
import sys
import numpy as np

sys.path.append('../')

# Datasets
from apnet.datasets import MedleySolosDb, GoogleSpeechCommands
from dcase_models.data.datasets import UrbanSound8k, TAUUrbanAcousticScenes2019

# Models
from dcase_models.model.models import SB_CNN, MLP
from apnet.model import APNet, AttRNNSpeechModel

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from dcase_models.util.files import load_json, load_pickle


available_models = {
    'APNet' :  APNet,
    'SB_CNN' : SB_CNN,
    'MLP' : MLP,
    'AttRNNSpeechModel' : AttRNNSpeechModel
}

available_datasets = {
    'UrbanSound8k' :  UrbanSound8k,
    'MedleySolosDb' : MedleySolosDb,
    'GoogleSpeechCommands' : GoogleSpeechCommands,
    'TAUUrbanAcousticScenes2019' : TAUUrbanAcousticScenes2019
}

def main():
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
        '-m', '--model', type=str,
        help='model name (e.g. APNet, MLP, SB_CNN ...)',
        default='APNet'
    )
    parser.add_argument(
        '-fold', '--fold_name', type=str, 
        help='fold name',
        default='test'
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
        '-gpu', '--gpu_visible', type=str, help='gpu_visible',
        default='0')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_visible
    print(__doc__)

    if args.dataset not in available_datasets:
        raise AttributeError('Dataset not available')

    model_name = args.model
    if args.model not in available_models:
        base_model = args.model.split('/')[0]
        if base_model not in available_models:
            raise AttributeError('Model not available')
        else:
            model_name = base_model

    model_folder = os.path.join(args.models_path, args.dataset, args.model)

    parameters_file = os.path.join(model_folder, 'config.json')
    params = load_json(parameters_file)

    params_dataset = params['datasets'][args.dataset]
    dataset_class = available_datasets[args.dataset]
    dataset_path = os.path.join(args.dataset_path, params_dataset['dataset_path'])
    dataset = dataset_class(dataset_path)

    model_folder = os.path.join(args.models_path, args.dataset, args.model)
    exp_folder = os.path.join(model_folder, args.fold_name)
    results_file = os.path.join(exp_folder, 'results.pickle')
    results = load_pickle(results_file)

    annotations = [np.argmax(x) for x in results['annotations']]
    predictions = [np.argmax(x) for x in results['predictions']]

    cm = confusion_matrix(annotations, predictions)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=dataset.label_list
    )
    disp.plot(xticks_rotation='vertical')
    plt.savefig('confusion_matrix.png')

    # print(annotations)
    # print(predictions)
    # print(dataset.label_list)

if __name__ == '__main__':
    main()
