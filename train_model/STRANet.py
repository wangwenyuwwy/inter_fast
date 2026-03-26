import warnings

warnings.filterwarnings(
    "ignore",
    message="Importing from timm.models.layers is deprecated, please import via timm.layers",
    category=FutureWarning,
)

import argparse
import os
from solver import Solver

os.environ['CUDA_VISIBLE_DEVICE'] = '0'


def main(config):
    config.model_path = os.path.abspath(config.model_path)
    config.label_root = os.path.abspath(config.label_root)
    config.crop_root = os.path.abspath(config.crop_root)
    config.num_epochs_decay = max(1, int(config.num_epochs * 0.2))

    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)

    solver = Solver(config, [0])
    solver.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--num_epochs_decay', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train_budget', type=int, default=150000)
    parser.add_argument('--validation_interval', type=int, default=2500)
    parser.add_argument('--max_validation_batches', type=int, default=0)
    parser.add_argument('--label_root', type=str, default='../gen_dataset')
    parser.add_argument('--crop_root', type=str, default='../dataset/train_set')
    parser.add_argument('--model_path', type=str, default='./trained_models/EnhancedFeatureExtractor_DSC_Net_retrain')
    parser.add_argument('--cuSize', type=int, default=0, help='cuSize=0 32x32, cuSize=1 16x16, cuSize=2 16x32 cuSize=3 8x32 cuSize=4 8x16 cuSize=5 chroma')
    config = parser.parse_args()
    main(config)