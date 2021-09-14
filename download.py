import argparse
import os

def main(config):

  inference = [
  'wget -O data/Lorient-1k_spectralData.npy https://zenodo.org/record/4687057/files/Lorient-1k_spectralData.npy?download=1',
  'wget -O data/Lorient-1k_melSpectrograms.npy https://zenodo.org/record/5153616/files/Lorient-1k_melSpectrograms.npy?download=1',
  'wget -O data/Lorient-1k_presence.npy https://zenodo.org/record/4687057/files/Lorient-1k_presence.py?download=1',
  'wget -O data/Lorient-1k_time_of_presence.npy https://zenodo.org/record/4687057/files/Lorient-1k_time_of_presence.py?download=1'
  ]

  if config.task == 'inference':
    datasets = inference
  
  for dataset in datasets:
    os.system(dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='inference', help='Download the datasets for the task: evaluation, training, pretext')
    config = parser.parse_args()

    main(config)
