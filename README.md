# censeModels
 todo
## Setup

    python3.7 -m venv env/
    source env/bin/activate
    pip install -r requirements.txt
    
## Download sample dataset
    python download.py

## Eval fast

Ensure Lorient-1k.npy files are in ./data/
See code for options

    python inference.py --exp TVBCense_Fast

## Eval slow

    python fast_to_slow.py
    python inference.py --exp TVBCense_Slow --dataset Lorient-1k_slow

## Run without evaluation

    python inference.py --exp TVBCense_Fast --dataset Lorient-1k -no_metrics -force_recompute
    python inference.py --exp TVBCense_Slow --dataset Lorient-1k_slow -no_metrics -force_recompute

Each produces 2 files located in the eval_output directory: \_presence.npy with boolean values indicating presence of a given source and \_scores.npy floating point values with likelihood of presence
