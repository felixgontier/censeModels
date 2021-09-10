# censeModels
 todo
## Setup

    python3.7 -m venv env/
    source env/bin/activate
    pip install -r requirements.txt

## Eval fast

Ensure Lorient-1k .npy files are in ./data/
See code for options

    python inference.py --exp TVBCense_Fast

## Eval slow

    python fast_to_slow.py
    python inference.py --exp TVBCense_Slow --dataset Lorient-1k_slow
