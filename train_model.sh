# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate price_match_env

# Train NLP model
python train.py --val false --model_type nlp

# Train IMG model
python train.py --cv_type group --model_type img --trainfolds 0 1 2 3