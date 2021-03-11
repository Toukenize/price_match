# Shopee Price Match

## Environment setup

Install environment with `conda install env.yml`

Alternative, if it doesn't work due to incompatible `cudatoolkit`, you can edit the `env.yml` by specifying a compatible `cudatoolkit` (more info [here](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)).

> Utility function to export .yml : `conda env export | grep -v "^prefix: " > env.yml`

## Test Your Environment

Normally, only GPU dependent modules are problematic.

### Test `Tensorflow` installation:

1. Activate the environment `conda activate price_match`
2. Bring up python shell `python`
3. Import and check `import tensorflow as tf ; tf.test.is_gpu_available() # Should return True`
4. If you receive error similar to this `Could not load dynamic library 'libcudart.so.11.0'`, then you need to set your environment variable to point to the the correct folder that contains the library (likely in `/home/<your_username>/anaconda3/envs/price_match/lib/`). If you only want this variable to be set when your conda env is active, follow this guide [here](https://stackoverflow.com/questions/46826497/conda-set-ld-library-path-for-env-only)



### Test `Pytorch` installation:

1. Step 1 and 2 of test `tensorflow` installation
2. Import and check `import torch ; torch.cuda.get_device_name() # Should return your NVIDA GPU name`
3. Pytorch no problem one

