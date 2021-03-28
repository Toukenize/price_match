# Shopee Price Match

## 1. Environment setup

1. Install environment with `conda install env.yml`

   Alternative, if it doesn't work due to incompatible `cudatoolkit`, you can edit the `env.yml` by specifying a compatible `cudatoolkit` (more info [here](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)).

   > Utility function to export .yml : `conda env export | grep -v "^prefix: " > env.yml`

### If you are going to use `jupyter notebook`

1. Activate your environment `conda activate price_match_env` 
2. Install the kernel `python -m ipykernel install --user --display_name "price_match_env"`
3. Spin off your jupyter notebook as usual `jupyter notebook`

## 2. Test Your Environment

Normally, only GPU-dependent modules are problematic.

### Test `Tensorflow` installation:

1. Activate the environment `conda activate price_match`
2. Bring up python shell `python`
3. Import and check `import tensorflow as tf ; tf.test.is_gpu_available() # Should return True`
4. If you receive error similar to this `Could not load dynamic library 'libcudart.so.11.0'`, then you need to set your environment variable to point to the the correct folder that contains the library (likely in `/home/<your_username>/anaconda3/envs/price_match/lib/`). If you only want this variable to be set when your conda env is active, follow this guide [here](https://stackoverflow.com/questions/46826497/conda-set-ld-library-path-for-env-only)



### Test `Pytorch` installation:

1. Step 1 and 2 of test `tensorflow` installation
2. Import and check `import torch ; torch.cuda.get_device_name() # Should return your NVIDA GPU name`

### Test `Xgboost` installation:

1. Step 1 and 2 of test `tensorflow` installation

2. Since there is no utility function that helps to check if gpu is available for `xgboost`, need to write some sample code to test:
   
   ```
   import numpy as np
   import xgboost as xgb
   
   n = 10_000
   m = 100
   X = np.random.randn(n, m)
   y = np.random.randn(n)
   exp_models = []
   
   for i in range(3):
   
       # As long as this runs with no problem, gpu support should be ok
       clf = xgb.XGBRegressor(
           tree_method='gpu_hist', eta=0.1, max_depth=6, verbosity=0)
       trained_model = clf.fit(X, y, verbose=False)
   ```
   
   
