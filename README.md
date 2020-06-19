##Stochastic Segmentation Networks


### BraTS 2017
To download the data go to  ```https://www.med.upenn.edu/sbia/brats2017/registration.html``` and follow the instructions provided by the challenge's organisers. 
We preprocess the data using bias-field correction, histogram matching and normalisation, and we computed binary brain masks.
The preprocessing script will be included in this repository in the future.
**Important:** Binary brain masks are necessary for two reasons:
1) The model is training on patches, and so we need to know where the brain is to sample patches inside the brain and avoid sampling patches contained only air.
2) A limitation of our method is that is it is numerically unstable in areas with infinite covariance such as the air outside the brain.
To avoid exploding covariance, we mask out the outside of the brain when computing the covariance matrix.

After downloading and preprocessing the data you can use the data splits we used in the paper which are provided in the folder 
```data/BraTS2017```. Make sure you use the same suffixes we have. Set the ```path``` variable to the path of your data and run to following commands to replace
the path in the files with your path:

    path=/vol/vipdata/data/brain/brats/2017_kostas/preprocessed_v2/Brats17TrainingData
    sed -i "s~<path>~$path~g" data/BraTS2017/data_index_train.csv
    sed -i "s~<path>~$path~g" data/BraTS2017/data_index_valid.csv
    sed -i "s~<path>~$path~g" data/BraTS2017/data_index_test.csv


To train the stochastic segmentation networks run:

    python ssn/train.py --job-dir jobs/rank_10_mc_20_patch_110/train \
    --config-file data/config_files/rank_10_mc_20_patch_110.json \
    --train-csv-path data/BraTS2017/data_index_train.csv \
    --valid-csv-path data/BraTS2017/data_index_valid.csv \
    --num-epochs 1200 \
    --device 0 \
    --random-seeds "1"

You can change the ```job-dir``` to whatever you like and use the different config files provided to run the baseline and larger model described in the paper.
As described in the paper, we found no benefit in using the model with the larger patch for this application. 
Since it takes significantly longer to train it, you can likely skip that.
Change the ```device``` variable to the index of the gpu you wish to use or to ```"cpu"```.
    
**Important:** do not use the numbers printed in the console during training for evaluation. 
They are computed on patches, not on full images.

For inference run:

    python ssn/inference.py --job-dir jobs/rank_10_mc_20_patch_110/test \
    --config-file data/config_files/rank_10_mc_20_patch_110.json \
    --test-csv-path data/BraTS2017/data_index_test.csv \
    --device 0 \
    --saved-model-paths "data/saved_models/rank_10_mc_20_patch_110.pt"

You will need to change the ```config-file``` and ```saved-model-paths``` according to the model being used.
You can find the trained models used in our paper in the folder ```saved_models```.
The inference script will output the mean of the distribution as the prediction as well as the covariance diagonal and covariance factor.
To generate samples and evaluate your predictions, you will need to use the evaluation script. 
You will need to edit lines 18-21 of this script to include your own paths to experiments.

    python evaluation/evalute.py

You can pass options ```--detailed True``` to include extra evaluation (takes longer) and ```--make--thumbs True``` to generate png thumbs.

### Toy problem

asdfa

### LIDC 2D