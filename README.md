## Stochastic Segmentation Networks

[![arXiv](http://img.shields.io/badge/arXiv-2006.06015-B31B1B.svg)](https://arxiv.org/abs/2006.06015)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MiguelMonteiro/stochastic_segmentation_networks_demo/blob/master/ssn_demo.ipynb)

[\[Paper\]](https://arxiv.org/abs/2006.06015)
[\[Interactive Demo\]](https://colab.research.google.com/github/MiguelMonteiro/stochastic_segmentation_networks_demo/blob/master/ssn_demo.ipynb)

![Figure from paper](assets/images/image_1.png)


This repository contains the code for the paper:
> Monteiro, M., Folgoc, L., Castro, D.C., Pawlowski, N., Marques, B., Kamnitsas, K., van der Wilk, M. and Glocker, B., _Stochastic Segmentation Networks: Modelling Spatially Correlated Aleatoric Uncertainty_, 2020 [[arXiv]](https://arxiv.org/abs/2006.06015)


If you use these tools in your publications, please consider citing our paper:
```
@article{monteiro2020stochastic,
    title={Stochastic Segmentation Networks: Modelling Spatially Correlated Aleatoric Uncertainty},
    author={Miguel Monteiro and Loïc Le Folgoc and Daniel Coelho de Castro and Nick Pawlowski and Bernardo Marques and Konstantinos Kamnitsas and Mark van der Wilk and Ben Glocker},
    year={2020},
    eprint = {arXiv:2006.06015}
}
```

### Requirements
Install the necessary requirements:

    pip install requirements.txt


### BraTS 2017
To download the data go to  `https://www.med.upenn.edu/sbia/brats2017/registration.html` and follow the instructions provided by the challenge's organisers. 
We preprocess the data using bias-field correction, histogram matching and normalisation, and we computed binary brain masks.
The preprocessing script will be included in this repository in the future.
**Important:** Binary brain masks are necessary for two reasons:
1) The model is training on patches, and so we need to know where the brain is to sample patches inside the brain and avoid sampling patches contained only air.
2) A limitation of our method is that is it is numerically unstable in areas with infinite covariance such as the air outside the brain.
To avoid exploding covariance, we mask out the outside of the brain when computing the covariance matrix.

After downloading and preprocessing the data you can use the data splits we used in the paper which are provided in the folder 
`assets/BraTS2017_data`. Make sure you use the same suffixes we have. Set the `path` variable to the path of your data and run to following commands to replace
the path in the files with your path:

    path=/vol/vipdata/data/brain/brats/2017_kostas/preprocessed_v2/Brats17TrainingData
    sed -i "s~<path>~$path~g" assets/BraTS2017_data/data_index_train.csv
    sed -i "s~<path>~$path~g" assets/BraTS2017_data/data_index_valid.csv
    sed -i "s~<path>~$path~g" assets/BraTS2017_data/data_index_test.csv

#### Training
To train the stochastic segmentation networks run:

    python ssn/train.py --job-dir jobs/rank_10_mc_20_patch_110/train \
    --config-file assets/config_files/rank_10_mc_20_patch_110.json \
    --train-csv-path assets/BraTS2017_data/data_index_train.csv \
    --valid-csv-path assets/BraTS2017_data/data_index_valid.csv \
    --num-epochs 1200 \
    --device 0 \
    --random-seeds "1"

You can change the `job-dir` to whatever you like and use the different config files provided to run the baseline and larger model described in the paper.
As described in the paper, we found no benefit in using the model with the larger patch for this application. 
Since it takes significantly longer to train it, you can likely skip that.
Change the `device` variable to the index of the gpu you wish to use or to `"cpu"`.
    
**Important:** do not use the numbers printed in the console during training for evaluation. 
They are computed on patches, not on full images.

#### Inference
For inference run:

    python ssn/inference.py --job-dir jobs/rank_10_mc_20_patch_110/test \
    --config-file assets/config_files/rank_10_mc_20_patch_110.json \
    --test-csv-path assets/BraTS2017_data/data_index_test.csv \
    --device 0 \
    --saved-model-paths "assets/saved_models/rank_10_mc_20_patch_110.pt"

You will need to change the `config-file` and `saved-model-paths` according to the model being used.
You can find the pre-trained models used in our paper in the folder `assets/saved_models`.
The inference script will output the mean of the distribution as the prediction as well as the covariance diagonal and covariance factor.

#### Evaluation
To generate samples and evaluate your predictions, you will need to use the evaluation script. 

    python evaluation/evaluate.py --path-to-prediction-csv jobs/rank_10_mc_20_patch_110/test/predictions/prediction.csv
     
The results of the evaluations will be stored in the same folder as the `prediction.csv` file.
Results are saved to disk, so a second run of the same script takes only seconds. 
Note that most of the time is being used in computing the generalised energy distance and sample diversity and not in sampling.


By default vector image thumbs are generated from which you can create your own figures (with Inkscape for example). 
Set `--make--thumbs 0` if don't wish to overwrite your images or save time.
You will need to pass `--is-deterministic 1` if you wish to evaluate the benchmark model.
If you pass `--detailed 1` with the deterministic model it will generate samples from the independent 
categorical distributions at the end of the network to illustrate why this is not ideal. 
It will also calculate the sample diversity generalised energy distance for the single deterministic prediction.
If you pass `--detailed 1` with stochastic models it will also generate sample vector images varying the temperature for each class. See paper for examples.
Note that passing `--detailed 1` significantly increases evaluation times. 
#### Sampling
The evaluation script is slow because the generalised energy distance and sample diversity computations are pairwise.
If you just want to generate samples without evaluation run
    
    python evaluation/generate_samples.py --path-to-prediction-csv jobs/rank_10_mc_20_patch_110/test/predictions/prediction.csv --num-samples 10

All the same extra arguments as the evaluation script apply.

### Toy problem

To run the toy problem
    
    python ssn/toy_problem.py
    
Pass `--overwrite True` to overwrite previous runs. You can find the output in `jobs/toy_problem`.

### LIDC 2D

The LIDC 2D results presented in the paper were generated using a [fork of the PHi-Seg code](https://github.com/MiguelMonteiro/PHiSeg-code).
Evaluation script is `phiseg_full_evaluation.py`. 
Loss function in `phiseg/phiseg_model.py`.
Model in `phiseg/model_zoo/likelihoods.py`.
Additional configs in `phiseg/experiments`.
You need to modify `config/system.py`, `phiseg_full_evaluation.py` and `phiseg/experiments/*` with your own local paths.