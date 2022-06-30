# Steps to setup and run OpenNMT

## Install necessary packages to run
```
pip3 install torch torchvision torchaudio
```

* Ensure that python version is >3.6 by running the below command

```
python -V
```

## Install OpenNMT and packages for advanced features
```
pip install OpenNMT-py
```

* Now we install extra packages for advanced features from the requirements file from https://github.com/OpenNMT/OpenNMT-py/blob/master/requirements.opt.txt and then run the below command. I have placed this in the documents directory, **PLEASE VERIFY THAT THIS DOCUMENT IS UP TO DATE WHEN YOU RUN IT**

```
pip install -r Documents/requirements.opennmt.txt
```

## Now let's run openNMT
**Before running the below commands, make sure you are using a GPU JupyterLab session with at least 4 cores and 140GB mem**
```
# Early stopping test:
export PATH="$PATH:/home/<your_computing_id>/.local/bin"

# Change into the OpenNMT/Initial_model directory if you aren't already there, then run...
onmt_build_vocab -config ICD_AIS-Initial_test.yaml -n_sample 100000
onmt_train -config ICD_AIS-Initial_test.yaml # This command takes a while, go for a 10 minute walk, keep laptop plugged in :)
onmt_translate -model run/model_step_13500.pt -src ../../Data/test_icd_pre_I9_A05.csv -output ../../Results/test_ais_pred_early_stop.csv -gpu 0 
onmt_translate -model run/model_step_13500.pt -src ../../Data/test_icd_pre_I9_A05-100.csv -output ../../Results/test_ais_pred_early_stop-100.csv -gpu 0  -attn_debug &> ../../Results/test_ais_pred_early_stop-100-attn.csv

# View attention of traning data
onmt_translate -model run/model_step_13500.pt -src ../../Data/train_icd_pre_I9_A05-100.csv -output ../../Results/train_ais_pred_early_stop-100.csv -gpu 0  -attn_debug &> ../../Results/train_ais_pred_early_stop-100-attn.csv

# Testing on training data:
onmt_translate -model run/model_step_13500.pt -src ../../Data/train_icd_pre_I9_A05-10000.csv -output ../../Results/train_ais_pred_early_stop-10000.csv -gpu 0

# Get only D-codes
# Change into OpenNMT/D_only_model directory before
sed -e 's/[A-CE-Z][0-9.-]*\s//g' ../../Data/train_icd_pre_I9_A05.csv > train_icd_onlyD_I9_A05.csv

# Model for D-codes only:
onmt_build_vocab -config ICD_AIS-D_only_test.yaml -n_sample 100000
onmt_train -config ICD_AIS-D_only_test.yaml # Take another walk... or make a smoothie
onmt_translate -model run/model_step_12500.pt -src ../../Data/test_icd_onlyD_I9_A05.csv -output ../../Results/test_ais_pred_early_stop-d_only.csv -gpu 0 
onmt_translate -model run/model_step_12500.pt -src ../../Data/test_icd_onlyD_I9_A05-100.csv -output ../../Results/test_ais_pred_early_stop-d_only-100.csv -gpu 0  -attn_debug &> ../../Results/test_ais_pred_early_stop-d_only-100-attn.csv

# Get results for age categories
# Change into OpenNMT/D_only_model
onmt_build_vocab -config ICD_AIS-Agecat.yaml -n_sample 100000
onmt_train -config ICD_AIS-Agecat.yaml
onmt_translate -model run/model_step_13500.pt -src ../../Data/test_icd_agecat_I9_A05.csv -output ../../Results/test_ais_pred_early_stop-agecat.csv -gpu 0 
```

