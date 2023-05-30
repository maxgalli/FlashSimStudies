# FlashSim Studies 

FlashSim studies mostly concerning photons and SC quantities.

## Setup

```
mamba env create -f environment.yml
```

then clone ```nflows``` ([link](https://github.com/maxgalli/nflows.git)) and install it with the usual ```pip install -e .``` in case we need to edit it.

## Electrons

Instructions to run what was done by Filippo in wipfs.

Extraction:

```
cd /work/gallim/SIMStudies/wipfs/extract/electrons
root 
root [0] .L extraction.C
root [1] extraction()
```
which will save the ROOT files in the current directory.

Preprocess:
```
cd /work/gallim/SIMStudies/wipfs/preproce/electrons
python preprocessing.py
```
which will create a hdf5 file in ```/work/gallim/SIMStudies/wipfs/training/electrons```.

## Photons

### Dataset Preparation

For gamma + jet ([DAS link](https://cmsweb.cern.ch/das/request?view=list&limit=50&instance=prod%2Fglobal&input=dataset%3D%2FGJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_13TeV_Pythia8%2FRunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1%2FNANOAODSIM)):

```
cd /pnfs/psi.ch/cms/trivcat/store/user/gallim/FlashSim/GJet
/cvmfs/cms.cern.ch/common/dasgoclient -query="file dataset=/GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_13TeV_Pythia8/RunIISummer20UL18NanoAODv2-106X_upgrade2018_realistic_v15_L1v1-v1/NANOAODSIM" > files.txt
python get_files.py

cd /work/gallim/SIMStudies/FlashSimStudies/preprocessing
python extract_photons.py --executor dask
```

### First Update

At the moment of doing the training, nanoAOD with the SC quantities we added haven't been centrally produced, so we need to produce them ourselves. In order ot make the dataset we follow the same procedure described [here](https://gist.github.com/maxgalli/0886ec4290672ecf57031ac969c4ade5), which works with 10_6_26. 

Input MiniAOD: [link](https://cmsweb.cern.ch/das/request?view=list&limit=50&instance=prod%2Fglobal&input=dataset%3D%2FGJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_13TeV_Pythia8%2FRunIISummer20UL18MiniAOD-106X_upgrade2018_realistic_v11_L1v1-v3%2FMINIAODSIM)

NanoAOD produced: [link](https://cmsweb.cern.ch/das/request?input=dataset%3D%2FGJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_13TeV_Pythia8%2Fgallim-crab_FlashSimPhotonSample-ee6432161e5c185bb80950ddbb7162c0%2FUSER&instance=prod/phys03)

### Preprocessing

```
cd preprocessing 
python preprocess_photons.py
```

The script contains a dictionary of pipelines for each variable we want to preprocess. In ```preprocessed_photons``` the following objects are dumped:
- train and test parquet files
- ```pipelines.pkl```, a dictinoary containing a pipeline for each of the preprocessed variables
- inside ```preprocessed_photons/figures```, variables before and after preprocessing are plotted side by side

### Training

```
cd training
python train_photons.py --config-name <config-name>
```

To use the multi-GPU training and also select which GPUs to use, the best way I found consists in specifying the GPUs to use through ```CUDA_VISIBLE_DEVICES```, e.g.:

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_photons.py --config-name cfg7
```

#### Configurations comments

- cfg2: 4 conditioning, 2 vars with full 32M parameters model used for electrons and no preprocessing performed on the conditioning variables

## Quick Start (for Photons)

1. Create the environment following [these instructions](#setup). Note that some packages (such as tensorboard) might be missing - if this is the case, just install it by hand. The repo containg the models is needed, more specifically the branch ```electrons_new``` of ```wipfs``` ([link](https://github.com/francesco-vaselli/wipfs/tree/electrons_new)). The structure should be something that looks like this:
```
.
├── FlashSimStudies
│   ├── fstudies
│   ├── preprocessing
│   └── training
├── packages_to_install
│   └── nflows
└── wipfs
``` 

2. Gather [this](https://cmsweb.cern.ch/das/request?input=dataset%3D%2FGJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_13TeV_Pythia8%2Fgallim-crab_FlashSimPhotonSample-ee6432161e5c185bb80950ddbb7162c0%2FUSER&instance=prod/phys03) gamma + jet nanoAOD sample using XRootD (**Note**: this is a privately produced sample which was necessary because at the moment of performing the study the nanoAOD version with all the interesting variables was not ready yet, but now it probably is - worth checking!)
3. Extract photons: produce parquet files containing only the interesting variables
```
cd preprocessing
python extract_photons.py --executor dask
```
(**Note**: input and output directories are hardcoded, remember to change them! fastest way is to grep for ```gallim```; it might also be that an ```extracted_photons``` directory has to be created, if I wasn't smart enough to use ```os.makedirs("path/to/demo_folder", exist_ok=True)```)

4. Preprocessing: create train and test files (more details [here](#preprocessing))
```
python preprocess_photons.py
```

5. Training: hydra is used for configuration and commands are reported [above](#training). **Note**: the models used are the ones already included in ```wipfs```, hence the path to this directory is appended to the Python path. Again, grep for ```gallim``` and change it accordingly.

To check the results we use tensorboard:
```
tensorboard --logdir=outputs --port 6006
```
then connect locally via ssh forwarding using:
```
ssh -Y -N -f -L localhost:8002:localhost:6006 gallim@psi
```

## Improvements and Future Work

At the moment of writing, the target photon variables that we've simulated give good results in terms of Wassertein distance, but more work should be done:

- corner plots and nicer plots investigating the correlation between target and context variables should be added;
- more target variables should be added (along with useful preprocessing); to check which ones, just look at what has been done for the electrons;
- PU related variables are still missing from the conditioning list, hence they should be preprocessed (with a smearing) and added.