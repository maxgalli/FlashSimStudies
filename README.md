# FlashSim Studies 

FlashSim studies mostly concerning photons and SC quantities.

## Setup

```
mamba env create -f environment.yml
```

then clone ```nflows``` and install it with the usual ```pip install -e .``` in case we need to edit it.

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