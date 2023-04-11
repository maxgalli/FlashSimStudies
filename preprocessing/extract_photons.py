import awkward as ak
from coffea import processor
from coffea.nanoevents import NanoAODSchema
import pandas as pd
import os
import json
import argparse
from dask.distributed import Client
from dask_jobqueue import SLURMCluster


def parse_arguments():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--executor",
        choices=["iterative", "futures", "dask"],
        default="iterative",
        help="How to distribute the computation",
    )

    return parser.parse_args()


def find_root_files(path):
    """
    Finds all files ending with ".root" in each subfolder of the given path.
    Returns a list with the full paths to these files.
    """
    root_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".root"):
                root_files.append(os.path.join(root, file))
    return root_files


class PhotonProcessor(processor.ProcessorABC):
    def __init__(self, output_dir=None):
        self.output_dir = output_dir
        self.gen_variables = [
            "pt",
            "eta",
            "phi",
            "status",
            "statusFlags",
        ]
        self.reco_variables = [
            "pt",
            "eta",
            "phi",
            "mass",
            "r9",
            "sieie",
            "eCorr",
            "energyErr",
            "hoe",
            "mvaID",
            "pfRelIso03_all",
            "pfRelIso03_chg",
            "esEffSigmaRR",
            "esEnergyOverRawE",
            "etaWidth",
            "phiWidth",
            "pfChargedIsoPFPV",
            "pfChargedIsoWorstVtx",
            "pfPhoIso03",
            "s4",
            "sieie",
            "sieip",
            "trkSumPtHollowConeDR03",
            "x_calo",
            "y_calo",
            "z_calo",
        ]
        self.to_dump = [f"GenPho_{var}" for var in self.gen_variables] + [
            f"RecoPho_{var}" for var in self.reco_variables
        ]

    def process(self, events):
        # gen level
        gen = events.GenPart
        gen_photons = gen[gen.pdgId == 22]
        gen_electrons = gen[abs(gen.pdgId) == 11]
        gen_jets = events.GenJet

        # reco level
        photons = events.Photon

        # order by pt
        gen_photons = gen_photons[ak.argsort(gen_photons.pt, ascending=False)]
        gen_electrons = gen_electrons[ak.argsort(gen_electrons.pt, ascending=False)]
        gen_jets = gen_jets[ak.argsort(gen_jets.pt, ascending=False)]
        photons = photons[ak.argsort(photons.pt, ascending=False)]

        # get nearest gen photon for each reco photon
        nearest_gen_photons = photons.nearest(gen_photons)
        nearest_gen_electrons = nearest_gen_photons.nearest(gen_electrons)
        nearest_gen_jets = nearest_gen_photons.nearest(gen_jets)

        # PU
        pu = events.Pileup

        # other quantities
        other = ak.Array(
            {
                "GenPhoGenEle_deltar": nearest_gen_photons.delta_r(
                    nearest_gen_electrons
                ),
                "GenPhoGenEle_deltaphi": nearest_gen_photons.delta_phi(
                    nearest_gen_electrons
                ),
                "GenPhoGenEle_deltaeta": abs(
                    nearest_gen_photons.eta - nearest_gen_electrons.eta
                ),
                "GenPhoGenEle_ptratio": nearest_gen_photons.pt
                / nearest_gen_electrons.pt,
                "GenPhoGenJet_deltar": nearest_gen_photons.delta_r(nearest_gen_jets),
                "GenPhoGenJet_deltaphi": nearest_gen_photons.delta_phi(
                    nearest_gen_jets
                ),
                "GenPhoGenJet_deltaeta": abs(
                    nearest_gen_photons.eta - nearest_gen_jets.eta
                ),
                "GenPhoGenJet_ptratio": nearest_gen_photons.pt / nearest_gen_jets.pt,
                "RecoPhoGenPho_deltar": photons.delta_r(nearest_gen_photons),
                "RecoPhoGenPho_deltaphi": photons.delta_phi(nearest_gen_photons),
                "RecoPhoGenPho_deltaeta": abs(photons.eta - nearest_gen_photons.eta),
                "RecoPhoGenPho_ptratio": photons.pt / nearest_gen_photons.pt,
                "ClosestGenJet_pt": nearest_gen_jets.pt,
                "ClosestGenJet_mass": nearest_gen_jets.mass,
                "PU_nTrueInt": ak.ones_like(photons.pt) * pu.nTrueInt,
                "PU_nPU": ak.ones_like(photons.pt) * pu.nPU,
                "PU_gpudensity": ak.ones_like(photons.pt) * pu.gpudensity,
                "PU_pudensity": ak.ones_like(photons.pt) * pu.pudensity,
                "PU_sumEOOT": ak.ones_like(photons.pt) * pu.sumEOOT,
                "PU_sumLOOT": ak.ones_like(photons.pt) * pu.sumLOOT,
            }
        )

        # merge and make pandas df
        df = pd.DataFrame()
        for var in self.gen_variables:
            df[f"GenPho_{var}"] = ak.to_numpy(ak.flatten(nearest_gen_photons[var]))
        for var in self.reco_variables:
            df[f"RecoPho_{var}"] = ak.to_numpy(ak.flatten(photons[var]))
        for var in other.fields:
            df[var] = ak.to_numpy(ak.flatten(other[var]))
        nan_mask = df.isna()
        cols_with_nan = nan_mask.any()
        cols_with_nan = cols_with_nan[cols_with_nan == True].index.tolist()
        for col in cols_with_nan:
            how_many = df[col].isna().sum()
            print(f"Column {col} has {how_many}/{len(df)} NaNs")
        df = df.dropna()

        # save to hdf5
        if self.output_dir:
            f_out = events.behavior["__events_factory__"]._partition_key.replace(
                "/", "_"
            )
            output_path = os.path.join(self.output_dir, f"{f_out}.parquet")
            # df.to_hdf(output_path, key="df", mode="w")
            df.to_parquet(output_path, engine="pyarrow", compression="gzip")

        return {}

    def postprocess(self, accumulator):
        pass


def main(args):
    base_dir = (
        "/pnfs/psi.ch/cms/trivcat/store/user/gallim/FlashSimGammaPlusJet_oldbranch"
    )
    files = find_root_files(base_dir)
    #files = files[:10]
    fileset = {"GJet": files}
    print(fileset)
    output_dir = (
        "/work/gallim/SIMStudies/FlashSimStudies/preprocessing/extracted_photons"
    )

    if args.executor == "iterative":
        executor = processor.IterativeExecutor()
    elif args.executor == "futures":
        executor = processor.FuturesExecutor(workers=30)
    elif args.executor == "dask":
        cluster = SLURMCluster(
            # queue="short",
            queue="standard",
            # walltime="10:00:00",
            cores=1,
            processes=1,
            memory="6G",
            log_directory="slurm_logs",
            local_directory="slurm_logs",
        )
        cluster.adapt(minimum=10, maximum=50)
        #cluster.adapt(minimum=1, maximum=10)
        client = Client(cluster)
        client.wait_for_workers(10)

        executor = processor.DaskExecutor(client=client)

    run = processor.Runner(
        executor=executor,
        schema=NanoAODSchema,
    )
    out = run.run(
        fileset,
        treename="Events",
        processor_instance=PhotonProcessor(output_dir=output_dir),
    )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
