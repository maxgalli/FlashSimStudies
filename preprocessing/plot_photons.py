"""
Plot the extracted photon related quantities. In order to make it memory efficient, we fill one histogram 
per partition in parallel and then merge the histograms. This is done using dask.
"""
import dask
import dask.dataframe as dd
from dask.distributed import LocalCluster, Client
import matplotlib.pyplot as plt
import os
from hist import Hist


def fill_histograms(part, histograms_dict):
    # part = part.compute()
    for col in histograms_dict:
        histograms_dict[col].fill(part[col].values)
    return histograms_dict


def main():
    file_path_pattern = "/work/gallim/SIMStudies/FlashSimStudies/preprocessing/extracted_photons/*.parquet"
    output_dir = (
        "/eos/home-g/gallim/www/plots/SIMStudies/FlashSimStudies/extracted_photons"
    )

    # start a local cluster for parallel processing
    cluster = LocalCluster()
    client = Client(cluster)

    print("Reading files...")
    df = dd.read_parquet(file_path_pattern, engine="fastparquet")

    histograms = {}
    for col in df.columns:
        print(f"Making histogram for {col}")
        # https://docs.dask.org/en/stable/best-practices.html#avoid-calling-compute-repeatedly
        xmin, xmax = dask.compute(df[col].min(), df[col].max())
        histograms[col] = Hist.new.Reg(100, xmin, xmax, name=col, label=col).Double()

    print("Filling histograms...")
    futures = []
    for part in df.partitions:
        futures.append(dask.delayed(fill_histograms)(part, histograms))
    dicts = dask.compute(*futures)
    # merge the histograms
    for d in dicts:
        for col, hist in d.items():
            histograms[col] += hist

    print("Plotting histograms...")
    for col, hist in histograms.items():
        fig, ax = plt.subplots()
        ax.bar(hist.axes[0].centers, hist.view(), width=hist.axes[0].widths)
        ax.set_xlabel(col)
        ax.set_ylabel("Events")
        for format in ["png", "pdf"]:
            fig.savefig(os.path.join(output_dir, f"{col}.{format}"))

    # close the local cluster
    client.close()
    cluster.close()


if __name__ == "__main__":
    main()
