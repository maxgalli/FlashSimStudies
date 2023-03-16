import os
import pandas as pd

def main():
    input_dir = "/work/gallim/SIMStudies/FlashSimStudies/preprocessing/extracted_photons"
    # read the first 10 parquet files and dump a pandas dataframe
    files = [f"{input_dir}/{f}" for f in os.listdir(input_dir)][:10]
    df = pd.concat([pd.read_parquet(f) for f in files])
    print(len(df))
    # dump
    with open("small_photon_sample.parquet", "wb") as f:
        df.to_parquet(f, engine="pyarrow", compression="gzip")
    
if __name__ == "__main__":
    main()