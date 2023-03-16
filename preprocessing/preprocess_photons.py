import os

def main():
    input_dir = "/work/gallim/SIMStudies/FlashSimStudies/preprocessing/extracted_photons"
    # divide the files into three parts (train, validation, test)
    # train contains 50% of the files, validation 25%, test 25%
    train_files, val_files, test_files = [], [], []
    all_files = [f"{input_dir}/{f}" for f in os.listdir(input_dir)]
    for i, f in enumerate(all_files):
        if i % 4 == 0:
            test_files.append(f)
        elif i % 4 == 1:
            val_files.append(f)
        else:
            train_files.append(f)
    print(len(train_files), len(val_files), len(test_files))

if __name__ == "__main__":
    main()