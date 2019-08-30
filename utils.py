import os
from tqdm import tqdm


def clear_folder(folder_path):
    # deletes all contents of a folder
    print("Clearing folder...")
    for the_file in tqdm(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
    print("\nClearing finished.")
