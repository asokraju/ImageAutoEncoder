import os


def get_image_data(all_dirs):
    # List to store all image file paths
    all_image_paths = []

    # Loop through all directories and subdirectories in the data directory
    for data_dir in all_dirs:
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                # Check if the file is an image file (you can add more extensions as needed)
                if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                    # If the file is an image file, append its path to the list
                    all_image_paths.append(os.path.join(root, file))
        print(data_dir)
    image_count = len(all_image_paths)
    print("Total number of imges:", image_count)
    return all_image_paths