import os
import pandas as pd
import numpy as np

class HAMPreprocessor:
    def __init__(self, metadata_path: str, image_dirs: list):
        self.metadata_path = metadata_path
        self.image_dirs = image_dirs
        self.df = pd.read_csv(metadata_path)
    
    def validate_image_paths(self):
        print("Validating Image Paths...")
        valid_paths = []
        for img_id in self.df['image_id']:
            found = False
            for dir_path in self.image_dirs:
                full_path = os.path.join(dir_path, f"{img_id}.jpg")
                if os.path.exists(full_path):
                    valid_paths.append(full_path)
                    found = True
                    break
            if not found:
                valid_paths.append(None)

        self.df['image_path'] = valid_paths
        self.df.dropna(subset=['image_path'], inplace = True)
        print(f"Found {len(self.df)} valid images.")

    def impute_missing_ages(self):
        print("Imputing missing age values with class = conditional median...")
        self.df['age'] = self.df.groupby('dx')['age'].transform(
            lambda x: x.fillna(x.median())
        )
        class_counts = self.df['dx'].value_counts()
        print(class_counts)
    
    def preprocess(self):
        self.validate_image_paths()
        self.impute_missing_ages()
        print("Preprocessing complete.")
        return self.df


if __name__ == "__main__":

    image_dirs = [
        "/home/Ujjwal/Aryan/HAM10000/ham10000_images_part_1",
        "/home/Ujjwal/Aryan/HAM10000/HAM10000_images_part_1",
        "/home/Ujjwal/Aryan/HAM10000/ham10000_images_part_2",
        "/home/Ujjwal/Aryan/HAM10000/HAM10000_images_part_2"
    ]

    processor = HAMPreprocessor("/home/Ujjwal/Aryan/HAM10000/HAM10000_metadata.csv", image_dirs)
    df_processed = processor.preprocess()
    df_processed.to_csv("HAMProcessed.csv", index=False)