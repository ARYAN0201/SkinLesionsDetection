import numpy as np
import pandas as pd

class DataBalancer:
    def __init__(self, df, target_count):
        self.df = df
        self.target_count = target_count

    def calculate_stats(self, image_arrays):
        pixels = np.concatenate([img.flatten() for img in image_arrays])
        return np.mean(pixels), np.var(pixels)
    
    def sample_class(self, class_df, image_arrays):
        # Sample with replacement to reach target_count
        sampled_df = class_df.sample(n=self.target_count, replace=True, random_state=42)
        return sampled_df
    
    def balance(self):
        balanced_dfs = []
        for cls in self.df['dx'].unique():
            class_df = self.df[self.df['dx'] == cls]
            balanced_class_df = self.sample_class(class_df, None)
            balanced_dfs.append(balanced_class_df)
        balanced_df = pd.concat(balanced_dfs).reset_index(drop=True)
        return balanced_df
