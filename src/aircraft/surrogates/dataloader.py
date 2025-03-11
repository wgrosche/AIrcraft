import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple
from collections import namedtuple
import matplotlib.pyplot as plt

__all__ = ['AeroDataset', 'create_dataloader', 'get_airplane_params']

AircraftSpecs = namedtuple('AircraftSpecs', ['chord', 'wing_area', 'mass'])

def get_airplane_params(df:pd.DataFrame) -> dict:
    airplane_params = {}
    for index, row in df.iterrows():
        airplane_params[row["var_name"]] = row["var_val"]
    
    return airplane_params

class AeroDataset(Dataset):
    def __init__(self, data_dir: str, input_features: List[str], output_features: List[str], 
                 transform=None, target_transform=None):
        
        self.data_dir = Path(data_dir)
        self.input_features = input_features
        self.output_features = output_features
        self.transform = transform
        self.target_transform = target_transform

        self.data = self.load_dataset()


    def load_dataset(self) -> pd.DataFrame:
        dataset = pd.DataFrame(columns=self.input_features + self.output_features)
        for file in os.listdir(self.data_dir):
            if file.endswith('.csv'):
                if file == 'data_real.csv':
                    continue
                temp = pd.read_csv(self.data_dir / file)
                print(temp.head())
                dataset = pd.concat([dataset, temp[self.input_features + self.output_features]], axis=0)

        return dataset

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.data.iloc[idx]
        inputs = row[self.input_features].to_numpy(dtype=np.float32)
        targets = row[self.output_features].to_numpy(dtype=np.float32)

        if self.transform:
            inputs = self.transform(inputs)
        if self.target_transform:
            targets = self.target_transform(targets)

        return torch.from_numpy(inputs), torch.from_numpy(targets)

def create_dataloader(dataset: Dataset, batch_size: int = 16, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def main():#
    BASEPATH = os.path.dirname(os.path.abspath(__file__)).split('src')[0]
    DATA_PATH = os.path.join(BASEPATH, 'data', 'processed')
    # load dataset
    input_features = ['q','alpha','beta','aileron','elevator']
    output_features = ['CD', 'CY', 'CL', 'Cl', 'Cm', 'Cn']
    dataset = AeroDataset(DATA_PATH, input_features, output_features)

    # create dataloader
    dataloader = create_dataloader(dataset, batch_size=2056, shuffle=True)

    # plot some data
    inputs, targets = next(iter(dataloader))
    print(inputs.shape, targets.shape)

    # plot some data
    _, ax = plt.subplots(2, 3, figsize=(15, 10))
    for i in range(6):
        ax[i//3, i%3].scatter(inputs[:,1], targets[:, i])
        ax[i//3, i%3].set_xlabel(input_features[1])
        ax[i//3, i%3].set_ylabel(output_features[i])
    plt.show()


if __name__ == '__main__':
    main()
