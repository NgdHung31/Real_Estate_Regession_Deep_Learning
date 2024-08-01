import torch
import lightning as L
import pandas as pd
import numpy as np
import pickle
from argparse import ArgumentParser
from train import LitDataModule, LitDNNModule  # Assuming these are in train.py

def prepare_data(data):
    with open('encoder.pkl', 'rb') as f:
        Encoder = pickle.load(f)

    Categorical_columns = ['Property_type', 'Legal_status', 'Furniture', 'Project_name', 'District']

    # Transform your data using the encoder
    data[Categorical_columns] = Encoder.transform(data[Categorical_columns])
    
    data = data.values
    
    with open('scaler.pkl', 'rb') as f:
        s_scaler = pickle.load(f)
        
    data = s_scaler.transform(data.astype(np.float32))
    data = torch.tensor(data, dtype=torch.float)
    
    return data 

def test_model(hparams):
    Data = pd.read_csv(hparams.data_path)
    Data = prepare_data(Data)

    model = LitDNNModule.load_from_checkpoint(hparams.checkpoint_name)    
    data_module = LitDataModule(batch_size=64, num_workers=6, data=Data)
    
    model_trainer = L.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu")
    model_trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./checkpoint/final.ckpt')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the testing data CSV file')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=6)
    hparams = parser.parse_args()
    
    test_model(hparams=hparams)
