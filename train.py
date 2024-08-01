import pandas as pd
import numpy as np
import pickle
import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from lightning import seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from collections import OrderedDict
from argparse import ArgumentParser
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from sklearn import metrics

def prepare_data(data):
    data['Toilets'] = data['Toilets'].replace('Nhiều hơn 6 phòng', 7)
    data['Toilets'] = pd.to_numeric(data['Toilets'], errors='coerce')

    Categorical_columns = ['Property_type', 'Legal_status', 'Furniture', 'Project_name', 'District']

    Encoder = OrdinalEncoder()
    Encoder.set_params(encoded_missing_value=-1)

    # Fit the encoder to your data
    Encoder.fit(data[Categorical_columns])

    # Transform your data using the encoder
    data[Categorical_columns] = Encoder.transform(data[Categorical_columns])
    
    ##### SAVE ENCODER
    
    return data

class LitDataModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers, data):
        super(LitDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data = data

    def setup(self, stage=None):
        x = self.data.drop('Price', axis=1).values
        y = self.data['Price'].values

        s_scaler = StandardScaler()
        x = s_scaler.fit_transform(x)
        
        ##### SAVE SCALER

        dataset = TensorDataset(torch.tensor(
            x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))

        train_size = int(0.9 * len(dataset))
        test_size = int(0.05 * len(dataset))
        val_size = int(len(dataset) - train_size - test_size)
        print('Train size: ', train_size)
        print('Test size: ', test_size)
        print('Validation size: ', val_size)
        self.train_dataset, self.valid_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        dataloader = DataLoader(self.train_dataset,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=self.num_workers,
                                pin_memory=True,
                                drop_last=False)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(self.valid_dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                pin_memory=True,
                                drop_last=False)
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(self.test_dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                pin_memory=True,
                                drop_last=False)
        return dataloader

class LitDNNModule(L.LightningModule):
    def __init__(self, learning_rate=0.02):
        super(LitDNNModule, self).__init__()
        self.learning_rate = learning_rate
        self.loss_fn = nn.MSELoss()
        self.layers = nn.Sequential(OrderedDict(
            self.get_fclayer_list([10, 1024, 256, 128, 128, 64])))

        # Define metrics
        self.mse = MeanSquaredError()
        self.rmse = MeanSquaredError(squared=False)  # RMSE
        self.mae = MeanAbsoluteError()

    # creates a list of hidden layers with given number of neuron in each layer and connects it to the output layer.
    # Relu is used as the activation funtion. No activation function is applied for the last layer output
    def get_fclayer_list(self, hidden_layers, outputs=1):
        input_layers, output_layers = hidden_layers[:-1], hidden_layers[1:]
        layers = []
        for i, (l1, l2) in enumerate(zip(input_layers, output_layers)):
            layers.append((f'fc{i}', nn.Linear(l1, l2)))
            layers.append((f'prelu{i}', nn.PReLU()))
            layers.append((f"dropout{i}", nn.Dropout(0.1)))
        layers.append(('fc_out', nn.Linear(output_layers[-1], outputs)))
        return layers

    def forward(self, x):
        return self.layers(x)

    def shared_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        y = y.view(y.size(0), -1)
        loss = self.loss_fn(y, y_pred)
        return loss, y_pred

    def configure_optimizers(self):
        optimizer = optim.Adadelta(
            self.parameters(), lr=self.learning_rate, rho=0.9, eps=1e-6, weight_decay=0)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=20, eta_min=0.0, verbose=True
        # )

        return {"optimizer": optimizer,
                # "lr_scheduler": {
                #     "scheduler": scheduler,
                #     "monitor": "train_loss",
                #     "frequency": 1,
                # },
                }

    def training_step(self, batch, batch_idx):
        loss, y_pred = self.shared_step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_pred = self.shared_step(batch, batch_idx)
        self.log("val_loss", loss)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.to('cpu')
        y = y.view(y.size(0), -1)
        loss, y_pred = self.shared_step(batch, batch_idx)
        y_pred = y_pred.to('cpu')
        VarScore = metrics.explained_variance_score(y, y_pred)
        mse = self.mse(y_pred, y)
        rmse = self.rmse(y_pred, y)
        mae = self.mae(y_pred, y)
        self.log('Varian Score: ', VarScore)
        self.log('MSE Score', mse)
        self.log('RMSE Score', rmse)
        self.log('MAE Score', mae)
        return y_pred, VarScore, mse, rmse, mae

def train_model(hparams):
    seed_everything(77)
    
    # Load your data here
    data = pd.read_csv(hparams.data_path)
    
    data_module = LitDataModule(batch_size=hparams.batch_size, num_workers=hparams.num_workers, data=data)
    ml_module = LitDNNModule(learning_rate=hparams.learning_rate, batch=32)
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=1e-7, patience=80, verbose=True, mode='min', log_rank_zero_only=True)
    
    model_trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        default_root_dir=hparams.default_root_dir,
        callbacks=[early_stop_callback, lr_monitor],
        max_epochs=hparams.max_epochs
    )
    
    model_trainer.fit(ml_module, datamodule=data_module)
    model_trainer.save_checkpoint(hparams.checkpoint)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./checkpoint/final.ckpt')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the training data CSV file')
    parser.add_argument('--default_root_dir', type=str, default='./logs')
    parser.add_argument('--max_epochs', type=int, default=600)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=0.05)
    hparams = parser.parse_args()
    
    train_model(hparams=hparams)
