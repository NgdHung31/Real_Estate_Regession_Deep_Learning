# import required libraries
import pandas as pd
import numpy as np

# read the csv file
Data = pd.read_csv("./content/dataset_non_outliers_Price_Area.csv", low_memory=False)
Data = Data.drop(
    labels=[
        "Frontage",
        "Street",
        "Ward",
        "Province",
        "Width",
        "Length",
        "House_orientation",
        "Balcony_orientation",
        "Access_road",
        "Posting_date",
        "Expiry_date",
        "Type_of_listing",
    ],
    axis=1,
)

Data.loc[Data["Property_type"] == "Chung cư", "Number_of_floors"] = 1
Data.loc[pd.isnull(Data["Project_name"]), "Project_name"] = "Other"
Data = Data.rename(
    columns={
        "Number_of_floors": "Floors",
        "Number_of_bedrooms": "Bedrooms",
        "Number_of_toilets": "Toilets",
    }
)

# Convert object value to numeric value
Data["Toilets"] = Data["Toilets"].replace("Nhiều hơn 6 phòng", 7)
Data["Toilets"] = pd.to_numeric(Data["Toilets"], errors="coerce")
Data = Data.dropna()
from sklearn.preprocessing import OrdinalEncoder

# Create an instance of OrdinalEncoder
Encoder = OrdinalEncoder()
Encoder.set_params(encoded_missing_value=-1)

Categorical_columns = [
    "Property_type",
    "Legal_status",
    "Furniture",
    "Project_name",
    "District",
]

# Fit the encoder to your data
Encoder.fit(Data[Categorical_columns])

# Transform your data using the encoder
Data[Categorical_columns] = Encoder.transform(Data[Categorical_columns])
## Sử dụng model Machine Learning để dự đoán giá trị bị thiếu
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

cols = ["Property_type", "Area", "Floors", "Bedrooms", "Toilets"]
impute_it = IterativeImputer()
Data[cols] = impute_it.fit_transform(Data[cols])
import torch
import lightning as L

# from sklearn import preprocessing
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
from sklearn import metrics
from sklearn.model_selection import train_test_split


class LitDataModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers, data):
        super(LitDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data = data
        # X = data.drop("Price", axis=1).values
        # y = data["Price"].values

    def setup(self, stage=None):
        x = self.data.drop("Price", axis=1).values
        y = self.data["Price"].values

        x_train, x_temp, y_train, y_temp = train_test_split(
            x, y, test_size=0.1, random_state=101
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_temp, y_temp, test_size=0.5, random_state=101
        )

        s_scaler = StandardScaler()
        x_train = s_scaler.fit_transform(x_train.astype(np.float64))
        x_val = s_scaler.transform(x_val.astype(np.float64))
        x_test = s_scaler.transform(x_test.astype(np.float64))

        self.train_dataset = TensorDataset(
            torch.tensor(x_train, dtype=torch.float),
            torch.tensor(y_train, dtype=torch.float),
        )

        self.valid_dataset = TensorDataset(
            torch.tensor(x_val, dtype=torch.float),
            torch.tensor(y_val, dtype=torch.float),
        )

        self.test_dataset = TensorDataset(
            torch.tensor(x_test, dtype=torch.float),
            torch.tensor(y_test, dtype=torch.float),
        )

        print("Train size: ", 0.8)
        print("Test size: ", 0.1)
        print("Validation size: ", 0.1)

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return dataloader


class LitDNNModule(L.LightningModule):
    def __init__(self, learning_rate=0.02, batch=32):
        super(LitDNNModule, self).__init__()
        self.learning_rate = learning_rate
        self.batch = batch
        self.loss_fn = nn.MSELoss()
        self.layers = nn.Sequential(
            OrderedDict(self.get_fclayer_list([10, 15, 20, 15, 10, 5]))
        )

    # creates a list of hidden layers with given number of neuron in each layer and connects it to the output layer.
    # Relu is used as the activation funtion. No activation function is applied for the last layer output
    def get_fclayer_list(self, hidden_layers, outputs=1):
        input_layers, output_layers = hidden_layers[:-1], hidden_layers[1:]
        layers = []
        for i, (l1, l2) in enumerate(zip(input_layers, output_layers)):
            layers.append((f"fc{i}", nn.Linear(l1, l2)))
            layers.append((f"leakyrelu{i}", nn.LeakyReLU()))
        layers.append(("fc_out", nn.Linear(output_layers[-1], outputs)))
        return layers

    def forward(self, x):
        y = self.layers(x)
        return y

    # def configure_optimizers(self):
    #    return optim.AdamW(self.parameters(), lr=self.learning_rate)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=20, eta_min=0.0, verbose=True
        )
        # def lr_scheduler_step(self, scheduler, metric):
        #     if metric is None:
        #         scheduler.step()
        #         last_lr = scheduler.get_last_lr()[0]
        #         self.print(f"Current learning rate: {last_lr}")
        #     else:
        #         scheduler.step(metric)
        #         last_lr = scheduler.get_last_lr()[0]
        #         self.print(f"Current learning rate: {last_lr}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

    def shared_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        y = y.view(y.size(0), -1)
        loss = self.loss_fn(y, y_pred)
        return loss, y_pred

    # def on_validation_epoch_end(self):
    #     last_lr = self.lr_schedulers.get_last_lr()[0]
    #     # current_lr = lr_scheduler.optimizer.param_groups[0]['lr']
    #     print(f"Current learning rate: {last_lr}")

    def training_step(self, batch, batch_idx):
        loss, y_pred = self.shared_step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_pred = self.shared_step(batch, batch_idx)
        self.log("val_loss", loss)
        return {"val_loss", loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.to("cpu")
        y = y.view(y.size(0), -1)
        loss, y_pred = self.shared_step(batch, batch_idx)
        y_pred = y_pred.to("cpu")
        VarScore = metrics.explained_variance_score(y, y_pred)
        self.log("Varian Score: ", VarScore)
        return y_pred, VarScore


from argparse import Namespace
import torch
from lightning import seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.tuner import Tuner

# Define hparams
hparams = Namespace(
    checkpoint_name="./checkpoint/final.ckpt",
    data_folder="./content",
    test_input="test.csv",
    test_output="test_pred.csv",
    default_root_dir="./logs",
    max_epochs=450,
    gpus=(-1 if torch.cuda.is_available() else 0),
    auto_select_gpus=True,
    deterministic=True,
    batch_size=128,
    num_workers=6,
    learning_rate=0.02,
    fast_dev_run=False,
)

lr_monitor = LearningRateMonitor(logging_interval="epoch")
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=1e-7,
    patience=80,
    verbose=True,
    mode="min",
    log_rank_zero_only=True,
)
seed_everything(77)

ml_module = LitDNNModule(learning_rate=0.1, batch=64)
data_module = LitDataModule(batch_size=64, num_workers=6, data=Data)
# data_module.setup()
model_trainer = L.Trainer(
    accelerator="gpu",
    default_root_dir="./logs",
    callbacks=[early_stop_callback, lr_monitor],
    max_epochs=10000,
)
model_trainer.logger = L.pytorch.loggers.TensorBoardLogger("logs/", name="exp")

# tuner = Tuner(model_trainer)
# lr_finder = tuner.lr_find(ml_module, datamodule=data_module)
# print(lr_finder.results)


def train_model(hparams):
    # model_trainer = pl.Trainer.from_argparse_args(hparams)

    model_trainer.fit(ml_module, datamodule=data_module)
    model_trainer.save_checkpoint(hparams.checkpoint_name)


def test_model(hparams):
    model = LitDNNModule.load_from_checkpoint(hparams.checkpoint_name)
    model_trainer.test(model, datamodule=data_module)


# Call train_model() with hparams
train_model(hparams=hparams)
# test_model(hparams=hparams)
test_model(hparams=hparams)
