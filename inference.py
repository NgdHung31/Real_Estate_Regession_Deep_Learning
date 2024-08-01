import pandas as pd
import numpy as np
import pickle
import torch
from argparse import Namespace
from argparse import ArgumentParser
from train import LitDNNModule  # Assuming LitDNNModule is in train.py
 
def prepare_data(data):
    with open('encoder.pkl', 'rb') as f:
        Encoder = pickle.load(f)

    Categorical_columns = ['Property_type', 'Legal_status', 'Furniture', 'Project_name', 'District']

    # Transform your data using the encoder
    data[Categorical_columns] = Encoder.transform(data[Categorical_columns])
    
    return data 

def data_preprocessing(hparams):
    model = LitDNNModule.load_from_checkpoint(hparams.checkpoint)
    model.eval()
    
    # Load test data
    test_data = pd.DataFrame({
        'Property_type': [hparams.property_type],
        'Area': [hparams.area],
        'Floors': [hparams.floors],
        'Bedrooms': [hparams.bedrooms],
        'Toilets': [hparams.toilets],
        'Legal_status': [hparams.legal_status],
        'Furniture': [hparams.furniture],
        'Project_name': [hparams.project_name],
        'District': [hparams.district],
        'Distance': [hparams.distance],
    })
    
    test_data = prepare_data(test_data)
    X_test = test_data.values
    
    with open('scaler.pkl', 'rb') as f:
        s_scaler = pickle.load(f)
        
    X_test = s_scaler.transform(X_test.astype(np.float32))
    X_test = torch.tensor(X_test, dtype=torch.float)
    
    with torch.no_grad():
        predictions = model(X_test)
    
    predictions = predictions.numpy()
    #pd.DataFrame(predictions, columns=['Price']).to_csv(hparams.result_path, index=False)
    return predictions[0][0]

def inference_model(property_type, area,  floors, bedrooms, toilets, legal_status, furniture, project_name, district, distance):
    hparams = Namespace(
        checkpoint="./checkpoint/final.ckpt",
        property_type=property_type,
        area=area,
        floors=floors,
        bedrooms=bedrooms,
        toilets=toilets,
        legal_status=legal_status,
        furniture=furniture,
        project_name=project_name,
        district=district,       
        distance=distance,
    )
    return data_preprocessing(hparams)