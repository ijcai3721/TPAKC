import os
import sys
import pandas as pd
from statistics import mean
import numpy as np
from pprint import pprint
REPO_NAME = "tactis"
def get_repo_basepath():
    cd = os.path.abspath(os.curdir)
    return cd[:cd.index(REPO_NAME) + len(REPO_NAME)]
REPO_BASE_PATH = get_repo_basepath()
sys.path.append(REPO_BASE_PATH)
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:521"
# os.environ['CUDA_VISIBLE_DEVICES']='0'
from pts import Trainer
import torch
from tactis.gluon.estimator import TACTiSEstimator
from tactis.gluon.dataset import generate_backtesting_datasets, __FixedMultivariateGrouper
from tactis.gluon.metrics import compute_validation_metrics
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from tactis.gluon.plots import plot_four_forecasts
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.dataset.common import ListDataset

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') # don't have GPU
    return device

def df_to_tensor(df):
    device = get_device()
    return torch.from_numpy(df).float().to(device)


history_factor = 1
batchh = 5
backtest_id = 0
dataset = "traffic"
metadata, train_data, test_data = generate_backtesting_datasets(dataset, backtest_id, history_factor) # electricity_hourly, fred_md, kdd_cup_2018_without_missing,traffic

print("Trial Run Initialising........")

df_input_tensor = df_to_tensor(train_data.list_data[0]['target']).t()

estimator = TACTiSEstimator(
    model_parameters = {
        "series_embedding_dim": 5,
        "input_encoder_layers": 2,
        "input_encoding_normalization": True,
        "data_normalization": "standardization",
        "loss_normalization": "series",
        "positional_encoding":{
            "dropout": 0.0,
        },
        "temporal_encoder":{
            "attention_layers": 2,
            "attention_heads": 2, 
            "attention_dim": 24,
            "attention_feedforward_dim": 24,
            "dropout": 0.0,
        },
        "copula_decoder":{
            "min_u": 0.05,
            "max_u": 0.95,
            "attentional_copula": {
                "attention_heads": 1, 
                "attention_layers": 2,
                "attention_dim": 48,
                "mlp_layers": 5,
                "mlp_dim": 128,
                "resolution": 20,
            },
            "dsf_marginal": {
                "mlp_layers": 1,
                "mlp_dim": 48,
                "flow_layers": 3,
                "flow_hid_dim": 16,
            },
        },
    },
    num_series = train_data.list_data[0]["target"].shape[0],
    history_length = history_factor * metadata.prediction_length,
    prediction_length = metadata.prediction_length,
    freq = metadata.freq,
    trainer = Trainer(
        epochs = 50,
        batch_size = batchh,
        num_batches_per_epoch = 512,
        learning_rate = 1e-5,
        weight_decay = 0,
        maximum_learning_rate = 1e-3,
        clip_gradient = 1e3,
        device = get_device(),
    ),
    cdf_normalization = False,
    num_parallel_samples = 100,
)

predictor = estimator.train(train_data)
predictor.batch_size = 5
print("Corrected_TACTIS_1 RAW_EXCEL2_sig_0.05 const_30 running ",dataset)
for i in range(0,6):
    print("backtest:",i)
    backtest_id = i
    metadata, train_data, test_data = generate_backtesting_datasets(dataset, backtest_id, history_factor)
    metrics = compute_validation_metrics(
        predictor=predictor,
        dataset=test_data,
        window_length=estimator.history_length + estimator.prediction_length,
        num_samples=100,
        split=False,
    )
    print(metrics)
