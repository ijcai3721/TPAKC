import torch
from tactis.gluon.estimator import TACTiSEstimator
from tactis.gluon.trainer import TACTISTrainer
from tactis.gluon.dataset import generate_hp_search_datasets, generate_prebacktesting_datasets, generate_backtesting_datasets
from tactis.gluon.metrics import compute_validation_metrics
from tactis.gluon.plots import plot_four_forecasts
from gluonts.evaluation.backtest import make_evaluation_predictions

history_factor = 1
batchh = 5
dataset = "kdd_cup_2018_without_missing"

metadata, train_data, valid_data = generate_hp_search_datasets(dataset, history_factor)

estimator = TACTiSEstimator(
    model_parameters = {
        "flow_series_embedding_dim": 5,
        "copula_series_embedding_dim": 5,
        "flow_input_encoder_layers": 2,
        "copula_input_encoder_layers": 2,
        "input_encoding_normalization": True,
        "data_normalization": "standardization",
        "loss_normalization": "series",
        "bagging_size": 20,
        "positional_encoding":{
            "dropout": 0.0,
        },
        "flow_temporal_encoder":{
            "attention_layers": 2,
            "attention_heads": 1,
            "attention_dim": 16,
            "attention_feedforward_dim": 16,
            "dropout": 0.0,
        },
        "copula_temporal_encoder":{
            "attention_layers": 2,
            "attention_heads": 1,
            "attention_dim": 16,
            "attention_feedforward_dim": 16,
            "dropout": 0.0,
        },
        "copula_decoder":{
            "min_u": 0.05,
            "max_u": 0.95,
            "attentional_copula": {
                "attention_heads": 3,
                "attention_layers": 1,
                "attention_dim": 48,
                "mlp_layers": 5,
                "mlp_dim": 128,
                "resolution": 20,
                "activation_function": "relu"
            },
            "dsf_marginal": {
                "mlp_layers": 2,
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
    trainer = TACTISTrainer(
        epochs_phase_1 = 50,
        epochs_phase_2 = 50,
        batch_size = batchh,
        num_batches_per_epoch = 512,
        learning_rate = 1e-3,
        weight_decay = 1e-4,
        maximum_learning_rate = 1e-3,
        clip_gradient = 1e3,
        device = torch.device("cuda:0"),
    ),
    cdf_normalization = False,
    num_parallel_samples = 100,
)

backtest_id = 0
metadata, backtest_train_data, backtest_valid_data = generate_prebacktesting_datasets(dataset, backtest_id, history_factor)
_, _, backtest_test_data = generate_backtesting_datasets(dataset, backtest_id, history_factor)

model = estimator.train(backtest_train_data, backtest_valid_data)

nll = estimator.validate(backtest_test_data, backtesting=True)

estimator.model_parameters["skip_copula"] = False

# Create predictor
transformation = estimator.create_transformation()
device = estimator.trainer.device
predictor = estimator.create_predictor(
    transformation=transformation,
    trained_network=model,
    device=device,
    experiment_mode="forecasting",
    history_length=history_factor * metadata.prediction_length,
)
for i in range(0,6):
    print("backtest:",i)
    backtest_id = i
    metadata, train_data, test_data = generate_backtesting_datasets(dataset, backtest_id, history_factor)
    metrics, ts_wise_metrics = compute_validation_metrics(
        predictor=predictor,
        dataset=test_data,
        window_length=estimator.history_length + estimator.prediction_length,
        prediction_length=estimator.prediction_length,
        num_samples=100,
        split=False,
    )
    print(metrics)
