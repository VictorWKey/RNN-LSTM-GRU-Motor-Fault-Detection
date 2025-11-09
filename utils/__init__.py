"""
Utilidades para preparación de datos y manejo de datasets.
"""
from .data_utils import (
    load_motor_signals,
    create_windows_classification,
    create_windows_regression,
    normalize_data,
    SignalDataset,
    TimeSeriesDataset,
    get_data_loaders_classification,
    get_data_loaders_regression,
    generate_synthetic_timeseries
)

from .training_utils import (
    train_epoch,
    validate_epoch,
    train_model,
    compute_classification_metrics,
    compute_regression_metrics,
    save_checkpoint,
    load_checkpoint
)

from .visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_predictions_vs_actual,
    plot_time_series_prediction,
    plot_model_comparison,
    save_results_table
)

__all__ = [
    # Data utils
    'load_motor_signals',
    'create_windows_classification',
    'create_windows_regression',
    'normalize_data',
    'SignalDataset',
    'TimeSeriesDataset',
    'get_data_loaders_classification',
    'get_data_loaders_regression',
    'generate_synthetic_timeseries',
    # Training utils
    'train_epoch',
    'validate_epoch',
    'train_model',
    'compute_classification_metrics',
    'compute_regression_metrics',
    'save_checkpoint',
    'load_checkpoint',
    # Visualization
    'plot_training_history',
    'plot_confusion_matrix',
    'plot_predictions_vs_actual',
    'plot_time_series_prediction',
    'plot_model_comparison',
    'save_results_table'
]
