"""
Paquete de utilidades.
"""
from .data_utils import (
    make_windows_regression,
    make_windows_classification,
    TimeSeriesDataset,
    SignalClassificationDataset,
    normalize_data
)
from .training_utils import (
    train_epoch,
    evaluate,
    train_model,
    compute_classification_metrics
)
from .visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_predictions_vs_actual,
    plot_time_series_prediction
)

__all__ = [
    'make_windows_regression',
    'make_windows_classification',
    'TimeSeriesDataset',
    'SignalClassificationDataset',
    'normalize_data',
    'train_epoch',
    'evaluate',
    'train_model',
    'compute_classification_metrics',
    'plot_training_history',
    'plot_confusion_matrix',
    'plot_predictions_vs_actual',
    'plot_time_series_prediction'
]
