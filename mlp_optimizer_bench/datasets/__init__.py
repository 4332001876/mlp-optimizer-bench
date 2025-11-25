# mlp_optimizer_bench/datasets/__init__.py

from .toy_classification import *
from .mnist import *

from .high_order_classification import (
    HighOrderClassificationConfig,
    HighOrderClassificationDataset,
    generate_high_order_classification,
    get_high_order_classification_dataloaders,
)

from .function_regression import (
    FunctionRegressionConfig,
    FunctionRegressionDataset,
    generate_function_regression,
    get_function_regression_dataloaders,
)

from .sequence_parity import (
    SequenceParityConfig,
    SequenceParityDataset,
    generate_sequence_parity,
    get_sequence_parity_dataloaders,
)