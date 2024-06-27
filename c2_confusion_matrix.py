"""
Artur Marques, 2024

Example client for my own confusion matrix tools

"""
from confusion_matrix2 import \
    confusion_matrix, \
    confusion_matrix_to_string, \
    confusion_matrix_metrics,\
    get_all_model_performance_metrics_for_all_classes,\
    present_performance_metrics

"""
Example problem
spam or no spam?
dataset = ["S", "NS", "S", "NS", "S", "NS", "NS", "S"]
predictions = ["S", "NS", "NS", "NS", "S", "S", "NS", "S"]
"""

dataset =       ["C", "D", "F", "F", "C", "D", "D", "C"]
predictions =   ["C", "F", "D", "F", "C", "C", "D", "C"]

print("Will now test the confusion_matrix function with:")
print(f"dataset: {dataset}")
print(f"predictions: {predictions}")

cm = confusion_matrix(
    dataset,
    predictions
)
print(cm)

cms = confusion_matrix_to_string(
    p_cm=cm,
    p_dataset=dataset
)
print(cms)

metrics = confusion_matrix_metrics(
    p_cm=cm,
    p_dataset=dataset
)
print(metrics)

performance = get_all_model_performance_metrics_for_all_classes(
    p_metrics=metrics,
    p_dataset=dataset
)

# print(performance)
present_performance_metrics(performance)