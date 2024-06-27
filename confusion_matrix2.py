"""
Artur Marques, 2024
A set of tools for creating confusion matrices
Includes tools for metrics (accuracy, precision, recall and f1-score), with limited explainability regarding the calculations

Example usages are on a different file

"""

def count_times_when_at_the_same_index_of_actual_and_preds_cols_the_stated_values_are_found(
    p_some_actual_class_value, # usually a single string, representing a class name
    p_some_prediction_class_value, # usually a single string, representing a class name
    p_actual:list, # usually the targets column of a dataset, represented as a list
    p_preds:list # usually a list, paralell to p_actual, with the predictions a model achieved for samples whose actual values are at p_actual
):
    count = 0
    # walk across the entire dataset
    for idx in range(len(p_actual)):
        # the idea is to count how many times some class X is being predicted as X in the predictions

        # so "is the current value class X"?
        # only check for matches for the slots in the actual values collection whose value is p_some_actual_value
        value_worth_checking = p_actual[idx] == p_some_actual_class_value

        if value_worth_checking:
            # if the prediction, in the predictions collection, matches the value p_some_prediction
            # then it is a "match"
            # because, at the same index of the actual and prediction collections, one finds the wanted actual value and the wanted predicted value
            match = p_some_prediction_class_value == p_preds[idx]
            if match: count += 1
        # if
    # for
    return count
# def count_times_when_at_the_same_index_of_actual_and_preds_cols_the_stated_values_are_found

#-----------------------------------------------------------------------------------------------------------------
def confusion_matrix(
    p_ds, # the dataset
    p_preds # the corresponding predicitions, by some model
):
    ret = dict()

    # labels = enumerate(sorted(set(dataset))) # not reusable
    labels = list(enumerate(sorted(set(p_ds))))
    for label in labels:
        print(label)
    # for

    # the following 2 iterators (2 fors) will build a square matrix of counts
    # so, we are NOT iterating over an existing matrix, but building a new one, by doing the proper counts

    # build/iterate over all the rows (all the actual values)
    for idx_row, row_label in labels:
        # build/iterate over all the columns (all the predictions)
        for idx_col, col_label in labels:
            current_tuple_idxs = (idx_row, idx_col)
            current_actual_vs_prediction_labels = (row_label, col_label)
            correct_or_wrong = "correctly" if row_label == col_label else "wrongly"

            count = count_times_when_at_the_same_index_of_actual_and_preds_cols_the_stated_values_are_found(
                row_label,
                col_label,
                p_ds,
                p_preds
            )

            msg = f"@{current_tuple_idxs} is {count} = number of times the actual value of {row_label} is {correct_or_wrong} predicted as {col_label}"
            print(msg)

            ret[current_tuple_idxs] = count
        # for
    # for
    return ret
# def confusion_matrix

#-----------------------------------------------------------------------------------------------------------------
def confusion_matrix_to_string(
    p_cm:dict,  # the confusion matrix, as a dictionary, where keys are tuples representing (row, col) addresses and values are the countings
    p_dataset
):
    ret = ""
    # set eliminantes repetitions
    # enumerate gets an iterator (usable only once)
    # builds a list from the iterator (reusable)
    list_of_idx_label_tuples = list(enumerate(sorted(set(p_dataset))))

    headings = "\t"
    for idx, heading in list_of_idx_label_tuples:
        headings+=f"{heading}\t"
    # for
    headings+="\n"

    rows = ""
    for idx_row, label_row in list_of_idx_label_tuples:
        row = f"{label_row}\t"
        for idx_col, label_col in list_of_idx_label_tuples:
            current_matrix_key = (idx_row, idx_col)
            row+=f"{p_cm[current_matrix_key]}\t"
        # for
        row+="\n"
        rows+=row
    # for

    ret = headings + rows

    return ret
# def confusion_matrix_to_string

#-----------------------------------------------------------------------------------------------------------------
"""
False Positives (FP) for a given class c
refer to the instances where the model incorrectly predicts the class c 
when the actual class is something else. 

In other words, these are the cases where other classes are misclassified as class c.

To find the False Positives for each class from the confusion matrix, 
sum the counts of all non-diagonal elements in the COLUMN corresponding to that class.

"""
def confusion_matrix_get_FPs_per_col(
    p_cm:dict, # keys are tuples representing (row, col) addresses ; values are the confusion matrix numbers at the cells
    p_dataset
):
    FPs = dict()
    list_of_idx_label_tuples = list(enumerate(sorted(set(p_dataset))))

    classes = list(sorted(set(p_dataset)))

    how_many_different_classes = len(set(p_dataset))

    for col_id in range(how_many_different_classes):
        FP_KEY = f"FP ({classes[col_id]})"
        FPs[FP_KEY] = 0
        for row_id in range(how_many_different_classes):
            current_address = (row_id, col_id)
            is_diagonal_cell = row_id == col_id
            if not is_diagonal_cell:
                cell_value = p_cm[current_address]
                FPs[FP_KEY] += cell_value

                actual = classes[row_id]
                predicted = classes[col_id]
                FP_DETAIL_KEY = f"FP ({actual}->{predicted})"
                if FP_DETAIL_KEY in FPs.keys(): # existing key
                    FPs[FP_DETAIL_KEY]+=cell_value
                else: # new key
                    FPs[FP_DETAIL_KEY] = cell_value
                # if
            # if
        # for
    # for

    return FPs
# def confusion_matrix_get_FPs_per_col

#-----------------------------------------------------------------------------------------------------------------
"""
False Negatives (FN) for a given class c
refer to the instances where the actual class is c, 
but the model predicts a different class. 

In other words, these are the cases where the model fails to identify the instances of class c.

To find the False Negatives for each class from the confusion matrix, 
sum the counts of all non-diagonal elements in the ROW corresponding to that class.

"""
def confusion_matrix_get_FNs_per_row(
    p_cm:dict, # keys are tuples representing (row, col) addresses ; values are the confusion matrix numbers at the cells
    p_dataset
):
    FNs = dict()
    list_of_idx_label_tuples = list(enumerate(sorted(set(p_dataset))))

    classes = list(sorted(set(p_dataset)))

    how_many_different_classes = len(set(p_dataset))

    for row_id in range(how_many_different_classes):
        FN_KEY = f"FN ({classes[row_id]})"
        FNs[FN_KEY] = 0
        for col_id in range(how_many_different_classes):
            current_address = (row_id, col_id)
            is_diagonal_cell = row_id == col_id
            if not is_diagonal_cell:
                cell_value = p_cm[current_address]
                FNs[FN_KEY] += cell_value

                actual = classes[row_id]
                predicted = classes[col_id]
                FN_DETAIL_KEY = f"FN ({actual}->{predicted})"
                if FN_DETAIL_KEY in FNs.keys(): # existing key
                    FNs[FN_DETAIL_KEY]+=cell_value
                else: # new key
                    FNs[FN_DETAIL_KEY] = cell_value
                # if
            # if
        # for
    # for

    return FNs
# def confusion_matrix_get_FNs_per_row

#-----------------------------------------------------------------------------------------------------------------
def confusion_matrix_get_TPs_from_diagonal(
    p_cm:dict, # keys are tuples representing (row, col) addresses ; values are the confusion matrix numbers at the cells
    p_dataset
):
    TPs = dict()

    list_of_idx_label_tuples = list(enumerate(sorted(set(p_dataset))))

    # compute the "true positives" per class (TP),
    # from the values in the diagonal
    for idx_row, label_row in list_of_idx_label_tuples:
        for idx_col, label_col in list_of_idx_label_tuples:
            current_matrix_key = (idx_row, idx_col)
            current_matrix_value = p_cm[current_matrix_key]

            is_this_cell_on_the_diagonal = idx_row == idx_col
            if is_this_cell_on_the_diagonal:
                TP_KEY = f"TP ({label_row})"
                TPs[TP_KEY] = current_matrix_value
            # if
        # for
    # for

    return TPs
# def confusion_matrix_get_TPs_from_diagonal

#-----------------------------------------------------------------------------------------------------------------
"""
To compute the True Negatives (TNs) for each class in a confusion matrix, 
you need to identify the instances where the model correctly predicts that a sample does NOT belong to a particular class. 

In other words, for a given class c, 
TNs are the counts of all instances that
are neither predicted as c nor actually belong to c.

So: "NOT actual c" *and* "NOT predicted as c"

"""
def confusion_matrix_get_TNs(p_cm: dict, p_dataset):
    """
    Calculate True Negatives (TNs) for each class in the confusion matrix.

    :param p_cm: Dictionary representing the confusion matrix with keys as tuples (row, col) and values as counts.
    :param p_dataset: List of actual class labels present in the dataset.
    :return: Dictionary with TN values for each class.
    """
    TNs = dict()
    classes = list(sorted(set(p_dataset)))
    how_many_different_classes = len(classes)

    for class_id in range(how_many_different_classes):
        TN_KEY = f"TN ({classes[class_id]})"
        TNs[TN_KEY] = 0
        for row_id in range(how_many_different_classes):
            for col_id in range(how_many_different_classes):
                # Check if the cell is not in the row or column of the target class
                if row_id != class_id and col_id != class_id:
                    current_address = (row_id, col_id)
                    cell_value = p_cm.get(current_address, 0)  # Use .get to handle missing keys
                    TNs[TN_KEY] += cell_value
                # if
            # for
        # for
    # for

    return TNs
# def confusion_matrix_get_TNs

#-----------------------------------------------------------------------------------------------------------------
def confusion_matrix_accuracy_from_metrics_for_class(
    p_metrics:dict,
    p_class:str
):
    """
    Ac = Accuracy for class c
    (TPc + TNc) / (TPc + TNc + FPc + FNc)
    """
    key_for_TP_for_class = f"TP ({p_class})"
    key_for_TN_for_class = f"TN ({p_class})"
    key_for_FP_for_class = f"FP ({p_class})"
    key_for_FN_for_class = f"FN ({p_class})"
    TP = p_metrics[key_for_TP_for_class]
    TN = p_metrics[key_for_TN_for_class]
    FP = p_metrics[key_for_FP_for_class]
    FN = p_metrics[key_for_FN_for_class]

    accuracy = (TP + TN) / (TP + TN + FP + FN)

    explanation = "accuracy = (TP + TN) / (TP + TN + FP + FN)"
    explanation += "\n"
    explanation+=f"accuracy = ( {TP} + {TN} ) / ( {TP} + {TN} + {FP} + {FN} )"
    explanation += "\n"
    explanation+=f"accuracy = { TP + TN } / { TP + TN + FP + FN }"
    explanation += "\n"
    explanation += f"accuracy = {accuracy}"
    explanation += "\n"

    return accuracy, explanation
# def confusion_matrix_accuracy_from_metrics_for_class

#-----------------------------------------------------------------------------------------------------------------
def confusion_matrix_precision_from_metrics_for_class(
    p_metrics:dict,
    p_class:str
):
    """
    Pc = Precision for class c
    TPc / (TPc + FPc)
    """
    key_for_TP_for_class = f"TP ({p_class})"
    key_for_TN_for_class = f"TN ({p_class})"
    key_for_FP_for_class = f"FP ({p_class})"
    key_for_FN_for_class = f"FN ({p_class})"
    TP = p_metrics[key_for_TP_for_class]
    TN = p_metrics[key_for_TN_for_class]
    FP = p_metrics[key_for_FP_for_class]
    FN = p_metrics[key_for_FN_for_class]

    precision = TP / (TP + FP)

    explanation = "precision = TP / (TP + FP)"
    explanation += "\n"
    explanation += f"precision = {TP} / ( {TP} + {FP} )"
    explanation += "\n"
    explanation += f"precision = {TP} / {TP + FP}"
    explanation += "\n"
    explanation += f"precision = {precision}"
    explanation += "\n"

    return precision, explanation
# def confusion_matrix_precision_from_metrics_for_class

#-----------------------------------------------------------------------------------------------------------------
def confusion_matrix_recall_from_metrics_for_class(
    p_metrics:dict,
    p_class:str
):
    """
    Rc = Recall for class c
    TPc / (TPc + FNc)
    """
    key_for_TP_for_class = f"TP ({p_class})"
    key_for_TN_for_class = f"TN ({p_class})"
    key_for_FP_for_class = f"FP ({p_class})"
    key_for_FN_for_class = f"FN ({p_class})"
    TP = p_metrics[key_for_TP_for_class]
    TN = p_metrics[key_for_TN_for_class]
    FP = p_metrics[key_for_FP_for_class]
    FN = p_metrics[key_for_FN_for_class]

    recall = TP / (TP + FN)

    explanation = "recall = TP / (TP + FN)"
    explanation += "\n"
    explanation += f"recall = {TP} / ( {TP} + {FN} )"
    explanation += "\n"
    explanation += f"recall = {TP} / {TP + FN}"
    explanation += "\n"
    explanation += f"recall = {recall}"
    explanation += "\n"

    return recall, explanation
# def confusion_matrix_recall_from_metrics_for_class

#-----------------------------------------------------------------------------------------------------------------
def confusion_matrix_f1_score_from_metrics_for_class(
    p_metrics:dict,
    p_class:str
):
    """
    F1-score for class c
    2 * (Pc * Rc) / (Pc + Rc)
    """
    precision, exp_precision = confusion_matrix_precision_from_metrics_for_class(p_metrics, p_class)
    recall, exp_recall = confusion_matrix_recall_from_metrics_for_class(p_metrics, p_class)

    num = precision * recall
    den = precision + recall

    f1 = num / den
    f1_score = 2*f1

    explanation = "f1_score = 2 * ( (precision * recall) / (precision + recall) )"
    explanation += "\n"
    explanation += f"f1_score = 2 * ( {precision} * {recall} ) / ( {precision} + {recall} )"
    explanation += "\n"
    explanation += f"f1_score = 2 * ( {precision * recall} ) / ( {precision + recall} )"
    explanation += "\n"
    explanation += f"f1_score = 2 * {f1}"
    explanation += "\n"
    explanation += f"f1_score = {f1_score}"
    explanation += "\n"

    return f1_score, explanation
# def confusion_matrix_f1_score_from_metrics_for_class

#-----------------------------------------------------------------------------------------------------------------
def get_all_model_performance_metrics_for_all_classes(
    p_metrics:dict, # keys are tuples representing (row, col) addresses ; values are the confusion matrix numbers at the cells
    p_dataset
):
    all_model_performance_metrics = dict()

    classes = list(sorted(set(p_dataset)))
    for c in classes:
        current_metric_key = f"ACCURACY ({c})"
        current_exp_metric_key = f"EXP_ACCURACY ({c})"
        accuracy, exp_accuracry = confusion_matrix_accuracy_from_metrics_for_class(p_metrics, c)
        all_model_performance_metrics[current_metric_key] = accuracy
        all_model_performance_metrics[current_exp_metric_key] = exp_accuracry

        current_metric_key = f"PRECISION ({c})"
        current_exp_metric_key = f"EXP_PRECISION ({c})"
        precision, exp_precision = confusion_matrix_precision_from_metrics_for_class(p_metrics, c)
        all_model_performance_metrics[current_metric_key] = precision
        all_model_performance_metrics[current_exp_metric_key] = exp_precision

        current_metric_key = f"RECALL ({c})"
        current_exp_metric_key = f"EXP_RECALL ({c})"
        recall, exp_recall = confusion_matrix_recall_from_metrics_for_class(p_metrics, c)
        all_model_performance_metrics[current_metric_key] = recall
        all_model_performance_metrics[current_exp_metric_key] = exp_recall

        current_metric_key = f"F1-SCORE ({c})"
        current_exp_metric_key = f"EXP_F1-SCORE ({c})"
        f1_score, exp_f1_score = confusion_matrix_f1_score_from_metrics_for_class(p_metrics, c)
        all_model_performance_metrics[current_metric_key] = f1_score
        all_model_performance_metrics[current_exp_metric_key] = exp_f1_score
    # for

    return all_model_performance_metrics
# def get_all_model_performance_metrics_for_all_classes
#-----------------------------------------------------------------------------------------------------------------
def confusion_matrix_metrics(
    p_cm:dict, # keys are tuples representing (row, col) addresses ; values are the confusion matrix numbers at the cells
    p_dataset
):
    metrics = dict()

    tps = confusion_matrix_get_TPs_from_diagonal(p_cm, p_dataset)
    metrics.update(tps)

    fps = confusion_matrix_get_FPs_per_col(p_cm, p_dataset)
    metrics.update(fps)

    fns = confusion_matrix_get_FNs_per_row(p_cm, p_dataset)
    metrics.update(fns)

    tns = confusion_matrix_get_TNs(p_cm, p_dataset)
    metrics.update(tns)

    return metrics

    """
    Ac = Accuracy for class c
    (TPc + TNc) / (TPc + TNc + FPc + FNc)
    Pc = Precision for class c
    TPc / (TPc + FPc)
    Rc = Recall for class c
    TPc / (TPc + FNc)
    F1-score for class c
    2 * (Pc * Rc) / (Pc + Rc)

    """

# def confusion_matrix_metrics

def present_performance_metrics(
    p_performance_metrics:dict
):
    for k,v in p_performance_metrics.items():
        print(k)
        print(v)
        print("-"*20)
    # for
# def present_performance_metrics