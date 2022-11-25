import math
import numpy as np


def calculate_entropy(data_set: np.ndarray, attr_index: int, outcome_index: int):
    """
        Calculates the entropy of given attribute from the given data set
        The outcome index is the place of the outcome which used for finding out the outcome of the datas in the
        data set

        Creates a dictionary "count" before calculating the numbers of "Yes" and "No" results. It is used later in the
        implementation of prune methods
        Arguments:
            data_set [dict]: the dictionary of the entropy results of an attribute.
                                Used as a dictionary for different values in the column
            attr_index [int]: index of the attribute whose entropy is calculated
            outcome_index [int]: index of the result attribute, "Yes" and "No"s in this case
        Returns:
            entropy_result [dict]: The entropy value of the given attribute and its values
            count [dict]: The count of "Yes" and "No" results of the given attribute and its values
    """

    # Creates the count dictionary with different values of the given attribute
    count = {"general": (0, 0)}
    for data_index in range(len(data_set)):
        count[data_set[data_index][attr_index]] = (0, 0)

    # Calculates the number of "Yes" and "No" at the index of outcome for the attribute values
    for index in range(len(data_set)):
        data = data_set[index][attr_index]
        if data_set[index][outcome_index] == "Yes":
            count[data] = (count[data][0] + 1, count[data][1])
            count["general"] = (count["general"][0] + 1, count["general"][1])
        else:
            count[data] = (count[data][0], count[data][1] + 1)
            count["general"] = (count["general"][0], count["general"][1] + 1)

    # Entropy calculation
    entropy_result = {}
    for key in count.keys():
        data = count[key]
        positive = data[0] / (data[0] + data[1])
        negative = data[1] / (data[0] + data[1])

        if not positive or not negative:
            entropy_result[key] = 0
            continue
        entropy_result[key] = (-positive * math.log2(positive)
                               ) - (negative * math.log2(negative))

    return entropy_result, count


def calculate_gain(entropy_result: dict, count: dict) -> float:
    """
        From a given entropy value and number of counts of the attribute calculates the information gain

        Arguments:
            entropy_result [dict]: the dictionary of the entropy results of an attribute.
                                Used as a dictionary for different values in the column
            count [dict]: the count of "Yes", "No" results of an attribute and its values
        Returns:
            result [float]: The information gain of the given attribute
    """
    result = entropy_result["general"]
    general_total = count["general"][0] + count["general"][1]
    for key in entropy_result.keys():
        if key == "general":
            continue
        attr_total = count[key][0] + count[key][1]
        result -= (attr_total / general_total) * entropy_result[key]

    return result


def find_max_gain(data_set, attr_list, counts):
    """
        Finds the attribute from the given attribute list which has the most information gain
        Arguments:
            data_set [np.array]: the data set of the model
            attr_list [list]: the attributes of the selected node
            counts [dict]: the number of "Yes" and "No" of the attribute and its values
        Returns:
            selected_attr [str]: The name of the selected attribute
            max_gain [int]: The value of the information gain of the selected attribute
    """
    selected_attr = None
    max_gain = 0
    for i in range(len(attr_list)):
        # Skip if the index is "Attrition", outcome attribute
        if i == 1:
            continue
        # Calculate the entropy, count and then gain
        entropy, count = calculate_entropy(data_set, i, 1)
        gain = calculate_gain(entropy, count)
        counts[attr_list[i]] = count
        if gain > max_gain:
            max_gain = gain
            selected_attr = attr_list[i]
    return selected_attr, max_gain


def filter_data_set(data_set: np.ndarray, attr_index: int, attr_value: str) -> list:
    """
        Filters the data set depending on the attribute value at the attribute index
        Arguments:
            data_set [np.array]: the data set of the model
            attr_index [int]: The attribute that is used while filtering the data set
            attr_value [int]: The value of the given attribute in data set
        Returns:
            filtered_result [float]: The filtered data set depending on the attribute and its value
    """
    filtered_result = []
    for data in data_set:
        if data[attr_index] == attr_value:
            filtered_result.append(data)

    return filtered_result


def _create_bins(lower_bound: int, higher_bound: int, interval_num: int) -> list:
    """
        Takes min, max values of the columns and interval number
        Creates number of intervals depending on the max, min values
        Arguments:
            lower_bound [int]: Min. value in the column
            higher_bound [int]: Max. value in the column
            interval_num [int]: Number of intervals
        Returns:
            bins [list]: List of the tuples of the intervals
    """
    bins = []
    interval_size = (higher_bound - lower_bound) / interval_num
    for num in range(interval_num):
        bins.append((lower_bound + (interval_size * num), (lower_bound + (interval_size * (num + 1)))))

    return bins


def discrete_data(data_set: np.ndarray, cont_indexes: list) -> None:
    """
        Discrete the continuous attribute in the given data set
        First creates intervals depending on the max, min values of the given column
        Then matches the values in the column with labels

        Arguments:
            data_set [np.array]: the given data set
            cont_indexes [list]: the indexes of the cont. attributes in the given data set
        Returns:
            accuracy [float]: The ratio between correct predictions and number of predictions
    """
    labels = ["Low", "Mid", "High"]
    for index in cont_indexes:
        data_array = data_set[:, index]
        max_val = np.max(data_array)
        min_val = np.min(data_array)
        # Calculates the intervals
        bins = _create_bins(min_val, max_val, len(labels))

        for data_num in range(len(data_array)):
            for interval_num in range(len(bins)):
                # Matches the values in the column with the intervals
                if interval_num == len(bins) - 1:
                    if bins[interval_num][1] >= data_array[data_num] >= bins[interval_num][0]:
                        data_array[data_num] = labels[interval_num]
                        break
                else:
                    if bins[interval_num][1] > data_array[data_num] >= bins[interval_num][0]:
                        data_array[data_num] = labels[interval_num]
                        break


def calculate_accuracy(test_set: np.ndarray, predictions: list) -> float:
    """
        Calculates accuracy with matching test_set and predictions.
        Arguments:
            test_set [np.array]: the test set of the model
            predictions [list]: the predictions of the test set from the tree
        Returns:
            accuracy [float]: The ratio between correct predictions and number of predictions
    """
    TP_TN = 0
    for test_index in range(len(test_set)):
        if test_set[test_index][1] == predictions[test_index]:
            TP_TN += 1
    accuracy = TP_TN / len(predictions)
    return accuracy


def find_miss_classification(test_set: np.ndarray, predictions: list) -> list:
    """
        Finds the classification errors from given test and predictions
        Arguments:
            test_set [np.array]: the test set of the model
            predictions [list]: the predictions of the test set from the tree
        Returns:
            miss_class [list]: Indexes of the classification errors in the test_set/predictions
    """
    miss_class = []
    for index in range(len(test_set)):
        if test_set[index][1] != predictions[index]:
            miss_class.append((test_set, predictions))
    return miss_class


def calculate_recall(test_set: np.ndarray, predictions: list) -> float:
    """
        Calculates Precision metric score of the model with matching (T)rue (P)ositive and
        (F)alse (N)egative values
        Arguments:
            test_set [np.array]: the test set of the model
            predictions [list]: the predictions of the test set from the tree
        Returns:
            recall [float]: TP / (TP+FN)
    """
    TP = 0
    FN = 0
    for i in range(len(test_set)):
        if test_set[i][1] == "Yes" and predictions[i] == "Yes":
            TP += 1
        elif test_set[i][1] == "Yes" and predictions[i] == "No":
            FN += 1
    return TP/(TP+FN)


def calculate_precision(test_set: np.ndarray, predictions: list) -> float:
    """
        Calculates Precision metric score of the model with matching (T)rue (P)ositive and
        (F)alse (P)ositive values
        Arguments:
            test_set [np.ndarray]: the test set of the model
            predictions [list]: the predictions of the test set from the tree
        Returns:
            precision [float]: TP / (TP+FP)
    """
    TP = 0
    FP = 0
    for i in range(len(test_set)):
        if test_set[i][1] == "Yes" and predictions[i] == "Yes":
            TP += 1
        elif predictions[i] == "Yes" and test_set[i][1] == "No":
            FP += 1
    return TP/(TP+FP)


def calculate_f1_score(recall: float, precision: float) -> float:
    """
        Calculates F1 metric score of the model

        Arguments:
            recall [flot]: recall value of the model
            precision [float]: precision value of the model
        Returns:
            f1_score [float]: 2 * (recall * precision) / (recall + precision)
        """
    return 2 * (recall * precision) / (recall + precision)


def shuffle_split(data_set: np.ndarray, k: int) -> np.array:
    """
    Shuffles and splits the data into k folds

    Arguments:
        data_set [np.ndarray]: the data set from the given file
        k [int]: number of folds
    Returns:
        k_fold_data [np.array]: k amount of lists, which include (number_of_data/k) elements
    """
    np.random.shuffle(data_set)
    split_set = np.array_split(data_set, k)
    return split_set

