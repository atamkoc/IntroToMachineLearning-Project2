import pandas as pd
import numpy as np
import time
import utility
from DecisionTree import DecisionTree

# Reading the data from given file
df = pd.read_csv("data.xls")

# Drop EmployeeNumber, irrelevant attribute
df = df.drop(['EmployeeNumber'], axis=1)
df_array = np.array(df)
df_columns = list(df.columns)

# Extracting the continuous attributes
cont_labels = []
for i in range(len(df_array[0])):
    if type(df_array[0][i]) == int:
        cont_labels.append(i)

utility.discrete_data(df_array, cont_labels)

# Decision Tree Classification with 5-Fold
k = 5
k_fold_data = utility.shuffle_split(df_array, k)
for data_num in range(len(k_fold_data)):
    # Split the train data 0.8, test data 0.2 from folds
    train_data = []
    for train_index in range(len(k_fold_data)):
        if data_num == train_index:
            continue
        train_data.extend(k_fold_data[train_index])
    test_data = k_fold_data[data_num]

    # Tree creation
    d_tree = DecisionTree()
    d_tree.create_tree(train_data, df_columns)

    start_time = time.time()

    predictions = []
    for test in test_data:
        predictions.append(d_tree.predict(test))

    end_time = time.time()

    accuracy = utility.calculate_accuracy(test_data, predictions)
    precision = utility.calculate_precision(test_data, predictions)
    recall = utility.calculate_recall(test_data, predictions)
    f1_score = utility.calculate_f1_score(recall, precision)

    print(f"##### Fold {data_num + 1} Statistics #####")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}\n")
    print(f"The computation time is {end_time - start_time} seconds\n\n")


# Before Pruning The Tree
training_data = []
for data_num in range(len(k_fold_data)-2):
    training_data.extend(k_fold_data[data_num])
validation_data = k_fold_data[-1]
test_data = k_fold_data[-2]

d_tree = DecisionTree()
d_tree.create_tree(training_data, df_columns)

start_time = time.time()

predictions = []
for test in test_data:
    predictions.append(d_tree.predict(test))

end_time = time.time()

accuracy_before = utility.calculate_accuracy(test_data, predictions)
precision_before = utility.calculate_precision(test_data, predictions)
recall_before = utility.calculate_recall(test_data, predictions)
f1_score_before = utility.calculate_f1_score(recall_before, precision_before)

print(f"Accuracy: {accuracy_before}")
print(f"Precision: {precision_before}")
print(f"Recall: {recall_before}")
print(f"F1 Score: {f1_score_before}\n")
print(f"The computation time is {end_time-start_time} seconds\n\n")

# Pruning the Tree
d_tree.find_twigs(d_tree.root)
pruned_nodes = []
while True:
    predictions = []
    for data in validation_data:
        predictions.append(d_tree.predict(data))
    last_accuracy = utility.calculate_accuracy(validation_data, predictions)

    # Deletes the least information gain from the tree but stores it locally in case of reverting the operation
    del_node, parent_node = d_tree.prune()
    pruned_nodes.append(del_node)
    predictions = []
    for data in validation_data:
        predictions.append(d_tree.predict(data))
    current_accuracy = utility.calculate_accuracy(validation_data, predictions)

    if last_accuracy <= current_accuracy:
        # Deletes the node completely, continues to the next step
        del del_node
        continue
    d_tree.revert_prune(del_node, parent_node)
    pruned_nodes.remove(del_node)
    break

# Predictions after the prune and performance metrics
start_time = time.time()

predictions = []
for test in test_data:
    predictions.append(d_tree.predict(test))

end_time = time.time()

accuracy = utility.calculate_accuracy(test_data, predictions)
precision = utility.calculate_precision(test_data, predictions)
recall = utility.calculate_recall(test_data, predictions)
f1_score = utility.calculate_f1_score(recall, precision)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1_score}\n")
print(f"The computation time is {end_time - start_time} seconds\n\n")

# Pruned Nodes
for node in pruned_nodes:
    path = [node]
    temp_node = node
    while temp_node.parent is not None:
        temp_node = temp_node.parent
        path.append(temp_node)
    for index in range(len(path)-1, -1, -1):
        if len(path[index].children) == 1:
            print(f"{path[index].title}) ^ ", end="")
        else:
            if index == 0:
                print(f"({path[index].title} (PRUNED))")
                print(f"- Most common value of ({path[index].title}) is ({path[index].find_most_common()})")
                print(f"- The counts of Yes-No options in order {path[index].count['general']}\n\n")
            else:
                print(f"({path[index].title} -> ", end="")

# Comparison of the metrics
print("\t\t%s\t\t%s" % ("Before", "After"))
print("Accuracy\t%.3f\t\t%.3f" % (accuracy_before, accuracy))
print("Precision\t%.3f\t\t%.3f" % (precision_before, precision))
print("Recall\t\t%.3f\t\t%.3f" % (recall_before, recall))
print("F1 Score\t%.3f\t\t%.3f" % (f1_score_before, f1_score))