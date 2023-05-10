import numpy as np
import pandas as pd
from joblib import dump, load
min_value_income = 0
max_value_income = 0
min_value_credit = 0
max_value_credit = 0


def convert():
    # read the CSV file into a pandas dataframe
    df = pd.read_csv('Final Data Training.csv')

    # select the column you want to modify
    column_name = 'Income'

    # find the minimum and maximum values in the column
    min_val_income = df[column_name].min()
    max_val_income = df[column_name].max()

    # define the breakpoints for the three classes
    class_labels = ['A', 'B', 'C', 'D', 'E']
    num_classes = len(class_labels)
    interval_size = (max_val_income - min_val_income) / num_classes
    breakpoints = [min_val_income + i *
                   interval_size for i in range(num_classes)] + [max_val_income]

    # create a new column to store the class labels
    class_column_name = 'class_column'
    df[class_column_name] = pd.cut(
        df[column_name], bins=breakpoints, labels=class_labels, include_lowest=True)

    # update the original column with the new values
    df[column_name] = df[class_column_name]

    # remove the new column and rename the original column
    df = df.drop(class_column_name, axis=1)
    df = df.rename(columns={column_name: column_name})

    # print the intervals and their corresponding labels
    for i in range(num_classes):
        print(
            f"Interval {i+1}: {breakpoints[i]} - {breakpoints[i+1]}, Label: {class_labels[i]}")

    # select the column you want to modify
    column_name = 'Credit'

    # find the minimum and maximum values in the column
    min_val_credit = df[column_name].min()
    max_val_credit = df[column_name].max()

    # define the breakpoints for the three classes
    class_labels = ['L', 'M', 'N', 'O', 'P']
    num_classes = len(class_labels)
    interval_size = (max_val_credit - min_val_credit) / num_classes
    breakpoints = [min_val_credit + i *
                   interval_size for i in range(num_classes)] + [max_val_credit]

    # create a new column to store the class labels
    class_column_name = 'class_column'
    df[class_column_name] = pd.cut(
        df[column_name], bins=breakpoints, labels=class_labels, include_lowest=True)

    # update the original column with the new values
    df[column_name] = df[class_column_name]

    # remove the new column and rename the original column
    df = df.drop(class_column_name, axis=1)
    df = df.rename(columns={column_name: column_name})

    # print the intervals and their corresponding labels
    for i in range(num_classes):
        print(
            f"Interval {i+1}: {breakpoints[i]} - {breakpoints[i+1]}, Label: {class_labels[i]}")

    # save the modified dataframe back to a new CSV file
    df.to_csv('Final Data Trained.csv', index=False)


def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = -np.sum([(counts[i]/np.sum(counts))*np.log2(counts[i] /
                      np.sum(counts)) for i in range(len(elements))])
    return entropy


def info_gain(data, split_attribute, target_name="Fraud"):
    total_entropy = entropy(data[target_name])

    values, counts = np.unique(data[split_attribute], return_counts=True)
    weighted_entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(
        data[split_attribute] == values[i]).dropna()[target_name]) for i in range(len(values))])

    information_gain = total_entropy - weighted_entropy
    return information_gain


def id3(data, original_data, features, target_attribute_name="Fraud", parent_node_class=None):

    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    elif len(data) == 0:
        return np.unique(original_data[target_attribute_name])[np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])]

    elif len(features) == 0:
        return parent_node_class

    else:
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(
            np.unique(data[target_attribute_name], return_counts=True)[1])]

        item_values = [info_gain(data, feature, target_attribute_name)
                       for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        tree = {best_feature: {}}

        features = [i for i in features if i != best_feature]

        for value in np.unique(data[best_feature]):
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = id3(sub_data, data, features,
                          target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree

        return tree


def print_tree(tree, indent=''):
    if isinstance(tree, dict):
        for key, value in tree.items():
            print(f'{indent}{key}')
            print_tree(value, indent + '  ')
    else:
        print(f'{indent}==> {tree}')


def main():
    convert()
    data = pd.read_csv("Final Data Trained.csv")
    features = data.columns[:-1]
    print(data.to_string())
    tree = id3(data, data, features)

    # Save the classifier to a file
    filename = 'decision_tree.joblib'
    dump(tree, filename)

    print("Decision Tree:", tree)
    print_tree(tree)


if __name__ == '__main__':
    main()
