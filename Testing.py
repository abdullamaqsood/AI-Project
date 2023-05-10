import pandas as pd
from joblib import dump, load


def predict(query, tree, default=1.0):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            if key == 'Children':
                query[key] = float(query[key])
            try:
                result = tree[key][query[key]]
            except:
                return default

            if isinstance(result, dict):
                return predict(query, result)
            else:
                return result


def main():
    data = pd.read_csv("Final Data Training.csv")
    features = data.columns[:-1]
    min_value_income = data['Income'].min()
    max_value_income = data['Income'].max()
    min_value_credit = data['Credit'].min()
    max_value_credit = data['Credit'].max()

    # Load the classifier from the file
    filename = 'decision_tree.joblib'
    tree = load(filename)

    # Initialize confusion matrix
    confusion_matrix = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

    fraud_records = []  # Initialize empty list for fraud records
    real_fraud = []
    data = pd.read_csv("Final Data Testing.csv")
    for index, row in data.iterrows():
        query = {}
        for feature in features:
            value = row[feature]
            if feature == 'Income':
                interval = (max_value_income-min_value_income)/5
                value = float(value)
                if value >= min_value_income or value <= max_value_income:
                    sum = min_value_income
                    classname = ['A', 'B', 'C', 'D', 'E']
                    for i in range(5):
                        sum = interval + sum
                        if value <= sum:
                            query[feature] = classname[i]
                            break
                elif value > max_value_income:
                    query[feature] = 'E'
                elif value < min_value_income:
                    query[feature] = 'A'
            elif feature == 'Credit':
                value = float(value)
                interval = (max_value_credit-min_value_credit)/5
                if value >= min_value_credit or value <= max_value_credit:
                    sum = min_value_credit
                    classname = ['L', 'M', 'N', 'O', 'P']
                    for i in range(5):
                        sum = interval + sum
                        if value <= sum:
                            query[feature] = classname[i]
                            break
                elif value > max_value_credit:
                    query[feature] = 'P'
                elif value < min_value_credit:
                    query[feature] = 'L'
            else:
                query[feature] = value

        prediction = predict(query, tree)
        actual = row['Fraud']

        if prediction == 1.0 and actual == 1.0:
            confusion_matrix['TP'] += 1
            real_fraud.append(index+2)
        elif prediction == 0.0 and actual == 0.0:
            confusion_matrix['TN'] += 1
        elif prediction == 1.0 and actual == 0.0:
            confusion_matrix['FP'] += 1
        elif prediction == 0.0 and actual == 1.0:
            confusion_matrix['FN'] += 1

        if prediction == 1.0:
            # append record number to fraud_records list if prediction is fraud
            fraud_records.append(index+2)

    # output the list of fraud record numbers
    print("Fraud Records:", fraud_records)
    print()
    # output the list of actual fraud record numbers
    print("Correct Fraud Records:", real_fraud)
    print()

    # Calculate the accuracy using the confusion matrix
    accuracy = (confusion_matrix['TP'] + confusion_matrix['TN']) / (confusion_matrix['TP'] +
                                                                    confusion_matrix['TN'] + confusion_matrix['FP'] + confusion_matrix['FN'])
    print(f"Accuracy: {accuracy:.2f}")
    print()

    # Print the confusion matrix
    print("Confusion Matrix:", confusion_matrix)


if __name__ == '__main__':
    main()
