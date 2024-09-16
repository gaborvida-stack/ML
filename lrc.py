try:
    from sklearn.datasets import load_iris, load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
except ImportError as err:
    print("Error occurred: {}".format(err))


def load_data(dataset_name):
    if dataset_name == 'iris':
        return load_iris()
    elif dataset_name == 'wine':
        return load_wine()
    else:
        raise ValueError("Unknown dataset: {}".format(dataset_name))

def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model, scaler

def evaluate_model(model, scaler, X_test, y_test):
    X_test = scaler.transform(X_test)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    return accuracy, report, matrix


def main():
    print("Choose a dataset (iris/wine):")
    dataset_name = input().strip().lower()
    data = load_data(dataset_name)
    
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    
    model, scaler = train_model(X_train, y_train)
    
    accuracy, report, matrix = evaluate_model(model, scaler, X_test, y_test)
    
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", matrix)

if __name__ == "__main__":
    main()