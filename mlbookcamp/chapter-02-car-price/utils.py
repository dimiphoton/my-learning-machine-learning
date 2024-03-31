import joblib
import matplotlib.pyplot as plt
import seaborn as sns


def save_model(model, file_name):
    joblib.dump(model, file_name)

def load_model(file_name):
    model = joblib.load(file_name)
    return model

def plot_predictions(predictions, true_values):
    plt.figure(figsize=(6, 4))
    sns.histplot(true_values, label='target', color='#222222', alpha=0.6, bins=40)
    sns.histplot(predictions, label='prediction', color='#aaaaaa', alpha=0.8, bins=40)
    plt.legend()
    plt.ylabel('Frequency')
    plt.xlabel('Log(Price + 1)')
    plt.title('Predictions vs actual distribution')
    plt.show()