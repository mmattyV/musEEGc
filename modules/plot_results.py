from modules.constants import *

def plot_training_history(accuracies, losses, model):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    for phase in ['train', 'val']:
        plt.plot(accuracies[phase], label=f'{phase} accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    for phase in ['train', 'val']:
        plt.plot(losses[phase], label=f'{phase} loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(f'../plots/{model}_loss_plot.png')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, model):
    y_true = [y.cpu() for y in y_true]
    y_pred = [y.cpu() for y in y_pred]

    flattened_y_true = [item for tensor in y_true for item in tensor.tolist()]
    flattened_y_pred = [item for tensor in y_pred for item in tensor.tolist()]

    unique_labels = np.unique(np.concatenate((flattened_y_true, flattened_y_pred)))
    label_names = [f"Class {label}" for label in unique_labels]

    # Generate the confusion matrix
    cm = confusion_matrix(flattened_y_true, flattened_y_pred, labels=unique_labels)

    # Plotting the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {model}')
    plt.savefig(f'../plots/{model}_confusion_mat.png')
    plt.show()
    