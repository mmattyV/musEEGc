from modules.constants import *  # Import necessary constants and configuration settings.

def plot_training_history(accuracies, losses, model):
    """
    Plot the training and validation accuracy and loss over epochs for a given model.
    Args:
    - accuracies: Dictionary containing lists of accuracies for 'train' and 'val' phases.
    - losses: Dictionary containing lists of losses for 'train' and 'val' phases.
    - model: String, name of the model, used for naming the saved plot file.
    """
    plt.figure(figsize=(12, 6))  # Set the figure size for the plot

    # Plot accuracy history
    plt.subplot(1, 2, 1)  # Create a subplot (1 row, 2 columns, first plot)
    for phase in ['train', 'val']:
        plt.plot(accuracies[phase], label=f'{phase} accuracy')  # Plot accuracy for each phase
    plt.title('Accuracy over Epochs')  # Set title
    plt.xlabel('Epoch')  # Set the x-axis label
    plt.ylabel('Accuracy')  # Set the y-axis label
    plt.legend()  # Add a legend

    # Plot loss history
    plt.subplot(1, 2, 2)  # Create a subplot (1 row, 2 columns, second plot)
    for phase in ['train', 'val']:
        plt.plot(losses[phase], label=f'{phase} loss')  # Plot loss for each phase
    plt.title('Loss over Epochs')  # Set title
    plt.xlabel('Epoch')  # Set the x-axis label
    plt.ylabel('Loss')  # Set the y-axis label
    plt.legend()  # Add a legend

    plt.savefig(f'../plots/{model}_loss_plot.png')  # Save the figure to a file
    plt.show()  # Display the figure

def plot_confusion_matrix(y_true, y_pred, model):
    """
    Plot a confusion matrix based on the true labels and predictions.
    Args:
    - y_true: List of tensors containing the true labels.
    - y_pred: List of tensors containing the predicted labels.
    - model: String, name of the model, used for naming the saved plot file.
    """

    # Get unique labels present in the data to label the confusion matrix
    unique_labels = np.unique(np.concatenate((y_true, y_pred)))
    label_names = [f"Class {label}" for label in unique_labels]  # Generate readable class names

    # Generate the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    # Plotting the confusion matrix using Matplotlib and sklearn's ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(cmap=plt.cm.Blues)  # Use a blue color map for the display
    plt.title(f'Confusion Matrix for {model}')  # Set title
    plt.savefig(f'../plots/{model}_confusion_mat.png')  # Save the figure to a file
    plt.show()  # Display the figure
