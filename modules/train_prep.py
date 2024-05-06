from modules.constants import *  # Import constants such as decay_weight, base_lr, epoch_decay.

def exp_lr_scheduler(optimizer, epoch, init_lr=base_lr, lr_decay_epoch=epoch_decay):
    """
    Adjust the learning rate of the optimizer based on epoch number and decay settings.
    Args:
    - optimizer: PyTorch optimizer whose learning rate needs adjustment.
    - epoch: Current epoch number during training.
    - init_lr: Initial learning rate (default taken from base_lr constant).
    - lr_decay_epoch: Frequency of epochs after which to apply decay (default from epoch_decay constant).

    Returns:
    - optimizer: Optimizer with updated learning rate.
    """
    # Calculate the new learning rate by exponential decay formula.
    lr = init_lr * (decay_weight**(epoch // lr_decay_epoch))

    # Log the updated learning rate at the start of a decay epoch.
    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    # Apply the calculated learning rate to all parameter groups in the optimizer.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def train_model(model, criterion, optimizer, lr_scheduler, dset_loaders, dset_sizes, num_epochs=10, device=None):
    """
    Train and evaluate a model with the given data loaders, criterion, optimizer, and scheduler.
    Args:
    - model: PyTorch model to be trained.
    - criterion: Loss function.
    - optimizer: Optimizer.
    - lr_scheduler: Learning rate scheduler function.
    - dset_loaders: Dictionary of data loaders for 'train' and 'val' sets.
    - dset_sizes: Dictionary containing sizes of 'train' and 'val' datasets.
    - num_epochs: Number of epochs to train (default is 10).
    - device: Device on which to train the model ('cuda', 'mps', 'cpu').

    Returns:
    - best_model: The model with the highest accuracy on the validation set.
    - accuracies: Dictionary storing the accuracy history for both training and validation phases.
    - losses: Dictionary storing the loss history for both training and validation phases.
    - best_val_preds: Predictions from the best model on the validation set.
    - best_val_labels: Ground truth labels for the best validation predictions.
    """
    since = time.time()  # Record the start time for evaluating training duration.

    # Set the device for training based on availability and user specification.
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    
    model.to(device)  # Move the model to the specified device.

    best_model = copy.deepcopy(model)  # Initialize the best model as a deep copy of the original model.
    best_acc = 0.0  # Initial best accuracy.

    # Initialize dictionaries to store accuracies and losses for both training and validation phases.
    accuracies = {'train': [], 'val': []}
    losses = {'train': [], 'val': []}

    epoch_bar = tqdm(total=num_epochs, position=1)  # Progress bar for epochs.

    # Begin training over specified number of epochs.
    for epoch in range(num_epochs):
        print('-' * 10)
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Iterate over each phase: training and validation.
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch)  # Adjust learning rate based on the scheduler.
                model.train()  # Set model to training mode.
            else:
                val_preds = []  # List to store predictions for validation phase.
                val_labels = []  # List to store labels for validation phase.
                model.eval()   # Set model to evaluation mode.

            running_loss = 0.0  # Sum of losses for the current phase.
            running_corrects = 0  # Count of correct predictions.

            print(f'\nIn {phase} phase:')
            batch_bar = tqdm(total=dset_sizes[phase], position=0)  # Progress bar for current phase batches.
            
            # Iterate over data in batches.
            for inputs, labels in dset_loaders[phase]:
                inputs = inputs.float().to(device)  # Move inputs to device and ensure data type is float.
                labels = labels.long().to(device)  # Move labels to device and ensure data type is long.

                optimizer.zero_grad()  # Zero the gradients.

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  # Get model outputs.
                    _, preds = torch.max(outputs, 1)  # Get the predicted classes.

                    if phase == 'val':
                        val_labels.extend(labels)
                        val_preds.extend(preds)

                    loss = criterion(outputs, labels)  # Calculate loss.

                    # Backward pass and optimize only if in training phase.
                    if phase == 'train':
                        loss.backward()  # Compute gradients.
                        optimizer.step()  # Update parameters.

                # Accumulate statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                batch_bar.update(b_size)  # Update batch progress bar.
            
            batch_bar.close()

            # Calculate average loss and accuracy for the current phase.
            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]
            accuracies[phase].append(epoch_acc.cpu().numpy())  # Store accuracy.
            losses[phase].append(epoch_loss)  # Store loss.

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Check if the current model is the best model; if so, update the best model.
            if phase == 'val' and epoch_acc > best_acc:
                best_val_labels = [label.cpu().numpy() for label in val_labels]
                best_val_preds = [pred.cpu().numpy() for pred in val_preds]
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)
                print('New best accuracy = {:.4f}'.format(best_acc))
        
        epoch_bar.update(1)  # Update the epoch progress bar.
    
    epoch_bar.close()

    # Print total training time.
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))

    # Return the best model and training/validation histories.
    return best_model, accuracies, losses, best_val_preds, best_val_labels

# Usage:
# model, criterion, optimizer, and lr_scheduler need to be defined
# dset_loaders should be a dictionary with 'train' and 'val' DataLoader objects
# dset_sizes should be a dictionary with the sizes of these datasets
# device can be set manually or will be set to CUDA if available, and MPS if on compatible macOS devices
