from modules.constants import *

def exp_lr_scheduler(optimizer, epoch, init_lr=base_lr, lr_decay_epoch=epoch_decay):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""
    lr = init_lr * (decay_weight**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def train_model(model, criterion, optimizer, lr_scheduler, dset_loaders, dset_sizes, num_epochs=10, device=None):
    since = time.time()

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    
    model.to(device)

    best_model = copy.deepcopy(model)
    best_acc = 0.0

    accuracies = {'train': [], 'val': []}
    losses = {'train': [], 'val': []}

    epoch_bar = tqdm(total=num_epochs, position=1)

    for epoch in range(num_epochs):
        print('-' * 10)
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch)
                model.train()  # Set model to training mode
            else:
                val_preds = []
                val_labels = []
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            print(f'\nIn {phase} phase:')
            batch_bar = tqdm(total=dset_sizes[phase], position=0)
            for inputs, labels in dset_loaders[phase]:

                inputs = inputs.float().to(device)
                labels = labels.long().to(device)

                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'val':
                        val_labels.append(labels)
                        val_preds.append(preds)

                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                batch_bar.update(b_size)
            
            batch_bar.close()

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]
            accuracies[phase].append(epoch_acc)
            losses[phase].append(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_val_labels = val_labels
                best_val_preds = val_preds
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)
                print('New best accuracy = {:.4f}'.format(best_acc))
        
        epoch_bar.update(1)
    
    epoch_bar.close()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))

    return best_model, accuracies, losses, best_val_preds, best_val_labels

# Usage:
# model, criterion, optimizer, and lr_scheduler need to be defined
# dset_loaders should be a dictionary with 'train' and 'val' DataLoader objects
# dset_sizes should be a dictionary with the sizes of these datasets
# device can be set manually or will be set to CUDA if available, and MPS if on compatible macOS devices