import torch
import utils
from pathlib        import Path


saved_dir = Path("../saved")

# for training and finding the best model based on validation loss
def train_eval(setup):
    
    print("="*15, "Training", "="*15)
    
    best_valid_loss = float('inf')

    train_losses, train_accs = [], []
    valid_losses, valid_accs = [], []

    for epoch in range(setup.n_epochs):
        
        train_loss, train_acc = _train(setup.bool_reshape, 
                                       setup.model, 
                                       setup.train_loader, 
                                       setup.optimizer, 
                                       setup.criterion, 
                                       setup.device)
        valid_loss, valid_acc = _eval(setup.bool_reshape, 
                                      setup.model, 
                                      setup.val_loader, 
                                      setup.criterion, 
                                      setup.device)

        # for plotting
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        
        # for saving
        filename = utils.get_filename(setup.config_file, setup.dataset_config_file, setup.seed)

        # save model with best validation loss
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(setup.model.state_dict(), f'{saved_dir}/{filename}.pt')

        print(f'Epoch: {epoch+1:02}')
        print(f'Train Loss: {train_loss:.3f}  |  Train Acc: {train_acc*100:.2f}%')
        print(f'Valid Loss: {valid_loss:.3f}  |  Valid Acc: {valid_acc*100:.2f}%')
        
    return train_losses, train_accs, valid_losses, valid_accs

# for testing
def test(setup):
        
    print("="*15, "Testing ", "="*15)
    # for loading
    filename = utils.get_filename(setup.config_file, setup.dataset_config_file, setup.seed)
    
    setup.model.load_state_dict(torch.load(f'{saved_dir}/{filename}.pt'))
    test_loss, test_acc = _eval(setup.bool_reshape, setup.model, setup.test_loader, setup.criterion, setup.device)
    print(f'Test  Loss: {test_loss:.3f}  |  Test  Acc: {test_acc*100:.2f}%')

# =====internal use============
# for training
def _train(bool_reshape, model, loader, optimizer, criterion, device):
    
    epoch_loss, epoch_acc = 0, 0
    model.train()  # useful for batchnorm and dropout

    for i, (X, Y) in enumerate(loader):

        # GPU
        # images shape is [100, 1, 28, 28] [batch_size, channel, height, width]
        if(bool_reshape):
            X = X.reshape(X.shape[0], -1).to(device)
        else:
            X = X.to(device)
            
        Y = Y.to(device)  # label shape is  [100, ]

        # Forward pass
        Y_hat = model(X)  # one-hot encoded because output_size is 10 from the network

        # Loss and Acc
        loss = criterion(Y_hat, Y)
        acc  = _acc(Y_hat, Y)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc  += acc.item()

    return epoch_loss / len(loader), epoch_acc / len(loader)

# for evaluation
def _eval(bool_reshape, model, loader, criterion, device):

    epoch_loss, epoch_acc = 0, 0
    model.eval()

    with torch.no_grad():
        for i, (X, Y) in enumerate(loader):

            # GPU
            # images shape is [100, 1, 28, 28] [batch_size, channel, height, width]
            if(bool_reshape):
                X = X.reshape(X.shape[0], -1).to(device)
            else:
                X = X.to(device)
            
            Y = Y.to(device)  # label shape is  [100, ]

            # Forward pass
            Y_hat = model(X)  # one-hot encoded because output_size is 10 from the network

            # Loss and Acc
            loss = criterion(Y_hat, Y)
            acc  = _acc(Y_hat, Y)
            
            epoch_loss += loss.item()
            epoch_acc  += acc.item()

    return epoch_loss / len(loader), epoch_acc / len(loader)

# for computing accuracy
def _acc(Y_hat, Y):
    
    _, predicted = torch.max(Y_hat.data, 1)  # returns max value, indices
    correct = (predicted == Y).float() 
    acc = correct.sum() / len(correct)
    
    return acc