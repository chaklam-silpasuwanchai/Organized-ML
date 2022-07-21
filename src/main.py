import torch.optim  as optim
import utils, datasets
import argparse

from datetime       import datetime
from train          import train_eval, test
from torch          import nn
from pathlib        import Path

def _run_experiment(config_file, seed):
    
    print("="*14, "Experiment", "="*14)
    utils.print_nicely("time", datetime.now().strftime("%d:%b:%Y-%H:%M:%S"))

    config_dir = Path("../configs")
    
    #1. load config
    config = utils.load_yaml(config_dir, config_file)  #config of that architecture
    shared_config = utils.load_yaml(config_dir, "shared_config.yaml")  #shared config
    
    #2. set seeds and device
    utils.set_seeds(seed)
    device = utils.set_device()
    
    #3. load dataset
    train_loader, val_loader, test_loader = datasets.load(shared_config.batch_size_train, shared_config.batch_size_test)
        
    #4. load model
    model = utils.load_model(config, device)
    
    #5. set optimizer and loss function
    optimizer = optim.SGD(model.parameters(), lr=shared_config.learning_rate, momentum=shared_config.momentum)
    criterion = nn.CrossEntropyLoss()
    
    #6. train model
    train_losses, train_accs, valid_losses, valid_accs =  train_eval(shared_config.n_epochs, train_loader, val_loader, 
                                                                    model, criterion, optimizer, device, config_file)
    
    #7. eval model
    test(model, test_loader, criterion, device, config_file)
        
    #8. visualize results
    utils.plot(train_losses, valid_losses, config_file, "loss", seed)
    utils.plot(train_accs,   valid_accs  , config_file, "acc" , seed)

if __name__ == "__main__":
  # initialize ArgumentParser class of argparse
  parser = argparse.ArgumentParser()
 
  # add different arguments and their types
  parser.add_argument('-c', '--config_file', type=str, required=True)
  parser.add_argument('-s', '--seed', type=int, required=True)

  # read arguments from command line
  args = parser.parse_args()
    
  # run with arguments specified by command line arguments
  _run_experiment(
      config_file=args.config_file,
      seed = args.seed
      )