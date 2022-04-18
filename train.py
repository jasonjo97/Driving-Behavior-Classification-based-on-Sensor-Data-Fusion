import numpy as np 
import time 

import torch 
import torch.nn as nn 
import torch.optim as optim 

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

from datasets.images import CustomDataset, StatsRecorder, NormalizedDataset
from utils.data import load_data, load_data_vision
from models.classification import CustomizedAlexNet


def preprocessing(path):
    """ Load time series data with data labels """
    X_gps_imu, y, drivers = load_data(path)
    X_vision = load_data_vision(path)

    X_vision = np.array(X_vision).reshape((len(X_vision[0]),-1))

    y = np.array(y)
    drivers= np.array(drivers)

    print("Shape of X_vision : {}".format(X_vision.shape))
    print("Shape of y : {}".format(y.shape))
    print("Shape of drivers : {}".format(drivers.shape))

    return X_gps_imu, X_vision, drivers, y


def train(model, criterion, optimizer, dataloaders, threshold, plot=False): 
    """ Train the model using BCE loss with batch element weights """
    train_losses, train_accuracies = [],[]
    valid_losses, valid_accuracies = [],[]
    best_acc = 0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1,num_epochs))
        print('-'*70)
        
        train_loss, train_acc = 0,0
        val_loss, val_acc = 0,0 
        
        train_targets_lst, val_targets_lst = [],[] 
        train_logits_lst, val_logits_lst  = torch.tensor([]), torch.tensor([])
        
        for phase in ['train', 'validation']:
            if phase == 'train':
                print('Training ...')
                t_train_start = time.time()
                model.train()        
            else: 
                print('Validation ...')
                t_eval_start = time.time()
                model.eval()
                
            for step, (inputs, targets, weights) in enumerate(dataloaders[phase]):
                if phase =='train':
                    optimizer.zero_grad()
                    logits = model(inputs)
                    nonweighted_loss = criterion(logits, targets)
                    loss = (nonweighted_loss * weights).mean()
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1) # gradient clipping 
                    optimizer.step()
                    
                    train_loss += loss.item() # accumulate loss 
                    train_targets_lst.extend(targets.tolist())
                    train_logits_lst = torch.cat([train_logits_lst, logits])

                    if (step % 20 == 0) & (step != 0):
                        print("Batch {}/{} - Loss : {}".format(step, len(dataloaders[phase]), train_loss/step))
                    
                else: 
                    with torch.no_grad(): 
                        logits = model(inputs)
                        nonweighted_loss = criterion(logits, targets)
                        loss = (nonweighted_loss * weights).mean()

                        val_loss += loss.item()
                        val_targets_lst.extend(targets.tolist())
                        val_logits_lst = torch.cat([val_logits_lst, logits])

                        if (step % 20 == 0) & (step != 0): 
                            print('Batch {}/{} -- Loss : {}'.format(step, len(dataloaders[phase]), val_loss/step))
        
            # calculate average loss/accuracy
            if phase == 'train':
                avg_train_loss = train_loss/len(dataloaders[phase])
                predicted = (train_logits_lst >= threshold).int()
                avg_train_acc = 100 * f1_score(train_targets_lst, predicted.numpy())

                train_losses.append(avg_train_loss)
                train_accuracies.append(avg_train_acc) 
                print('  Average training loss : {}'.format(avg_train_loss))
                print('  Average training accuracy : {} %'.format(avg_train_acc))
                print(confusion_matrix(train_targets_lst, predicted.numpy()))

                t_train_end = time.time() 
                print('  Elapsed Time : {}'.format(t_train_end-t_train_start))
                
            else: 
                avg_val_loss = val_loss/len(dataloaders[phase])
                predicted = (val_logits_lst >= threshold).int()
                avg_val_acc = 100 * f1_score(val_targets_lst, predicted.numpy())

                valid_losses.append(avg_val_loss)
                valid_accuracies.append(avg_val_acc)
                print('  Average validation loss : {}'.format(avg_val_loss))
                print('  Average validation accuracy : {} %'.format(avg_val_acc))
                print(confusion_matrix(val_targets_lst, predicted.numpy()))
                t_eval_end = time.time() 
                print('  Elapsed Time : {}'.format(t_eval_end-t_eval_start))

                # save the model with best accuracy  
                if best_acc < avg_val_acc: 
                    best_acc = avg_val_acc 
                    torch.save(model.state_dict(), './models/model.pt')
                    print('model checkpoint saved !') 
        
    if plot == True: 
        plt.subplot(2,1,1)
        plt.plot(np.arange(num_epochs), train_losses, 'r')
        plt.plot(np.arange(num_epochs), valid_losses, 'b')
        plt.legend(['Train Loss','Validation Loss'])
        plt.show()
        
        plt.subplot(2,1,2)
        plt.plot(np.arange(num_epochs), train_accuracies, 'r')
        plt.plot(np.arange(num_epochs), valid_accuracies, 'b')
        plt.legend(['Train Accuracy','Validation Accuracy'])
        plt.show()
        
    print("Training Complete")
    return model 


if __name__ == "__main__":
    X_gps_imu, X_vision, drivers, y = preprocessing('C:/Users/Desktop/trip_data_final/year=2021/5월_trip_final/')

    # train/validation/test set split  
    ind_train, ind_test = train_test_split(np.arange(len(X_gps_imu[0])), test_size=0.4, random_state=101)
    ind_val, ind_test = train_test_split(ind_test, test_size=0.5, random_state=101)

    indices = {'train': ind_train, 'validation' : ind_val}

    # customized datasets/dataloaders 
    batch_size = 100
    class_weights = torch.tensor(compute_class_weight('balanced', classes = np.unique([y[j] for j in indices["train"]]), 
                                                               y = [y[j] for j in indices["train"]]), dtype=torch.float)
    datasets = {x : CustomDataset([[X_gps_imu[i][j] for j in indices[x]] for i in range(9)], [y[j] for j in indices[x]], class_weights) 
            for x in ['train', 'validation']}
    dataloaders = {x : torch.utils.data.DataLoader(datasets[x], batch_size, shuffle=True) for x in ['train', 'validation']}
    
    # Update mean & std statistics throughout the whole training dataset 
    stats = StatsRecorder()
    with torch.no_grad():
        for idx, data in enumerate(dataloaders["train"]): 
            stats.update(data[0])
            if (idx+1) % 10 == 0: 
                print("Batch {}/{} Complete!".format(idx+1, len(dataloaders["train"])))
    
    # apply normalization
    normalized_datasets = {x : NormalizedDataset([[X_gps_imu[i][j] for j in indices[x]] for i in range(9)], [y[j] for j in indices[x]], 
                                    [stats.mean, stats.std], class_weights) for x in ['train', 'validation']}
    normalized_dataloaders = {x : torch.utils.data.DataLoader(normalized_datasets[x], batch_size, shuffle=True) for x in ['train', 'validation']}

    # define hyperparameters 
    num_epochs = 100
    learning_rate = 0.025
    model = CustomizedAlexNet()
    criterion = nn.BCELoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr = learning_rate)

    # model training 
    trained_model = train(model, criterion, optimizer, normalized_dataloaders, threshold=0.5, plot=True)