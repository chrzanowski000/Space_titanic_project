import torch
import time
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
import numpy as np
import pandas as pd


def train(network, train_data, test_data, epoch_num, learning_rate, device, test = False):
    '''
    description
    '''
    #define globaly used dtype and device
    device_ = device
    dtype_ = torch.float32
    # configuring the net
    model = network
    model = model.to(dtype=dtype_, device=device_)   
    criterion = torch.nn.BCELoss()          # binary cross entropy loss function
    learning_rate = learning_rate
    weight_decay = 0.001 # for ADAM optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)

    # learning loop

    correct_list = []
    loss_list = []
    accuracy_list = []
    for epoch in range(epoch_num):
        print(f'epoch nr {epoch}')
        predictions = np.array([])
        targets = np.array([])
        for x, y in train_data:
                optimizer.zero_grad()           # clear gradient of loss function
                x = x.to(dtype=dtype_, device=device_)
                y = y.to(dtype=dtype_, device=device_)
                results = model(x)              # calculate predictions
                #results = torch.where(results < 0.5, 0, 1).to(dtype=dtype_, device=device_)
                loss = criterion(results, y)    # calculate loss
                loss.backward()                 # calculate gradient
                optimizer.step()                # update parameters
                
                predictions = np.append(predictions, results.cpu().detach().numpy())
                predictions =  np.where(predictions < 0.5, 0, 1)
                targets = np.append(targets, y.cpu().detach().numpy())
                
        loss_list.append(loss.cpu().detach().numpy()) # store loss
        acc = metrics.accuracy_score(predictions,targets)
        print(f'loss: {loss.cpu().detach().numpy()}')
        accuracy_list.append(acc)
        print(f'accuracy: {acc}')
        
    # plot loss function on test data
    
    if test:
            print(metrics.classification_report(targets, predictions, digits=2))
            
            fig, ax = plt.subplots()     
            epoch_array = np.arange(1,epoch_num+1)
            loss_list_array = np.array(loss_list)
            #print(loss_list_array)
            sns.lineplot(x=epoch_array,y=loss_list_array, ax=ax)
            sns.lineplot(x=epoch_array,y=accuracy_list, ax=ax)
            plt.xlabel("Number of epochs")
            plt.grid()
            plt.legend(["Loss", "Accuracy"], bbox_to_anchor = [1,1])
            #plt.savefig('loss.pdf')
            plt.savefig('Loss.jpeg')
            #plt.show()
            
            """
            fig, ax = plt.subplots() 
            plt.scatter(targets, targets)
            plt.scatter(predictions, predictions)
            plt.show()
            plt.savefig('TP.pdf')
            # the precision and recall, among other metrics
            metrics_table=metrics.classification_report(targets, predictions, digits=4)
            wrong_list = [np.shape(test_data)[0] - c for c in correct_list]
            print(metrics_table)
            
            names = []
            plt.plot(correct_list, range(epoch_num), color = "green")
            names.append("Correct guesses")
            plt.plot(wrong_list, range(epoch_num), color = "red")
            names.append("Wrong guesses")
            plt.yscale("log")
            plt.ylabel("Number of guesses")
            plt.xlabel("Epoch number")
            plt.title("Accuracy of the net on test data")
            plt.legend(names, bbox_to_anchor = [1, 1])
            plt.grid()
            plt.show()
            plt.savefig('CW.pdf')
            """
    return model