def train(network, train_data, test_data, test = False, epoch_num = 10):
    '''
    description
    '''
    model = network
    # configuring the net

    criterion = torch.nn.BCELoss()          # binary cross entropy loss function
    learning_rate = 3e-4                    # initial lr, internet says best for Adam
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    # learning loop

    correct_list = []
    predictions = np.array([])
    targets = np.array([])
    for epoch in range(epoch_num):
            print(f'epoch nr {epoch}')
            for x, y in train_data:
                    optimizer.zero_grad()           # clear gradient of loss function
                    x = x.double()
                    y = y.double()
                    results = model(x).double()     # calculate predictions
                    loss = criterion(results, y)    # calculate loss
                    loss.backward()                 # calculate gradient
                    optimizer.step()                # update parameters
                    
                    predictions = np.append(predictions, results.cpu().detach().numpy())
                    predictions =  np.where(predictions < 0.5, 0, 1)
                    targets = np.append(targets, y.cpu().detach().numpy())
    # plot loss function on test data
    
    if test:
            plt.scatter(targets, targets)
            plt.scatter(predictions, predictions)
            plt.show()
            # the precision and recall, among other metrics
            metrics_table=metrics.classification_report(targets, predictions, digits=4)
            #wrong_list = [np.shape(test_data)[0] - c for c in correct_list]
            print(metrics_table)
            """
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
            plt.show()"""
    
    # and finally
    return model