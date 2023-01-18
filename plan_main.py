import load_data
import Analyze
import initialize_network
import train
def main():
    train_data, test_data = load_data.load_data(train='train.csv', test_data='test.csv', test_labels='sample_submission.csv')
    #Analyze.analyze(file_name='train.csv')
    network = initialize_network.Net(27, 100, 80, 100, 1)
    train.train(network = network, train_data=train_data, test_data=test_data, test=True, epoch_num=3)
    
if __name__ == "__main__":
    main()
