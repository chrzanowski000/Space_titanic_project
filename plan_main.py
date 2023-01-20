import load_data
import Analyze
import initialize_network
import train
import argparse

def phrase_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e","--epoch_num", help="Input number of epochs", default=5, type=int)
    args = parser.parse_args()
    return args.epoch_num

def main():
    epoch_num_=phrase_args()
    train_data, test_data = load_data.load_data(train='train.csv', test_data='test.csv', test_labels='sample_submission.csv')
    print(f'calculating for: {epoch_num_} epochs')
    #Analyze.analyze(file_name='train.csv')
    #network = initialize_network.Net(27, 100, 80, 100, 1)
    network = initialize_network.Net(27, 1)
    train.train(network = network, train_data=train_data, test_data=test_data, test=True, epoch_num=epoch_num_)
    
if __name__ == "__main__":
    main()



