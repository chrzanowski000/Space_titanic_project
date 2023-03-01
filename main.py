import data_maker
import Analyze
import initialize_network
import train
import argparse
import Data_Set

def phrase_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e","--epoch_num", help="Input number of epochs", default=5, type=int)
    parser.add_argument("-lr","--lr_rate", help="Input learning rate", default=5, type=float)
    parser.add_argument("-a","--analyze", help="Do you wan to analyze data", default=False, type=bool)
    args = parser.parse_args()
    return args.epoch_num, args.analyze, args.lr_rate

def main():
    epoch_num_, analyze_, lr_rate_= phrase_args()
    data_class = data_maker.load_Dataset(train='train.csv', test_data='test.csv', test_labels='sample_submission.csv')
    train_data, test_data = data_class.make_data()
    print(train_data)
    
    if analyze_:
        Analyze.analyze(file_name='train.csv')
    network = initialize_network.Net(27, 1)
    print(f'calculating for: {epoch_num_} epochs')
    print(f'learning rate:{lr_rate_}')
    train.train(network = network, train_data=train_data, test_data=test_data,learning_rate=lr_rate_, epoch_num=epoch_num_, test=True)
    
if __name__ == "__main__":
    main()



