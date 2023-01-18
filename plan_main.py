import load_data
import Analyze
import initialize_network
import train
def main():
    train_data, test_data = load_data.load_data()
    Analyze.analyze(file_name)
    initialize_network.initialize_network()
    train.train(train_data=train_data, test_data=test_data, test=False, epoch_num=10, )
