import data_maker
import Analyze
import initialize_network
import train
import argparse
import dataset
import torch
from torch.utils.data import DataLoader #Dataloader module
import torchvision.transforms as transforms #transformations

def phrase_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e","--epoch_num", help="Input number of epochs", default=5, type=int)
    parser.add_argument("-lr","--lr_rate", help="Input learning rate", default=0.003
                        , type=float)
    parser.add_argument("-a","--analyze", help="Do you wan to analyze data", default=False, type=bool)
    args = parser.parse_args()
    return args.epoch_num, args.analyze, args.lr_rate

def main():
    
    if torch.cuda.is_available():
        device_ = torch.device("cuda")
        print (f"Using {device_}")
        #Checking GPU RAM allocated memory
        print('allocated CUDA memory: ',torch.cuda.memory_allocated())
        print('cached CUDA memory: ',torch.cuda.memory_cached())
        torch.cuda.empty_cache() # clear CUDA memory
        torch.backends.cudnn.benchmark = True #let cudnn chose the most efficient way of calculating Convolutions
        
    elif torch.backends.mps.is_available():
        print ("CUDA device not found.")
        device_ = torch.device("mps")
        print (f"Using {device_}")
    else:
        print ("CUDA device not found.")
        print ("MPS device not found.")
        device_ = torch.device("cpu")
        print (f"Using {device_}")

    #define globaly used dtype and device
    #device_ = torch.device('cpu')
    dtype_ = torch.float64
    
    
    
    epoch_num_, analyze_, lr_rate_= phrase_args()
    data_class = data_maker.load_Dataset(train='train.csv', test_data='test.csv', test_labels='sample_submission.csv')
    train_data, test_data = data_class.make_data()
    
    train_dataset=dataset.Dataset_maker(train_data)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, num_workers=2)
    test_dataset=dataset.Dataset_maker(test_data)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=8, shuffle=True, num_workers=2)
    
    
    if analyze_:
        Analyze.analyze(file_name='train.csv')
    network = initialize_network.Net(27, 1)
    print(f'calculating for: {epoch_num_} epochs')
    print(f'learning rate:{lr_rate_}')
    train.train(network = network, train_data=train_dataloader, test_data=test_dataloader,learning_rate=lr_rate_, epoch_num=epoch_num_,device=device_, test=True)
    
if __name__ == "__main__":
    main()



