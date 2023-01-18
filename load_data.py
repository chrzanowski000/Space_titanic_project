def load_data(train, test_data, test_labels):
    import numpy as np
    import pandas as pd
    import torch

    df_train = pd.read_csv(train)
    df_test = pd.read_csv(test_data)
    df_ytest = pd.read_csv(test_labels)
    df_test['transported']=df_ytest.iloc[:,1].values

    df_train=pd.DataFrame.dropna(df_train)
    df_test=pd.DataFrame.dropna(df_test)

    def encoding_categories(sequences, categories):  #I put into this function a list of categories we want to encode
        results = np.zeros((len(sequences), len(categories))) # Creates an all-zero matrix of shape (len(sequences), len(categories))
        for i, sequence in enumerate(sequences): # Sets specific indices of results to 1
            for j in range(len(categories)):
                if sequence==categories[j]:
                    results[i,j]=1
        return np.array(results)

    def encoding(df):
        #convert true/false values to 1/0
        df.iloc[:,13]=1*df.iloc[:,13]
        df.iloc[:,2]=1*df.iloc[:,2]
        df.iloc[:,6]=1*df.iloc[:,6]
        
        #normalize numeric values by their maximum value
        for i in [5,7,8,9,10,11]:  
            df.iloc[:,i] = df.iloc[:,i]/max(df.iloc[:,i])

        home_planet=encoding_categories(df.iloc[:,1],['Europa','Earth','Mars'])
        destination=encoding_categories(df.iloc[:,4],['TRAPPIST-1e','PSO J318.5-22','55 Cancri e'])

        #Passenger ID is divided into group number (describing the whole group) and group ID (passenger's number inside their group)
        group, group_id=[],[]
        for x in df.iloc[:,0].values:
            group.append(int(x.split('_')[0]))
            group_id.append(int(x.split('_')[1]))  
        group=np.array(group)
        group_id=np.array(group_id)

        #Cabin number is divided into Deck/Number/Side, where Side can be either P for Port or S for Starboard. (explained in kaggle)
        deck, num, side =[],[],[]
        for x in df.iloc[:,3].values:
            if isinstance(x, float):
                deck.append(np.nan) #if there's a missing value, I assign nan to deck, num and side
                num.append(np.nan) 
                side.append(np.nan)
            else:
                deck.append(x.split('/')[0])
                num.append(int(x.split('/')[1]))
                side.append(x.split('/')[2])
        num=np.array(num)
        deck=encoding_categories(deck,['A','B','C','D','E','F','G','T'])
        side=encoding_categories(side,['P','S'])

        data= np.transpose(np.array([group, group_id, home_planet[:,0],home_planet[:,1],home_planet[:,2],df.iloc[:,2].values,num,
            deck[:,0], deck[:,1], deck[:,2],deck[:,3],deck[:,4],deck[:,5],deck[:,6],deck[:,7],side[:,0],
            side[:,1],destination[:,0],destination[:,1],destination[:,2],df.iloc[:,5].values,df.iloc[:,6].values,
            df.iloc[:,7].values,df.iloc[:,8].values,df.iloc[:,9].values,df.iloc[:,10].values,df.iloc[:,11].values ]))
        
        return data

    test_data=[]
    x=encoding(df_test)
    for i in range(len(x)):
        test_data.append([torch.tensor(np.array(tuple(x[i]))), torch.tensor([df_test.iloc[i,13]])])

    train_data=[]
    x=encoding(df_train)
    for i in range(len(x)):
        train_data.append([torch.tensor(np.array(tuple(x[i]))), torch.tensor([df_train.iloc[i,13]])])

    return train_data, test_data

    


