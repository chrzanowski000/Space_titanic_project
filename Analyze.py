def analyze(df_train):
    '''
    df_train w formacie pandas.DataFrame
    '''

    # pie plots for categorical variables

    fig, axis = plt.subplots(1, 2)

    for num, name in enumerate(["HomePlanet", "Destination"]):
        num_of_occ = list(df_train[name].value_counts().values)
        lab = list(df_train[name].value_counts().index)
        num_of_occ.append(df_train[name].isna().sum())
        lab.append("No data")

        axis[num].pie(x = num_of_occ, labels = lab, autopct='%1.1f%%')
        axis[num].set_title(name)

    plt.show()

    # pie plots for boolean variables

    fig, axis = plt.subplots(1, 3)

    for num, name in enumerate(["VIP", "Transported", "CryoSleep"]):
        num_of_occ = list(df_train[name].value_counts().values)
        lab = list(df_train[name].value_counts().index)
        num_of_occ.append(df_train[name].isna().sum())
        lab.append("No data")

        axis[num].pie(x = num_of_occ, labels = lab, autopct='%1.1f%%')
        axis[num].set_title(name)

    plt.show()

    # histograms for numerical variables

    for name in "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck":

        fig, axis = plt.subplots(1, 2)
        plt.suptitle(name + " histogram")
        axis[0].hist(df_train[name].values, color ="violet", bins = 40)
        axis[0].set_xlabel(name)
        axis[0].set_ylabel("Occurence")
        data_not_ok = df_train[name].isna().sum()
        data_ok = len(df_train[name].values) - data_not_ok
        axis[1].pie([data_ok, data_not_ok], labels = ["OK", "Missing data"], autopct='%1.1f%%', colors = ["green", "red"] )

        plt.show()

    # now a quick check, if NaN values are distributed randomly or if about some people we have completely no data.

    nan_list = []
    num_people = df_train.shape[0]
    nan_people = []

    for i in range(num_people):
        nan_list.append(df_train.iloc[i].isna().sum())

    nan_list = np.array(nan_list)
    unique, counts = np.unique(nan_list, return_counts=True)

    print("MISSING VALUES")
    for i in range(len(unique)):
        print(str(counts[i]) + " people have " + str(unique[i]) + " missing values.")