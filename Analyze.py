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

    # cabin id is sequence of two letters and a number. Maybe those are random values, but maybe they are related to some "sectors" or something
    # like that. I'd give a shot to this idea. If chance of being transported has NO relation to those letters than probability of transportation
    # has for each letter binomial distribution with p = 0.504. Let's test it.

    # at first we divide "cabin" variable into three separate variables
    if_transported = []
    first_letter = []
    for i in range(num_people):
        try: # for case of NaN
            first_letter.append(df_train["Cabin"].values[i][0])
            if_transported.append(df_train["Transported"].values[i])
        except: continue

    last_letter = []
    for i in range(num_people):
        try:
            last_letter.append(df_train["Cabin"].values[i][-1])
        except: continue

    middle_number = []
    for i in range(num_people):
        try:
            middle_number.append(df_train["Cabin"].values[i][2:-2])
        except: continue

    F_letters, F_occurence = np.unique(np.array(first_letter), return_counts = True)
    M_numbers, M_occurence = np.unique(np.array(middle_number), return_counts = True)
    L_letters, L_occurence = np.unique(np.array(last_letter), return_counts = True)

    # now we calculate percentage of transported people for each letter/number

    f_true = []
    for f in F_letters:
        x = [if_transported[i] for i in range(len(first_letter)) if first_letter[i] == f]
        f_true.append(x.count(True) / first_letter.count(f))

    l_true = []
    for l in L_letters:
        x = [if_transported[i] for i in range(len(last_letter)) if last_letter[i] == l]
        l_true.append(x.count(True) / last_letter.count(l))

    m_true = []
    for m in M_numbers:
        x = [if_transported[i] for i in range(len(middle_number)) if middle_number[i] == m]
        m_true.append(x.count(True) / middle_number.count(m))

    # I'll omit "number" variable and focus only on letters

    # according to de Laplace-Moivre theorem we can approximate Bernoulli distribution with probability p and n samples with normal distribution with
    # mean = pn and variance = np(1-p)

    print("P-value for hipothesis that chance of being transported is not correlated to first letter of \'Cabin\'.")

    for i in range(len(f_true)):
        sd = np.sqrt(0.504 * 0.496 / F_occurence[i])
        p = 2 * min(norm.cdf(f_true[i], loc = 0.504, scale = sd), norm.cdf(-f_true[i] + 2 * 0.504, loc = 0.504, scale = sd))
        print("For letter \'" + F_letters[i] + "\' p-value is equal to " + str(p))

    print("\nP-value for hipothesis that chance of being transported is not correlated to first letter of \'Cabin\'.")
    for i in range(len(l_true)):
        sd = np.sqrt(0.504 * 0.496 / L_occurence[i])
        p = 2 * min(norm.cdf(l_true[i], loc = 0.504, scale = sd), norm.cdf(-l_true[i] + 2 * 0.504, loc = 0.504, scale = sd))
        print("For letter \'" + L_letters[i] + "\' p-value is equal to " + str(p))

    print("\nConclusion: the probability of transportation depended zajebiście from the Cabin ID.")
    print("It might have also depended from numerical part of ID, but I have no idea how to deal with it.")

    # and now lets plot correlations

    corr_matrix = df_train.corr(numeric_only = True)
    sns.heatmap(corr_matrix)
    plt.title('Correlation matrix')
    plt.show()

    # we could also check and plot dependence of "transported" with respect to other variables, but come on... it's machine learning, let computer figure 