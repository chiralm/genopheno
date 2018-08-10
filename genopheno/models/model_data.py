import random


class ModelDataBuilder:
    def __init__(self, phenotypes, data_split):
        self._data_split = data_split
        self._training = {}
        self._testing = {}

        # split into testing and training data
        for pheno, pheno_df in phenotypes.iteritems():
            test_df, train_df = self.__split_test_train(pheno_df)
            self._testing[pheno] = test_df
            self._training[pheno] = train_df

    def __split_test_train(self, pheno_df):
        # Get list of users ids that we have genotypes for
        user_ids = pheno_df.columns.values.tolist()
        user_ids.remove("Gene_info")

        # shuffle the genotype user_ids so the test and training users are selected randomly
        random.seed(1)
        random.shuffle(user_ids)

        # split the users into training and test sets
        test_count = len(user_ids) * self._data_split / 100
        test_users = user_ids[:test_count]
        train_users = user_ids[test_count:]

        train_users_df = pheno_df.loc[:, train_users]
        test_users_df = pheno_df.loc[:, test_users]

        # add back the gene info column
        train_users_df["Gene_info"] = pheno_df["Gene_info"]
        test_users_df["Gene_info"] = pheno_df["Gene_info"]

        return test_users_df, train_users_df

    def apply_to_training(self, func):
        for key in self._training.keys():
            self._training[key] = func(key, self._training[key])

    def apply_to_testing(self, func):
        for key in self._testing.keys():
            self._testing[key] = func(key, self._training[key])

    def apply(self, func):
        self.apply_to_training(func)
        self.apply_to_testing(func)

    def reduce_training(self, reducer):
        return reducer(self._training)
