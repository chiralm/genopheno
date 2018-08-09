import random


class ModelDataBuilder:
    def __init__(self, phenotypes, data_split):
        self._data_split = data_split
        self._training = {}
        self._testing = {}

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

        return pheno_df.loc[:, test_users], pheno_df.loc[:, train_users]

    def apply_to_training(self, func):
        for key in self._training.keys():
            self._training[key] = func(key, self._training[key])

    def reduce_training(self, reducer):
        return reducer(self._training)
