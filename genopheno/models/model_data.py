import random


class ModelDataBuilder:
    def __init__(self, phenotypes, data_split):
        self.data_split = data_split
        self.training = {}
        self.testing = {}

        for pheno, pheno_df in phenotypes.iteritems():
            self.__split_test_train(pheno_df)
            # self.training[pheno] = test_sample

    def __split_test_train(self, pheno_df):
        # Get list of users ids that we have genotypes for
        user_ids = pheno_df.columns.values.tolist()
        user_ids.remove("Gene_info")

        # shuffle the genotype user_ids so the test and training users are selected randomly
        random.seed(1)
        random.shuffle(user_ids)

        # split the users into training and test sets
        test_count = len(user_ids) * self.data_split / 100
        test_users = user_ids[:test_count]
        train_users = user_ids[test_count:]

        # return test and training dataset
        # todo

    def save(self, path):
        pass
