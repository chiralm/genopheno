import random
import pandas as pd


class ModelDataBuilder:
    """
    A builder to mutate the preprocessed data for modeling. The phenotype dataframes are in the format:
         Gene_info, user1, user2, ..., userN
    snp1,
    snp2,
    ...,
    snpN

    Where the Gene_info column contains the SNP metadata. All user columns contain the number of mutations that user
    has for the SNP, ranging from 0 to 2.
    """
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
        """
        Splits a phenotype dataframe into testing and training subsets
        :param pheno_df: The dataframe to split
        :return: A tuple containing the testing and training dataframes
        """
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

        test_users_df = pheno_df.loc[:, test_users]
        train_users_df = pheno_df.loc[:, train_users]

        # add back the gene info column
        test_users_df["Gene_info"] = pheno_df["Gene_info"]
        train_users_df["Gene_info"] = pheno_df["Gene_info"]

        return test_users_df, train_users_df

    def apply_to_training(self, func):
        """
        Applies a function to each training dataframe
        :param func: The function to apply. It should return the resulting phenotype dataframe
        """
        for key in self._training.keys():
            self._training[key] = func(key, self._training[key])

    def apply_to_testing(self, func):
        """
        Applies a function to each testing dataframe
        :param func: The function to apply. It should return the resulting phenotype dataframe
        """
        for key in self._testing.keys():
            self._testing[key] = func(key, self._testing[key])

    def apply(self, func):
        """
        Applies a function to each training and testing dataframe
        :param func: The function to apply. It should return the resulting phenotype dataframe
        """
        self.apply_to_training(func)
        self.apply_to_testing(func)

    def reduce_training(self, reducer):
        return reducer(self._training)

    def reduce_testing(self, reducer):
        return reducer(self._testing)

    def build(self):
        # todo drop non-user columns, transpose, add phenotype column
        self.apply(self.__finalize_df)

        # merge
        training = self.reduce_training(self.__merge_phenos)
        testing = self.reduce_testing(self.__merge_phenos)

        return ModelData(training, testing)

    def __finalize_df(self, pheno, pheno_df):
        # todo remove non-user columns
        columns = pheno_df.columns.values

        # transpose and add the phenotype label
        transposed_data = pheno_df.transpose()
        transposed_data['phenotype'] = pheno

        return transposed_data


    def __merge_phenos(self, phenotypes):
        return pd.concat(phenotypes.values())


class ModelData:
    def __init__(self, training, testing):
        self.training = training
        self.testing = testing