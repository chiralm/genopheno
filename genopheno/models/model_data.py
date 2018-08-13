import random
import pandas as pd
from collections import namedtuple


ModelData = namedtuple('ModelData', 'training testing')


class ModelDataBuilder:
    """
    A builder to mutate the preprocessed data for modeling. The input phenotype dataframes are in the format:
         Gene_info, user1, user2, ..., userN
    snp1,
    snp2,
    ...,
    snpN

    Where the Gene_info column contains the SNP metadata. All user columns contain the number of mutations that user
    has for the SNP, ranging from 0 to 2.

    This splits the data frames into testing and training data, allows for any custom manipulation of the data frames and
    formats the data frames into the final format required by modeling.

          snp1, snp2, ..., snp3, phenotype
    user1,
    user2,
    ...
    user3
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
        :param func: The function to apply. It should return the resulting phenotype dataframe.
        The func takes two args, the phenotype string label and the dataframe.
        """
        for key in self._training.keys():
            self._training[key] = func(key, self._training[key])

    def apply_to_testing(self, func):
        """
        Applies a function to each testing dataframe
        :param func: The function to apply. It should return the resulting phenotype dataframe.
        The func takes two args, the phenotype string label and the dataframe.
        """
        for key in self._testing.keys():
            self._testing[key] = func(key, self._testing[key])

    def apply(self, func):
        """
        Applies a function to each training and testing dataframe
        :param func: The function to apply. It should return the resulting phenotype dataframe.
        The func takes two args, the phenotype string label and the dataframe.
        """
        self.apply_to_training(func)
        self.apply_to_testing(func)

    def reduce_training(self, reducer):
        """
        Applies a function to reduce the training data into a single value
        :param reducer: The function that will reduce the phenotypes. It should return the reduced value.
        The reducer takes one arg, a map of the training dataframes.
        :return: The reducer function return value
        """
        return reducer(self._training)

    def reduce_testing(self, reducer):
        """
        Applies a function to reduce the testing data into a single value
        :param reducer: The function that will reduce the phenotypes. It should return the reduced value.
        The reducer takes one arg, a map of the testing dataframes.
        :return: The reducer function return value
        """
        return reducer(self._testing)

    def build(self):
        """
        Builds the final data frames required for modeling
        :return: An object containing the modeling training and testing data frames
        """
        # Finalize metadata related info in the data frames
        self.apply(self.__finalize_df)

        # Merge the separate phenotypes into one data frame for training and one for testing
        training = self.reduce_training(self.__merge_phenos)
        testing = self.reduce_testing(self.__merge_phenos)

        return ModelData(training, testing)

    @staticmethod
    def __finalize_df(pheno, pheno_df):
        """
        This finalizes the metadata in the dataframes. First, non-user columns are dropped (i.e. Gene_info).
        Then the data frames are transposed so that users are the rows and SNPs are the values. A column for
        phenotype is also added.
        """
        # todo remove non-user columns
        columns_to_drop = filter(lambda x: not x.isdigit(), pheno_df.columns.values)
        pheno_df.drop(labels=columns_to_drop, axis=1, inplace=True)

        # transpose and add the phenotype label
        transposed_data = pheno_df.transpose()
        transposed_data['phenotype'] = pheno

        return transposed_data

    @staticmethod
    def __merge_phenos(phenotypes):
        return pd.concat(phenotypes.values())
