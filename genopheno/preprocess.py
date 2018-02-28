import argparse
import pandas as pd
import numpy as np
from preprocessing import snp
from util import *
from preprocessing.users import UserPhenotypes


def __merge_user_mutations(users, phenotype, snp_details):
    """
    Calculates the mutations for each user SNP and merges them into one data frame.
    :param users: The users to include in the mutations data frame
    :param phenotype: The phenotype label
    :param snp_details: The data frame containing the SNP details
    :return: A data frame containing all user mutations for all SNPs
    """
    # The final data structure doesn't need ref or alt, only if the user has a mutation or not.
    merged_user_data = snp_details.drop(['Ref', 'Alt'], axis=1)

    def merge_user(user_to_merge, all_user_data):
        """
        Merges the user data with the data frame containing data about all user SNPs
        :param user_to_merge: The user to merge
        :param all_user_data: The data frame containing data about all user SNPs
        :return: The merged data
        """
        user_data = user_to_merge.allele_transformation(snp_details)
        if not user_data.empty:
            return pd.merge(all_user_data, user_data, on=['Rsid'], how='outer')
        else:
            print '[WARNING] User {} did not have any SNPs in the SNP database. ' \
                  'Skipping the user.'.format(user_to_merge.id)
            return all_user_data

    for i in range(len(users)):
        user = users[i]
        merged_user_data = timed_invoke(
            "processing user {} with phenotype '{}' ({}/{})".format(user.id, phenotype, i + 1, len(users)),
            lambda: merge_user(user, merged_user_data)
        )

    return merged_user_data


def __calc_snp_percents(user_mutations):
    """
    Calculates the average mutation percentage (no, partial and full mutations) for each SNP
    :param user_mutations: The data frame containing user mutations
    :return: The mutations data frame with mutation percentages
    """
    zero_pct = []
    one_pct = []
    two_pct = []

    def count_snps(row):
        zero_count = 0
        one_count = 0
        two_count = 0

        # count number of mutations for each gene
        for mut_count in row[2::].values:
            if mut_count == 0:
                zero_count += 1
            elif mut_count == 1:
                one_count += 1
            elif mut_count == 2:
                two_count += 1

        # calculate the percents of each mutation
        total = zero_count + one_count + two_count
        zero_pct.append(0 if zero_count == 0 else float(zero_count) / total * 100)
        one_pct.append(0 if one_count == 0 else float(one_count) / total * 100)
        two_pct.append(0 if two_count == 0 else float(two_count) / total * 100)

    user_mutations.apply(count_snps, axis=1)
    user_mutations['pct_fm'] = two_pct
    user_mutations['pct_nm'] = zero_pct
    user_mutations['pct_pm'] = one_pct

    return user_mutations


def __write_final(phenotype, all_user_data, output_dir):
    """
    Writes the user SNP data to a CSV file
    :param phenotype: The phenotype the data represents
    :param all_user_data: The SNP data for all users
    :param output_dir: The directory to write the data to
    """
    # set index to rsid
    all_user_data.set_index(['Rsid'], inplace=True)
    
    # Save as a CSV file
    file_path = os.path.join(output_dir, "preprocessed_{}.csv.gz".format(phenotype))
    all_user_data.to_csv(file_path, header=True, compression='gzip')


def run(user_data_dir, snp_data_dir, known_pheno_file, output_dir):
    """
    Preprocesses the user data for model building
    :param user_data_dir: The directory containing all user genomic files
    :param snp_data_dir: The directory containing all SNP VCF files
    :param known_pheno_file: The file containing the known user phenotype classifications
    :param output_dir: The directory to write the preprocessed files to
    :return:
    """
    # Expand file paths
    user_data_dir = expand_path(user_data_dir)
    snp_data_dir = expand_path(snp_data_dir)
    known_pheno_file = expand_path(known_pheno_file)
    output_dir = expand_path(output_dir)

    # Make sure output directory exists before doing work
    clean_output(output_dir)

    def timed_run():
        # Build SNPs data frame
        snp_details = timed_invoke('building SNP data frame', lambda: snp.build_database(snp_data_dir, output_dir))

        # Build users information
        users_phenotypes = UserPhenotypes(known_pheno_file, user_data_dir)

        def reducer(phenotype, users):
            """
            Processes a list of users categorized by phenotype into the final data structure form
            :param phenotype: The phenotype of the users
            :param users: The users with the phenotype
            """
            all_user_data = __merge_user_mutations(users, phenotype, snp_details)
            all_user_data = timed_invoke('calculating mutation percentages', lambda: __calc_snp_percents(all_user_data))
            timed_invoke("saving preprocessed file for phenotype '{}'".format(phenotype),
                         lambda: __write_final(phenotype, all_user_data, output_dir))
            return all_user_data

        timed_invoke('building final data structure', lambda: users_phenotypes.reduce_phenotypes(reducer))

    timed_invoke('preprocessing data', lambda: timed_run())

    print 'Output written to "{}"'.format(output_dir)


if __name__ == '__main__':
    # Parse input
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--user-geno",
        "-u",
        metavar="<directory path>",
        default="resources/data/users",
        help="The directory containing user genomic data. Each file contains data for one user."
             " 23andMe and Ancestry.com data formats are supported."
             " File names must start with the numeric user ID followed by an underscore and end"
             " with either 23andme.txt or ancestry.txt."
             "\n\nExamples:"
             "\nuser44_file19_yearofbirth_1970_sex_XY.23andme.txt"
             "\nuser44_file19_yearofbirth_1970_sex_XY.ancestry.txt"
             "\n\nDefault: resources/data/users"
        )

    parser.add_argument(
        "--known-phenos",
        "-p",
        metavar="<file path>",
        default="resources/data/known_phenotypes.csv",
        help="The file path to the file that contains the known phenotypes. This is used to train the model."
             " This must be a CSV file with the following format with columns user_id and phenotype."
             "\nuser_id is the numeric user ID. The user ID should match a user ID in the --user-geno directory."
             "\nphenotype is the phenotype classification for the user. For example, if using eye color then the"
             "phenotype column would contain a value of 'Brown' or 'Blue_Green'."
             "\n\nExample:"
             "\nuser_id,phenotype"
             "\n44,Brown"
             "\n124,Blue_Green"
             "\n55,Blue_Green"
             "\n\nDefault: resources/data/known_phenotypes.csv"
    )

    parser.add_argument(
        "--snp",
        "-s",
        metavar="<directory path>",
        default="resources/data/snp",
        help="The directory containing the SNP data for each genome. The supported file format is VCF. Each file"
             "must end in .vcf. Optionally, the files can be compressed with gzip and must end in .gz."
             "\n\nDefault: resources/data/snp"
    )

    parser.add_argument(
        "--output",
        "-o",
        metavar="<directory path>",
        default="resources/data/preprocessed",
        help="The directory that the out files should be written to. This will include all files required for the"
             "machine learning input."
             "\n\nDefault: resources/data/preprocessed"
    )

    args = parser.parse_args()
    run(args.user_geno, args.snp, args.known_phenos, args.output)
