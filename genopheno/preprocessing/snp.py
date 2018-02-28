import pandas as pd
import numpy as np
import re
import os
from math import isnan
from os import listdir
from os.path import isfile, join


RSID_COLUMN = 'Rsid'
REF_COLUMN = 'Ref'
ALT_COLUMN = 'Alt'
GENEINFO_COLUMN = 'Gene_info'
CHROM_COLUMN = 'Chrom'
POS_COLUMN = 'Pos'
QUAL_COLUMN = 'Qual'
FILTER_COLUMN = 'Filter'


def build_database(snp_data_dir, output_dir):
    """
    Builds a data frame containing data for all SNPs from individual SNP files.
    :param snp_data_dir: The directory containing the individual SNP files.
    The files must be in VCF format and can optionally be compressed using gzip. Files must either end in .gz or .vcf.
    :param output_dir: The directory to save the processed SNPs in
    :return: A data frame containing all SNP data. The data frame includes columns Rsid,Ref,Alt,Gene_info.
    """
    # Combine all SNP files into one data frame
    snp_details = __combine_snp_data(snp_data_dir)

    # Remove duplicate data
    snp_details.drop_duplicates(subset=[RSID_COLUMN], inplace=True)

    # Remove all SNPs with missing gene info
    snp_details.dropna(subset=[GENEINFO_COLUMN], inplace=True)

    # Save the snp data frame. It is needed in the prediction step.
    snp_details.to_csv(os.path.join(output_dir, 'snp_database.csv.gz'), compression='gzip', index=False)

    return snp_details


def __combine_snp_data(snp_data_dir):
    """
    Combines SNP files into a data frame.
    Columns 'Chrom', 'Pos', 'Qual', 'Filter' are dropped from the SNP files.
    :param snp_data_dir: The directory containing the individual SNP files.
    The files must be in VCF format and can optionally be compressed using gzip. Files must either end in .gz or .vcf.
    :return: A data frame containing all SNP data. The data frame includes columns Rsid,Ref,Alt,Gene_info.
    """
    snp_details = pd.DataFrame(columns=[RSID_COLUMN, REF_COLUMN, ALT_COLUMN, GENEINFO_COLUMN])

    snp_file_names = [f for f in listdir(snp_data_dir) if isfile(join(snp_data_dir, f))]
    for snp_file in snp_file_names:
        snp_file_path = os.path.join(snp_data_dir, snp_file)
        try:
            if '.gz' in snp_file:
                data = pd.read_csv(snp_file_path, compression="gzip", skiprows=12, sep='\t',
                                   names=[CHROM_COLUMN, POS_COLUMN, RSID_COLUMN, REF_COLUMN, ALT_COLUMN, QUAL_COLUMN,
                                          FILTER_COLUMN, GENEINFO_COLUMN])
            else:
                data = pd.read_csv(snp_file_path, skiprows=12, sep='\t',
                                   names=[CHROM_COLUMN, POS_COLUMN, RSID_COLUMN, REF_COLUMN, ALT_COLUMN, QUAL_COLUMN,
                                          FILTER_COLUMN, GENEINFO_COLUMN])

        except Exception as e:
            print '[WARNING] "{}" VCF file invalid. Skipping it. Reason: {}'.format(snp_file_path, e)
            continue

        # Remove columns that are not needed
        data = data.drop([CHROM_COLUMN, POS_COLUMN, QUAL_COLUMN, FILTER_COLUMN], 1)

        # Extract relevant gene info
        data[GENEINFO_COLUMN] = data[GENEINFO_COLUMN].apply(__extract_gene_info)

        # Add data to the aggregate snp data frame
        snp_details = pd.concat([snp_details, data], ignore_index=True)

    return snp_details


def __extract_gene_info(gene_info):
    """
    Extracts the GENEINFO value from the SNP input Geno_info column.

    The SNP input Gene_Info column looks like this:
    'RSPOS=162227882;GENEINFO=5071:PARK2;dbSNPBuildID=36;SAO=0;GMAF=0.429513;VC=snp;VLD;VP=0501000800051705003F0100'
    In this case 5071:PARK2 would be returned.

    In some cases no GENEINFO is included:
    'RSPOS=198269732;dbSNPBuildID=36;SAO=0;GMAF=0.496006;VC=snp;VLD;VP=0501280000051505003F0100'
    In this case NaN is returned.
    :param gene_info:
    :return: The value for the GENEINFO attribute in the SNP Geno_info column
    """
    if isinstance(gene_info, float) and isnan(gene_info):
        return gene_info

    match = re.search('GENEINFO=([^;]*);', gene_info)
    if match:
        return match.group(1)
    else:
        return np.nan


def extract_rsid(gene_rsid):
    """
    Extracts rsid from formatted gene name
    :param gene_rsid:
    :return:
    """
    match = re.match('[\w_]*_([\w]*)', gene_rsid)
    if match:
        return match.group(1)
    else:
        return np.nan


def format_snps(rsid, snp_details):
    """
    Formats rsid into gene_<gene info>_<rsid>
    :param rsid:
    :param snp_details:
    :return:
    """
    gene_info = snp_details[snp_details["Rsid"] == rsid]["Gene_info"]
    formatted = 'gene_' + gene_info.str.replace(r'\W', '_').item() + '_' + rsid
    return formatted
