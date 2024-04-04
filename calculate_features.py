import json
from pathlib import Path
from random import randint
import pandas as pd
import os
import pickle
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from propy import CTD
from propy import AAComposition


class DataSelection:
    def __init__(self, folder, positive_data, negative_data, pn, nn):
        """
        Select the number of positive and negative samples.
        :param folder:
        :param positive_data: json file with the positive cases
        :param negative_data: json file with the negative cases
        :param pn: positive number. If 'all', the class uses all positive cases
        :param nn: dimension of negative cases. 1 = same size as positives, 2 = double the negatives, etc...
        """
        self.__location__ = os.path.realpath(os.path.join(os.getcwd(), folder))
        self.all_data = {}
        self.positive_dataset = positive_data
        self.negative_dataset = negative_data
        self.pn = pn
        self.nn = nn
        self.dataset_name = None
        self.df = None
        print('Dataset building ... ')

    def read_data(self):
        """
        Read data files
        :return: svm2802
        """
        with open(os.path.join(self.__location__, self.positive_dataset), 'r') as p:
            self.positive_dataset = json.load(p)
        with open(os.path.join(self.__location__, self.negative_dataset), 'r') as n:
            self.negative_dataset = json.load(n)
        if self.pn == 'all':
            self.positive_dataset = self.positive_dataset
        else:
            while len(self.positive_dataset) > self.pn:
                self.positive_dataset.pop(
                    list(self.positive_dataset.keys())[randint(0, len(self.positive_dataset) - 1)])

        print('Positive data dimension:', len(self.positive_dataset))
        print('Negative data dimension:', len(self.negative_dataset))

    def construct_dataset(self):
        """
        Construct dataset based on positive and negative samples
        :return: svm2802
        """
        duplicate_check = []
        for key, v in self.positive_dataset.items():
            if key in self.negative_dataset:
                duplicate_check.append(key)
        if len(duplicate_check) == 0:
            print('Preprocessing Successful')
        else:
            print('You have duplicates in both files')
        self.dataset_name = 'd' + str((len(self.positive_dataset) * (self.nn + 1)))

        dataset_file = Path(self.__location__, self.dataset_name)

        if not dataset_file.is_file():
            for prot in self.positive_dataset.keys():
                self.all_data[prot] = [self.positive_dataset[prot][0], 1]
            i = 0
            while i < self.nn * len(self.positive_dataset):
                prot = list(self.negative_dataset.keys())[randint(0, len(self.negative_dataset) - 1)]
                if prot not in self.all_data.keys():
                    self.all_data[prot] = [self.negative_dataset[prot][0], 0]
                    i += 1
            count_1 = 0
            count_0 = 0
            for key in self.all_data.keys():
                if self.all_data[key][1] == 0:
                    count_0 += 1
                if self.all_data[key][1] == 1:
                    count_1 += 1

            print('Number of positive samples:', count_1)
            print('Number of negative samples:', count_0)
            print('Number of total samples:', len(self.all_data))

            self.df = pd.DataFrame({'Acc_number': list(self.all_data.keys()),
                                    'Annotation': [elem[0] for elem in self.all_data.values()],
                                    'DPO': [elem[1] for elem in self.all_data.values()]})
            self.df = self.df.set_index('Acc_number')

        else:
            with open(os.path.join(self.__location__, dataset_file), 'rb') as d:
                self.df = pickle.load(d)
                print('Data Imported')

    def add_orf_features(self):
        orf_content = {}
        for i in range(len(self.df)):
            p_name = self.df.index[i]
            if self.df.iloc[i]['DPO'] == 1:
                orf_content[p_name] = count_orf(self.positive_dataset[p_name][2])
            elif self.df.iloc[i]['DPO'] == 0:
                orf_content[p_name] = count_orf(self.negative_dataset[p_name][2])
        for ki in self.df.index:
            self.df.loc[ki, orf_content[ki].keys()] = orf_content[ki].values()

    def add_len_aa(self):
        aa_len_dic = {}
        for i in range(len(self.df)):
            p_name = self.df.index[i]
            if self.df.iloc[i]['DPO'] == 1:
                size = len(self.positive_dataset[p_name][1])
                aa_len_dic[p_name] = size
            elif self.df.iloc[i]['DPO'] == 0:
                size = len(self.negative_dataset[p_name][1])
                aa_len_dic[p_name] = size
        for ki in self.df.index:
            self.df.loc[ki, 'AA_Len'] = int(aa_len_dic[ki])

    def add_aa_count(self):
        aa_content = {}
        for i in range(len(self.df)):
            p_name = self.df.index[i]
            if self.df.iloc[i]['DPO'] == 1:
                aa_content[p_name] = count_aa(self.positive_dataset[p_name][1])
            elif self.df.iloc[i]['DPO'] == 0:
                aa_content[p_name] = count_aa(self.negative_dataset[p_name][1])
        for ka in self.df.index:
            self.df.loc[ka, aa_content[ka].keys()] = aa_content[ka].values()

    def add_aromaticity(self):
        aroma_dic = {}
        for i in range(len(self.df)):
            p_name = self.df.index[i]
            if self.df.iloc[i]['DPO'] == 1:
                aroma_dic[p_name] = ProteinAnalysis(self.positive_dataset[p_name][1]).aromaticity()
            elif self.df.iloc[i]['DPO'] == 0:
                aroma_dic[p_name] = ProteinAnalysis(self.negative_dataset[p_name][1]).aromaticity()
        for ki in self.df.index:
            self.df.loc[ki, 'Aromaticity'] = aroma_dic[ki]

    def add_isoelectric_point(self):
        iso_dic = {}
        for i in range(len(self.df)):
            p_name = self.df.index[i]
            if self.df.iloc[i]['DPO'] == 1:
                iso_dic[p_name] = ProteinAnalysis(self.positive_dataset[p_name][1]).isoelectric_point()
            elif self.df.iloc[i]['DPO'] == 0:
                iso_dic[p_name] = ProteinAnalysis(self.negative_dataset[p_name][1]).isoelectric_point()
        for ki in self.df.index:
            self.df.loc[ki, 'IsoelectricPoint'] = iso_dic[ki]

    def add_secondary_structure_fraction(self):
        st_dic_master = {}
        for i in range(len(self.df)):
            p_name = self.df.index[i]
            if self.df.iloc[i]['DPO'] == 1:
                st_dic_master[p_name] = sec_st_fr(self.positive_dataset[p_name][1])
            elif self.df.iloc[i]['DPO'] == 0:
                st_dic_master[p_name] = sec_st_fr(self.negative_dataset[p_name][1])
        for ka in self.df.index:
            self.df.loc[ka, st_dic_master[ka].keys()] = st_dic_master[ka].values()

    def add_composition_transition_distribution(self):
        ctd_dic = {}
        for i in range(len(self.df)):
            p_name = self.df.index[i]
            if self.df.iloc[i]['DPO'] == 1:
                ctd_dic[p_name] = CTD.CalculateCTD(self.positive_dataset[p_name][1])
            elif self.df.iloc[i]['DPO'] == 0:
                ctd_dic[p_name] = CTD.CalculateCTD(self.negative_dataset[p_name][1])
        for ku in self.df.index:
            self.df.loc[ku, ctd_dic[ku].keys()] = ctd_dic[ku].values()

    def add_di_peptide_composition(self):
        dp = {}
        for i in range(len(self.df)):
            p_name = self.df.index[i]
            if self.df.iloc[i]['DPO'] == 1:
                dp[p_name] = AAComposition.CalculateDipeptideComposition(self.positive_dataset[p_name][1])
            elif self.df.iloc[i]['DPO'] == 0:
                dp[p_name] = AAComposition.CalculateDipeptideComposition(self.negative_dataset[p_name][1])
        for ka in self.df.index:
            self.df.loc[ka, dp[ka].keys()] = dp[ka].values()

    def load_features(self):
        """
        Apply features to all dataset
        :return: svm2802
        """
        self.add_orf_features()
        self.add_len_aa()
        self.add_aromaticity()
        self.add_isoelectric_point()
        self.add_aa_count()
        self.add_secondary_structure_fraction()
        self.add_composition_transition_distribution()
        self.add_di_peptide_composition()

    def save_data(self):
        """
        Saves the files in pickle format. EX: d1090
        :return: svm2802
        """
        print(self.dataset_name)
        with open(os.path.join(self.__location__, self.dataset_name), 'wb') as p:
            pickle.dump(self.df, p)
        print('Dataset saved', os.getcwd())


def count_orf(orf_seq):
    dic = {'DNA-A': 0, 'DNA-C': 0, 'DNA-T': 0, 'DNA-G': 0, 'DNA-GC': 0}
    for letter in range(len(orf_seq)):
        for k in range(0, 4):
            if orf_seq[letter] in list(dic.keys())[k][-1]:
                dic[list(dic.keys())[k]] += 1
    dic['DNA-GC'] = ((dic['DNA-C'] + dic['DNA-G']) / (
            dic['DNA-A'] + dic['DNA-C'] + dic['DNA-T'] + dic['DNA-G'])) * 100
    return dic


def count_aa(aa_seq):
    dic = {'G': 0, 'A': 0, 'L': 0, 'V': 0, 'I': 0, 'P': 0, 'F': 0, 'S': 0, 'T': 0, 'C': 0,
           'Y': 0, 'N': 0, 'Q': 0, 'D': 0, 'E': 0, 'R': 0, 'K': 0, 'H': 0, 'W': 0, 'M': 0}
    for letter in range(len(aa_seq)):
        if aa_seq[letter] in dic.keys():
            dic[aa_seq[letter]] += 1
    return dic


def sec_st_fr(aa_seq):
    st_dic = {'Helix': 0, 'Turn': 0, 'Sheet': 0}
    stu = ProteinAnalysis(aa_seq).secondary_structure_fraction()
    st_dic['Helix'] = stu[0]
    st_dic['Turn'] = stu[1]
    st_dic['Sheet'] = stu[2]
    return st_dic

