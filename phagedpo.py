import pickle
from Bio import SeqIO
import os
import pandas as pd
import numpy as np
from local_ctd import CalculateCTD
from local_AAComposition import CalculateDipeptideComposition
from Bio.SeqUtils.ProtParam import ProteinAnalysis


class PDPOPrediction:

    def __init__(self, folder='location', mdl='', seq_file='fasta_file.fasta', ttable=11):
        """
        Initialize PhageDPO prediction.
        :param folder: data path
        :param mdl: ml model, in this case SVM
        :param seq_file: fasta file
        :param ttable: Translational table. By default, The Bacterial, Archaeal and Plant Plastid Code Table 11
        """
        self.records = []
        self.data = {}
        self.df_output = None
        self.seqfile = seq_file
        self.__location__ = os.path.realpath(os.path.join(os.getcwd(), folder))

        with open(os.path.join(self.__location__, mdl), 'rb') as m:
            self.model0 = pickle.load(m)
            self.model = self.model0.named_steps['clf']
            self.scaler = self.model0.named_steps['scl']
            self.selectk = self.model0.named_steps['selector']
            self.name = 'model'

        for seq in SeqIO.parse(os.path.join(self.__location__, self.seqfile), 'fasta'):
            record = []
            DNA_seq = seq.seq
            AA_seq = DNA_seq.translate(table=ttable)
            descr_seq = seq.description.replace(' ', '')
            self.data[descr_seq] = [DNA_seq._data, AA_seq._data]
            record.append(seq.description)
            record.append(DNA_seq._data)
            record.append(AA_seq._data)
            self.records.append(record)

        columns = ['ID', 'DNAseq', 'AAseq']
        self.df = pd.DataFrame(self.records, columns=columns)
        #self.df = self.df.set_index('ID')
        self.df.update(self.df.DNAseq[self.df.DNAseq.apply(type) == list].str[0])
        self.df.update(self.df.AAseq[self.df.AAseq.apply(type) == list].str[0])

    def Datastructure(self):
        """
        Create dataset with all features
        """
        def count_orf(orf_seq):
            """
            Function to count open reading frames
            :param orf_seq: sequence to analyze
            :return: dictionary with open reading frames
            """
            dic = {'DNA-A': 0, 'DNA-C': 0, 'DNA-T': 0, 'DNA-G': 0, 'DNA-GC': 0}
            for letter in range(len(orf_seq)):
                for k in range(0, 4):
                    if str(orf_seq[letter]) in list(dic.keys())[k][-1]:
                        dic[list(dic.keys())[k]] += 1
            dic['DNA-GC'] = ((dic['DNA-C'] + dic['DNA-G']) / (
                    dic['DNA-A'] + dic['DNA-C'] + dic['DNA-T'] + dic['DNA-G'])) * 100
            return dic

        def count_aa(aa_seq):
            """
            Function to count amino acids
            :param aa_seq: sequence to analyze
            :return: dictionary with amino acid composition
            """
            dic = {'G': 0, 'A': 0, 'L': 0, 'V': 0, 'I': 0, 'P': 0, 'F': 0, 'S': 0, 'T': 0, 'C': 0,
                   'Y': 0, 'N': 0, 'Q': 0, 'D': 0, 'E': 0, 'R': 0, 'K': 0, 'H': 0, 'W': 0, 'M': 0}
            for letter in range(len(aa_seq)):
                if aa_seq[letter] in dic.keys():
                    dic[aa_seq[letter]] += 1
            return dic

        def sec_st_fr(aa_seq):
            """
            Function to analyze secondary structure. Helix, Turn and Sheet
            :param aa_seq: sequence to analyze
            :return: dictionary with composition of each secondary structure
            """
            st_dic = {'Helix': 0, 'Turn': 0, 'Sheet': 0}
            stu = ProteinAnalysis(aa_seq).secondary_structure_fraction()
            st_dic['Helix'] = stu[0]
            st_dic['Turn'] = stu[1]
            st_dic['Sheet'] = stu[2]
            return st_dic


        self.df_output = self.df.copy()
        self.df_output.drop(['DNAseq', 'AAseq'], axis=1, inplace=True)
        dna_feat = {}
        aa_len = {}
        aroma_dic = {}
        iso_dic = {}
        aa_content = {}
        st_dic_master = {}
        CTD_dic = {}
        dp = {}
        self.df1 = self.df[['ID']].copy()
        self.df.drop(['ID'], axis=1, inplace=True)
        for i in range(len(self.df)):
            i_name = self.df.index[i]
            dna_feat[i] = count_orf(self.df.iloc[i]['DNAseq'])
            aa_len[i] = len(self.df.iloc[i]['AAseq'])
            aroma_dic[i] = ProteinAnalysis(self.df.iloc[i]['AAseq']).aromaticity()
            iso_dic[i] = ProteinAnalysis(self.df.iloc[i]['AAseq']).isoelectric_point()
            aa_content[i] = count_aa(self.df.iloc[i]['AAseq'])
            st_dic_master[i] = sec_st_fr(self.df.iloc[i]['AAseq'])
            CTD_dic[i] = CalculateCTD(self.df.iloc[i]['AAseq'])
            dp[i] = CalculateDipeptideComposition(self.df.iloc[i]['AAseq'])
        for j in self.df.index:
            self.df.loc[j, dna_feat[j].keys()] = dna_feat[j].values() #dic with multiple values
            self.df.loc[j, 'AA_Len'] = int(aa_len[j]) #dic with one value
            self.df.loc[j, 'Aromaticity'] = aroma_dic[j]
            self.df.loc[j, 'IsoelectricPoint'] = iso_dic[j]
            self.df.loc[j, aa_content[j].keys()] = aa_content[j].values()
            self.df.loc[j, st_dic_master[j].keys()] = st_dic_master[j].values()
            self.df.loc[j, CTD_dic[j].keys()] = CTD_dic[j].values()
            self.df.loc[j, dp[j].keys()] = dp[j].values()
        self.df.drop(['DNAseq', 'AAseq'], axis=1, inplace=True)

    def Prediction(self):
        """
        Predicts the percentage of each CDS being depolymerase.
        :return: model prediction
        """
        scores = self.model0.predict_proba(self.df.iloc[:, :])
        pos_scores = np.empty((self.df.shape[0], 0), float)
        for x in scores:
            pos_scores = np.append(pos_scores, round(x[1]*100))
        self.df_output.reset_index(inplace=True)
        self.df_output.rename(columns={'index': 'CDS'}, inplace=True)
        self.df_output['CDS'] += 1
        self.df_output['{} DPO Prediction (%)'.format(self.name)] = pos_scores
        self.df_output.to_html('output.html', index=False, justify='center')




