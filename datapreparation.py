import numpy as np
from neuralnet import *
import pandas as pd

# Przygotowanie danych

# słowniki zamiana danych typu string na liczbowe
job_dict = {'admin.': 1, 'blue-collar': 2, 'entrepreneur': 3, 'housemaid': 4, 'management': 5, 'retired': 6,
                    'self-employed': 7, 'services': 8, 'student': 9, 'technician': 10, 'unemployed': 11, 'unknown': 0}

marital_dict = {'married': 1, 'single': 2, 'divorced': 3, 'unknown': 0}

education_dict = {'basic.4y': 1, 'basic.6y': 2, 'basic.9y': 3, 'high.school': 4, 'illiterate': 5,
                  'professional.course': 6, 'university.degree': 7, 'unknown': 0}

month_dict = {'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'oct': 10, 'nov': 11, 'dec': 12, 'mar': 3, 'apr': 4, 'sep': 9}

day_of_week_dict = {'fri': 5, 'wed': 3, 'mon': 1, 'thu': 2, 'tue': 4}

default_dict = {'no': 0, 'yes': 1, 'unknown': 2}

housing_dict = {'no': 0, 'yes': 1, 'unknown': 2}

loan_dict = {'no': 0, 'yes': 1, 'unknown': 2}

contact_dict = {'cellular': 1, 'telephone': 2}

poutcome_dict = {'nonexistent': 0, 'failure': 1, 'success': 2}

y_dict = {'no': 0, 'yes': 1}

dictionaries = {'job': job_dict, 'marital': marital_dict, 'education': education_dict,
                'month': month_dict, 'day_of_week': day_of_week_dict, "default": default_dict,
                'housing': housing_dict, 'loan': loan_dict, 'contact': contact_dict, 'poutcome': poutcome_dict,
                'y': y_dict}

division_dict = {'age': 100, 'job': 10, 'marital': 10, 'education': 10, 'default': 10,
                 'housing': 10, 'loan': 10, 'contact': 10, 'month': 10, 'day_of_week': 10,
                 'duration': 10_000, 'campaign': 10, 'pdays': 10, 'previous': 10, 'poutcome': 10,
                 'emp.var.rate': 10, 'cons.price.idx': 100, 'cons.conf.idx': 100,
                 'euribor3m': 10, 'nr.employed': 10_000, 'y': 1}

class DataPreparation:
    """klasa przygotowania danych wejściowych, by te nadawały się do modelu"""
    def __init__(self, file_path):
        """

        :param file_path: ścieżka do pliku csv z danymi wejściowymi
        """
        self.file_path = file_path
        self.df = pd.read_csv(file_path, sep=";")


    def prepare_data(self):
        self.df = pd.read_csv('bank-additional.csv', sep=";")
        #print(self.df)
        # zamiana w kolumnie pdays z 999 na 0 w przypadku braku kontaktu
        self.df.loc[(self.df['pdays'] == 999, 'pdays')] = 22
        self.df['cons.conf.idx'] = self.df['cons.conf.idx'].abs()
        self.df['emp.var.rate'] = self.df['emp.var.rate'].abs()


        #zamieniamy dane tekstowe na liczbowe dla każego słownika
        for key, dictionary in dictionaries.items():
            for k, v in dictionary.items():
                self.df.loc[(self.df[key] == k, key)] = v


        #dzielimy przez odpowiednie wartości
        for k in division_dict:
            #print(self.df.loc[(self.df[k] == k, k)])
            self.df.loc[:, k] = self.df.loc[:, k] / division_dict.get(k)

        # usuwanie kolumn
        '''
        del self.df['emp.var.rate']
        del self.df['cons.price.idx']
        del self.df['cons.conf.idx']
        del self.df['euribor3m']
        del self.df['nr.employed']
        '''
        # 'emp.var.rate', 'cons.price.idx':, 'cons.conf.idx':,
        # 'euribor3m', 'nr.employed':,

        return self.df
