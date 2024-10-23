import pandas as pd


# load clinical data xslx into a dataframe
def load_clinical_data(path):
    df = pd.read_excel(path)
    return df


def load_hpo_target_phenotypes(path):
    df = pd.read_excel(path)
    return df


def load_cuis_only_data(path):
    df = pd.read_csv(path)
    return df
