import pandas as pd

file_path = 'data-tokenized.tsv'

def read_file(file_path):

    data = pd.read_csv(file_path, sep='\t', encoding='latin-1')

    # replace missing annotations with label 'sufficient'
    data['ANNOTATION'].fillna('sufficient', inplace=True)

    # convert to binary labels for classification task
    data['label'] = data['ANNOTATION'].map({'sufficient': 1, 'insufficient': 0})

    argument_list = data['TEXT'].to_list()
    labels = data['label'].to_list()

    return data, argument_list, labels

#print(read_file(file_path))

