import pandas as pd
import numpy as np
from joblib import dump
import csv

from ml_pipeline import experiment


def main(): 
    for dataset in ['dataset1', 'dataset2']:
        for representation in [
            'counts',
            'tfidf',
            'embed'
        ]:
            print(f'>> Running experiment for dataset {dataset} representation {representation}')

            d_train, d_test = (
                pd.read_csv(f'../pynlp/data/{dataset}/{split}Data.csv', delimiter='\t')
                for split in ['train', 'test']
            )
            model = 'libsvc' if representation in ['counts', 'tfidf'] else 'sigmoid'
            res = experiment.run(
                task_name='vua_format',
                data_dir=f'../pynlp/data/{dataset}/',
                pipeline_name=f'svm_{model}_{representation}_lex',
                print_predictions=False
            )
            scores_train, scores_test = (
                res.decision_function(d['Text']) for d in [d_train, d_test]
            )
            # TODO: Should ensure all result/<dataset>/<representation> folders exist 
            for d, scores, split in zip([d_train, d_test], [scores_train, scores_test], ['train', 'test']):
                d['Score'] = scores
                d['Prediction'] =  np.greater_equal(scores, 0)
                d['Prediction'] = d['Prediction'].replace({True: 'OFF', False: 'NOT'})
                d.to_csv(
                    f'result/{dataset}/{representation}/{split}Data.csv',
                    sep='\t',
                    index=False,
                    quoting=csv.QUOTE_ALL
                )
            
            dump(res['clf'], f'result/{dataset}/{representation}/model.joblib')



if __name__ == "__main__":
    main()
