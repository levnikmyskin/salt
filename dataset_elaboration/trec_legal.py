import pandas as pd
import os
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Iterable


def extract_topic_did_label(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\s+', names=['topic', 'i', 'did', 'label'])
    df.loc[(df.label > 1), 'label'] = 0
    return df[['topic', 'did', 'label']]


def from_text_to_tfidf(text: Iterable[str], is_files=False):
    if is_files:
        return TfidfVectorizer(max_df=.9, input='filename', encoding="ISO-8859-1").fit_transform(text)
    return TfidfVectorizer(max_df=.9).fit_transform(text)


if __name__ == '__main__':
    qrels_path = '.data/legal/qrels'
    texts = '.data/legal/doctexts/collection.json'

    dataset = {}
    for topic in os.listdir(qrels_path):
        qrel = extract_topic_did_label(os.path.join(qrels_path, topic))
