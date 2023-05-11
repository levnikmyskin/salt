from dataset_elaboration.trec_legal import *
from tqdm import tqdm
from typing import Generator, Tuple
import pickle
import concurrent.futures


def elaborate_topic(topic, qrel, text_path, saving_path, text_dids):
    dids = set(d for d in qrel[qrel.topic == topic].did if d in text_dids)
    if not dids:
        return
    x = from_text_to_tfidf(
        map(lambda d: os.path.join(text_path, str(d).zfill(6)), dids), is_files=True
    )
    y = (
        qrel[(qrel.topic == topic) & (qrel.did.isin(dids))]
        .drop_duplicates("did")
        .label.to_numpy()
    )
    with open(os.path.join(saving_path, f"{topic}.pkl"), "wb") as f:
        pickle.dump((x, y), f)


def load_dataset(
    processed_path=".data/jebbush-sample/processed",
) -> Generator[Tuple[str, NDArray[float], NDArray[int]], None, None]:
    for p in os.listdir(processed_path):
        with open(os.path.join(processed_path, p), "rb") as f:
            x, y = pickle.load(f)
        yield p.split(".pkl")[0], x, y


if __name__ == "__main__":
    qrels_path = ".data/jebbush-sample/athome4.qrel.sample"
    text_path = ".data/jebbush-sample/athome4_test"
    saving_path = ".data/jebbush-sample/processed"

    text_dids = set(map(int, os.listdir(text_path)))
    qrel = extract_topic_did_label(qrels_path)
    dataset = {}
    os.makedirs(saving_path, exist_ok=True)
    jobs = max(len(qrel.topic.unique()), 40)
    with concurrent.futures.ProcessPoolExecutor(jobs) as p:
        futures = []
        for topic in qrel.topic.unique():
            futures.append(
                p.submit(
                    elaborate_topic, topic, qrel, text_path, saving_path, text_dids
                )
            )

        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            _.result()
