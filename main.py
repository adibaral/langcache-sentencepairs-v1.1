from datasets import load_dataset
import os
import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets
from datasets import Features, Value


def convert_label_to_int(example):
    """
    Map the label to an integer.

    For the PIT-2015 dataset, we follow this convention:

    train:
        paraphrases: (3, 2) (4, 1) (5, 0)
        non-paraphrases: (1, 4) (0, 5)
        debatable: (2, 3)  which you may discard if training binary classifier
    """
    try:
        example["label"] = int(example["label"])
    except:
        if example["label"] in ["(3, 2)", "(4, 1)", "(5, 0)"]:
            example["label"] = 1
        elif example["label"] in ["(1, 4)", "(0, 5)", "(2, 3)"]:
            example["label"] = 0
    return example


def standardize_labels(dataset):
    dataset = dataset.map(convert_label_to_int)
    dataset = dataset.cast(
        Features(
            {
                "sentence1": Value("string"),
                "sentence2": Value("string"),
                "label": Value("int8"),
            }
        )
    )
    return dataset


def load_paws_dataset():
    """
    Load the PAWS dataset from the Hugging Face Hub.

    Subset: unlabeled_final
    Splits: train, validation, test
    """
    dataset = load_dataset("google-research-datasets/paws", "unlabeled_final")
    train_dataset = dataset["train"]
    val_dataset = None
    test_dataset = dataset["validation"]
    # drop the columns that are not needed
    train_dataset = train_dataset.select_columns(["sentence1", "sentence2", "label"])
    test_dataset = test_dataset.select_columns(["sentence1", "sentence2", "label"])
    # standardize labels
    train_dataset = standardize_labels(train_dataset)
    test_dataset = standardize_labels(test_dataset)
    return train_dataset, val_dataset, test_dataset


def load_mrpc_dataset():
    """
    Load the MRPC dataset from the Hugging Face Hub.

    Splits: train, validation, test
    """
    dataset = load_dataset("nyu-mll/glue", "mrpc")
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]
    # drop the columns that are not needed
    train_dataset = train_dataset.select_columns(["sentence1", "sentence2", "label"])
    val_dataset = val_dataset.select_columns(["sentence1", "sentence2", "label"])
    test_dataset = test_dataset.select_columns(["sentence1", "sentence2", "label"])
    # standardize labels
    train_dataset = standardize_labels(train_dataset)
    val_dataset = standardize_labels(val_dataset)
    test_dataset = standardize_labels(test_dataset)
    return train_dataset, val_dataset, test_dataset


def load_qqp_dataset():
    """
    Load the QQP dataset from the Hugging Face Hub.

    Splits: train, validation, test
    """
    dataset = load_dataset("nyu-mll/glue", "qqp")
    train_dataset = dataset["train"]
    test_dataset = dataset["validation"]
    # rename the columns
    train_dataset = train_dataset.rename_columns(
        {"question1": "sentence1", "question2": "sentence2"}
    )
    test_dataset = test_dataset.rename_columns(
        {"question1": "sentence1", "question2": "sentence2"}
    )
    # drop the columns that are not needed
    train_dataset = train_dataset.select_columns(["sentence1", "sentence2", "label"])
    test_dataset = test_dataset.select_columns(["sentence1", "sentence2", "label"])
    # standardize labels
    train_dataset = standardize_labels(train_dataset)
    test_dataset = standardize_labels(test_dataset)
    return train_dataset, None, test_dataset


def load_stsb_dataset():
    """
    Load the STS-B dataset from the Hugging Face Hub.
    """
    dataset = load_dataset("glue", "stsb")
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]
    # drop the columns that are not needed
    train_dataset = train_dataset.select_columns(["sentence1", "sentence2", "label"])
    val_dataset = val_dataset.select_columns(["sentence1", "sentence2", "label"])
    test_dataset = test_dataset.select_columns(["sentence1", "sentence2", "label"])
    # map the labels. Anything > 3.5 is a paraphrase, anything <= 3.5 is a non-paraphrase
    train_dataset = train_dataset.map(lambda x: {"label": 1 if x["label"] > 3.5 else 0})
    val_dataset = val_dataset.map(lambda x: {"label": 1 if x["label"] > 3.5 else 0})
    test_dataset = test_dataset.map(lambda x: {"label": 1 if x["label"] > 3.5 else 0})
    # standardize labels
    train_dataset = standardize_labels(train_dataset)
    val_dataset = standardize_labels(val_dataset)
    test_dataset = standardize_labels(test_dataset)
    return train_dataset, val_dataset, test_dataset


def load_opusparcus_dataset():
    """
    Load the OpusParCus dataset from the Hugging Face Hub.
    """
    dataset = load_dataset("GEM/opusparcus", lang="en", quality=80)
    train_dataset = dataset["train"]
    val_dataset = dataset["validation.full"]
    test_dataset = dataset["test.full"]
    # rename the columns
    train_dataset = train_dataset.rename_columns(
        {"sent1": "sentence1", "sent2": "sentence2", "annot_score": "label"}
    )
    val_dataset = val_dataset.rename_columns(
        {"sent1": "sentence1", "sent2": "sentence2", "annot_score": "label"}
    )
    test_dataset = test_dataset.rename_columns(
        {"sent1": "sentence1", "sent2": "sentence2", "annot_score": "label"}
    )
    # drop the columns that are not needed
    train_dataset = train_dataset.select_columns(["sentence1", "sentence2", "label"])
    val_dataset = val_dataset.select_columns(["sentence1", "sentence2", "label"])
    # map the labels. On the train, everything should be 1.0. On valid and test, [0, 2] -> non-paraphrases, [3, 4] -> paraphrases (can be float)
    train_dataset = train_dataset.map(lambda x: {"label": 1.0})
    val_dataset = val_dataset.map(lambda x: {"label": 1.0 if x["label"] >= 3 else 0.0})
    test_dataset = test_dataset.map(
        lambda x: {"label": 1.0 if x["label"] >= 3 else 0.0}
    )
    # standardize labels
    train_dataset = standardize_labels(train_dataset)
    val_dataset = standardize_labels(val_dataset)
    test_dataset = standardize_labels(test_dataset)
    return train_dataset, val_dataset, test_dataset


def load_parade_dataset(dir: str = "data/parade"):
    """
    Load the PARADE dataset from the local directory.

    Splits: train, validation, test
    """
    train_path = os.path.join(dir, "PARADE_train.txt")
    val_path = os.path.join(dir, "PARADE_validation.txt")
    test_path = os.path.join(dir, "PARADE_test.txt")
    # load the tsv files
    train_df = pd.read_csv(train_path, sep="\t")
    val_df = pd.read_csv(val_path, sep="\t")
    test_df = pd.read_csv(test_path, sep="\t")
    # rename the columns
    train_df.rename(
        columns={
            "Definition1": "sentence1",
            "Definition2": "sentence2",
            "Binary labels": "label",
        },
        inplace=True,
    )
    val_df.rename(
        columns={
            "Definition1": "sentence1",
            "Definition2": "sentence2",
            "Binary labels": "label",
        },
        inplace=True,
    )
    test_df.rename(
        columns={
            "Definition1": "sentence1",
            "Definition2": "sentence2",
            "Binary labels": "label",
        },
        inplace=True,
    )
    # drop the other columns that are not needed
    train_df = train_df[["sentence1", "sentence2", "label"]]
    val_df = val_df[["sentence1", "sentence2", "label"]]
    test_df = test_df[["sentence1", "sentence2", "label"]]
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    # standardize labels
    train_dataset = standardize_labels(train_dataset)
    val_dataset = standardize_labels(val_dataset)
    test_dataset = standardize_labels(test_dataset)

    return train_dataset, val_dataset, test_dataset


def load_ttic31190_dataset(
    dir: str = "/opt/dlami/nvme/crossencoder-training-data/ttic-31190",
):
    """
    Load the TTIC-31190 dataset from the local directory.

    Splits: train, validation, test
    """
    # TODO: Sample negative examples from the training set to make it balanced
    train_path = os.path.join(dir, "train.tsv")
    val_path = os.path.join(dir, "dev.tsv")
    test_path = os.path.join(dir, "devtest.tsv")

    # load the tsv files (no header, so specify column names)
    # train.tsv has only 2 columns (no labels), dev.tsv and devtest.tsv have 3 columns
    train_df = pd.read_csv(
        train_path,
        sep="\t",
        header=None,
        names=["sentence1", "sentence2"],
        quoting=3,
        encoding="utf-8",
    )  # quoting=3 means no quoting
    train_df["label"] = 1  # All training examples are positive pairs
    val_df = pd.read_csv(
        val_path,
        sep="\t",
        header=None,
        names=["sentence1", "sentence2", "label"],
        quoting=3,
        encoding="utf-8",
    )  # quoting=3 means no quoting
    test_df = pd.read_csv(
        test_path,
        sep="\t",
        header=None,
        names=["sentence1", "sentence2", "label"],
        quoting=3,
        encoding="utf-8",
    )  # quoting=3 means no quoting
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    # standardize labels
    train_dataset = standardize_labels(train_dataset)
    val_dataset = standardize_labels(val_dataset)
    test_dataset = standardize_labels(test_dataset)
    return train_dataset, val_dataset, test_dataset


def load_pit2015_dataset(
    dir: str = "data/pit2015",
):
    """
    Load the PIT-2015 dataset from the local directory.

    Splits: train, validation, test
    """
    train_path = os.path.join(dir, "train.data")
    val_path = os.path.join(dir, "dev.data")
    test_data_path = os.path.join(dir, "test.data")
    # test_labels_path = os.path.join(dir, "test_labels.tsv")
    # load the data files
    #  Topic_Id | Topic_Name | Sent_1 | Sent_2 | Label | Sent_1_tag | Sent_2_tag |
    train_df = pd.read_csv(
        train_path,
        sep="\t",
        header=None,
        names=[
            "Topic_Id",
            "Topic_Name",
            "Sent_1",
            "Sent_2",
            "Label",
            "Sent_1_tag",
            "Sent_2_tag",
        ],
    )
    val_df = pd.read_csv(
        val_path,
        sep="\t",
        header=None,
        names=[
            "Topic_Id",
            "Topic_Name",
            "Sent_1",
            "Sent_2",
            "Label",
            "Sent_1_tag",
            "Sent_2_tag",
        ],
    )
    test_df = pd.read_csv(
        test_data_path,
        sep="\t",
        header=None,
        names=[
            "Topic_Id",
            "Topic_Name",
            "Sent_1",
            "Sent_2",
            "Label",
            "Sent_1_tag",
            "Sent_2_tag",
        ],
    )
    # test_labels_df = pd.read_csv(test_labels_path, sep="\t", header=None, names=["Topic_Id", "Topic_Name", "Label"])
    # rename the columns
    train_df.rename(
        columns={"Sent_1": "sentence1", "Sent_2": "sentence2", "Label": "label"},
        inplace=True,
    )
    val_df.rename(
        columns={"Sent_1": "sentence1", "Sent_2": "sentence2", "Label": "label"},
        inplace=True,
    )
    test_df.rename(
        columns={"Sent_1": "sentence1", "Sent_2": "sentence2", "Label": "label"},
        inplace=True,
    )
    # drop the columns that are not needed
    train_df = train_df[["sentence1", "sentence2", "label"]]
    val_df = val_df[["sentence1", "sentence2", "label"]]
    test_df = test_df[["sentence1", "sentence2", "label"]]
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    # standardize labels
    train_dataset = standardize_labels(train_dataset)
    val_dataset = standardize_labels(val_dataset)
    test_dataset = standardize_labels(test_dataset)
    # convert test labels to [0, 1], this is the standard convention:
    #   paraphrases: 4 or 5
    #   non-paraphrases: 0 or 1 or 2
    #   debatable: 3 (discarded in Paraphrase Identification evaluation)
    # but we will consider debatable as non-paraphrases
    test_dataset = test_dataset.map(
        lambda x: {"label": 1 if x["label"] in [4, 5] else 0}
    )
    return train_dataset, val_dataset, test_dataset


def load_apt_dataset(dir: str = "data/apt"):
    """
    Load the APT dataset from the local directory.
    """
    train_path = os.path.join(dir, "train.tsv")
    test_path = os.path.join(dir, "test.tsv")
    # load the tsv files
    train_df = pd.read_csv(train_path, sep="\t")
    test_df = pd.read_csv(test_path, sep="\t")
    # rename the columns
    train_df.rename(
        columns={"text_a": "sentence1", "text_b": "sentence2", "labels": "label"},
        inplace=True,
    )
    test_df.rename(
        columns={"text_a": "sentence1", "text_b": "sentence2", "labels": "label"},
        inplace=True,
    )
    # drop the columns that are not needed
    train_df = train_df[["sentence1", "sentence2", "label"]]
    test_df = test_df[["sentence1", "sentence2", "label"]]
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    # standardize labels
    train_dataset = standardize_labels(train_dataset)
    test_dataset = standardize_labels(test_dataset)
    return train_dataset, None, test_dataset


def load_sick_dataset(dir: str = "data/sick"):
    """
    Load the SICK dataset from the local directory.
    """
    file_path = os.path.join(dir, "SICK.txt")
    # load the tsv file
    df = pd.read_csv(file_path, sep="\t")
    # rename the columns
    df.rename(
        columns={
            "sentence_A": "sentence1",
            "sentence_B": "sentence2",
            "entailment_label": "label",
        },
        inplace=True,
    )
    # get the train, validation and test splits
    train_df = df[df["SemEval_set"] == "TRAIN"]
    val_df = df[df["SemEval_set"] == "TRIAL"]
    test_df = df[df["SemEval_set"] == "TEST"]
    # drop the columns that are not needed
    train_df = train_df[["sentence1", "sentence2", "label"]]
    val_df = val_df[["sentence1", "sentence2", "label"]]
    test_df = test_df[["sentence1", "sentence2", "label"]]
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    # drop the __index_level_0__ column
    train_dataset = train_dataset.remove_columns(["__index_level_0__"])
    val_dataset = val_dataset.remove_columns(["__index_level_0__"])
    test_dataset = test_dataset.remove_columns(["__index_level_0__"])
    # map the labels. If label is CONTRADICTION, then it is a non-paraphrase, else it is a paraphrase
    train_dataset = train_dataset.map(
        lambda x: {"label": 0 if x["label"] == "CONTRADICTION" else 1}
    )
    val_dataset = val_dataset.map(
        lambda x: {"label": 0 if x["label"] == "CONTRADICTION" else 1}
    )
    test_dataset = test_dataset.map(
        lambda x: {"label": 0 if x["label"] == "CONTRADICTION" else 1}
    )
    # standardize labels
    train_dataset = standardize_labels(train_dataset)
    val_dataset = standardize_labels(val_dataset)
    test_dataset = standardize_labels(test_dataset)
    return train_dataset, val_dataset, test_dataset


def load_parasci_dataset(dir: str):
    """
    Load the Parasci dataset from the local directory.
    """
    train_dir = os.path.join(dir, "train")
    val_dir = os.path.join(dir, "val")
    test_dir = os.path.join(dir, "test")
    train_src_path = os.path.join(train_dir, "train.src")
    train_tgt_path = os.path.join(train_dir, "train.tgt")
    val_src_path = os.path.join(val_dir, "val.src")
    val_tgt_path = os.path.join(val_dir, "val.tgt")
    test_src_path = os.path.join(test_dir, "test.src")
    test_tgt_path = os.path.join(test_dir, "test.tgt")
    # load the files
    with open(train_src_path, "r", encoding="utf-8") as f:
        train_sentence_1 = f.readlines()
    with open(train_tgt_path, "r", encoding="utf-8") as f:
        train_sentence_2 = f.readlines()
    with open(val_src_path, "r", encoding="utf-8") as f:
        val_sentence_1 = f.readlines()
    with open(val_tgt_path, "r", encoding="utf-8") as f:
        val_sentence_2 = f.readlines()
    with open(test_src_path, "r", encoding="utf-8") as f:
        test_sentence_1 = f.readlines()
    with open(test_tgt_path, "r", encoding="utf-8") as f:
        test_sentence_2 = f.readlines()
    # create the labels
    train_label = [1] * len(train_sentence_1)
    val_label = [1] * len(val_sentence_1)
    test_label = [1] * len(test_sentence_1)
    # create the datasets
    train_dataset = Dataset.from_dict(
        {
            "sentence1": train_sentence_1,
            "sentence2": train_sentence_2,
            "label": train_label,
        }
    )
    val_dataset = Dataset.from_dict(
        {"sentence1": val_sentence_1, "sentence2": val_sentence_2, "label": val_label}
    )
    test_dataset = Dataset.from_dict(
        {
            "sentence1": test_sentence_1,
            "sentence2": test_sentence_2,
            "label": test_label,
        }
    )
    # standardize labels
    train_dataset = standardize_labels(train_dataset)
    val_dataset = standardize_labels(val_dataset)
    test_dataset = standardize_labels(test_dataset)
    return train_dataset, val_dataset, test_dataset


def read_paralex_dataset(
    dir: str = "/opt/dlami/nvme/crossencoder-training-data/wikianswers-paraphrases-1.0",
):
    """
    Read the Paralex dataset from the local directory.
    """
    file_path = os.path.join(dir, "word_alignments.txt")
    # load the tsv file
    df = pd.read_csv(
        file_path, sep="\t", header=None, names=["sentence1", "sentence2", "alignment"]
    )
    # the dataset has (sentence1, sentence2) and also (sentence2, sentence1)
    # remove the duplicates by creating a sorted tuple of the two sentences
    df["sorted_pair"] = df.apply(
        lambda row: tuple(sorted([row["sentence1"], row["sentence2"]])), axis=1
    )
    df = df.drop_duplicates(subset=["sorted_pair"])
    df = df.drop(columns=["sorted_pair"])

    df["label"] = 1
    # drop the alignment column
    df = df.drop(columns=["alignment"])
    # create the datasets
    train_dataset = Dataset.from_pandas(df)
    # standardize labels
    train_dataset = standardize_labels(train_dataset)
    # drop __index_level_0__ column
    train_dataset = train_dataset.remove_columns(["__index_level_0__"])
    return train_dataset, None, None


def remove_null_examples(dataset):
    """
    Remove examples with null values in the sentence1 or sentence2 columns.
    """
    return dataset.filter(
        lambda x: x.get("sentence1") is not None and x.get("sentence2") is not None
    )


if __name__ == "__main__":
    # embedding_model = SentenceTransformer("Alibaba-NLP/gte-modernbert-base")
    callables = {
        "paws": load_paws_dataset,
        "mrpc": load_mrpc_dataset,
        "qqp": load_qqp_dataset,
        "parade": load_parade_dataset,
        # "ttic31190": load_ttic31190_dataset, # TODO: Need to sample negatives
        "pit2015": load_pit2015_dataset,
        # "opusparcus": load_opusparcus_dataset, # TODO: There is some error here. The dataset is not loading.
        "apt": load_apt_dataset,
        "stsb": load_stsb_dataset,
        "sick": load_sick_dataset,
        # "parasci-acl": lambda: load_parasci_dataset("crossencoder-training-data/parasci/Data/ParaSCI-ACL"), # TODO: Need to sample negatives
        # "parasci-arxiv": lambda: load_parasci_dataset("crossencoder-training-data/parasci/Data/ParaSCI-arXiv"), # TODO: Need to sample negatives
        # "paralex": read_paralex_dataset, # TODO: Need to sample negatives
    }
    # create a single large dataset. it should have a subset for each dataset, and an "all" subset
    # each subset should have a train, validation and test split, with the "all" subset having all the data from the other subsets
    dataset = DatasetDict()
    train_datasets, val_datasets, test_datasets = [], [], []
    for name, callable in callables.items():
        train_dataset, val_dataset, test_dataset = callable()
        train_dataset = remove_null_examples(train_dataset)
        dataset[name] = DatasetDict()
        dataset[name]["train"] = train_dataset
        if val_dataset is not None:
            val_dataset = remove_null_examples(val_dataset)
            dataset[name]["validation"] = val_dataset
        if test_dataset is not None:
            test_dataset = remove_null_examples(test_dataset)
            dataset[name]["test"] = test_dataset
        dataset[name].push_to_hub(
            f"aditeyabaral-redis/langcache-sentencepairs-v1.1", config_name=name
        )

        # add source dataset name and idx to the dataset with formatted source_id
        train_padding = len(str(len(train_dataset) - 1))
        train_dataset = train_dataset.map(
            lambda ex, idx: {"source_idx": idx, "source": name}, with_indices=True
        )
        train_datasets.append(train_dataset)
        if val_dataset is not None:
            val_padding = len(str(len(val_dataset) - 1))
            val_dataset = val_dataset.map(
                lambda ex, idx: {"source_idx": idx, "source": name}, with_indices=True
            )
            val_datasets.append(val_dataset)
        if test_dataset is not None:
            test_padding = len(str(len(test_dataset) - 1))
            test_dataset = test_dataset.map(
                lambda ex, idx: {"source_idx": idx, "source": name}, with_indices=True
            )
            test_datasets.append(test_dataset)

    # print dataset
    print(dataset)

    # merge the datasets
    merged_dataset = DatasetDict()
    merged_dataset["train"] = concatenate_datasets(train_datasets)
    merged_dataset["validation"] = concatenate_datasets(val_datasets)
    merged_dataset["test"] = concatenate_datasets(test_datasets)
    
    # add formatted idx column to each dataset
    train_total = len(merged_dataset["train"])
    val_total = len(merged_dataset["validation"])
    test_total = len(merged_dataset["test"])
    
    # Determine padding width based on total number of examples
    train_padding = len(str(train_total - 1))
    val_padding = len(str(val_total - 1))
    test_padding = len(str(test_total - 1))
    
    merged_dataset["train"] = merged_dataset["train"].map(
        lambda ex, idx: {"id": f"langcache_train_{idx:0{train_padding}d}"}, with_indices=True
    )
    merged_dataset["validation"] = merged_dataset["validation"].map(
        lambda ex, idx: {"id": f"langcache_validation_{idx:0{val_padding}d}"}, with_indices=True
    )
    merged_dataset["test"] = merged_dataset["test"].map(
        lambda ex, idx: {"id": f"langcache_test_{idx:0{test_padding}d}"}, with_indices=True
    )

    for split in ["train", "validation", "test"]:
        merged_dataset[split] = merged_dataset[split].cast(
            Features(
                {
                    "id": Value("string"),
                    "source_idx": Value("int32"),
                    "source": Value("string"),
                    "sentence1": Value("string"),
                    "sentence2": Value("string"),
                    "label": Value("int8"),
                }
            )
        )
    
    # Reorder columns to desired order: id, source_idx, source, sentence1, sentence2, label
    column_order = ["id", "source_idx", "source", "sentence1", "sentence2", "label"]
    merged_dataset["train"] = merged_dataset["train"].select_columns(column_order)
    merged_dataset["validation"] = merged_dataset["validation"].select_columns(column_order)
    merged_dataset["test"] = merged_dataset["test"].select_columns(column_order)
    
    print(merged_dataset)
    # push the dataset to the hub
    merged_dataset.push_to_hub(
        "aditeyabaral-redis/langcache-sentencepairs-v1.1", config_name="all"
    )
