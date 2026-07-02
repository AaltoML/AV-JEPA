import os
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    name: str
    num_classes: int
    multi_label: bool
    train_tar: str
    train_csv: str
    train_size: int
    test_tar: str
    test_csv: str
    test_size: int
    spec_mean: float
    spec_std: float
    csv_format: str


AUDIOSET_DATA = os.environ.get("AUDIOSET_DIR", "./data/AudioSet")
AUDIOSET_TARS = f"{AUDIOSET_DATA}/shards/data_{{000..453}}.tar"
AUDIOSET_TARS_256 = f"{AUDIOSET_DATA}/shards_256/data_{{000..453}}.tar"

VGGSOUND_DATA = os.environ.get("VGGSOUND_DIR", "./data/VGGSound")

DATASETS = {
    "vggsound": DatasetConfig(
        name="vggsound",
        num_classes=309,
        multi_label=False,
        train_tar=f"{VGGSOUND_DATA}/train_tars/vggsound_train_{{00..71}}.tar",
        train_csv=f"{VGGSOUND_DATA}/train.csv",
        train_size=183_730,
        test_tar=f"{VGGSOUND_DATA}/test_tars/vggsound_test_{{00..03}}.tar",
        test_csv=f"{VGGSOUND_DATA}/test.csv",
        test_size=15_446,
        spec_mean=-20.437003,
        spec_std=24.496246,
        csv_format="vggsound",
    ),
    "vggsound_256": DatasetConfig(
        name="vggsound_256",
        num_classes=309,
        multi_label=False,
        train_tar=f"{VGGSOUND_DATA}/train_tars_256/vggsound_train_{{00..71}}.tar",
        train_csv=f"{VGGSOUND_DATA}/train.csv",
        train_size=183_730,
        test_tar=f"{VGGSOUND_DATA}/test_tars_256/vggsound_test_{{00..03}}.tar",
        test_csv=f"{VGGSOUND_DATA}/test.csv",
        test_size=15_446,
        spec_mean=-17.689175,
        spec_std=24.122319,
        csv_format="vggsound",
    ),
    "audioset": DatasetConfig(
        name="audioset",
        num_classes=527,
        multi_label=True,
        train_tar=AUDIOSET_TARS,
        train_csv=f"{AUDIOSET_DATA}/unbalanced_train_segments.csv",
        train_size=1_910_000,
        test_tar=AUDIOSET_TARS,
        test_csv=f"{AUDIOSET_DATA}/eval_segments.csv",
        test_size=20_371,
        spec_mean=-17.977254,
        spec_std=24.468409,
        csv_format="audioset",
    ),
    "audioset_256": DatasetConfig(
        name="audioset_256",
        num_classes=527,
        multi_label=True,
        train_tar=AUDIOSET_TARS_256,
        train_csv=f"{AUDIOSET_DATA}/unbalanced_train_segments.csv",
        train_size=1_910_000,
        test_tar=AUDIOSET_TARS_256,
        test_csv=f"{AUDIOSET_DATA}/eval_segments.csv",
        test_size=20_371,
        spec_mean=-17.977254,
        spec_std=24.468409,
        csv_format="audioset",
    ),
    "audioset_20k": DatasetConfig(
        name="audioset_20k",
        num_classes=527,
        multi_label=True,
        train_tar=AUDIOSET_TARS,
        train_csv=f"{AUDIOSET_DATA}/balanced_train_segments.csv",
        train_size=22_163,
        test_tar=AUDIOSET_TARS,
        test_csv=f"{AUDIOSET_DATA}/eval_segments.csv",
        test_size=20_371,
        spec_mean=-17.977254,
        spec_std=24.468409,
        csv_format="audioset",
    ),
}
