from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class PTBXLConfig:
    ptbxl_root: str = 'C:\\Users\\imneo\\Documents\\Skripsi\\data\\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1'
    sampling_rate: int = 100
    target_length_sec: float = 10.0
    n_leads: int = 12
    target_classes: List[str] = field(default_factory=lambda: ['NORM', 'MI', 'STTC', 'CD', 'HYP'])
    use_bandpass_filter: bool = True
    lowcut: float = 0.5
    highcut: float = 40.0
    filter_order: int = 4
    normalize_mode: str = 'global'
    crop_mode_train: str = 'random'
    crop_mode_eval: str = 'center'
    augment_train: bool = True
    noise_std: float = 0.01
    scale_range: Tuple[float, float] = (0.9, 1.1)
    shift_max: float = 0.1
    use_lead_dropout: bool = True
    lead_dropout_prob: float = 0.15
    lead_dropout_max_leads: int = 2
    test_fold: int = 10
    val_fold: int = 9
    seed: int = 42

    @property
    def target_length(self) -> int:
        return int(self.target_length_sec * self.sampling_rate)
