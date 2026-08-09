from enum import Enum, IntEnum
from dataclasses import dataclass, field

class Col(str, Enum):
    IMG_NAME = 'image_id'
    MELANOMA = 'melanoma'

class MelanomaType(IntEnum, Enum):
    BENIGN = 0
    MALIGNANT = 1

@dataclass
class DataImage:
    image_id: str = ""
    image_path: str = ""
    label: MelanomaType = MelanomaType.BENIGN
    mask_path: str = ""

@dataclass
class DataImagesGroup:
    data_images: list[DataImage] = field(default_factory=list)
