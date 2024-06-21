from kedro.io import AbstractDataset

class FileWithDirAsLabel(AbstractDataset):
    def __init__(self, filepath: str):
        self.path = filepath

    def _load(self) -> dict:
        p = Path(self.path)
        return {'path': self.path, 'label': p.}