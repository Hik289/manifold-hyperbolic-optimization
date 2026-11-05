"""Read an existing, provenance-checked image manifest without copying the cache."""
import hashlib
import io
import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class ManifestImages(Dataset):
    def __init__(self, records, transform, classes):
        self.records, self.transform, self.classes = records, transform, classes
        self.arrow_cache = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        if row.get('storage') == 'hf_arrow':
            from datasets import Dataset as HFDataset, concatenate_datasets, Image as HFImage
            folder, ref = row['path'].split('#')
            split, offset = ref.split('/')
            key = (folder, split)
            if key not in self.arrow_cache:
                files = sorted(Path(folder).glob(f'imagenet-100-{split}*.arrow'))
                if not files:
                    raise FileNotFoundError(f'No Arrow shards for {key}')
                self.arrow_cache[key] = concatenate_datasets(
                    [HFDataset.from_file(str(f)) for f in files]).cast_column('image', HFImage(decode=False))
            picture = self.arrow_cache[key][int(offset)]['image']
            source = io.BytesIO(picture['bytes']) if picture.get('bytes') is not None else picture['path']
        else:
            source = row['path']
        with Image.open(source) as image:
            x = self.transform(image.convert('RGB'))
        return x, int(row['label'])


def load_manifest_images(folder, transform):
    folder = Path(folder)
    ready = json.loads((folder / 'ready.json').read_text())
    raw = (folder / 'manifest.jsonl').read_bytes()
    if not ready.get('ready') or hashlib.sha256(raw).hexdigest() != ready['manifest_sha256']:
        raise ValueError('Image manifest is unvalidated or has changed')
    rows = [json.loads(line) for line in raw.splitlines()]
    if ready['n_classes'] != 100 or set(r['label'] for r in rows) != set(range(100)):
        raise ValueError('Expected all 100 ImageNet-100 classes')
    sets = []
    for split in ['train', 'validation', 'test']:
        part = [r for r in rows if r['split'] == split]
        if len(part) != ready['counts'][split]:
            raise ValueError(f'Image count mismatch: {split}')
        sets.append(ManifestImages(part, transform, ready['classes']))
    paths = [set(r['path'] for r in data.records) for data in sets]
    if any(paths[i] & paths[j] for i in range(3) for j in range(i)):
        raise ValueError('Image manifest splits overlap')
    return (*sets, 100)
