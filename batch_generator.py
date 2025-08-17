import os
import json
import numpy as np
from PIL import Image

class BatchGenerator:
    TRAIN = "train"
    TEST = "test"
    VAL = "val"

    def __init__(self, mode, batch_size, class_to_prune=None, unbalance=None, dataset_dir="noaug"):
        self.mode = mode
        self.batch_size = batch_size
        self.class_to_prune = class_to_prune
        self.unbalance = unbalance
        self.dataset_dir = dataset_dir
        json_file = os.path.join(dataset_dir, "annotations", f"{mode}.json")
        with open(json_file, 'r') as f:
            self.coco_data = json.load(f)
        self.image_dir = os.path.join(dataset_dir, mode)
        self.image_shape = [3, 32, 32]  # RGB, 32x32
        self.classes = [cat["id"] for cat in self.coco_data["categories"]]
        self.class_names = {cat["id"]: cat["name"] for cat in self.coco_data["categories"]}
        self.dataset_x, self.dataset_y, self.per_class_ids = self._load_dataset()

    def _load_dataset(self):
        images = {img["id"]: img for img in self.coco_data["images"]}
        dataset_x = []
        dataset_y = []
        per_class_ids = {c: [] for c in self.classes}

        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            category_id = ann["category_id"]
            bbox = ann["bbox"]
            img_info = images[img_id]
            img_path = os.path.join(self.image_dir, img_info["file_name"])
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img)
            x, y, w, h = [int(v) for v in bbox]
            cropped = img_array[y:y+h, x:x+w, :]
            cropped_img = Image.fromarray(cropped).resize((32, 32))
            cropped_array = np.array(cropped_img)
            cropped_array = (cropped_array / 127.5) - 1.0  # Normalize to [-1, 1]
            dataset_x.append(cropped_array)
            dataset_y.append(category_id)
            per_class_ids[category_id].append(len(dataset_x) - 1)

        dataset_x = np.array(dataset_x)
        dataset_y = np.array(dataset_y)
        if self.class_to_prune is not None and self.unbalance is not None:
            keep_indices = []
            class_counts = {c: len(per_class_ids[c]) for c in self.classes}
            max_samples = max(class_counts.values())
            for c in self.classes:
                indices = per_class_ids[c]
                if c == self.class_to_prune:
                    n_keep = int(self.unbalance * max_samples)
                    if n_keep < len(indices):
                        indices = np.random.choice(indices, n_keep, replace=False)
                keep_indices.extend(indices)
            dataset_x = dataset_x[keep_indices]
            dataset_y = dataset_y[keep_indices]
            per_class_ids = {c: [] for c in self.classes}
            for i, label in enumerate(dataset_y):
                per_class_ids[label].append(i)

        # Convert to channels_first
        dataset_x = np.transpose(dataset_x, (0, 3, 1, 2))
        return dataset_x, dataset_y, per_class_ids

    def get_image_shape(self):
        return self.image_shape

    def get_label_table(self):
        return self.classes

    def per_class_count(self):
        return {c: len(self.per_class_ids[c]) for c in self.classes}

    def get_class_probability(self):
        counts = self.per_class_count()
        total = sum(counts.values())
        return np.array([counts[c] / total for c in self.classes])

    def get_num_samples(self):
        return len(self.dataset_y)

    def get_samples_for_class(self, class_id, n_samples):
        indices = np.random.choice(self.per_class_ids[class_id], n_samples, replace=True)
        return self.dataset_x[indices]

    def next_batch(self):
        indices = np.arange(len(self.dataset_y))
        np.random.shuffle(indices)
        for start_idx in range(0, len(self.dataset_y), self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            yield self.dataset_x[batch_indices], self.dataset_y[batch_indices]