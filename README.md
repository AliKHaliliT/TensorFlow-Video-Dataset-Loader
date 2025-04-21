# TensorFlow Video Dataset Loader
<div style="display: flex; gap: 10px; flex-wrap: wrap;">
    <img src="https://img.shields.io/github/license/AliKHaliliT/TensorFlow-Video-Dataset-Loader" alt="License">
    <img src="https://img.shields.io/github/last-commit/AliKHaliliT/TensorFlow-Video-Dataset-Loader" alt="Last Commit">
    <img src="https://img.shields.io/github/issues/AliKHaliliT/TensorFlow-Video-Dataset-Loader" alt="Open Issues">
</div>
<br/>

A customizable video dataset generator for TensorFlow that allows for efficient loading, preprocessing, and batching of video data for classification tasks. Ideal for research and production pipelines dealing with video-based supervised learning.

## Usage

### Installation

Ensure you have the required dependencies:

```bash
pip install tensorflow==2.18.0 tqdm==4.67.1 pandas==2.2.3 numpy==2.0.2 opencv-python==4.10.0.84
```

### Directory Structure

Your dataset should be organized as follows:

```
dataset_root/
│
├── class_1/
│   ├── video1.mp4
│   └── video2.mp4
│
├── class_2/
│   ├── video3.mp4
│   └── video4.mp4
│
...
```

Each subdirectory under `dataset_root` represents a class, and the videos inside are samples for that class.

### Example

```python
from video_dataset_loader import VideoDatasetLoader


video_loader = VideoDatasetLoader(
    data_path='path/to/dataset_root',
    batch_size=4,
    vid_size=(128, 128),
    normalization=True,
    shuffle=True,
    shuffle_buffer_size=500,
    reshuffle_each_iteration=True,
    drop_remainder=True
)


dataset = video_loader()

for video_batch, label_batch in dataset:
    print(video_batch.shape, label_batch.shape)
```

### Parameters

| Parameter | Type | Default | Description |
|----------|------|---------|-------------|
| `data_path` | `str` | — | Path to the dataset directory |
| `batch_size` | `int` | — | Batch size for training |
| `dtype` | `tf.dtypes.DType` | `tf.float32` | Output dtype of the video tensors |
| `vid_size` | `tuple(int, int)` | `None` | Resize dimensions `(width, height)` |
| `normalization` | `bool` | `False` | Normalize pixel values to `[0, 1]` |
| `shuffle` | `bool` | `True` | Shuffle data before training |
| `shuffle_buffer_size` | `int` | `1000` | Buffer size for shuffling |
| `reshuffle_each_iteration` | `bool` | `True` | Reshuffle on each epoch |
| `drop_remainder` | `bool` | `True` | Drop last batch if it's smaller than `batch_size` |

### Notes

- This generator is compatible with distributed TensorFlow training.
- To ensure consistent behavior in distributed environments, set both `reshuffle_each_iteration` and `drop_remainder` to `True`.
- Improper use of normalization in combination with resizing or mixed precision training can cause convergence issues.

## License

This work is under an [MIT](https://choosealicense.com/licenses/mit/) License.