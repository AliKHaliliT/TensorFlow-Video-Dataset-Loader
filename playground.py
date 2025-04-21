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