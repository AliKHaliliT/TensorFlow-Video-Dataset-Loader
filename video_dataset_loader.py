import tensorflow as tf
from typing import Optional
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import cv2


class VideoDatasetLoader:

    """

    A custom data generator for supervised classification tasks. The generator reads video files from a
    directory structure where each subdirectory represents a class, and the videos inside those directories
    are the samples. The generator returns a batch of video frames (x) and their corresponding labels (y).

    When using this data generator in a distributed setting, it is strongly recommended
    to set `reshuffle_each_iteration` and `drop_remainder` to `True`. This prevents unintended
    silent behaviors and ensures proper utilization of the entire dataset.

    
    Usage
    -----
    The class instance must be called.

    """

    def __init__(self, data_path: str, batch_size: int,
                 dtype: tf.dtypes.DType = tf.float32,
                 vid_size: Optional[tuple[int, int]] = None, 
                 normalization: bool = False,
                 shuffle: bool = True,
                 shuffle_buffer_size: int = 1000,
                 reshuffle_each_iteration: bool = True,
                 drop_remainder: bool = True) -> None:
        
        """

        Constructor of the Video Data Generator class.

        
        Parameters
        ----------
        data_path : str
            Path to the directory containing the video folders, where each subdirectory represents a class.

        batch_size : int
            Batch size for the dataset. 
            It is advisable to select `batch_size` carefully to minimize unintended behaviors in distributed training. 
            For example, consider the number of processors being used and adjust the `batch_size` accordingly.

        dtype : tf.dtypes.DType, optional
            Data type for the video pixel values. The default value is `tf.float32`.
            The data type should be compatible with the model dtype requirements.

        vid_size : tuple, optional
            Size to resize the videos to. If `None`, no resizing is performed.

        normalization : bool, optional
            Whether to normalize the video pixel values to `[0, 1]`. The default value is `False`.
            Use this parameter carefully when resizing functionality is active or when mixed precision training is applied with an optimizer wrapped in a loss-scaling mechanism. 
            Improper use can hinder convergence, causing the loss to remain constant or oscillate within a small range. 
            Additionally, pay attention to the loss function—especially if it relies on distance ratio calculations. 
            In such cases, the value range is critical, as it can lead to vanishing gradients. 
            Similar issues may arise when resizing methods alter pixel values based on the resize dimensions.

        shuffle : bool, optional
            Whether to shuffle the dataset. The default value is `True`.
        
        shuffle_buffer_size : int, optional
            Buffer size for shuffling the dataset. The default value is `1000`.
            Only applicable if shuffle is `True`. 
            A larger buffer allows more elements to be held in memory, enabling a more thorough shuffle.
            However, the `buffer_size` directly impacts RAM usage because all the buffered elements are held in memory.
            Choose a value based on the available RAM and the size of the dataset.

        reshuffle_each_iteration : bool, optional
            Determines whether to reshuffle the dataset at every iteration. The default value is `True`.
            It is recommended to set this to `True` when `drop_remainder` is also `True` to avoid leaving unused samples.

        drop_remainder : bool, optional
            Specifies whether to drop the last incomplete batch. The default value is `True`.
            If your program requires batches with consistent outer dimensions, set `drop_remainder` to `True` to prevent generating smaller batches.
            Please note that for programs needing statically defined shapes (e.g., when using XLA), `drop_remainder=True` is required. 
            Without it, the output dataset will have an unknown leading dimension due to a potentially smaller final batch.
            
            
        Returns
        -------
        None.

        """

        if not isinstance(data_path, str) or not os.path.isdir(data_path):
            raise ValueError(f"data_path must be a valid directory path. Received: {data_path} with type {type(data_path)}")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer. Received: {batch_size} with type {type(batch_size)}")
        if not isinstance(dtype, tf.dtypes.DType):
            raise TypeError(f"dtype must be a valid TensorFlow data type considering the model dtype requirements. Received: {dtype} with type {type(dtype)}")
        if vid_size is not None and (not isinstance(vid_size, tuple) or len(vid_size) != 2 or 
                                     not all(isinstance(x, int) and x > 0 for x in vid_size)):
            raise ValueError(f"vid_size must be a tuple of two positive integers. Received: {vid_size} with type {type(vid_size)}")
        if not isinstance(normalization, bool):
            raise TypeError(f"normalize must be a boolean. Received: {normalization} with type {type(normalization)}")
        if not isinstance(shuffle, bool):
            raise TypeError(f"shuffle must be a boolean. Received: {shuffle} with type {type(shuffle)}")
        if not isinstance(shuffle_buffer_size, int) or shuffle_buffer_size <= 0:
            raise ValueError(f"shuffle_buffer_size must be a positive integer. Received: {shuffle_buffer_size} with type {type(shuffle_buffer_size)}")
        if not isinstance(reshuffle_each_iteration, bool):
            raise TypeError(f"reshuffle_each_iteration must be a boolean. Received: {reshuffle_each_iteration} with type {type(reshuffle_each_iteration)}")
        if not isinstance(drop_remainder, bool):
            raise TypeError(f"drop_remainder must be a boolean. Received: {drop_remainder} with type {type(drop_remainder)}")


        self.data_path = data_path
        self.batch_size = batch_size
        self.dtype = dtype
        self.vid_size = vid_size
        self.normalization = normalization
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.reshuffle_each_iteration = reshuffle_each_iteration
        self.drop_remainder = drop_remainder
        self.video_paths = None
        self.labels = None
        self.dataset = None


    def _load_videos_and_labels(self) -> None:

        """

        Method to load the video paths and their corresponding labels from the data directory.


        Parameters
        ----------
        None.


        Returns
        -------
        None.

        """

        video_paths = []
        labels = []

        for class_name in tqdm(os.listdir(self.data_path)):

            class_path = os.path.join(self.data_path, class_name)

            if os.path.isdir(class_path):

                for video_file in os.listdir(class_path):

                    video_path = os.path.join(class_path, video_file)
                    
                    if os.path.isfile(video_path):
                        video_paths.append(video_path)
                        labels.append(class_name)
                    else:
                        print()
                        print("-" * 50)
                        print(f"Invalid file in folder: {video_path}")
                        print("-" * 50)
        
        self.video_paths = pd.DataFrame(video_paths, columns=["video_path"])
        self.labels = pd.get_dummies(labels)


    def _load_and_preprocess_video(self, path: str) -> np.array:

        """

        Method to load and preprocess a single video.

        
        Parameters
        ----------
        path : str
            Path to the video file.

            
        Returns
        -------
        video: np.array
            The preprocessed video as a numpy array.

        """

        # Convert the path to a proper string (since tf.py_function passes tf.string)
        cap = cv2.VideoCapture(path.numpy().decode("utf-8"))
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if self.vid_size:
                frame = cv2.resize(frame, self.vid_size)
            frames.append(frame)

        cap.release()


        return np.array(frames)


    def _prepare_and_serve_video(self, path: str) -> tf.Tensor:

        """

        Method to prepare a single video array to serve.

        
        Parameters
        ----------
        path : str
            Path to the video file.

            
        Returns
        -------
        video: tf.Tensor:
            The final video tensor.

        """

        frames = tf.py_function(func=self._load_and_preprocess_video, inp=[path], Tout=self.dtype)
        if self.normalization:
            frames /= 255.0
        frames.set_shape([None, None, None, None]) # Consider making this static if issues arise with dynamic or unspecified dtypes


        return frames
        

    def _create_dataset(self) -> None:

        """

        Method to create the dataset from the loaded video paths and labels.

        
        Parameters
        ----------
        None.


        Returns
        -------
        None.

        """

        if self.video_paths is None or self.labels is None:
            raise ValueError("Video paths or labels not loaded. There might be an issue with _load_videos_and_labels().")


        # Create the dataset slicing with file paths and labels
        dataset = tf.data.Dataset.from_tensor_slices((
            self.video_paths["video_path"].values,
            self.labels.values
        ))

        # Apply shuffling BEFORE loading
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size, reshuffle_each_iteration=self.reshuffle_each_iteration)

        # Now map the dataset to load and preprocess the videos
        dataset = dataset.map(
            lambda path, label: (self._prepare_and_serve_video(path), label),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Apply batching and prefetching
        self.dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=self.drop_remainder).prefetch(tf.data.AUTOTUNE)


    def _get_dataset(self) -> tf.data.Dataset:

        """

        Method to retrieve the created dataset.

        
        Parameters
        ----------
        None.

        
        Returns
        -------
        dataset : tf.data.Dataset
            The processed dataset of videos and their corresponding labels.

        """

        if self.dataset is None:
            raise ValueError("Dataset not created. There might be an issue with _create_dataset().")
        

        return self.dataset
    

    def __call__(self) -> tf.data.Dataset:

        """

        Method to execute the data loading and processing pipeline.

        
        Parameters
        ----------
        None.


        Returns
        -------
        dataset: tf.data.Dataset
            The processed dataset of videos and their corresponding labels.

        """

        self._load_videos_and_labels()
        self._create_dataset()


        return self._get_dataset()