#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import shutil

class MultiVideoEncodingManager:
    """
    Context manager that ensures proper video encoding and data cleanup even if exceptions occur.

    This manager handles:
    - Batch encoding for any remaining episodes when recording interrupted
    - Cleaning up temporary image files from interrupted episodes
    - Removing empty image directories

    Args:
        dataset: The LeRobotDataset instance
    """

    def __init__(self, datasets: dict):
        self.datasets = datasets

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Handle any remaining episodes that haven't been batch encoded
        for dataset in self.datasets.values():
            if dataset.episodes_since_last_encoding > 0:
                if exc_type is not None:
                    logging.info("Exception occurred. Encoding remaining episodes before exit...")
                else:
                    logging.info("Recording stopped. Encoding remaining episodes...")

                start_ep = dataset.num_episodes - dataset.episodes_since_last_encoding
                end_ep = dataset.num_episodes
                logging.info(
                    f"Encoding remaining {dataset.episodes_since_last_encoding} episodes, "
                    f"from episode {start_ep} to {end_ep - 1}"
                )
                dataset._batch_save_episode_video(start_ep, end_ep)

            # Finalize the dataset to properly close all writers
            dataset.finalize()

            # Clean up episode images if recording was interrupted
            if exc_type is not None:
                interrupted_episode_index = dataset.num_episodes
                for key in dataset.meta.video_keys:
                    img_dir = dataset._get_image_file_path(
                        episode_index=interrupted_episode_index, image_key=key, frame_index=0
                    ).parent
                    if img_dir.exists():
                        logging.debug(
                            f"Cleaning up interrupted episode images for episode {interrupted_episode_index}, camera {key}"
                        )
                        shutil.rmtree(img_dir)

            # Clean up any remaining images directory if it's empty
            img_dir = dataset.root / "images"
            # Check for any remaining PNG files
            png_files = list(img_dir.rglob("*.png"))
            if len(png_files) == 0:
                # Only remove the images directory if no PNG files remain
                if img_dir.exists():
                    shutil.rmtree(img_dir)
                    logging.debug("Cleaned up empty images directory")
            else:
                logging.debug(f"Images directory is not empty, containing {len(png_files)} PNG files")

        return False  # Don't suppress the original exception
