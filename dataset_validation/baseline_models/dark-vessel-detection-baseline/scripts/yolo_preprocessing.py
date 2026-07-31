import os
import cv2

class YoloPreprocessing:
    @staticmethod
    def convert_to_yolo_format(matched_data, output_dir):
        """
        Convert existing dataset to YOLO format.

        Args:
            matched_data: List of tuples (image, label, image_name, label_name)
            output_dir: Directory to save images and labels
        """
        # Create output directories for images and labels
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

        for image, label, image_name, label_name in matched_data:
            # Save image as jpg
            img_path = os.path.join(
                output_dir, 'images', f"{os.path.splitext(image_name)[0]}.jpg"
            )
            cv2.imwrite(img_path, image)

            yolo_labels = []
            img_height, img_width = image.shape[:2]
            root = label.getroot()

            # Iterate over each ship in the label XML
            for ship in root.findall(".//Ship"):
                bbox = ship.find("BoundingBox")
                if bbox is not None:
                    try:
                        # Read bounding box coordinates
                        top = int(float(bbox.find("Top").text))
                        left = int(float(bbox.find("Left").text))
                        right = int(float(bbox.find("Right").text))
                        bottom = int(float(bbox.find("Bottom").text))

                        # Convert to YOLO format (normalized center x/y, width, height)
                        x_center = ((left + right) / 2) / img_width
                        y_center = ((top + bottom) / 2) / img_height
                        width = (right - left) / img_width
                        height = (bottom - top) / img_height

                        yolo_labels.append(
                            f"0 {x_center} {y_center} {width} {height}"
                        )
                    except Exception as e:
                        print(f"Error processing bounding box: {e}")

            # Save YOLO label file
            label_path = os.path.join(
                output_dir, 'labels', f"{os.path.splitext(image_name)[0]}.txt"
            )
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_labels))

        print(
            f"Converted {len(matched_data)} samples to YOLO format in {output_dir}"
        )

    @staticmethod
    def split_dataset(
        data_dir,
        negative_dir=None,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1
    ):
        """
        Split the dataset into train, validation, and test sets, including negative samples.

        Args:
            data_dir: Directory containing 'images' and 'labels' folders with positive samples
            negative_dir: Directory path containing negative samples (without objects)
            train_ratio: Percentage for training set
            val_ratio: Percentage for validation set
            test_ratio: Percentage for test set

        Returns:
            Dictionary with image paths for each split
        """
        import yaml
        import random
        from sklearn.model_selection import train_test_split

        # Get list of positive images
        positive_images = [
            f for f in os.listdir(os.path.join(data_dir, 'images'))
            if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tiff')
        ]

        negative_images = []
        # Get list of negative images if provided
        if negative_dir and os.path.exists(os.path.join(negative_dir, 'images')):
            negative_images = [
                f for f in os.listdir(os.path.join(negative_dir, 'images'))
                if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tiff')
            ]
            print(f"Found {len(negative_images)} negative samples")

        # Split positive images
        pos_train, pos_temp = train_test_split(
            positive_images, train_size=train_ratio, random_state=42
        )
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        pos_val, pos_test = train_test_split(
            pos_temp, train_size=val_ratio_adjusted, random_state=42
        )

        neg_train, neg_temp = [], []
        neg_val, neg_test = [], []

        # Split negative images if available
        if negative_images:
            neg_train, neg_temp = train_test_split(
                negative_images, train_size=train_ratio, random_state=42
            )
            neg_val, neg_test = train_test_split(
                neg_temp, train_size=val_ratio_adjusted, random_state=42
            )

        # Combine positive and negative splits
        train_files = pos_train + neg_train
        val_files = pos_val + neg_val
        test_files = pos_test + neg_test

        # Shuffle the splits
        random.seed(42)
        random.shuffle(train_files)
        random.shuffle(val_files)
        random.shuffle(test_files)

        print(
            f"Dataset split: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test"
        )

        # Copy negative images and create empty label files if needed
        if negative_images and negative_dir != data_dir:
            import shutil
            for img_file in negative_images:
                src = os.path.join(negative_dir, 'images', img_file)
                dst = os.path.join(data_dir, 'images', img_file)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

                label_name = os.path.splitext(img_file)[0] + '.txt'
                empty_label_path = os.path.join(data_dir, 'labels', label_name)
                if not os.path.exists(empty_label_path):
                    with open(empty_label_path, 'w') as f:
                        pass

                # Create dataset YAML config
                dataset_yaml = {
                    'path': os.path.abspath(data_dir).replace('\\', '/'),
                    'train': train_files,  # Changed from 'images' to actual list
                    'val': val_files,      # Changed from 'images' to actual list  
                    'test': test_files,    # Changed from 'images' to actual list
                    'nc': 1,
                    'names': ['ship']
                }

                # Save YAML config
                with open(os.path.join(data_dir, 'dataset.yaml'), 'w') as f:
                    yaml.dump(dataset_yaml, f)

                return {'train': train_files, 'val': val_files, 'test': test_files}