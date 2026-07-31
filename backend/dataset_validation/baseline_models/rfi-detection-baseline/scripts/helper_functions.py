"""
This module contains utilities to load data, load XML, and plot images.
It assumes the data files are TIFF files, labels are XML, and plots would use OpenCV.
It also prints the length of files in each directory and shows them.
Additionally, it checks the sizes of the files and tells you what files do not match 512 x 512 px.
"""

import os
import re
import cv2
from PIL import Image
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import xml.etree.ElementTree as ET


class Utilities:
    """
    A class for helper functions.

    Methods
    -------
    num_files(path: str) -> int:
        Returns the count of the items in a given directory.

    plot_image(image: np.ndarray, file_name: str) -> None:
        Plots an image using OpenCV.

    load_tiff_files(path: str) -> list:
        Loads TIFF files from a specified directory.

    product_filter(images: list, product_type: str, polarization: str = None) -> list:
        Filters images based on product type and optionally polarization.

    load_xml_files(path: str) -> list:
        Loads XML files from a specified directory.

    match_images_to_labels(images: list, labels: list) -> list:
        Matches images to labels based on their filenames.
    """
    @staticmethod
    def print_all_fields_in_xml(element, indent=0):
        """
        Print all fields in an XML element recursively.

        Args:
            element (xml.etree.ElementTree.Element): The XML element to print.
            indent (int, optional): The indentation level for nested elements (default is 0).
        """
        print(" " * indent + f"Tag: {element.tag}, Text: {element.text.strip() if element.text else 'None'}")
        for child in element:
            Utilities.print_all_fields_in_xml(child, indent + 2)

    @staticmethod
    def num_files(path: str) -> int:
        """
        Returns the count of the items in a given directory.

        Args:
            path (str): Directory path.

        Returns:
            int: The number of files in the directory.
        """
        try:
            num_files = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
            print(f"Number of files in the directory: {num_files}")
            return num_files
        except FileNotFoundError:
            print(f"Error: Directory not found: {path}")
        except NotADirectoryError:
            print(f"Error: Not a directory: {path}")
        except Exception as e:
            print(f"An error occurred: {e}")
        return 0

    @staticmethod
    def plot_image(image: np.ndarray, file_name: str) -> None:
        """
        Plots an image using OpenCV.

        Args:
            image (np.ndarray): The image to be plotted.
            file_name (str): The name of the file to display in the window.
        """
        if image is None:
            print("Error: Image not found.")
        else:
            cv2.imshow(file_name, image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    @staticmethod
    def load_tiff_files(path: str) -> list:
        """
        Load TIFF files from a specified directory.

        Args:
            path (str): The directory path containing the TIFF files.

        Returns:
            list: A list of loaded images. Each image is represented as a NumPy array.
        """
        image_data = []
        try:
            for filename in os.listdir(path):
                try:
                    image_path = os.path.join(path, filename)
                    img = cv2.imread(image_path)
                    if img is not None:
                        image_data.append((img, filename))
                    else:
                        print(f"Error: Could not load image {filename}")
                except Exception as e:
                    print(f"Error loading image {filename}: {e}")
            print('Images Loaded Successfully')
        except FileNotFoundError:
            print(f"Error: Directory not found: {path}")
        except NotADirectoryError:
            print(f"Error: Not a directory: {path}")
        except Exception as e:
            print(f"An error occurred: {e}")

        return image_data

    @staticmethod
    def product_filter(images: list, product_type: str, polarization: str = None) -> list:
        """
        Filters images based on product type (SLC or GRD) and optionally polarization.

        Args:
            images (list): A list of (img, filename) tuples.
            product_type (str): "SLC" or "GRD" to filter by product type.
            polarization (str, optional): Polarization to filter by (e.g., "VH", "VV"). If None, polarization filtering is skipped.

        Returns:
            list: A list of (img, filename) tuples that match the criteria.
        """
        filtered_products = []
        for img, filename in images:
            if product_type in filename:
                if polarization is None or polarization in filename:
                    filtered_products.append((img, filename))

        print(
            f'Length of {product_type} images'
            f'{" in " + polarization + " polarization" if polarization else ""}:'
            f' {len(filtered_products)}'
        )

        return filtered_products

    @staticmethod
    def load_xml_files(path: str) -> list:
        """
        Load XML files from a specified directory.

        Args:
            path (str): The directory path containing the XML files.

        Returns:
            list: A list of loaded labels.
        """
        labels = []
        try:
            for filename in os.listdir(path):
                try:
                    label_path = os.path.join(path, filename)
                    print(f'Processing Label Path: {label_path}')
                    try:
                        label = ET.parse(label_path)
                        labels.append((label, filename))
                    except ET.ParseError as parse_error:
                        print(f"Error: Could not parse XML {filename}. ParseError: {parse_error}")
                except Exception as e:
                    print(f"Error loading label {filename}: {e}")
            print('Labels Loaded Successfully')
        except FileNotFoundError:
            print(f"Error: Directory not found: {path}")
        except NotADirectoryError:
            print(f"Error: Not a directory: {path}")
        except Exception as e:
            print(f"An error occurred: {e}")

        return labels
        
    @staticmethod
    def match_images_to_labels(images_list, labels_list):
        """
        Match images to labels based on their names and return detailed output.

        Args:
            images_list (list): List of tuples (image_data, image_name).
            labels_list (list): List of tuples (parsed_xml, label_name).

        Returns:
            list: List of tuples (image_data, parsed_xml, image_name, label_name) for matched pairs, sorted by numeric ID.
        """
        labels_dict = {label_name: parsed_xml for parsed_xml, label_name in labels_list}
        matched_pairs = []
        table_data = {'Image Name': [], 'Label Name': [], 'Image Slice': [], 'Label Slice': []}

        for image_data, image_name in images_list:
            # Extract the numeric ID from the image filename
            parts = image_name.split('_')
            if len(parts) >= 4:
                id_number = parts[3].lstrip('0')  # Remove leading zeros
                label_key = f'DB_OPENSAR_RFI_{id_number}.xml'

                if label_key in labels_dict:
                    matched_pairs.append((image_data, labels_dict[label_key], image_name, label_key))
                    table_data['Image Name'].append(image_name)
                    table_data['Label Name'].append(label_key)
                    img_slice = str(image_data[0, :3].tolist()) if isinstance(image_data, np.ndarray) else str(image_data)[:20]
                    table_data['Image Slice'].append(img_slice)
                    xml_str = labels_dict[label_key].getroot().tag[:50] if labels_dict[label_key] else "N/A"
                    table_data['Label Slice'].append(xml_str)
                else:
                    print(f"No label found for image: {image_name}")
            else:
                print(f"Invalid image name format: {image_name}")

        # Sort matched pairs by numeric ID extracted from the image name
        matched_pairs.sort(key=lambda x: int(x[2].split('_')[3].lstrip('0')))

        print(f"Number of matched labels: {len(matched_pairs)}")
        print(f"Number of samples (images): {len(images_list)}")

        df = pd.DataFrame(table_data)
        print("\nMatched Pairs Summary:")
        print(df.to_string(index=False))

        return matched_pairs
    
    @staticmethod
    def sort_images_by_Scene(matched_tuple, target_scene,):
        """
        Group images that match the given scene from the matched matched_tuple list.

        Args:
            matched_tuple (list): List of tuples containing (image, label, image_name, label_name).
            target_scene (str): The target SARProduct scene to filter and plot.
        Returns:
            filtered_matches (list)
        """
        # Filter images that match the target scene
        filtered_matches = []
        for image, label, image_name, label_name in matched_tuple:
            # Parse the XML label
            root = label.getroot()
            sar_product = root.find(".//SARProduct")
            if sar_product is not None and sar_product.text == target_scene:
                filtered_matches.append((image, label, image_name, label_name))

        if not filtered_matches:
            print(f"No images found for the scene: {target_scene}")
            return
        
        return filtered_matches
    

    def plot_patch_and_mask(patches_list, masks_dir, num_samples=None, figsize=(20, 15), max_images_per_plot=15):
        """
        Match image tuples to mask files and create multiplots with one pair per row.
        
        Args:
            patches_list (list): List of tuples containing (image_data, image_name)
            masks_dir (str): Path to the binary masks directory
            num_samples (int): Number of sample pairs to plot (None for all)
            figsize (tuple): Figure size for the plot
            max_images_per_plot (int): Maximum number of image pairs to show in a single plot
            
        Returns:
            int: Number of matched pairs found
        """
        # Get list of files in the masks directory
        mask_files = sorted(os.listdir(masks_dir))
        
        # Find matching files using numerical pattern in filenames
        matched_files = []
        files_without_masks = []
        
        # Extract just the image names from the tuples
        for image_data, image_name in patches_list:
            # Extract the numerical part from patch filename
            match_patch = re.search(r'(\d+)', image_name)
            if match_patch:
                num_id = match_patch.group(1)
                # Find corresponding mask file with same number
                found_mask = False
                for mask_file in mask_files:
                    if f"_{num_id}_" in mask_file:
                        matched_files.append((image_data, image_name, mask_file))
                        found_mask = True
                        break
                
                if not found_mask:
                    files_without_masks.append(image_name)
        
        # Select samples (all if num_samples is None)
        samples = matched_files if num_samples is None else matched_files[:min(num_samples, len(matched_files))]
        
        # Split samples into batches to avoid creating too large figures
        batch_size = min(max_images_per_plot, len(samples))
        num_batches = (len(samples) + batch_size - 1) // batch_size
        
        print(f"Found {len(matched_files)} matching pairs. Plotting in {num_batches} batch(es)")
        
        # Process each batch
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(samples))
            batch_samples = samples[start_idx:end_idx]
            
            # One pair per row: num_rows = number of samples in batch
            num_rows = len(batch_samples)
            
            # Create subplot grid with 2 columns (patch and mask) for each row
            fig, axes = plt.subplots(num_rows, 2, figsize=figsize)
            
            # Make axes 2D if needed
            if num_rows == 1:
                axes = axes.reshape(1, -1)
            
            # Plot each matching pair in this batch
            for i, (image_data, image_name, mask_file) in enumerate(batch_samples):
                # Load mask image
                mask_img = np.array(Image.open(os.path.join(masks_dir, mask_file)))
                
                # Plot patch - we already have the image data in memory
                if len(image_data.shape) == 3:  # RGB or BGR
                    axes[i, 0].imshow(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
                else:  # Grayscale
                    axes[i, 0].imshow(image_data, cmap='gray')
                    
                axes[i, 0].set_title(f"Image: {image_name}", fontsize=12)
                axes[i, 0].axis('off')
                
                # Plot mask
                axes[i, 1].imshow(mask_img, cmap='gray')
                axes[i, 1].set_title(f"Mask: {mask_file}", fontsize=12)
                axes[i, 1].axis('off')
            
            # Adjust subplot layout to minimize whitespace
            plt.subplots_adjust(wspace=0.05, hspace=0.3)
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
            plt.suptitle(f"Batch {batch_idx+1}/{num_batches} (Samples {start_idx+1}-{end_idx} of {len(samples)})", 
                        fontsize=16, y=0.99)
            plt.show()
        
        # Print files without matching masks
        if files_without_masks:
            print("\nFiles without matching masks:")
            for file in files_without_masks:
                print(f"  - {file}")
        
        return len(matched_files)