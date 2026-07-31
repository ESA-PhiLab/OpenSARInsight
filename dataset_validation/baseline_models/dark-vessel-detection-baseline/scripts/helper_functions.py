"""
This module contains utilities to load data, load XML, and plot images.
It assumes the data files are TIFF files, labels are XML, and plots would use OpenCV.
It also prints the length of files in each directory and shows them.
Additionally, it checks the sizes of the files and tells you what files do not match 512 x 512 px.
"""

import os
import cv2
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
                label_key = f'DB_OPENSAR_VD_{id_number}.xml'

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
    
    @staticmethod
    def select_patches_from_scene(matched_tuple, target_scene):
            """
            Plots images that match the given scene from the matched matched_tuple list.

            Args:
                matched_tuple (list): List of tuples containing (image, label, image_name, label_name).
                target_scene (str, optional): The target SARProduct scene to filter and plot. If None, all images are plotted.

            Returns:
                filtered_matches (list): List of tuples that match the target scene.
            """
            # Filter images if target_scene is provided
            filtered_matches = []
            if target_scene:
                for image, label, image_name, label_name in matched_tuple:
                    # Parse the XML label
                    root = label.getroot()
                    sar_product = root.find(".//SARProduct")
                    if sar_product is not None and sar_product.text == target_scene:
                        filtered_matches.append((image, label, image_name, label_name))
                
                if not filtered_matches:
                    print(f"No images found for the scene: {target_scene}")
                    return []
            else:
                filtered_matches = matched_tuple
                
            return filtered_matches
    @staticmethod
    def plot_images_from_scene(matched_tuple, target_scene=None, images_per_row=3):
        """
        Plots images that match the given scene from the matched matched_tuple list.

        Args:
            matched_tuple (list): List of tuples containing (image, label, image_name, label_name).
            target_scene (str, optional): The target SARProduct scene to filter and plot. If None, all images are plotted.
            images_per_row (int): Number of images to display per row.

        Returns:
            None
        """
        # Filter images if target_scene is provided
        filtered_matches = []
        if target_scene:
            for image, label, image_name, label_name in matched_tuple:
                # Parse the XML label
                root = label.getroot()
                sar_product = root.find(".//SARProduct")
                if sar_product is not None and sar_product.text == target_scene:
                    filtered_matches.append((image, label, image_name, label_name))
            
            if not filtered_matches:
                print(f"No images found for the scene: {target_scene}")
                return
        else:
            filtered_matches = matched_tuple

        # Calculate the number of rows needed
        num_rows = (len(filtered_matches) + images_per_row - 1) // images_per_row

        # Create the figure and axes
        fig, axes = plt.subplots(num_rows, images_per_row, figsize=(20, 5 * num_rows))
        
        # Handle the case where there's only one row or one image
        if num_rows == 1 and images_per_row == 1:
            axes = np.array([[axes]])  # Make it 2D
        elif num_rows == 1:
            axes = axes.reshape(1, -1)  # Make it 2D with one row
        elif images_per_row == 1:
            axes = axes.reshape(-1, 1)  # Make it 2D with one column
            
        # Flatten the axes array for easier indexing
        axes_flat = axes.flatten()

        for idx, (image, label, image_name, label_name) in enumerate(filtered_matches):
            if idx >= len(axes_flat):
                print(f"Warning: More images ({len(filtered_matches)}) than plot spaces ({len(axes_flat)})")
                break
                
            ax = axes_flat[idx]

            # Parse the label to extract bounding boxes and centroids
            root = label.getroot()
            bounding_boxes = []
            centroids = []
            
            for ship in root.findall(".//Ship"):
                bbox = ship.find("BoundingBox")
                cpos = ship.find("Centroid_Position")
                
                if bbox is not None:
                    # Safely convert values, handling potential errors
                    try:
                        top_elem = bbox.find("Top")
                        left_elem = bbox.find("Left")
                        right_elem = bbox.find("Right")
                        bottom_elem = bbox.find("Bottom")
                        
                        # Check if all elements exist and have valid values
                        if (top_elem is not None and left_elem is not None and 
                            right_elem is not None and bottom_elem is not None):
                            
                            # Convert to float first to handle decimal values, then to int
                            top = int(float(top_elem.text))
                            left = int(float(left_elem.text))
                            right = int(float(right_elem.text))
                            bottom = int(float(bottom_elem.text))
                            
                            # Validate coordinates (ensure right > left and bottom > top)
                            if right > left and bottom > top:
                                bounding_boxes.append((top, left, right, bottom))
                            else:
                                print(f"Warning: Invalid box dimensions in {image_name}: ({top},{left},{right},{bottom})")
                        else:
                            print(f"Warning: Missing bounding box elements in {image_name}")
                    except (ValueError, TypeError, AttributeError) as e:
                        print(f"Error processing bounding box in {image_name}: {e}")
                            
                if cpos is not None:
                    try:
                        scene_sample_elem = cpos.find("Scene_Sample")
                        scene_line_elem = cpos.find("Scene_Line")
                        
                        if scene_sample_elem is not None and scene_line_elem is not None:
                            scene_sample = int(float(scene_sample_elem.text))
                            scene_line = int(float(scene_line_elem.text))
                            centroids.append((scene_sample, scene_line))
                    except (ValueError, TypeError, AttributeError) as e:
                        print(f"Error processing centroid in {image_name}: {e}")

            # Plot the image
            ax.imshow(image, cmap="gray")
            
            # Use basename of the file for cleaner title
            title = image_name
            if isinstance(title, str) and '\\' in title:
                title = title.split('\\')[-1]
            ax.set_title(title, fontsize=10)

            # Plot bounding boxes and centroids
            for (top, left, right, bottom) in bounding_boxes:
                # Draw a rectangle with thicker lines and distinct color
                rect = plt.Rectangle((left, top), right - left, bottom - top, 
                                    linewidth=2.5, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                
            for (scene_sample, scene_line) in centroids:
                ax.scatter(scene_sample, scene_line, c="blue", marker="x", s=50)

            # Add caption with ship count and annotation details
            ship_count = len(root.findall(".//Ship"))
            bbox_count = len(bounding_boxes)
            centroid_count = len(centroids)
            caption = f"Ships: {ship_count} (Boxes: {bbox_count}, Points: {centroid_count})"
            ax.set_xlabel(caption, fontsize=10)

        # Hide any unused subplots
        for ax in axes_flat[len(filtered_matches):]:
            ax.axis("off")

        # Show the plot
        plt.tight_layout()
        plt.show()