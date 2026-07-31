import os
import matplotlib as plt


class InferenceScripts:
    @staticmethod
    def visualize_predictions(model, test_files, yolo_data_dir, conf_threshold=0.01, max_images=5):
        """
        Robust visualization function with better error handling and lower confidence threshold
        """
        import cv2
        import os
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        images_dir = os.path.join(yolo_data_dir, 'images')
        labels_dir = os.path.join(yolo_data_dir, 'labels')
        
        # Filter to only existing files
        existing_files = [f for f in test_files if os.path.exists(os.path.join(images_dir, f))]
        
        if not existing_files:
            print("No existing test files found!")
            return
        
        print(f"Processing {min(max_images, len(existing_files))} images...")
        
        for idx, image_file in enumerate(existing_files[:max_images]):
            image_path = os.path.join(images_dir, image_file)
            
            print(f"Processing image {idx+1}: {image_file}")
            
            # Load ground truth
            label_file = os.path.join(labels_dir, os.path.splitext(image_file)[0] + '.txt')
            gt_boxes = []
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls, x_center, y_center, width, height = map(float, parts)
                            gt_boxes.append((x_center, y_center, width, height))
            
            # Test multiple confidence thresholds
            best_results = None
            best_conf = conf_threshold
            
            for test_conf in [0.01, 0.05, 0.1, 0.2]:
                results = model.predict(image_path, conf=test_conf, verbose=False)[0]
                if results.boxes is not None and len(results.boxes) > 0:
                    best_results = results
                    best_conf = test_conf
                    break
            
            if best_results is None:
                results = model.predict(image_path, conf=conf_threshold, verbose=False)[0]
            else:
                results = best_results
                print(f"  Found detections at confidence {best_conf}")
            
            # Load and display image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to load image: {image_path}")
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_height, img_width = img.shape[:2]
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            
            # Ground Truth plot
            ax1.imshow(img)
            ax1.set_title(f"Ground Truth ({len(gt_boxes)} ships)", fontsize=16, fontweight='bold')
            
            for i, (x_center, y_center, width, height) in enumerate(gt_boxes):
                x1 = int((x_center - width/2) * img_width)
                y1 = int((y_center - height/2) * img_height)
                w = int(width * img_width)
                h = int(height * img_height)
                
                rect = patches.Rectangle((x1, y1), w, h, linewidth=3, edgecolor='lime', facecolor='none')
                ax1.add_patch(rect)
                ax1.text(x1, y1-10, f"Ship #{i+1}", color='lime', fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
            
            # Predictions plot
            num_predictions = len(results.boxes) if results.boxes is not None else 0
            ax2.imshow(img)
            ax2.set_title(f"Predictions ({num_predictions} detections, conf > {best_conf})", fontsize=16, fontweight='bold')
            
            if results.boxes is not None and len(results.boxes) > 0:
                for i, box in enumerate(results.boxes):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor='red', facecolor='none')
                    ax2.add_patch(rect)
                    ax2.text(x1, y1-10, f'Conf: {conf:.3f}', color='red', fontsize=12, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
            
            ax1.axis('off')
            ax2.axis('off')
            
            plt.tight_layout()
            plt.suptitle(f"Image {idx+1}: {os.path.basename(image_path)}", fontsize=18, fontweight='bold', color='red')
            plt.show()
            plt.close()
            
            print(f"  Ground Truth: {len(gt_boxes)} ships")
            print(f"  Predictions: {num_predictions} ships")
            print("-" * 50)

    # Check file availability
    @staticmethod
    def check_file_availability(test_files, yolo_data_dir):
        """Check which test files exist and which are missing"""
        images_dir = os.path.join(yolo_data_dir, 'images')
        labels_dir = os.path.join(yolo_data_dir, 'labels')
        
        existing_files = []
        missing_files = []
        
        for img_file in test_files:
            img_path = os.path.join(images_dir, img_file)
            label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + '.txt')
            
            if os.path.exists(img_path):
                existing_files.append(img_file)
            else:
                missing_files.append(img_file)
        
        print(f"Total test files: {len(test_files)}")
        print(f"Existing files: {len(existing_files)}")
        print(f"Missing files: {len(missing_files)}")
        
        if missing_files:
            print("\nFirst 10 missing files:")
            for f in missing_files[:10]:
                print(f"  {f}")
        
        return existing_files

    # Test different confidence thresholds
    @staticmethod
    def test_confidence_thresholds(model, image_path, thresholds=[0.01, 0.02, 0.03, 0.04, 0.05, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]):
        """Test different confidence thresholds to see if model detects anything"""
        print(f"Testing image: {os.path.basename(image_path)}")
        
        for thresh in thresholds:
            results = model.predict(image_path, conf=thresh, verbose=False)[0]
            num_detections = len(results.boxes) if results.boxes is not None else 0
            print(f"  Conf {thresh}: {num_detections} detections")
            
            if num_detections > 0:
                for i, box in enumerate(results.boxes):
                    conf = box.conf[0].item()
                    print(f"    Detection {i+1}: conf={conf:.4f}")