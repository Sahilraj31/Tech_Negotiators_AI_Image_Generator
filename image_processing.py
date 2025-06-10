import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
import re
from pymongo import MongoClient
import gridfs
from PIL import Image
import io
import base64

class ImageDatabase:
    """Database handler for storing and retrieving images with descriptions"""
    
    def __init__(self, connection_string="mongodb://localhost:27017/", db_name="image_swapping_db"):
        """Initialize MongoDB connection and GridFS for image storage"""
        self.client = MongoClient(connection_string)
        self.db = self.client[db_name]
        self.fs = gridfs.GridFS(self.db)
        self.collection = self.db.images
        
        # Create indexes for faster searching
        self.collection.create_index("object_type")
        self.collection.create_index("background_type")
        self.collection.create_index("description")
        
    def store_image(self, image_path, object_type, background_type, description=""):
        """Store image in database with metadata
        
        Args:
            image_path: Path to the image file
            object_type: Type of object in image (e.g., 'boy', 'cow', 'car')
            background_type: Type of background (e.g., 'mountains', 'park', 'city')
            description: Additional description
        """
        try:
            # Read and store image in GridFS
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Store image in GridFS
            image_id = self.fs.put(image_data, filename=os.path.basename(image_path))
            
            # Store metadata in collection
            metadata = {
                'image_id': image_id,
                'filename': os.path.basename(image_path),
                'object_type': object_type.lower(),
                'background_type': background_type.lower(),
                'description': description.lower(),
                'keywords': self._generate_keywords(object_type, background_type, description)
            }
            
            result = self.collection.insert_one(metadata)
            print(f"✅ Stored image: {os.path.basename(image_path)} with ID: {result.inserted_id}")
            return result.inserted_id
            
        except Exception as e:
            print(f"❌ Error storing image {image_path}: {str(e)}")
            return None
    
    def _generate_keywords(self, object_type, background_type, description):
        """Generate search keywords from metadata"""
        keywords = []
        
        # Object type keywords
        object_keywords = {
            "boy": ["boy", "child", "kid", "son", "youth", "teen", "teenager", "person"],
            "girl": ["girl", "child", "kid", "daughter", "youth", "teen", "teenager", "person"],
            "child": ["child", "kid", "boy", "girl", "youth", "teen", "teenager", "person"],
            "person": ["person", "human", "man", "woman", "people", "adult", "boy", "girl"],
            "man": ["man", "person", "guy", "male", "adult", "human"],
            "woman": ["woman", "person", "lady", "female", "adult", "human"],
            "cow": ["cow", "cattle", "animal", "farm", "livestock"],
            "dog": ["dog", "puppy", "canine", "pet", "animal"],
            "cat": ["cat", "kitten", "feline", "pet", "animal"],
            "bird": ["bird", "avian", "flying", "feathers"],
            "car": ["car", "vehicle", "auto", "automobile"],
            "knight": ["knight", "warrior", "armor", "medieval"],
            "monkey": ["monkey", "monkeys", "apes"],
            "footballer": ["footballer", "sports man", "footballer kicking"],
            "sheep": ["sheep", "sheeps", "bhed"],
            "elephant": ["elephant", "elephants", "hathi"]
        }
        
        # Background keywords
        background_keywords = {
            "mountains": ["mountain", "mountains", "peak", "summit", "hill", "hills", "highlands", "alps", "range"],
            "park": ["park", "garden", "playground", "field", "yard", "outdoor", "nature"],
            "beach": ["beach", "shore", "coast", "ocean", "sea", "sand"],
            "forest": ["forest", "woods", "trees", "jungle", "woodland", "nature"],
            "farm": ["farm", "field", "countryside", "rural", "pasture", "barn", "ranch"],
            "city": ["city", "urban", "town", "downtown", "cityscape", "night_city"],
            "sky": ["sky", "clouds", "air", "heaven", "atmosphere"],
            "space": ["space", "universe", "stars", "galaxy", "cosmos"],
            "grass": ["grass", "lawn", "field", "meadow", "pasture"],
            "ground": ["ground", "earth", "soil", "dirt", "land"],
            "stadium": ["stadium", "arena", "field", "sports", "football"],
            "gym": ["gym", "fitness", "exercise", "workout"],
            "masjid": ["masjid", "mosque", "prayer", "islamic", "religious"],
            "field": ["field", "open", "meadow", "grass", "farm", "countryside"]
        }
        
        # Add object keywords
        if object_type.lower() in object_keywords:
            keywords.extend(object_keywords[object_type.lower()])
        keywords.append(object_type.lower())
        
        # Add background keywords
        if background_type.lower() in background_keywords:
            keywords.extend(background_keywords[background_type.lower()])
        keywords.append(background_type.lower())
        
        # Add description words
        if description:
            keywords.extend(description.lower().split())
        
        return list(set(keywords))  # Remove duplicates
    
    def search_images(self, query_type, search_term, limit=10):
        """Search for images by object or background type
        
        Args:
            query_type: 'object' or 'background'
            search_term: Term to search for
            limit: Maximum number of results
            
        Returns:
            list: List of image documents with scores
        """
        search_field = f"{query_type}_type"
        results = []
        
        # Direct match
        direct_matches = list(self.collection.find({search_field: search_term.lower()}).limit(limit))
        for doc in direct_matches:
            doc['score'] = 20  # High score for direct match
            results.append(doc)
        
        # Keyword match
        if len(results) < limit:
            keyword_matches = list(self.collection.find({
                "keywords": {"$in": [search_term.lower()]}
            }).limit(limit - len(results)))
            
            for doc in keyword_matches:
                if doc not in results:  # Avoid duplicates
                    doc['score'] = 10  # Medium score for keyword match
                    results.append(doc)
        
        # Text search in description
        if len(results) < limit:
            text_matches = list(self.collection.find({
                "description": {"$regex": search_term.lower(), "$options": "i"}
            }).limit(limit - len(results)))
            
            for doc in text_matches:
                if doc not in results:  # Avoid duplicates
                    doc['score'] = 5  # Low score for text match
                    results.append(doc)
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    def get_image_data(self, image_id):
        """Retrieve image data from GridFS
        
        Args:
            image_id: GridFS image ID
            
        Returns:
            numpy array: Image as RGB array
        """
        try:
            # Get image from GridFS
            grid_out = self.fs.get(image_id)
            image_data = grid_out.read()
            
            # Convert to PIL Image then to numpy array
            pil_image = Image.open(io.BytesIO(image_data))
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            return np.array(pil_image)
            
        except Exception as e:
            print(f"❌ Error retrieving image {image_id}: {str(e)}")
            return None
    
    def populate_from_folder(self, images_dir):
        """Populate database from existing images folder
        
        This method helps migrate from folder-based to database storage.
        It tries to extract object and background info from filenames.
        """
        if not os.path.exists(images_dir):
            print(f"❌ Images directory {images_dir} not found")
            return
        
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        print(f"📁 Found {len(image_files)} images to process...")
        
        for filename in image_files:
            image_path = os.path.join(images_dir, filename)
            
            # Try to parse filename for object and background info
            object_type, background_type = self._parse_filename(filename)
            
            # If parsing fails, ask user for input
            if not object_type or not background_type:
                print(f"\n🖼️  Processing: {filename}")
                object_type = input(f"Enter object type for {filename} (e.g., boy, cow, car): ").strip()
                background_type = input(f"Enter background type for {filename} (e.g., mountains, park, city): ").strip()
            
            description = f"{object_type} in {background_type}"
            
            # Store in database
            self.store_image(image_path, object_type, background_type, description)
    
    def _parse_filename(self, filename):
        """Try to extract object and background from filename patterns"""
        filename_lower = filename.lower().replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
        
        # Pattern: object_X_background_Y
        if '_background_' in filename_lower:
            parts = filename_lower.split('_background_')
            if len(parts) == 2:
                object_part = parts[0].replace('object_', '')
                background_part = parts[1]
                return object_part, background_part
        
        # Pattern: objectname_backgroundname
        common_objects = ['boy', 'girl', 'child', 'man', 'woman', 'cow', 'dog', 'cat', 'car', 'person']
        common_backgrounds = ['mountains', 'park', 'city', 'beach', 'forest', 'farm', 'grass', 'sky']
        
        for obj in common_objects:
            for bg in common_backgrounds:
                if obj in filename_lower and bg in filename_lower:
                    return obj, bg
        
        return None, None
    
    def get_all_images_info(self):
        """Get information about all stored images"""
        return list(self.collection.find({}, {'image_id': 1, 'filename': 1, 'object_type': 1, 'background_type': 1, 'description': 1}))
    
    def close_connection(self):
        """Close database connection"""
        self.client.close()

# Modified functions to work with database
def setup_detectron():
    """Setup Detectron2 configuration with enhanced object detection parameters"""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.65
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    return DefaultPredictor(cfg)


def get_refined_object_mask_and_box(image, class_label=None):
    """Get refined mask and bounding box for objects with improved boundary detection and better class matching"""
    predictor = setup_detectron()
    outputs = predictor(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    instances = outputs["instances"].to("cpu")
    if len(instances) == 0:
        print(f"⚠️ No objects detected in image")
        return None, None
    
    masks = instances.pred_masks.numpy()
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy()
    
    # Enhanced class mapping with better specificity and priority
    class_id_map = {
        # Humans - person class (0)
        "person": [0], "boy": [0], "girl": [0], "child": [0], "man": [0], "woman": [0],
        "knight": [0], "footballer": [0],
        
        # Animals - specific classes with higher priority
        "monkey": [16, 17, 18, 19, 20, 21, 22, 23],  # Prioritize animal classes for monkey
        "cow": [19], "cattle": [19], 
        "dog": [16], "cat": [15], "horse": [17], "sheep": [18],
        "bird": [14], "elephant": [20], "bear": [21], "zebra": [22], "giraffe": [23],
        
        # Vehicles
        "car": [2], "truck": [7], "bus": [5], "motorcycle": [3], "bicycle": [1]
    }
    
    # COCO class names for better debugging
    coco_class_names = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
        6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
        11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
        16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
        22: 'zebra', 23: 'giraffe'
    }
    
    selected_indices = None
    
    # Print detected objects for debugging
    detected_objects = [coco_class_names.get(cls, f"class_{cls}") for cls in classes]
    print(f"🔍 Detected objects: {detected_objects}")
    print(f"📊 Detection scores: {[f'{s:.3f}' for s in scores]}")
    
    # Filter by class if specified
    if class_label is not None:
        class_label_lower = class_label.lower()
        print(f"🎯 Looking specifically for: {class_label_lower}")
        
        if class_label_lower in class_id_map:
            target_classes = class_id_map[class_label_lower]
            
            # Special handling for monkey - prioritize any animal over person/car
            if class_label_lower == "monkey":
                # First try to find any animal class
                animal_classes = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]  # All animal classes
                animal_indices = np.isin(classes, animal_classes)
                if any(animal_indices):
                    selected_indices = animal_indices
                    print(f"✅ Found animal for monkey: {[coco_class_names.get(cls, cls) for cls in classes[animal_indices]]}")
                else:
                    # If no animals, try person as very last resort
                    person_indices = np.isin(classes, [0])
                    if any(person_indices):
                        selected_indices = person_indices
                        print(f"⚠️ Using person as fallback for monkey")
            else:
                # Normal class matching
                class_indices = np.isin(classes, target_classes)
                if any(class_indices):
                    selected_indices = class_indices
                    print(f"✅ Found {sum(class_indices)} {class_label} object(s)")
                else:
                    print(f"⚠️ No {class_label} detected")
                    
                    # Fallback logic for human-like objects
                    if class_label_lower in ["knight", "footballer", "boy", "girl", "child", "man", "woman"]:
                        person_indices = np.isin(classes, [0])  # Person class
                        if any(person_indices):
                            selected_indices = person_indices
                            print(f"✅ Found person-like object as fallback")
        else:
            print(f"⚠️ Unknown class {class_label}, trying fallback to any suitable object")
    
    # If no specific class found or no class specified, use smart selection
    if selected_indices is None:
        # Prioritize selection based on confidence and class relevance
        confidence_threshold = 0.3
        high_confidence_indices = scores > confidence_threshold
        
        if any(high_confidence_indices):
            # Among high confidence objects, prioritize non-vehicle classes if possible
            high_conf_classes = classes[high_confidence_indices]
            high_conf_scores = scores[high_confidence_indices]
            
            # Avoid vehicles if we have other options
            vehicle_classes = [1, 2, 3, 5, 6, 7, 8]  # bicycle, car, motorcycle, bus, train, truck, boat
            non_vehicle_mask = ~np.isin(high_conf_classes, vehicle_classes)
            
            if any(non_vehicle_mask):
                # Select best non-vehicle
                best_non_vehicle_idx = np.argmax(high_conf_scores[non_vehicle_mask])
                original_idx = np.where(high_confidence_indices)[0][np.where(non_vehicle_mask)[0][best_non_vehicle_idx]]
                selected_indices = np.zeros(len(classes), dtype=bool)
                selected_indices[original_idx] = True
                print(f"✅ Selected best non-vehicle object: {coco_class_names.get(classes[original_idx], 'unknown')}")
            else:
                # All are vehicles, select best one
                best_vehicle_idx = np.argmax(high_conf_scores)
                original_idx = np.where(high_confidence_indices)[0][best_vehicle_idx]
                selected_indices = np.zeros(len(classes), dtype=bool)
                selected_indices[original_idx] = True
                print(f"✅ Selected best vehicle object: {coco_class_names.get(classes[original_idx], 'unknown')}")
        else:
            # Lower confidence threshold and try again
            print("⚠️ No high-confidence objects, lowering threshold...")
            confidence_threshold = 0.15
            any_confidence_indices = scores > confidence_threshold
            if any(any_confidence_indices):
                selected_indices = any_confidence_indices
            else:
                # Take the best available
                selected_indices = np.array([True] * len(masks))
                print("⚠️ Using all available objects")
    
    # Apply selection
    if selected_indices is not None:
        masks = masks[selected_indices]
        boxes = boxes[selected_indices]
        scores = scores[selected_indices]
        classes = classes[selected_indices]
    
    if len(masks) == 0:
        print(f"❌ No valid objects found after filtering")
        return None, None
    
    # Select the best object (highest confidence among selected)
    best_idx = np.argmax(scores)
    final_mask = masks[best_idx].astype(np.uint8) * 255
    main_box = boxes[best_idx]
    selected_class = classes[best_idx]
    
    print(f"✅ Selected object: {coco_class_names.get(selected_class, 'unknown')} "
          f"with confidence: {scores[best_idx]:.3f}, "
          f"area: {(main_box[2] - main_box[0]) * (main_box[3] - main_box[1]):.0f}")
    
    # Apply advanced mask refinement
    kernel = np.ones((3,3), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    
    # Edge refinement with adaptive parameters
    mask_area = np.sum(final_mask > 0)
    if mask_area > 10000:  # Large objects
        blur_kernel = (7, 7)
        dilate_iterations = 2
    else:  # Small objects
        blur_kernel = (3, 3)
        dilate_iterations = 1
    
    final_mask = cv2.GaussianBlur(final_mask, blur_kernel, 0)
    
    # Expand mask slightly to ensure complete object coverage
    kernel = np.ones((3,3), np.uint8)
    final_mask = cv2.dilate(final_mask, kernel, iterations=dilate_iterations)
    
    return final_mask, main_box

def inpaint_background(image, mask):
    """Inpaint removed object area using improved background reconstruction"""
    # Convert mask to proper format for inpainting
    inpaint_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    
    # Expand mask slightly to cover edge artifacts
    kernel = np.ones((9,9), np.uint8)
    inpaint_mask = cv2.dilate(inpaint_mask, kernel, iterations=2)
    
    # Convert to BGR for inpainting
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Try TELEA method first
    inpainted = cv2.inpaint(image_bgr, inpaint_mask, 15, cv2.INPAINT_TELEA)
    
    # For large areas, also try NS method and blend results
    if np.sum(inpaint_mask) / 255 > 10000:  # Large area
        inpainted_ns = cv2.inpaint(image_bgr, inpaint_mask, 21, cv2.INPAINT_NS)
        # Blend both inpainting results for better quality
        alpha = 0.6
        inpainted = cv2.addWeighted(inpainted, alpha, inpainted_ns, 1-alpha, 0)
    
    return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)

def position_object_on_background(object_img, object_mask, object_box, background_img, target_box=None):
    """Position object on background with enhanced blending and positioning"""
    # Create a copy of the background to work with
    result = background_img.copy()
    
    # If no target box is specified, use the center of the background
    if target_box is None:
        bg_height, bg_width = background_img.shape[:2]
        obj_height = object_box[3] - object_box[1]
        obj_width = object_box[2] - object_box[0]
        
        # Calculate centered position with slight bottom offset for natural placement
        x_center = (bg_width - obj_width) // 2
        y_center = int((bg_height - obj_height) * 0.6)  # Place at 60% height for more natural look
        
        target_box = [x_center, y_center, x_center + obj_width, y_center + obj_height]
    
    # Extract object using mask
    obj_mask_3d = np.stack([object_mask > 127] * 3, axis=2)
    extracted_obj = np.zeros_like(object_img)
    extracted_obj[obj_mask_3d] = object_img[obj_mask_3d]
    
    # Create masked version of object
    obj_height = int(object_box[3] - object_box[1])
    obj_width = int(object_box[2] - object_box[0])
    
    # Crop the object and its mask to the bounding box
    cropped_obj = extracted_obj[int(object_box[1]):int(object_box[3]), 
                               int(object_box[0]):int(object_box[2])]
    cropped_mask = object_mask[int(object_box[1]):int(object_box[3]), 
                              int(object_box[0]):int(object_box[2])]
    
    # Calculate target dimensions
    target_width = int(target_box[2] - target_box[0])
    target_height = int(target_box[3] - target_box[1])
    
    # Resize object and mask to fit target box
    if obj_width > 0 and obj_height > 0 and target_width > 0 and target_height > 0:
        # Use INTER_LANCZOS4 for better quality upscaling
        resized_obj = cv2.resize(cropped_obj, (target_width, target_height), 
                               interpolation=cv2.INTER_LANCZOS4)
        resized_mask = cv2.resize(cropped_mask, (target_width, target_height), 
                                interpolation=cv2.INTER_LANCZOS4)
        
        # Create region of interest in result image
        y_start = max(0, int(target_box[1]))
        y_end = min(result.shape[0], int(target_box[1])+target_height)
        x_start = max(0, int(target_box[0]))
        x_end = min(result.shape[1], int(target_box[0])+target_width)
        
        # Adjust object and mask to fit within image bounds
        obj_y_start = max(0, -int(target_box[1]))
        obj_x_start = max(0, -int(target_box[0]))
        obj_y_end = target_height - max(0, int(target_box[1])+target_height - result.shape[0])
        obj_x_end = target_width - max(0, int(target_box[0])+target_width - result.shape[1])
        
        if y_end > y_start and x_end > x_start and obj_y_end > obj_y_start and obj_x_end > obj_x_start:
            roi = result[y_start:y_end, x_start:x_end]
            
            # Calculate blending mask with improved edge blending
            blend_mask = resized_mask[obj_y_start:obj_y_end, obj_x_start:obj_x_end].astype(float) / 255.0
            
            # Apply adaptive Gaussian blur based on object size
            mask_area = np.sum(blend_mask > 0.5)
            if mask_area > 5000:  # Large objects
                blur_size = (11, 11)
            else:  # Small objects
                blur_size = (7, 7)
            
            blend_mask = cv2.GaussianBlur(blend_mask, blur_size, 0)
            blend_mask = np.stack([blend_mask] * 3, axis=2)
            
            # Color adjustment for better blending
            obj_part = resized_obj[obj_y_start:obj_y_end, obj_x_start:obj_x_end]
            
            # Perform blending
            if roi.shape[:2] == obj_part.shape[:2]:
                # Enhanced blending with improved color matching
                obj_hsv = cv2.cvtColor(obj_part, cv2.COLOR_RGB2HSV)
                bg_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
                
                # Better color harmony adjustment
                # Match average hue and saturation more subtly
                bg_mean_h = np.mean(bg_hsv[:,:,0])
                bg_mean_s = np.mean(bg_hsv[:,:,1])
                obj_mean_h = np.mean(obj_hsv[:,:,0])
                obj_mean_s = np.mean(obj_hsv[:,:,1])
                
                # Subtle adjustment towards background color harmony
                hue_adjust = 0.1 * (bg_mean_h - obj_mean_h) / 180.0
                sat_adjust = 0.85 + 0.1 * (bg_mean_s - obj_mean_s) / 255.0
                sat_adjust = np.clip(sat_adjust, 0.7, 1.2)
                
                obj_hsv[:,:,0] = (obj_hsv[:,:,0] + hue_adjust * 180) % 180
                obj_hsv[:,:,1] = np.clip(obj_hsv[:,:,1] * sat_adjust, 0, 255)
                
                adjusted_obj = cv2.cvtColor(obj_hsv, cv2.COLOR_HSV2RGB)
                
                # Blend with adjusted object
                blended = roi * (1 - blend_mask) + adjusted_obj * blend_mask
                result[y_start:y_end, x_start:x_end] = blended
    
    return result

def swap_objects_with_positioning(image1, image2, obj1_class=None, obj2_class=None):
    """Swap objects between images with improved error handling and fallback options"""
    # Ensure images are same size for processing
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]
    
    if height1 != height2 or width1 != width2:
        image2_resized = cv2.resize(image2, (width1, height1))
    else:
        image2_resized = image2.copy()
    
    print(f"🔍 Detecting objects in images...")
    print(f"   Looking for '{obj1_class}' in first image")
    print(f"   Looking for '{obj2_class}' in second image")
    
    # Get refined masks and bounding boxes with optional class filtering
    mask1, box1 = get_refined_object_mask_and_box(image1, obj1_class)
    mask2, box2 = get_refined_object_mask_and_box(image2_resized, obj2_class)
    
    # More detailed error handling
    if mask1 is None or box1 is None:
        print(f"❌ Could not detect any suitable object in first image")
        if obj1_class:
            print(f"   Specifically looking for: {obj1_class}")
        raise ValueError(f"Could not detect objects in first image. Requested class: {obj1_class}")
    
    if mask2 is None or box2 is None:
        print(f"❌ Could not detect any suitable object in second image")
        if obj2_class:
            print(f"   Specifically looking for: {obj2_class}")
        raise ValueError(f"Could not detect objects in second image. Requested class: {obj2_class}")
    
    print(f"✅ Successfully detected objects in both images")
    
    # Fill backgrounds where objects were removed
    print(f"🎨 Inpainting backgrounds...")
    bg1 = inpaint_background(image1, mask1)
    bg2 = inpaint_background(image2_resized, mask2)
    
    # Place object 2 on background 1 at position of object 1
    print(f"🔄 Creating composites...")
    merged1 = position_object_on_background(image2_resized, mask2, box2, bg1, box1)
    
    # Place object 1 on background 2 at position of object 2
    merged2 = position_object_on_background(image1, mask1, box1, bg2, box2)
    
    # Extract objects for visualization
    obj1_mask = mask1 > 127
    obj2_mask = mask2 > 127
    
    obj1 = image1.copy()
    obj2 = image2_resized.copy()
    obj1[~np.stack([obj1_mask] * 3, axis=2)] = 0
    obj2[~np.stack([obj2_mask] * 3, axis=2)] = 0
    
    return (obj1, obj2, bg1, bg2, merged1, merged2)

def parse_simple_prompt(prompt):
    """Parse simple natural language prompts like 'boy in mountains' with enhanced object recognition"""
    # Enhanced objects list with better categorization
    objects = [
        # Humans
        "boy", "girl", "child", "person", "man", "woman", "people", "family", 
        "hiker", "hikers", "knight", "footballer", "player", "moon" 
        
        # Animals  
        "monkey", "monkeys", "ape", "apes", "cow", "cows", "cattle", "dog", "dogs", "puppy", 
        "cat", "cats", "kitten", "horse", "horses", "sheep", "sheeps", "goat", "goats", 
        "animal", "animals", "bird", "birds", "elephant", "elephants", "bear", "bears",
        "zebra", "giraffe", "lion", "tiger",
        
        # Vehicles
        "car", "cars", "vehicle", "truck", "bus", "motorcycle", "bicycle", "bike", "boat", "aeroplane",  
    ]
    
    locations = [
        "mountains", "mountain", "park", "garden", "beach", "field", "playground", 
        "forest", "lake", "river", "stream", "meadow", "pasture", "cliff", "valley",
        "waterfall", "ocean", "sea", "desert", "farm", "hill", "hills", "temple", 
        "city", "masjid", "mosque", "sky", "space", "grass", "ground", "stadium",
        "arena", "gym", "fitness", "yard", "countryside", "woods", "jungle", "dark_sky"
    ]
    
    actions = [
        "playing", "praying", "running", "walking", "sitting", "standing", "eating",
        "drinking", "sleeping", "climbing", "hiking", "swimming", "fishing", "jumping",
        "reading", "writing", "dancing", "singing", "talking", "laughing", "kicking",
        "throwing", "catching", "flying"
    ]
    
    # Convert to lowercase for comparison
    prompt_lower = prompt.lower()
    prompt_words = prompt_lower.split()
    
    # Initialize variables
    subject = None
    location = None
    action = None
    
    # Extract components from the prompt with priority
    for word in prompt_words:
        # Check for exact matches first
        if word in objects and subject is None:
            subject = word
        elif word in locations and location is None:
            location = word
        elif word in actions and action is None:
            action = word
    
    # Handle special cases and synonyms
    if subject:
        # Normalize monkey variations
        if subject in ["monkey", "monkeys", "ape", "apes"]:
            subject = "monkey"
        # Normalize people variations
        elif subject in ["people", "family", "hikers"]:
            subject = "person"
        # Normalize animal variations
        elif subject in ["cows", "cattle"]:
            subject = "cow"
        elif subject in ["dogs", "puppy"]:
            subject = "dog"
        elif subject in ["cats", "kitten"]:
            subject = "cat"
        elif subject in ["sheep", "sheeps"]:
            subject = "sheep"
    
    if location:
        # Normalize location variations
        if location == "mountain":
            location = "mountains"
        elif location in ["masjid", "mosque"]:
            location = "masjid"
        elif location in ["arena", "stadium"]:
            location = "stadium"
        elif location in ["gym", "fitness"]:
            location = "gym"
        elif location in ["woods", "jungle"]:
            location = "forest"
    
    # Special handling for "in gym" or "at stadium"
    if "gym" in prompt_lower:
        location = "gym"
    elif "stadium" in prompt_lower:
        location = "stadium"
    
    print(f"🔍 Parsed: Subject='{subject}', Location='{location}', Action='{action}'")
    return subject, location, action

def create_composite_for_pair_db(subject_doc, background_doc, subject, db):
    """Create a composite image from database documents with better error handling"""
    try:
        print(f"📸 Loading images from database...")
        print(f"   Subject: {subject_doc['filename']}")
        print(f"   Background: {background_doc['filename']}")
        
        # Load the images from database
        subject_image = db.get_image_data(subject_doc['image_id'])
        background_image = db.get_image_data(background_doc['image_id'])
        
        if subject_image is None:
            print(f"❌ Failed to load subject image: {subject_doc['filename']}")
            return None, False
            
        if background_image is None:
            print(f"❌ Failed to load background image: {background_doc['filename']}")
            return None, False
        
        print(f"✅ Successfully loaded both images")
        
        # Process the images - put subject in the background
        # Note: We want the subject on the background, so subject goes to background image
        object1, object2, background1, background2, merged1, merged2 = swap_objects_with_positioning(
            background_image, subject_image, None, subject)
        
        # We want the subject (from subject_image) on the background (from background_image)
        # So we use merged1 (subject_image object on background_image background)
        return merged1, True
        
    except ValueError as ve:
        print(f"⚠️ Object detection issue: {str(ve)}")
        return None, False
    except Exception as e:
        print(f"❌ Error creating composite for {subject_doc['filename']} + {background_doc['filename']}: {str(e)}")
        return None, False

def display_single_best_result_db(prompt, db):
    """Display only the single best result for a given prompt using database with better error handling"""
    # Parse the prompt
    subject, location, action = parse_simple_prompt(prompt)
    print(f"\n🔍 Parsed prompt - Subject: {subject}, Location: {location}, Action: {action}")
    
    if not subject:
        print("❌ Could not identify a subject in the prompt. Please specify an object (like 'boy', 'bird', 'cow', etc.)")
        return None
    
    if not location:
        print("❌ Could not identify a location in the prompt. Please specify a background (like 'mountains', 'park', 'masjid', etc.)")
        return None
    
    # Search for matching images in database
    print(f"🔎 Searching for subject: {subject}")
    subject_matches = db.search_images("object", subject, limit=10)  # Increased search limit
    
    print(f"🔎 Searching for location: {location}")
    location_matches = db.search_images("background", location, limit=10)  # Increased search limit
    
    print(f"📊 Found {len(subject_matches)} subject matches and {len(location_matches)} location matches")
    
    # Enhanced fallback logic
    if not subject_matches:
        print(f"⚠️ No direct matches for '{subject}', trying broader search...")
        # Try broader categories
        fallback_subjects = {
            "bird": ["animal"], 
            "person": ["boy", "girl", "man", "woman", "child"],
            "animal": ["cow", "dog", "cat", "sheep", "horse"]
        }
        
        for fallback in fallback_subjects.get(subject, ["person"]):
            subject_matches = db.search_images("object", fallback, limit=5)
            if subject_matches:
                print(f"✅ Found fallback matches for '{fallback}'")
                break
    
    if not location_matches:
        print(f"⚠️ No direct matches for '{location}', trying broader search...")
        # Try related locations
        fallback_locations = {
            "masjid": ["mosque", "religious", "temple"],
            "mosque": ["masjid", "religious", "temple"],
            "mountains": ["mountain", "hills", "nature"],
            "park": ["garden", "field", "outdoor"]
        }
        
        for fallback in fallback_locations.get(location, ["park", "field"]):
            location_matches = db.search_images("background", fallback, limit=5)
            if location_matches:
                print(f"✅ Found fallback matches for '{fallback}'")
                break
    
    # If still no matches, get any available images
    if not subject_matches:
        print("⚠️ Using any available subject images...")
        all_images = db.get_all_images_info()
        subject_matches = [img for img in all_images if img.get('object_type')][:5]
        for match in subject_matches:
            match['score'] = 1
    
    if not location_matches:
        print("⚠️ Using any available background images...")
        all_images = db.get_all_images_info()
        location_matches = [img for img in all_images if img.get('background_type')][:5]
        for match in location_matches:
            match['score'] = 1
    
    if not subject_matches or not location_matches:
        print("❌ No suitable images found in database for creating composite.")
        print("💡 Try adding more images to the database or use different search terms.")
        return None
    
    # Try multiple combinations until we find one that works
    print(f"🎯 Attempting to create composite with best matches...")
    
    max_attempts = min(3, len(subject_matches) * len(location_matches))
    attempt = 0
    
    for subject_match in subject_matches[:3]:  # Try top 3 subject matches
        for location_match in location_matches[:3]:  # Try top 3 location matches
            if attempt >= max_attempts:
                break
                
            attempt += 1
            print(f"\n🔄 Attempt {attempt}: {subject_match['filename']} + {location_match['filename']}")
            
            # Create composite
            composite, success = create_composite_for_pair_db(subject_match, location_match, subject, db)
            
            if success and composite is not None:
                print(f"✅ Successfully created composite!")
                
                # Display the result
                plt.figure(figsize=(10, 8))
                plt.imshow(composite)
                plt.title(f"Result for: '{prompt}'\n{subject_match['filename']} + {location_match['filename']}", 
                         fontsize=14, fontweight='bold')
                plt.axis('off')
                plt.tight_layout()
                plt.show()
                
                # Save the result
                output_dir = "output"
                os.makedirs(output_dir, exist_ok=True)
                prompt_slug = prompt.replace(' ', '_').lower()
                output_path = os.path.join(output_dir, f"{prompt_slug}_best.jpg")
                plt.imsave(output_path, composite)
                print(f"💾 Saved best result to {output_path}")
                
                return composite
            else:
                print(f"❌ Failed to create composite, trying next combination...")
    
    print(f"❌ Failed to create composite after {attempt} attempts.")
    print("💡 This might be due to:")
    print("   - Objects not being detected properly in the images")
    print("   - Images having incompatible formats or sizes")
    print("   - Try using different images or search terms")
    
    return None

def display_all_matching_results_db(prompt, subject_matches, location_matches, db):
    """Display only 2 best different combinations of matching images from database"""
    if not subject_matches or not location_matches:
        print("No matching images found for the prompt components.")
        return
    
    print(f"\nFound {len(subject_matches)} subject matches and {len(location_matches)} location matches")
    print("Creating 2 best different combinations...")
    
    # Take top 2 subjects and top 2 locations to ensure variety
    top_subjects = subject_matches[:2]  # Top 2 subjects
    top_locations = location_matches[:2]  # Top 2 locations
    
    # Create exactly 2 different combinations
    combinations = []
    
    # Combination 1: Best subject + Best location
    if len(top_subjects) > 0 and len(top_locations) > 0:
        combinations.append((top_subjects[0], top_locations[0]))
    
    # Combination 2: Different pairing to ensure variety
    if len(top_subjects) > 1 and len(top_locations) > 1:
        # Second subject + Second location
        combinations.append((top_subjects[1], top_locations[1]))
    elif len(top_subjects) > 1 and len(top_locations) > 0:
        # Second subject + First location
        combinations.append((top_subjects[1], top_locations[0]))
    elif len(top_subjects) > 0 and len(top_locations) > 1:
        # First subject + Second location
        combinations.append((top_subjects[0], top_locations[1]))
    
    # Ensure we have exactly 2 combinations and they are different
    if len(combinations) < 2 and len(combinations) > 0:
        # If we only have one combination, try to create a second one with different images
        if len(subject_matches) > 1 or len(location_matches) > 1:
            # Try to find a different combination
            for i, subj in enumerate(subject_matches[:3]):
                for j, loc in enumerate(location_matches[:3]):
                    new_combo = (subj, loc)
                    # Check if this combination is different from existing ones
                    is_different = True
                    for existing_combo in combinations:
                        if (new_combo[0]['filename'] == existing_combo[0]['filename'] and 
                            new_combo[1]['filename'] == existing_combo[1]['filename']):
                            is_different = False
                            break
                    
                    if is_different:
                        combinations.append(new_combo)
                        break
                if len(combinations) >= 2:
                    break
    
    # Limit to exactly 2 combinations
    combinations = combinations[:2]
    total_combinations = len(combinations)
    
    if total_combinations == 0:
        print("❌ Could not create any valid combinations.")
        return []
    
    print(f"Displaying {total_combinations} best different combinations")
    
    # Create the main figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    if total_combinations == 1:
        axes = [axes]
    
    fig.suptitle(f"Top {total_combinations} Different Combinations for: '{prompt}'", fontsize=16, fontweight='bold')
    
    # Parse prompt for subject identification
    subject, location, action = parse_simple_prompt(prompt)
    
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    successful_composites = []
    
    # Generate the combinations
    for idx, (subject_doc, location_doc) in enumerate(combinations):
        print(f"Processing combination {idx + 1}/{total_combinations}: "
              f"{subject_doc['filename']} + {location_doc['filename']}")
        
        # Create composite
        composite, success = create_composite_for_pair_db(subject_doc, location_doc, subject, db)
        
        if success and composite is not None:
            # Display in subplot
            ax = axes[idx] if total_combinations > 1 else axes[0]
            ax.imshow(composite)
            ax.set_title(f"Combo {idx + 1}\n{subject_doc['filename']}\n+ {location_doc['filename']}", 
                       fontsize=12, fontweight='bold')
            ax.axis('off')
            
            # Save individual result
            prompt_slug = prompt.replace(' ', '_').lower()
            output_filename = f"{prompt_slug}_combo_{idx + 1}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            plt.imsave(output_path, composite)
            
            successful_composites.append({
                'composite': composite,
                'subject_file': subject_doc['filename'],
                'location_file': location_doc['filename'],
                'output_path': output_path,
                'subject_score': subject_doc['score'],
                'location_score': location_doc['score'],
                'total_score': subject_doc['score'] + location_doc['score']
            })
            
            print(f"✅ Success - Saved to {output_path}")
        else:
            # Display placeholder for failed combination
            ax = axes[idx] if total_combinations > 1 else axes[0]
            ax.text(0.5, 0.5, 'Failed to\ncreate composite', 
                   ha='center', va='center', fontsize=12, color='red',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
            ax.set_title(f"Combo {idx + 1}\n{subject_doc['filename']}\n+ {location_doc['filename']}", 
                       fontsize=12, fontweight='bold')
            ax.axis('off')
            print(f"❌ Failed")
    
    # Hide unused subplot if only 1 combination
    if total_combinations == 1:
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n📊 Summary:")
    print(f"Total combinations attempted: {total_combinations}")
    print(f"Successful composites: {len(successful_composites)}")
    if total_combinations > 0:
        print(f"Success rate: {len(successful_composites)/total_combinations*100:.1f}%")
    
    if successful_composites:
        # Sort by total score (best matches first)
        successful_composites.sort(key=lambda x: x['total_score'], reverse=True)
        
        print(f"\n🏆 Best combinations (by matching score):")
        for i, comp in enumerate(successful_composites, 1):
            print(f"{i}. {comp['subject_file']} + {comp['location_file']} "
                  f"(Score: {comp['total_score']}) -> {comp['output_path']}")
    
    return successful_composites

# Usage example functions
def main_interactive_db():
    """Main interactive function for database-based image swapping"""
    print("🎨 AI Image Swapping System with Database Storage")
    print("=" * 50)
    
    # Initialize database
    db = ImageDatabase()
    
    while True:
        print("\nOptions:")
        print("1. Add images to database from folder")
        print("2. Search and create single best composite")
        print("3. Search and show all matching combinations")
        print("4. View database statistics")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            folder_path = input("Enter path to images folder: ").strip()
            if os.path.exists(folder_path):
                db.populate_from_folder(folder_path)
            else:
                print("❌ Folder not found!")
        
        elif choice == '2':
            prompt = input("\nEnter your prompt (e.g., 'boy in mountains'): ").strip()
            if prompt:
                result = display_single_best_result_db(prompt, db)
                if result is not None:
                    print("✅ Best result displayed and saved!")
        
        elif choice == '3':
            prompt = input("\nEnter your prompt (e.g., 'boy in mountains'): ").strip()
            if prompt:
                subject, location, action = parse_simple_prompt(prompt)
                subject_matches = db.search_images("object", subject, limit=5) if subject else []
                location_matches = db.search_images("background", location, limit=5) if location else []
                
                if subject_matches and location_matches:
                    results = display_all_matching_results_db(prompt, subject_matches, location_matches, db)
                    print(f"✅ Generated {len(results)} successful combinations!")
                else:
                    print("❌ No matching images found for this prompt.")
        
        elif choice == '4':
            images_info = db.get_all_images_info()
            print(f"\n📊 Database Statistics:")
            print(f"Total images: {len(images_info)}")
            
            # Count by object types
            object_types = {}
            background_types = {}
            for img in images_info:
                obj_type = img.get('object_type', 'unknown')
                bg_type = img.get('background_type', 'unknown')
                object_types[obj_type] = object_types.get(obj_type, 0) + 1
                background_types[bg_type] = background_types.get(bg_type, 0) + 1
            
            print(f"\nObject types: {dict(sorted(object_types.items()))}")
            print(f"Background types: {dict(sorted(background_types.items()))}")
        
        elif choice == '5':
            print("👋 Goodbye!")
            db.close_connection()
            break
        
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    # Example usage
    try:
        main_interactive_db()
    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("Please check your setup and try again.")