import streamlit as st
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
import tempfile
from pathlib import Path
import time

# Set page config
st.set_page_config(
    page_title="AI Image Swapping System",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    
    .status-info {
        color: #17a2b8;
        font-weight: bold;
    }
    
    .metric-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .image-info {
        background-color: #e9ecef;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        font-size: 0.9rem;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .upload-area {
        border: 2px dashed #1f77b4;
        border-radius: 1rem;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class ImageDatabase:
    """Database handler for storing and retrieving images with descriptions"""
    
    def __init__(self, connection_string="mongodb://localhost:27017/", db_name="image_swapping_db"):
        """Initialize MongoDB connection and GridFS for image storage"""
        try:
            self.client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.server_info()
            self.db = self.client[db_name]
            self.fs = gridfs.GridFS(self.db)
            self.collection = self.db.images
            
            # Create indexes for faster searching
            self.collection.create_index("object_type")
            self.collection.create_index("background_type")
            self.collection.create_index("description")
            
            self.connected = True
        except Exception as e:
            st.error(f"Failed to connect to MongoDB: {str(e)}")
            st.error("Please ensure MongoDB is running on localhost:27017")
            self.connected = False
        
    def store_image(self, image_data, filename, object_type, background_type, description=""):
        """Store image in database with metadata"""
        if not self.connected:
            return None
            
        try:
            # Store image in GridFS
            image_id = self.fs.put(image_data, filename=filename)
            
            # Store metadata in collection
            metadata = {
                'image_id': image_id,
                'filename': filename,
                'object_type': object_type.lower(),
                'background_type': background_type.lower(),
                'description': description.lower(),
                'keywords': self._generate_keywords(object_type, background_type, description)
            }
            
            result = self.collection.insert_one(metadata)
            return result.inserted_id
            
        except Exception as e:
            st.error(f"Error storing image {filename}: {str(e)}")
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
        """Search for images by object or background type"""
        if not self.connected:
            return []
            
        search_field = f"{query_type}_type"
        results = []
        
        try:
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
        except Exception as e:
            st.error(f"Error searching images: {str(e)}")
            return []
    
    def get_image_data(self, image_id):
        """Retrieve image data from GridFS"""
        if not self.connected:
            return None
            
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
            st.error(f"Error retrieving image {image_id}: {str(e)}")
            return None
    
    def get_all_images_info(self):
        """Get information about all stored images"""
        if not self.connected:
            return []
        try:
            return list(self.collection.find({}, {'image_id': 1, 'filename': 1, 'object_type': 1, 'background_type': 1, 'description': 1}))
        except Exception as e:
            st.error(f"Error getting images info: {str(e)}")
            return []
    
    def close_connection(self):
        """Close database connection"""
        if self.connected:
            self.client.close()

@st.cache_resource
def setup_detectron():
    """Setup Detectron2 configuration with enhanced object detection parameters"""
    try:
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.65
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
        return DefaultPredictor(cfg)
    except Exception as e:
        st.error(f"Error setting up Detectron2: {str(e)}")
        return None

def get_refined_object_mask_and_box(image, class_label=None):
    """Get refined mask and bounding box for objects with improved boundary detection"""
    predictor = setup_detectron()
    if predictor is None:
        return None, None
        
    outputs = predictor(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    instances = outputs["instances"].to("cpu")
    if len(instances) == 0:
        st.warning("⚠️ No objects detected in image")
        return None, None
    
    masks = instances.pred_masks.numpy()
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    classes = instances.pred_classes.numpy()
    
    # Enhanced class mapping
    class_id_map = {
        "person": [0], "boy": [0], "girl": [0], "child": [0], "man": [0], "woman": [0],
        "knight": [0], "footballer": [0],
        "monkey": [16, 17, 18, 19, 20, 21, 22, 23],
        "cow": [19], "cattle": [19], 
        "dog": [16], "cat": [15], "horse": [17], "sheep": [18],
        "bird": [14], "elephant": [20], "bear": [21], "zebra": [22], "giraffe": [23],
        "car": [2], "truck": [7], "bus": [5], "motorcycle": [3], "bicycle": [1]
    }
    
    # COCO class names
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
    st.info(f"🔍 Detected objects: {detected_objects}")
    st.info(f"📊 Detection scores: {[f'{s:.3f}' for s in scores]}")
    
    # Filter by class if specified
    if class_label is not None:
        class_label_lower = class_label.lower()
        st.info(f"🎯 Looking specifically for: {class_label_lower}")
        
        if class_label_lower in class_id_map:
            target_classes = class_id_map[class_label_lower]
            
            # Special handling for monkey
            if class_label_lower == "monkey":
                animal_classes = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
                animal_indices = np.isin(classes, animal_classes)
                if any(animal_indices):
                    selected_indices = animal_indices
                    st.success(f"✅ Found animal for monkey: {[coco_class_names.get(cls, cls) for cls in classes[animal_indices]]}")
                else:
                    person_indices = np.isin(classes, [0])
                    if any(person_indices):
                        selected_indices = person_indices
                        st.warning("⚠️ Using person as fallback for monkey")
            else:
                class_indices = np.isin(classes, target_classes)
                if any(class_indices):
                    selected_indices = class_indices
                    st.success(f"✅ Found {sum(class_indices)} {class_label} object(s)")
                else:
                    st.warning(f"⚠️ No {class_label} detected")
                    if class_label_lower in ["knight", "footballer", "boy", "girl", "child", "man", "woman"]:
                        person_indices = np.isin(classes, [0])
                        if any(person_indices):
                            selected_indices = person_indices
                            st.success("✅ Found person-like object as fallback")
        else:
            st.warning(f"⚠️ Unknown class {class_label}, trying fallback to any suitable object")
    
    # If no specific class found, use smart selection
    if selected_indices is None:
        confidence_threshold = 0.3
        high_confidence_indices = scores > confidence_threshold
        
        if any(high_confidence_indices):
            high_conf_classes = classes[high_confidence_indices]
            high_conf_scores = scores[high_confidence_indices]
            
            vehicle_classes = [1, 2, 3, 5, 6, 7, 8]
            non_vehicle_mask = ~np.isin(high_conf_classes, vehicle_classes)
            
            if any(non_vehicle_mask):
                best_non_vehicle_idx = np.argmax(high_conf_scores[non_vehicle_mask])
                original_idx = np.where(high_confidence_indices)[0][np.where(non_vehicle_mask)[0][best_non_vehicle_idx]]
                selected_indices = np.zeros(len(classes), dtype=bool)
                selected_indices[original_idx] = True
                st.success(f"✅ Selected best non-vehicle object: {coco_class_names.get(classes[original_idx], 'unknown')}")
            else:
                best_vehicle_idx = np.argmax(high_conf_scores)
                original_idx = np.where(high_confidence_indices)[0][best_vehicle_idx]
                selected_indices = np.zeros(len(classes), dtype=bool)
                selected_indices[original_idx] = True
                st.success(f"✅ Selected best vehicle object: {coco_class_names.get(classes[original_idx], 'unknown')}")
        else:
            st.warning("⚠️ No high-confidence objects, lowering threshold...")
            confidence_threshold = 0.15
            any_confidence_indices = scores > confidence_threshold
            if any(any_confidence_indices):
                selected_indices = any_confidence_indices
            else:
                selected_indices = np.array([True] * len(masks))
                st.warning("⚠️ Using all available objects")
    
    # Apply selection
    if selected_indices is not None:
        masks = masks[selected_indices]
        boxes = boxes[selected_indices]
        scores = scores[selected_indices]
        classes = classes[selected_indices]
    
    if len(masks) == 0:
        st.error("❌ No valid objects found after filtering")
        return None, None
    
    # Select the best object
    best_idx = np.argmax(scores)
    final_mask = masks[best_idx].astype(np.uint8) * 255
    main_box = boxes[best_idx]
    selected_class = classes[best_idx]
    
    st.success(f"✅ Selected object: {coco_class_names.get(selected_class, 'unknown')} "
               f"with confidence: {scores[best_idx]:.3f}")
    
    # Apply mask refinement
    kernel = np.ones((3,3), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    
    # Edge refinement
    mask_area = np.sum(final_mask > 0)
    if mask_area > 10000:
        blur_kernel = (7, 7)
        dilate_iterations = 2
    else:
        blur_kernel = (3, 3)
        dilate_iterations = 1
    
    final_mask = cv2.GaussianBlur(final_mask, blur_kernel, 0)
    kernel = np.ones((3,3), np.uint8)
    final_mask = cv2.dilate(final_mask, kernel, iterations=dilate_iterations)
    
    return final_mask, main_box

def inpaint_background(image, mask):
    """Inpaint removed object area using improved background reconstruction"""
    inpaint_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    
    kernel = np.ones((9,9), np.uint8)
    inpaint_mask = cv2.dilate(inpaint_mask, kernel, iterations=2)
    
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    inpainted = cv2.inpaint(image_bgr, inpaint_mask, 15, cv2.INPAINT_TELEA)
    
    if np.sum(inpaint_mask) / 255 > 10000:
        inpainted_ns = cv2.inpaint(image_bgr, inpaint_mask, 21, cv2.INPAINT_NS)
        alpha = 0.6
        inpainted = cv2.addWeighted(inpainted, alpha, inpainted_ns, 1-alpha, 0)
    
    return cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)

def position_object_on_background(object_img, object_mask, object_box, background_img, target_box=None):
    """Position object on background with enhanced blending"""
    result = background_img.copy()
    
    if target_box is None:
        bg_height, bg_width = background_img.shape[:2]
        obj_height = object_box[3] - object_box[1]
        obj_width = object_box[2] - object_box[0]
        
        x_center = (bg_width - obj_width) // 2
        y_center = int((bg_height - obj_height) * 0.6)
        
        target_box = [x_center, y_center, x_center + obj_width, y_center + obj_height]
    
    obj_mask_3d = np.stack([object_mask > 127] * 3, axis=2)
    extracted_obj = np.zeros_like(object_img)
    extracted_obj[obj_mask_3d] = object_img[obj_mask_3d]
    
    obj_height = int(object_box[3] - object_box[1])
    obj_width = int(object_box[2] - object_box[0])
    
    cropped_obj = extracted_obj[int(object_box[1]):int(object_box[3]), 
                               int(object_box[0]):int(object_box[2])]
    cropped_mask = object_mask[int(object_box[1]):int(object_box[3]), 
                              int(object_box[0]):int(object_box[2])]
    
    target_width = int(target_box[2] - target_box[0])
    target_height = int(target_box[3] - target_box[1])
    
    if obj_width > 0 and obj_height > 0 and target_width > 0 and target_height > 0:
        resized_obj = cv2.resize(cropped_obj, (target_width, target_height), 
                               interpolation=cv2.INTER_LANCZOS4)
        resized_mask = cv2.resize(cropped_mask, (target_width, target_height), 
                                interpolation=cv2.INTER_LANCZOS4)
        
        y_start = max(0, int(target_box[1]))
        y_end = min(result.shape[0], int(target_box[1])+target_height)
        x_start = max(0, int(target_box[0]))
        x_end = min(result.shape[1], int(target_box[0])+target_width)
        
        obj_y_start = max(0, -int(target_box[1]))
        obj_x_start = max(0, -int(target_box[0]))
        obj_y_end = target_height - max(0, int(target_box[1])+target_height - result.shape[0])
        obj_x_end = target_width - max(0, int(target_box[0])+target_width - result.shape[1])
        
        if y_end > y_start and x_end > x_start and obj_y_end > obj_y_start and obj_x_end > obj_x_start:
            roi = result[y_start:y_end, x_start:x_end]
            
            blend_mask = resized_mask[obj_y_start:obj_y_end, obj_x_start:obj_x_end].astype(float) / 255.0
            
            mask_area = np.sum(blend_mask > 0.5)
            if mask_area > 5000:
                blur_size = (11, 11)
            else:
                blur_size = (7, 7)
            
            blend_mask = cv2.GaussianBlur(blend_mask, blur_size, 0)
            blend_mask = np.stack([blend_mask] * 3, axis=2)
            
            obj_part = resized_obj[obj_y_start:obj_y_end, obj_x_start:obj_x_end]
            
            if roi.shape[:2] == obj_part.shape[:2]:
                obj_hsv = cv2.cvtColor(obj_part, cv2.COLOR_RGB2HSV)
                bg_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
                
                bg_mean_h = np.mean(bg_hsv[:,:,0])
                bg_mean_s = np.mean(bg_hsv[:,:,1])
                obj_mean_h = np.mean(obj_hsv[:,:,0])
                obj_mean_s = np.mean(obj_hsv[:,:,1])
                
                hue_adjust = 0.1 * (bg_mean_h - obj_mean_h) / 180.0
                sat_adjust = 0.85 + 0.1 * (bg_mean_s - obj_mean_s) / 255.0
                sat_adjust = np.clip(sat_adjust, 0.7, 1.2)
                
                obj_hsv[:,:,0] = (obj_hsv[:,:,0] + hue_adjust * 180) % 180
                obj_hsv[:,:,1] = np.clip(obj_hsv[:,:,1] * sat_adjust, 0, 255)
                
                adjusted_obj = cv2.cvtColor(obj_hsv, cv2.COLOR_HSV2RGB)
                
                blended = roi * (1 - blend_mask) + adjusted_obj * blend_mask
                result[y_start:y_end, x_start:x_end] = blended
    
    return result

def swap_objects_with_positioning(image1, image2, obj1_class=None, obj2_class=None):
    """Swap objects between images with improved error handling"""
    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]
    
    if height1 != height2 or width1 != width2:
        image2_resized = cv2.resize(image2, (width1, height1))
    else:
        image2_resized = image2.copy()
    
    st.info("🔍 Detecting objects in images...")
    if obj1_class:
        st.info(f"   Looking for '{obj1_class}' in first image")
    if obj2_class:
        st.info(f"   Looking for '{obj2_class}' in second image")
    
    mask1, box1 = get_refined_object_mask_and_box(image1, obj1_class)
    mask2, box2 = get_refined_object_mask_and_box(image2_resized, obj2_class)
    
    if mask1 is None or box1 is None:
        error_msg = f"Could not detect any suitable object in first image"
        if obj1_class:
            error_msg += f". Specifically looking for: {obj1_class}"
        st.error(error_msg)
        raise ValueError(error_msg)
    
    if mask2 is None or box2 is None:
        error_msg = f"Could not detect any suitable object in second image"
        if obj2_class:
            error_msg += f". Specifically looking for: {obj2_class}"
        st.error(error_msg)
        raise ValueError(error_msg)
    
    st.success("✅ Successfully detected objects in both images")
    
    st.info("🎨 Inpainting backgrounds...")
    bg1 = inpaint_background(image1, mask1)
    bg2 = inpaint_background(image2_resized, mask2)
    
    st.info("🔄 Creating composites...")
    merged1 = position_object_on_background(image2_resized, mask2, box2, bg1, box1)
    merged2 = position_object_on_background(image1, mask1, box1, bg2, box2)
    
    obj1_mask = mask1 > 127
    obj2_mask = mask2 > 127
    
    obj1 = image1.copy()
    obj2 = image2_resized.copy()
    obj1[~np.stack([obj1_mask] * 3, axis=2)] = 0
    obj2[~np.stack([obj2_mask] * 3, axis=2)] = 0
    
    return (obj1, obj2, bg1, bg2, merged1, merged2)

def parse_simple_prompt(prompt):
    """Parse simple natural language prompts like 'boy in mountains'"""
    objects = [
        "boy", "girl", "child", "person", "man", "woman", "people", "family", 
        "hiker", "hikers", "knight", "footballer", "player",
        "monkey", "monkeys", "ape", "apes", "cow", "cows", "cattle", "dog", "dogs", "puppy", 
        "cat", "cats", "kitten", "horse", "horses", "sheep", "bird", "birds", "elephant", "elephants", "bear", "zebra", "giraffe",
        "car", "cars", "vehicle", "truck", "bus", "motorcycle", "bike", "bicycle"
    ]
    
    backgrounds = [
        "mountains", "mountain", "peak", "hills", "hill", "alps", "highlands",
        "park", "garden", "playground", "field", "yard", "grass", "lawn",
        "beach", "shore", "coast", "ocean", "sea", "sand", "water",
        "forest", "woods", "trees", "jungle", "woodland", "nature",
        "farm", "countryside", "rural", "pasture", "barn", "ranch",
        "city", "urban", "town", "downtown", "cityscape", "night_city",
        "sky", "clouds", "air", "heaven", "atmosphere", "sunset", "sunrise",
        "space", "universe", "stars", "galaxy", "cosmos", "night",
        "ground", "earth", "soil", "dirt", "land", "road", "path",
        "stadium", "arena", "sports", "football", "gym", "fitness",
        "masjid", "mosque", "temple", "church", "building", "house"
    ]
    
    prompt_lower = prompt.lower()
    found_object = None
    found_background = None
    
    # Find object
    for obj in objects:
        if obj in prompt_lower:
            found_object = obj
            break
    
    # Find background
    for bg in backgrounds:
        if bg in prompt_lower:
            found_background = bg
            break
    
    return found_object, found_background

def display_image_grid(images, titles, cols=3):
    """Display images in a grid layout"""
    n_images = len(images)
    rows = (n_images + cols - 1) // cols
    
    for i in range(0, n_images, cols):
        cols_to_show = st.columns(cols)
        for j in range(cols):
            if i + j < n_images:
                with cols_to_show[j]:
                    st.image(images[i + j], caption=titles[i + j], use_column_width=True)

def create_collage(images, titles):
    """Create a collage of all result images"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('AI Image Swapping Results', fontsize=16, fontweight='bold')
    
    for i, (img, title) in enumerate(zip(images, titles)):
        row = i // 3
        col = i % 3
        axes[row, col].imshow(img)
        axes[row, col].set_title(title, fontsize=12, fontweight='bold')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig

def main():
    """Main Streamlit application"""
    # Header
    st.markdown('<h1 class="main-header">🎨 AI Image Swapping System</h1>', unsafe_allow_html=True)
    
    # Initialize database
    db = ImageDatabase()
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🤖 Smart Detection</h3>
            <p>Advanced AI object detection with Detectron2</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🔄 Seamless Swapping</h3>
            <p>Natural object swapping with background inpainting</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">  
            <h3>💾 Smart Storage</h3>
            <p>MongoDB-powered image database with intelligent search</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔄 Image Swapping", "💾 Database Manager", "🔍 Search & Swap", "📊 Analytics"])
    
    with tab1:
        st.header("🔄 Direct Image Swapping")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="upload-area">', unsafe_allow_html=True)
            st.subheader("📤 Upload First Image")
            uploaded_file1 = st.file_uploader("Choose first image", type=['jpg', 'jpeg', 'png'], key="img1")
            obj1_class = st.text_input("Object type in first image (optional)", 
                                     placeholder="e.g., boy, car, dog", key="obj1")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="upload-area">', unsafe_allow_html=True)
            st.subheader("📤 Upload Second Image")
            uploaded_file2 = st.file_uploader("Choose second image", type=['jpg', 'jpeg', 'png'], key="img2")
            obj2_class = st.text_input("Object type in second image (optional)", 
                                     placeholder="e.g., girl, tree, cat", key="obj2")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file1 and uploaded_file2:
            col1, col2 = st.columns(2)
            
            with col1:
                image1 = np.array(Image.open(uploaded_file1))
                st.image(image1, caption="First Image", use_column_width=True)
                st.markdown(f'<div class="image-info">📏 Size: {image1.shape[1]}x{image1.shape[0]} px</div>', unsafe_allow_html=True)
            
            with col2:
                image2 = np.array(Image.open(uploaded_file2))
                st.image(image2, caption="Second Image", use_column_width=True)
                st.markdown(f'<div class="image-info">📏 Size: {image2.shape[1]}x{image2.shape[0]} px</div>', unsafe_allow_html=True)
            
            if st.button("🔄 Swap Objects", key="swap_btn"):
                try:
                    with st.spinner("🤖 Processing images... This may take a moment"):
                        progress_bar = st.progress(0)
                        
                        progress_bar.progress(20)
                        st.info("🔍 Analyzing images and detecting objects...")
                        
                        obj1_input = obj1_class.strip() if obj1_class.strip() else None
                        obj2_input = obj2_class.strip() if obj2_class.strip() else None
                        
                        progress_bar.progress(40)
                        results = swap_objects_with_positioning(image1, image2, obj1_input, obj2_input)
                        
                        progress_bar.progress(80)
                        obj1, obj2, bg1, bg2, merged1, merged2 = results
                        
                        progress_bar.progress(100)
                        st.success("✅ Image swapping completed successfully!")
                        
                        # Display results
                        st.header("🎯 Swapping Results")
                        
                        # Create tabs for different views
                        result_tab1, result_tab2, result_tab3 = st.tabs(["🔄 Final Results", "🎭 Components", "📊 Collage"])
                        
                        with result_tab1:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("🎨 Object 2 → Background 1")
                                st.image(merged1, use_column_width=True)
                            with col2:
                                st.subheader("🎨 Object 1 → Background 2")
                                st.image(merged2, use_column_width=True)
                        
                        with result_tab2:
                            st.subheader("🧩 Extracted Components")
                            display_image_grid(
                                [obj1, obj2, bg1, bg2],
                                ["Extracted Object 1", "Extracted Object 2", "Background 1", "Background 2"],
                                cols=2
                            )
                        
                        with result_tab3:
                            st.subheader("🖼️ Complete Collage")
                            collage_fig = create_collage(
                                [obj1, obj2, bg1, bg2, merged1, merged2],
                                ["Object 1", "Object 2", "Background 1", "Background 2", "Result 1", "Result 2"]
                            )
                            st.pyplot(collage_fig)
                        
                        # Download buttons
                        st.header("💾 Download Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Convert to PIL for download
                            result1_pil = Image.fromarray(merged1.astype('uint8'))
                            buf1 = io.BytesIO()
                            result1_pil.save(buf1, format='PNG')
                            st.download_button(
                                label="📥 Download Result 1",
                                data=buf1.getvalue(),
                                file_name="swapped_result_1.png",
                                mime="image/png"
                            )
                        
                        with col2:
                            result2_pil = Image.fromarray(merged2.astype('uint8'))
                            buf2 = io.BytesIO()
                            result2_pil.save(buf2, format='PNG')
                            st.download_button(
                                label="📥 Download Result 2",
                                data=buf2.getvalue(),
                                file_name="swapped_result_2.png",
                                mime="image/png"
                            )
                        
                except Exception as e:
                    st.error(f"❌ Error during swapping: {str(e)}")
                    st.error("Please try with different images or check if objects are clearly visible")
    
    with tab2:
        st.header("💾 Database Manager")
        
        if not db.connected:
            st.error("❌ Database not connected. Please ensure MongoDB is running.")
            st.info("💡 To use the database features, make sure MongoDB is installed and running on localhost:27017")
        else:
            st.success("✅ Database connected successfully")
            
            # Upload to database
            st.subheader("📤 Add Image to Database")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                db_upload = st.file_uploader("Choose image for database", type=['jpg', 'jpeg', 'png'], key="db_upload")
                
                if db_upload:
                    preview_img = Image.open(db_upload)
                    st.image(preview_img, caption="Preview", use_column_width=True)
            
            with col2:
                object_type = st.selectbox("Object Type", [
                    "boy", "girl", "child", "person", "man", "woman",
                    "monkey", "cow", "dog", "cat", "bird", "elephant", "sheep",
                    "car", "knight", "footballer"
                ])
                
                background_type = st.selectbox("Background Type", [
                    "mountains", "park", "beach", "forest", "farm", "city",
                    "sky", "space", "grass", "ground", "stadium", "gym", "masjid", "field"
                ])
                
                description = st.text_area("Description (optional)", 
                                         placeholder="Additional details about the image...")
            
            if st.button("💾 Save to Database") and db_upload:
                try:
                    # Convert image to bytes
                    img_bytes = db_upload.getvalue()
                    filename = db_upload.name
                    
                    # Store in database
                    result_id = db.store_image(img_bytes, filename, object_type, background_type, description)
                    
                    if result_id:
                        st.success(f"✅ Image saved successfully! ID: {result_id}")
                    else:
                        st.error("❌ Failed to save image")
                        
                except Exception as e:
                    st.error(f"❌ Error saving image: {str(e)}")
            
            # Display stored images
            st.subheader("📋 Stored Images")
            
            if st.button("🔄 Refresh Database View"):
                st.rerun()
            
            try:
                stored_images = db.get_all_images_info()
                
                if stored_images:
                    st.info(f"📊 Total images in database: {len(stored_images)}")
                    
                    # Create a dataframe for better display
                    import pandas as pd
                    df_data = []
                    for img in stored_images:
                        df_data.append({
                            'Filename': img.get('filename', 'N/A'),
                            'Object': img.get('object_type', 'N/A'),
                            'Background': img.get('background_type', 'N/A'),
                            'Description': img.get('description', 'N/A')[:50] + '...' if len(img.get('description', '')) > 50 else img.get('description', 'N/A')
                        })
                    
                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("📭 No images stored in database yet")
                    
            except Exception as e:
                st.error(f"❌ Error retrieving stored images: {str(e)}")
    
    with tab3:
        st.header("🔍 Search & Swap")
        
        if not db.connected:
            st.error("❌ Database not connected. Search functionality unavailable.")
        else:
            st.info("🎯 Upload one image and find a matching image from the database to swap with")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📤 Upload Your Image")
                search_upload = st.file_uploader("Choose your image", type=['jpg', 'jpeg', 'png'], key="search_upload")
                
                if search_upload:
                    user_image = np.array(Image.open(search_upload))
                    st.image(user_image, caption="Your Image", use_column_width=True)
                    
                    user_obj_class = st.text_input("Object in your image", 
                                                 placeholder="e.g., boy, car, dog", key="user_obj")
            
            with col2:
                st.subheader("🔍 Search Database")
                
                # Simple search
                search_prompt = st.text_input("Search for", 
                                            placeholder="e.g., 'boy in mountains' or 'dog in park'", 
                                            key="search_prompt")
                
                # Advanced search
                with st.expander("🔧 Advanced Search"):
                    search_type = st.selectbox("Search by", ["object", "background"])
                    search_term = st.text_input("Search term", key="advanced_search")
                
                if st.button("🔍 Search"):
                    if search_prompt:
                        # Parse simple prompt
                        obj, bg = parse_simple_prompt(search_prompt)
                        if obj:
                            search_results = db.search_images("object", obj, limit=5)
                        elif bg:
                            search_results = db.search_images("background", bg, limit=5)
                        else:
                            st.warning("⚠️ Could not understand search prompt. Try 'object in background' format.")
                            search_results = []
                    elif search_term:
                        search_results = db.search_images(search_type, search_term, limit=5)
                    else:
                        st.warning("⚠️ Please enter a search term")
                        search_results = []
                    
                    if search_results:
                        st.success(f"✅ Found {len(search_results)} matching images")
                        
                        # Display search results
                        for i, result in enumerate(search_results):
                            with st.container():
                                st.markdown(f"**Result {i+1}** - Score: {result.get('score', 0)}")
                                
                                try:
                                    db_image = db.get_image_data(result['image_id'])
                                    if db_image is not None:
                                        col_img, col_info = st.columns([1, 1])
                                        
                                        with col_img:
                                            st.image(db_image, use_column_width=True)
                                        
                                        with col_info:
                                            st.write(f"**Object:** {result.get('object_type', 'N/A')}")
                                            st.write(f"**Background:** {result.get('background_type', 'N/A')}")
                                            st.write(f"**Description:** {result.get('description', 'N/A')}")
                                            
                                            if search_upload and st.button(f"🔄 Swap with Result {i+1}", key=f"swap_{i}"):
                                                try:
                                                    with st.spinner("🤖 Swapping images..."):
                                                        user_obj_input = user_obj_class.strip() if user_obj_class.strip() else None
                                                        db_obj_input = result.get('object_type')
                                                        
                                                        swap_results = swap_objects_with_positioning(
                                                            user_image, db_image, user_obj_input, db_obj_input
                                                        )
                                                        
                                                        obj1, obj2, bg1, bg2, merged1, merged2 = swap_results
                                                        
                                                        st.success("✅ Swap completed!")
                                                        
                                                        # Display results
                                                        result_col1, result_col2 = st.columns(2)
                                                        with result_col1:
                                                            st.image(merged1, caption="Database object → Your background", use_column_width=True)
                                                        with result_col2:
                                                            st.image(merged2, caption="Your object → Database background", use_column_width=True)
                                                
                                                except Exception as e:
                                                    st.error(f"❌ Swap failed: {str(e)}")
                                    else:
                                        st.error("❌ Could not load image from database")
                                        
                                except Exception as e:
                                    st.error(f"❌ Error displaying result: {str(e)}")
                                
                                st.markdown("---")
                    else:
                        st.info("📭 No matching images found")
    
    with tab4:
        st.header("📊 System Analytics")
        
        # System information
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-box">
                <h4>🤖 AI Model</h4>
                <p>Detectron2 + Mask R-CNN</p>
                <p>Device: """ + ("CUDA" if torch.cuda.is_available() else "CPU") + """</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-box">
                <h4>💾 Database</h4>
                <p>MongoDB + GridFS</p>
                <p>Status: """ + ("Connected" if db.connected else "Disconnected") + """</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if db.connected:
                try:
                    image_count = len(db.get_all_images_info())
                    st.markdown(f"""
                    <div class="metric-box">
                        <h4>📈 Statistics</h4>
                        <p>Stored Images: {image_count}</p>
                        <p>Available Classes: 23</p>
                    </div>
                    """, unsafe_allow_html=True)
                except:
                    st.markdown("""
                    <div class="metric-box">
                        <h4>📈 Statistics</h4>
                        <p>Unable to fetch stats</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-box">
                    <h4>📈 Statistics</h4>
                    <p>Database unavailable</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Performance tips
        st.subheader("💡 Performance Tips")
        st.info("""
        **For Best Results:**
        - Use clear, high-contrast images
        - Ensure objects are well-lit and unobstructed
        - Specify object types when possible
        - Use images with resolution between 800x600 and 2000x1500 pixels
        - Avoid images with too many overlapping objects
        """)
        
        # Supported objects
        st.subheader("🎯 Supported Object Types")
        supported_objects = [
            "People: boy, girl, child, person, man, woman, knight, footballer",
            "Animals: monkey, cow, dog, cat, bird, elephant, sheep, horse, bear, zebra, giraffe",
            "Vehicles: car, truck, bus, motorcycle, bicycle"
        ]
        
        for obj_type in supported_objects:
            st.markdown(f"• {obj_type}")
        
        # Supported backgrounds
        st.subheader("🌄 Supported Background Types")
        supported_backgrounds = [
            "Nature: mountains, park, beach, forest, farm, sky, grass, field",
            "Urban: city, stadium, gym, building, road",
            "Special: space, masjid, ground"
        ]
        
        for bg_type in supported_backgrounds:
            st.markdown(f"• {bg_type}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🎨 AI Image Swapping System | Powered by Detectron2 & Streamlit</p>
        <p>For best results, ensure MongoDB is running and use clear, well-lit images</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Cleanup
    if db.connected:
        db.close_connection()

if __name__ == "__main__":
    main()