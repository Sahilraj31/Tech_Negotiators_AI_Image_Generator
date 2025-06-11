import streamlit as st
from PIL import Image
import numpy as np
import io
import os
from image_processing import ImageDatabase, display_single_best_result_db, display_all_matching_results_db, parse_simple_prompt

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Image Generator", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎨"
)

# --- Enhanced Custom CSS for Beautiful Design ---
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global font styling */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container styling */
    .main > div {
        padding-top: 1rem;
    }
    
    /* Header styling with enhanced gradient */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 3rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        font-size: 1.3rem;
        opacity: 0.95;
        margin: 0;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    /* Enhanced sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
    }
    
    /* Enhanced feature card styling */
    .feature-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
        border: 1px solid rgba(102, 126, 234, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    }
    
    .feature-card h2, .feature-card h3 {
        color: #2d3748;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    /* Enhanced image container styling with prompt display */
    .image-container {
        background: linear-gradient(145deg, #ffffff 0%, #f7fafc 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        text-align: center;
        margin: 1.5rem 0;
        border: 1px solid rgba(102, 126, 234, 0.05);
        position: relative;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .image-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 45px rgba(0,0,0,0.15);
    }
    
    .image-container img {
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
    }
    
    .image-container img:hover {
        transform: scale(1.02);
    }
    
    /* Prompt label styling */
    .prompt-label {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
        display: inline-block;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        text-transform: capitalize;
        letter-spacing: 0.5px;
    }
    
    /* Enhanced result card for variations */
    .result-card {
        background: linear-gradient(145deg, #ffffff 0%, #f7fafc 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.08);
        text-align: center;
        margin: 1rem 0;
        border: 1px solid rgba(102, 126, 234, 0.05);
        position: relative;
        transition: all 0.3s ease;
        overflow: hidden;
    }
    
    .result-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 45px rgba(0,0,0,0.15);
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    }
    
    .result-card img {
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.12);
        transition: transform 0.3s ease;
        max-height: 350px;
        object-fit: contain;
    }
    
    .result-card img:hover {
        transform: scale(1.03);
    }
    
    .variation-title {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 1rem 0 0.5rem 0;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
    }
    
    .variation-subtitle {
        color: #718096;
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    
    /* Enhanced success/error message styling */
    .stSuccess > div {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        border-radius: 15px;
        border: none;
        padding: 1rem;
        font-weight: 500;
        box-shadow: 0 8px 20px rgba(72, 187, 120, 0.3);
    }
    
    .stError > div {
        background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
        color: white;
        border-radius: 15px;
        border: none;
        padding: 1rem;
        font-weight: 500;
        box-shadow: 0 8px 20px rgba(245, 101, 101, 0.3);
    }
    
    .stWarning > div {
        background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
        color: white;
        border-radius: 15px;
        border: none;
        padding: 1rem;
        font-weight: 500;
        box-shadow: 0 8px 20px rgba(237, 137, 54, 0.3);
    }
    
    .stInfo > div {
        background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
        color: white;
        border-radius: 15px;
        border: none;
        padding: 1rem;
        font-weight: 500;
        box-shadow: 0 8px 20px rgba(66, 153, 225, 0.3);
    }
    
    /* Enhanced button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 1rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Enhanced input styling */
    .stTextInput > div > div > input, .stTextarea > div > div > textarea {
        border-radius: 15px;
        border: 2px solid #e2e8f0;
        padding: 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        background: linear-gradient(145deg, #ffffff 0%, #f7fafc 100%);
    }
    
    .stTextInput > div > div > input:focus, .stTextarea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
        background: white;
    }
    
    /* Enhanced stats cards */
    .stats-card {
        background: linear-gradient(145deg, #ffffff 0%, #f7fafc 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
        margin: 0.5rem;
        border: 1px solid rgba(102, 126, 234, 0.05);
        transition: transform 0.3s ease;
    }
    
    .stats-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.12);
    }
    
    .stats-number {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        display: block;
        margin-bottom: 0.5rem;
    }
    
    .stats-label {
        color: #718096;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    /* Enhanced upload section */
    .upload-section {
        border: 3px dashed #667eea;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        background: linear-gradient(145deg, rgba(102, 126, 234, 0.03) 0%, rgba(118, 75, 162, 0.03) 100%);
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #5a67d8;
        background: linear-gradient(145deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
    }
    
    /* Enhanced navigation styling */
    .nav-item {
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        border-radius: 15px;
        font-weight: 500;
        transition: all 0.3s ease;
        border: 1px solid transparent;
    }
    
    .nav-item:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        color: #667eea;
        border-color: rgba(102, 126, 234, 0.2);
        transform: translateX(5px);
    }
    
    /* Section titles styling */
    .section-title {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        font-size: 1.8rem;
        margin-bottom: 1.5rem;
    }
    
    /* Example buttons styling */
    .example-btn {
        background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e0 100%);
        border: 1px solid #a0aec0;
        border-radius: 25px;
        padding: 0.75rem 1.5rem;
        margin: 0.25rem;
        font-weight: 500;
        transition: all 0.3s ease;
        color: #4a5568;
    }
    
    .example-btn:hover {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .main-header p {
            font-size: 1rem;
        }
        
        .feature-card {
            padding: 1.5rem;
        }
        
        .result-card {
            padding: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Database ---
@st.cache_resource
def init_database():
    return ImageDatabase()

db = init_database()

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>🎨 AI Image Generator</h1>
    <p>Create stunning composites with intelligent object detection and background swapping</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
st.sidebar.markdown("### 🚀 Navigation")
menu = st.sidebar.radio(
    "Choose Your Action",
    [
        "🔍 Best Composite",
        "🖼️ Multiple Composites", 
        "📁 Upload Images",
        "📊 Database Stats"
    ],
    key="main_menu"
)

# --- Helper Functions ---
def display_image_with_container(image, caption="", prompt="", max_width=400):
    """Display image in a styled container with proper sizing and prompt label"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        if prompt:
            st.markdown(f'<div class="prompt-label">"{prompt}"</div>', unsafe_allow_html=True)
        st.image(image, caption=caption, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

def display_variation_card(image, variation_num, subject_file, location_file, prompt=""):
    """Display a variation card with enhanced styling"""
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    if prompt:
        st.markdown(f'<div class="prompt-label">"{prompt}"</div>', unsafe_allow_html=True)
    st.image(image, use_column_width=True)
    st.markdown(f'<div class="variation-title">Variation {variation_num}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="variation-subtitle">{subject_file} + {location_file}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def create_stats_card(number, label):
    """Create a styled statistics card"""
    return f"""
    <div class="stats-card">
        <span class="stats-number">{number}</span>
        <span class="stats-label">{label}</span>
    </div>
    """

# --- Main Content Based on Menu Selection ---

if menu == "🔍 Best Composite":
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">🎯 Find Your Perfect Composite</h3>', unsafe_allow_html=True)
    st.markdown("Enter a simple description and let our AI create the best possible composite for you.")
    
    # Input section
    col1, col2 = st.columns([3, 1])
    with col1:
        prompt = st.text_input(
            "✨ Describe what you want to see",
            placeholder="e.g., 'boy in mountains', 'cow in park', 'knight in masjid'",
            help="Use simple descriptions like 'subject in location'"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        generate_btn = st.button("🚀 Generate", key="generate_best")
    
    if generate_btn and prompt:
        with st.spinner("🎨 Creating your masterpiece..."):
            result = display_single_best_result_db(prompt, db)
            
            if result is not None:
                st.success("✅ Composite created successfully!")
                display_image_with_container(result, f"🎨 Result for '{prompt}'", prompt)
                
                # Additional info
                st.info("💡 **Tip:** Try different combinations like 'dog in beach' or 'elephant in city' for more variations!")
            else:
                st.error("❌ Could not create composite. Try a different prompt or add more images to the database.")
                st.info("💡 **Suggestions:**\n- Make sure you have images in the database\n- Try simpler descriptions\n- Check if your subject and location exist in the database")
    
    elif generate_btn:
        st.warning("⚠️ Please enter a valid prompt to generate your composite.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Example prompts section
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">💡 Example Prompts</h3>', unsafe_allow_html=True)
    
    examples = [
        "🧑 boy in mountains", "🐄 cow in park", "⚔️ knight in masjid",
        "🐕 dog in beach", "🐘 elephant in city", "🐒 monkey in forest"
    ]
    
    cols = st.columns(3)
    for i, example in enumerate(examples):
        with cols[i % 3]:
            if st.button(example, key=f"example_{i}"):
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

elif menu == "🖼️ Multiple Composites":
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">🎭 Compare Multiple Composites</h3>', unsafe_allow_html=True)
    st.markdown("See different variations of your prompt with our top matching combinations.")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        prompt = st.text_input(
            "✨ Enter your creative prompt",
            placeholder="e.g., 'footballer in stadium', 'sheep in field'",
            help="We'll show you the top 2 different combinations"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        show_btn = st.button("🎨 Show Variations", key="show_multiple")
    
    if show_btn and prompt:
        with st.spinner("🎭 Creating multiple variations..."):
            subject, location, _ = parse_simple_prompt(prompt)
            
            if subject and location:
                subject_matches = db.search_images("object", subject, limit=5)
                location_matches = db.search_images("background", location, limit=5)
                
                if subject_matches and location_matches:
                    results = display_all_matching_results_db(prompt, subject_matches, location_matches, db)
                    
                    if results:
                        st.success(f"✅ Generated {len(results)} unique combinations!")
                        
                        # Display results in a grid
                        st.markdown('<h3 class="section-title">🖼️ Your Composite Variations</h3>', unsafe_allow_html=True)
                        
                        if len(results) >= 2:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                display_variation_card(
                                    results[0]['composite'], 
                                    1, 
                                    results[0]['subject_file'], 
                                    results[0]['location_file'],
                                    prompt
                                )
                            
                            with col2:
                                display_variation_card(  
                                    results[1]['composite'], 
                                    2, 
                                    results[1]['subject_file'], 
                                    results[1]['location_file'],
                                    prompt
                                )
                        else:
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                display_variation_card(
                                    results[0]['composite'], 
                                    1, 
                                    results[0]['subject_file'], 
                                    results[0]['location_file'],
                                    prompt
                                )
                    else:
                        st.warning("⚠️ No successful combinations were generated. Try a different prompt.")
                else:
                    st.error("❌ No matching images found. Please add more images to the database.")
            else:
                st.error("❌ Could not parse your prompt. Please use format like 'subject in location'.")
    
    elif show_btn:
        st.warning("⚠️ Please enter a valid prompt to see variations.")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif menu == "📁 Upload Images":
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">📸 Expand Your Image Database</h3>', unsafe_allow_html=True)
    st.markdown("Upload and tag new images to enhance your AI's creative possibilities.")
    
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "🖼️ Select Images", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True,
        help="Upload multiple images at once to build your database"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_files:
        st.markdown(f'<h3 class="section-title">📋 Tagging {len(uploaded_files)} Images</h3>', unsafe_allow_html=True)
        
        for idx, uploaded_file in enumerate(uploaded_files):
            bytes_data = uploaded_file.read()
            filename = uploaded_file.name
            
            # Create a container for each image
            with st.container():
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(bytes_data, caption=f"📷 {filename}", width=250)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"#### 🏷️ Tag: {filename}")
                    
                    # Tagging inputs
                    object_type = st.text_input(
                        "🎯 Object Type", 
                        placeholder="e.g., boy, cow, car, knight",
                        key=f"obj_{idx}_{filename}",
                        help="What is the main subject in this image?"
                    )
                    
                    background_type = st.text_input(
                        "🏞️ Background Type", 
                        placeholder="e.g., mountains, park, masjid, stadium",
                        key=f"bg_{idx}_{filename}",
                        help="What kind of background/location is this?"
                    )
                    
                    description = st.text_area(
                        "📝 Description", 
                        value=f"{object_type} in {background_type}" if object_type and background_type else "",
                        key=f"desc_{idx}_{filename}",
                        help="Additional details about the image"
                    )
                    
                    # Store button
                    if st.button(f"💾 Store {filename}", key=f"btn_{idx}_{filename}"):
                        if object_type and background_type:
                            temp_path = f"{filename}"
                            with open(temp_path, "wb") as f:
                                f.write(bytes_data)
                            
                            try:
                                db.store_image(temp_path, object_type, background_type, description)
                                os.remove(temp_path)
                                st.success(f"✅ Successfully stored {filename}!")
                            except Exception as e:
                                st.error(f"❌ Error storing {filename}: {str(e)}")
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                        else:
                            st.warning("⚠️ Please fill in both Object Type and Background Type.")
                
                st.markdown("---")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif menu == "📊 Database Stats":
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-title">📈 Database Analytics</h3>', unsafe_allow_html=True)
    
    all_images = db.get_all_images_info()
    total_images = len(all_images)
    
    # Main stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(create_stats_card(total_images, "Total Images"), unsafe_allow_html=True)
    
    # Count occurrences
    obj_types = {}
    bg_types = {}
    for img in all_images:
        obj = img.get("object_type", "unknown")
        bg = img.get("background_type", "unknown") 
        obj_types[obj] = obj_types.get(obj, 0) + 1
        bg_types[bg] = bg_types.get(bg, 0) + 1
    
    with col2:
        st.markdown(create_stats_card(len(obj_types), "Object Types"), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_stats_card(len(bg_types), "Background Types"), unsafe_allow_html=True)
    
    if total_images > 0:
        # Detailed breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Object Distribution")
            for obj_type, count in sorted(obj_types.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_images) * 100
                st.markdown(f"**{obj_type.title()}:** {count} ({percentage:.1f}%)")
        
        with col2:
            st.markdown("#### 🏞️ Background Distribution")
            for bg_type, count in sorted(bg_types.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_images) * 100
                st.markdown(f"**{bg_type.title()}:** {count} ({percentage:.1f}%)")
        
        # Recommendations
        st.markdown('<h3 class="section-title">💡 Recommendations</h3>', unsafe_allow_html=True)
        if total_images < 10:
            st.info("📸 **Add More Images:** Upload more images to improve composite quality and variety.")
        
        if len(obj_types) < 5:
            st.info("🎯 **Diversify Objects:** Add more object types like animals, people, or vehicles.")
        
        if len(bg_types) < 5:
            st.info("🏞️ **Expand Backgrounds:** Include more background types like nature, urban, or indoor scenes.")
    
    else:
        st.info("📁 **Empty Database:** Start by uploading some images to see statistics here!")
    
    st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 1rem;">
    <p>🎨 <strong>AI Image Generator</strong> | Built with ❤️ using Streamlit</p>
    <p><small>Upload images, create prompts, and watch the magic happen!</small></p>
</div>
""", unsafe_allow_html=True)
