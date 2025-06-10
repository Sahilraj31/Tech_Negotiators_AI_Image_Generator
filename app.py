import streamlit as st
from PIL import Image
import numpy as np
import io
import os
from image_processing import ImageDatabase, display_single_best_result_db, display_all_matching_results_db, parse_simple_prompt

# --- Initialize ---
st.set_page_config(page_title="AI Image Generator", layout="wide")
db = ImageDatabase()

# --- UI ---
st.title("🎨 AI Image Generator with Database")

menu = st.sidebar.radio("Choose Action", [
    "Search: Best Composite",
    "Search: 2 Composites",
    "Upload Images",
    "Database Stats"
])

# --- 1. Search and Display Best Composite ---
if menu == "Search: Best Composite":
    st.header("🔍 Find Best Composite")
    prompt = st.text_input("Enter prompt (e.g., 'boy in mountains')")

    if st.button("Generate Best Composite"):
        if prompt:
            with st.spinner("Generating composite..."):
                result = display_single_best_result_db(prompt, db)
                if result is not None:
                    st.success("✅ Composite created successfully!")
                    st.image(result, caption=f"Result for '{prompt}'", use_column_width=True)
                else:
                    st.error("❌ Could not create composite. Try different prompt or add more images.")
        else:
            st.warning("Please enter a valid prompt.")

# --- 2. Show Two Composites ---
elif menu == "Search: 2 Composites":
    st.header("🖼️ Show Two Different Composites")
    prompt = st.text_input("Enter prompt (e.g., 'dog in park')")

    if st.button("Show Top 2 Results"):
        if prompt:
            subject, location, _ = parse_simple_prompt(prompt)
            if subject and location:
                subject_matches = db.search_images("object", subject, limit=5)
                location_matches = db.search_images("background", location, limit=5)

                results = display_all_matching_results_db(prompt, subject_matches, location_matches, db)
                if results:
                    st.success(f"✅ Generated {len(results)} combinations!")
                    for res in results:
                        st.image(res['composite'], caption=f"{res['subject_file']} + {res['location_file']}", use_column_width=True)
                else:
                    st.warning("⚠️ No results generated.")
            else:
                st.error("Could not parse prompt. Try a simpler phrase.")
        else:
            st.warning("Please enter a valid prompt.")

# --- 3. Upload Images to Database ---
elif menu == "Upload Images":
    st.header("📁 Upload and Tag Images")
    uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read()
            filename = uploaded_file.name

            st.write(f"🖼️ {filename}")
            st.image(bytes_data, use_column_width=True)

            object_type = st.text_input(f"Object type for {filename}", key=f"obj_{filename}")
            background_type = st.text_input(f"Background type for {filename}", key=f"bg_{filename}")
            description = st.text_area(f"Description for {filename}", value=f"{object_type} in {background_type}", key=f"desc_{filename}")

            if st.button(f"Store {filename}", key=f"btn_{filename}"):
                temp_path = f"{filename}"
                with open(temp_path, "wb") as f:
                    f.write(bytes_data)
                db.store_image(temp_path, object_type, background_type, description)
                os.remove(temp_path)
                st.success(f"✅ Stored {filename} to database")

# --- 4. Show Database Stats ---
elif menu == "Database Stats":
    st.header("📊 Database Statistics")
    all_images = db.get_all_images_info()
    st.write(f"Total images in database: {len(all_images)}")

    # Count occurrences
    obj_types = {}
    bg_types = {}
    for img in all_images:
        obj = img.get("object_type", "unknown")
        bg = img.get("background_type", "unknown")
        obj_types[obj] = obj_types.get(obj, 0) + 1
        bg_types[bg] = bg_types.get(bg, 0) + 1

    st.subheader("Object Types")
    st.json(obj_types)

    st.subheader("Background Types")
    st.json(bg_types)
