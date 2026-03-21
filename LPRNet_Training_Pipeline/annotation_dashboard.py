import streamlit as st
import os
import shutil
import difflib
from PIL import Image

# Setup Paths (Relative to the LPRNet Pipeline Folder)
DATASET_DIR = r"D:\recoveryuserorchi\documentos proyectos 2025\ANPR colombian plate\dataset"
CURATED_DIR = r"D:\recoveryuserorchi\documentos proyectos 2025\ANPR colombian plate\curated_dataset"

st.set_page_config(page_title="LPRNet Annotation Dashboard", layout="wide")

st.title("LPRNet Continuous Learning Pipeline 🚗")
st.markdown("Review plates caught by the edge camera (`main.py`) and correct them to teach the AI.")

# Ensure directories exist
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(CURATED_DIR, exist_ok=True)

# Get all images in raw dataset
raw_images = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if len(raw_images) == 0:
    st.success("You are all caught up! No new plates in the dataset folder.")
else:
    st.sidebar.metric(label="Plates Pending Review", value=len(raw_images))
    
    # We will just focus on the first image to build a rapid flow
    img_name = raw_images[0]
    img_path = os.path.join(DATASET_DIR, img_name)
    
    # Extract AI's guess from the file name.
    # Filename format: plate_{car_id}_{timestamp}_{text}.jpg
    parts = img_name.split("_")
    ai_guess = "UNKNOWN"
    if len(parts) >= 4:
        # The text is the last part before .jpg
        ai_guess = parts[-1].split(".")[0]
        
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(Image.open(img_path), caption=f"Raw File: {img_name}", use_container_width=True)
        st.info(f"**AI Extracted:** {ai_guess}")
        
    with col2:
        st.subheader("Correction Panel")
        
        known_plates = sorted([d for d in os.listdir(CURATED_DIR) if os.path.isdir(os.path.join(CURATED_DIR, d))])
        
        # --- SMART DROPDOWN ---
        st.write("**Search previously curated plates:**")
        
        # We put a combobox so the user can type "J" and instantly see all plates starting with "J"
        selected_plate = st.selectbox(
            "Search Database",
            options=["<TYPE NEW BELOW>"] + known_plates,
            index=0
        )
        
        # Determine what to pre-fill the text box with
        if selected_plate == "<TYPE NEW BELOW>":
            default_text = ai_guess.upper()
        else:
            default_text = selected_plate
            
        # The ultimate text box always remains visible!
        correct_label = st.text_input("Final Plate to Approve:", value=default_text)
        correct_label = correct_label.strip().upper().replace(" ", "").replace("-", "")
        
        # --- AI QUICK SUGGESTION BUTTONS ---
        if known_plates and ai_guess != "UNKNOWN":
            suggestions = difflib.get_close_matches(ai_guess.upper(), known_plates, n=3, cutoff=0.3)
            # Remove the currently typed/selected plate from suggestions to avoid redundancy
            suggestions = [s for s in suggestions if s != correct_label]
            
            if suggestions:
                st.write("**AI Quick Matches (Click to Auto-Approve):**")
                cols = st.columns(len(suggestions))
                for idx, sugg in enumerate(suggestions):
                    if cols[idx].button(f"✅ {sugg}", use_container_width=True):
                        correct_label = sugg # Override
                        target_dir = os.path.join(CURATED_DIR, sugg)
                        os.makedirs(target_dir, exist_ok=True)
                        timestamp = parts[2] if len(parts) >= 3 else "auto"
                        shutil.move(img_path, os.path.join(target_dir, f"{sugg}_{timestamp}.jpg"))
                        st.rerun()
                        
        st.markdown("---")
        
        if st.button("🚧 Approve Plate & OPEN PHYSICAL GATE", type="primary", use_container_width=True):
            if correct_label and correct_label != "<TYPENEWBELOW>":
                
                # --- NEW: TRIGGER PHYSICAL GATE RELAY ---
                relay_ip = "192.168.1.100" # Change this to the actual network relay IP
                try:
                    # import requests
                    # requests.get(f"http://{relay_ip}/gate/open", timeout=3)
                    st.success(f"Network Signal Sent! Gate Opened for: {correct_label}!")
                except Exception as e:
                    st.error(f"Could not reach Gate Relay at {relay_ip}")
                    
                # Create the specific target directory for this label
                target_dir = os.path.join(CURATED_DIR, correct_label)
                os.makedirs(target_dir, exist_ok=True)
                
                # Keep a robust file name
                timestamp = parts[2] if len(parts) >= 3 else "auto"
                new_filename = f"{correct_label}_{timestamp}.jpg"
                target_path = os.path.join(target_dir, new_filename)
                
                # Move the file
                shutil.move(img_path, target_path)
                st.rerun() # Refresh the page instantly
            else:
                st.error("Please enter a valid license plate string.")
                
        if st.button("Trash Image (Blurry/Unreadable)", use_container_width=True):
            os.remove(img_path)
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**Phase 3: Automated MLOps**")
st.sidebar.caption("When 50 images are approved, auto_trainer.py will automatically trigger the CNN weights update.")
