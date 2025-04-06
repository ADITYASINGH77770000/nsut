import streamlit as st
import cv2
import torch
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import numpy as np
import tempfile
import os
from PIL import Image

# Load YOLOv5 Model (Pre-trained)
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s')

model = load_model()

# Function to send email notification
def send_email(subject, body, recipient_email):
    sender_email = "ar4564@srmist.edu.in"  # Replace with your email
    sender_password = "6MonkeysRLooking^"  # Use an app password for Gmail
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Set up the server and send the email
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Email sending failed: {e}")
        return False

# Function to process video frames, display them, and detect objects
def process_video(lost_item_desc, video_file, recipient_email):
    # Create a placeholder for displaying the video
    video_placeholder = st.empty()
    
    # Create a status text placeholder
    status_text = st.empty()
    
    # Create a temporary file for the video
    temp_video_path = os.path.join(tempfile.gettempdir(), video_file.name)
    
    # Save the uploaded video to a temporary file
    with open(temp_video_path, "wb") as f:
        f.write(video_file.getbuffer())
    
    # Open the saved video file with OpenCV
    cap = cv2.VideoCapture(temp_video_path)
    
    if not cap.isOpened():
        st.error(f"Error: Could not open video {video_file.name}.")
        return
    
    # Initialize variables
    frame_count = 0
    item_found = False
    matched_labels = []
    
    status_text.text("Processing video... Please wait.")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create a progress bar
    progress_bar = st.progress(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update progress bar
        frame_count += 1
        progress = int(frame_count / total_frames * 100)
        progress_bar.progress(progress)
        
        # Perform object detection on the frame
        results = model(frame)
        detections = results.xywh[0]  # Get the detection results
        
        detected_labels = []
        detected_objects_found = False
        
        for detection in detections:
            x_center, y_center, w, h, confidence, class_id = detection[:6]
            if confidence > 0.4:  # Confidence threshold
                label = model.names[int(class_id)]
                detected_labels.append(label)
                
                # Check if the detected object matches the lost item description
                if label.lower() in lost_item_desc.lower():
                    detected_objects_found = True
                    matched_labels.append(label)
                    
                    # Draw RED bounding box for matched items
                    color = (0, 0, 255)  # Red color for matched items
                else:
                    # Draw GREEN bounding box for other detected items
                    color = (0, 255, 0)
                
                # Draw bounding box and label on the frame
                start_point = (int(x_center - w/2), int(y_center - h/2))
                end_point = (int(x_center + w/2), int(y_center + h/2))
                cv2.rectangle(frame, start_point, end_point, color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", 
                           (start_point[0], start_point[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Convert the frame from BGR to RGB (for displaying in Streamlit)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame with bounding boxes in Streamlit
        video_placeholder.image(frame_rgb, caption=f"Frame {frame_count}/{total_frames}", use_column_width=True)
        
        # If matching object is found, send email and notify user
        if detected_objects_found and not item_found:
            item_found = True
            status_text.success("🎉 LOST ITEM FOUND! Sending email notification...")
            
            # Send email
            email_sent = send_email(
                subject="Lost Item Found!",
                body=f"Your lost item ({', '.join(set(matched_labels))}) has been found in the CCTV footage.",
                recipient_email=recipient_email
            )
            
            if email_sent:
                status_text.success(f"✅ Email notification sent to {recipient_email} about finding {', '.join(set(matched_labels))}.")
            else:
                status_text.error("❌ Failed to send email notification. Please check your email settings.")
        
        # To simulate real-time processing but not too fast
        # Adjust the sleep time based on video fps for a more natural display
        import time
        time.sleep(1/fps)  # Sleep to simulate real-time playback
    
    # Release video capture and delete temporary file
    cap.release()
    os.remove(temp_video_path)
    
    if not item_found:
        status_text.warning("⚠️ Lost item not found in the video footage.")
    
    return item_found, matched_labels

# Streamlit User Interface
st.title("Digital Lost & Found System")
st.markdown("### Upload your lost item details and CCTV footage to find it")

# Create tabs for better organization
tab1, tab2 = st.tabs(["Lost Item Details", "Results"])

with tab1:
    # User input fields
    lost_item_desc = st.text_input("Describe your lost item (e.g., backpack, phone, laptop):")
    st.caption("Tip: Be specific about your item. Example: 'black leather backpack', 'red smartphone'")
    
    lost_item_image = st.file_uploader("Upload an image of your lost item (optional)", type=["jpg", "png", "jpeg"])
    
    # Display the uploaded image if available
    if lost_item_image:
        st.image(Image.open(lost_item_image), caption="Uploaded Image", use_column_width=True)
    
    cctv_video = st.file_uploader("Upload CCTV video footage", type=["mp4", "avi", "mov"])
    
    recipient_email = st.text_input("Enter your email address for notifications:")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        start_button = st.button("Start Detection", type="primary")
    
    with col2:
        if not lost_item_desc or not cctv_video or not recipient_email:
            st.warning("Please provide all required inputs before starting detection.")

with tab2:
    if start_button:
        if not lost_item_desc or not cctv_video or not recipient_email:
            st.error("Please provide all required inputs: Lost Item Description, CCTV Video, and Email Address.")
        else:
            st.subheader("Detection Results")
            
            # Process the video and detect objects
            item_found, matched_labels = process_video(lost_item_desc, cctv_video, recipient_email)
            
            if item_found:
                st.success(f"Your lost item ({', '.join(set(matched_labels))}) was found! An email notification has been sent.")
            else:
                st.info("Processing complete. No matching items were found in the footage.")
                st.markdown("**Suggestions:**")
                st.markdown("- Try uploading a different video footage")
                st.markdown("- Make your item description more general (e.g., use 'bag' instead of 'backpack')")
                st.markdown("- Check if your item might be visible from a different camera angle")