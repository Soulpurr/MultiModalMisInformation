import streamlit as st
import subprocess
import os
import uuid

st.title("🎭 SimSwap Deepfake Generator")
st.markdown("Upload a **source face image** and a **target video** to create a deepfake.")

uploaded_image = st.file_uploader("Upload Source Image", type=["jpg", "jpeg", "png"])
uploaded_video = st.file_uploader("Upload Target Video", type=["mp4", "avi", "mov"])

# Config
arcface_path = "arcface_model/arcface_checkpoint.tar"
crop_size = 224
name = "people"
temp_path = "./tmp_results"
os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs(temp_path, exist_ok=True)

if st.button("Generate Deepfake"):
    if uploaded_image and uploaded_video:
        # Create unique file names
        unique_id = uuid.uuid4().hex[:8]
        img_ext = os.path.splitext(uploaded_image.name)[-1]
        vid_ext = os.path.splitext(uploaded_video.name)[-1]

        img_filename = f"img_{unique_id}{img_ext}"
        vid_filename = f"vid_{unique_id}{vid_ext}"
        output_filename = f"deepfake_{unique_id}.mp4"

        img_path = os.path.join("input", img_filename)
        vid_path = os.path.join("input", vid_filename)
        output_path = os.path.join("output", output_filename)

        # Save files to disk
        with open(img_path, "wb") as f:
            f.write(uploaded_image.read())
        with open(vid_path, "wb") as f:
            f.write(uploaded_video.read())

        # SimSwap command
        command = [
            "python", "test_video_swapsingle.py",
            "--isTrain", "false",
            "--crop_size", str(crop_size),
            "--name", name,
            "--Arc_path", arcface_path,
            "--pic_a_path", img_path,
            "--video_path", vid_path,
            "--output_path", output_path,
        ]

        # Progress bar for deepfake generation
        progress_bar = st.progress(0)
        with st.spinner("🛠️ Generating deepfake..."):
            # Run the subprocess and capture stdout and stderr
            process = subprocess.Popen(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True, 
                encoding='utf-8'
            )

            # Simulate progress based on output (you can adjust this based on your script's output)
            for stdout_line in iter(process.stdout.readline, ""):
                st.text(stdout_line.strip())  # Display stdout in Streamlit
                
                # Simulate progress update (you may need to modify based on real stdout feedback)
                if "progress" in stdout_line.lower():  # Replace this with actual condition based on stdout
                    progress_bar.progress(50)  # Update progress to 50% for demonstration (replace with actual calculation)
            
            # Handle stderr (error output)
            for stderr_line in iter(process.stderr.readline, ""):
                st.text(stderr_line.strip())  # Display stderr in Streamlit

            process.stdout.close()
            process.stderr.close()
            process.wait()  # Wait for process to finish

        # Check if the deepfake generation was successful
        if os.path.exists(output_path):
            st.success("✅ Deepfake generated successfully!")
            st.video(output_path)
            st.markdown(f"📁 **Saved as:** `{output_filename}`")

            with open(output_path, "rb") as video_file:
                st.download_button(
                    label="⬇️ Download Video",
                    data=video_file,
                    file_name=output_filename,
                    mime="video/mp4"
                )
        else:
            st.error("❌ Error during deepfake generation. No output file found.")
    else:
        st.warning("⚠️ Please upload both a source image and a target video.")
