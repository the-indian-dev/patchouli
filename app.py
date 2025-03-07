import os
import streamlit as st
import matplotlib.pyplot as plt
import tempfile
from lib import image_to_surf_2d
from OldLib import image_to_surf_2d as image_to_surf_2d_old
from PIL import Image
import base64
import re
import streamlit_analytics

st.set_page_config(
    page_title="Project Patchouli",
    page_icon="🌿",
    layout="wide"
)

code = """
<!-- Clarity Analytics -->
<script type="text/javascript">
    (function(c,l,a,r,i,t,y){
        c[a]=c[a]||function(){(c[a].q=c[a].q||[]).push(arguments)};
        t=l.createElement(r);t.async=1;t.src="https://www.clarity.ms/tag/"+i;
        y=l.getElementsByTagName(r)[0];y.parentNode.insertBefore(t,y);
    })(window, document, "clarity", "script", "qjj17mslbi");
</script>
<!-- SEO Meta Tags -->
<meta name="description" content="Project Patchouli - Convert images to YSFlight SURF files. Licensed under GPLv3.">
<meta name="keywords" content="YSFlight, SURF files, png2srf, repainting, decals, Project Patchouli, Ritabrata Das">
<meta name="author" content="Ritabrata Das">
<meta name="robots" content="index, follow">
<meta property="og:title" content="Project Patchouli - Image to YSFlight SURF Converter">
<meta property="og:description" content="Allows conversion of images into YSFlight surf format for the purpose of repaint and decals.">
<meta property="og:type" content="website">
<meta property="og:image" content="app/static/patchouli.gif">
"""

# Read environment variable for analytics password
analytics_password = st.secrets["A_PASSWORD"]

streamlit_analytics.start_tracking()

a=os.path.dirname(st.__file__)+'/static/index.html'
with open(a, 'r') as f:
    data=f.read()
    if len(re.findall('UA-', data))==0:
        with open(a, 'w') as ff:
            newdata=re.sub('<head>','<head>'+code,data)
            ff.write(newdata)

# Custom CSS for styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .download-link {
        text-decoration: none;
        color: white;
        background-color: #ff4b4b;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
        display: inline-block;
        margin-top: 20px;
    }
    .download-link:hover {
        background-color: #ff2b2b;
    }
</style>
""", unsafe_allow_html=True)

def process_image(image_file, color_levels, epsilon, countour_area, legacy, zFight, zIncrement):
    # Create temporary files for processing
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_input:
        temp_input.write(image_file.getbuffer())
        image_path = temp_input.name

    with tempfile.NamedTemporaryFile(prefix='patchouli_', suffix='.srf', delete=False) as temp_output:
        output_path = temp_output.name

    try:
        area = int(countour_area)
    except Exception as e:
        st.error(f"Invalid contour area: {str(e)}")
        return None, None, None, None

    try:
        zIncrement = float(zIncrement)
    except Exception as e:
        st.error(f"Invalid zIncrement value: {str(e)}")
        return None, None, None, None

    # Process the image
    try:
        if not legacy:
            original, reduced, contours, face_count = image_to_surf_2d(
                image_path,
                output_path,
                color_levels=color_levels,
                epsilon=epsilon,
                min_area=int(countour_area),
                zFight=zFight,
                zIncrement=zIncrement
            )
        else:
            original, reduced, contours, face_count = image_to_surf_2d_old(
                image_path,
                output_path,
                color_levels=color_levels,
                epsilon=epsilon,
                min_area=int(countour_area)
            )

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].set_title("Original Image")
        axes[0].imshow(original)
        axes[0].axis('off')

        axes[1].set_title(f"Reduced Colors ({color_levels})")
        axes[1].imshow(reduced)
        axes[1].axis('off')

        axes[2].set_title("Final Result")
        axes[2].imshow(contours)
        axes[2].axis('off')

        plt.tight_layout()

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_buf:
            plt.savefig(temp_buf, format='png', dpi=150)
            plt.close()
            temp_plot_path = temp_buf.name

        # Read SURF file content
        with open(output_path, 'r') as f:
            surf_content = f.read()

        # Read file for download
        with open(output_path, 'rb') as f:
            surf_data = f.read()

        # Clean up temporary files
        os.unlink(image_path)

        return temp_plot_path, surf_content, surf_data, face_count

    except Exception as e:
        # Clean up temporary files in case of error
        if os.path.exists(image_path):
            os.unlink(image_path)
        if os.path.exists(output_path):
            os.unlink(output_path)
        st.error(f"Error processing image: {str(e)}")
        return None, None, None, None

# Header section
col1, col2 = st.columns([5, 1])
with col1:
    st.title("Project Patchouli")
    st.subheader("Image to YSFlight SURF Converter")
    st.markdown("Made with ❤️ by [Ritabrata Das](https://theindiandev.in)")
    st.markdown("Licensed under GNU GPL v3.0")
with col2:
    try:
        st.image("assets/patchouli.gif", width=100)
    except:
        st.write("Patchouli image not found")

# Main content layout
left_col, right_col = st.columns([1, 1])

with left_col:
    st.header("Input")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    st.subheader("Parameters")
    st.write("Please consider tweaking with every option to get your desired result. Switching between legacy and new algorithm can produce drastically different results.")

    color_levels = st.slider(
        "Color Levels",
        min_value=2,
        max_value=32,
        value=8,
        step=1,
        help="Number of color clusters to detect"
    )

    epsilon = st.slider(
        "Simplification",
        min_value=0.1,
        max_value=10.0,
        value=2.0,
        step=0.1,
        help="Higher values produce fewer vertices"
    )

    countour_area = st.text_input(
        "Minimum Face Area (in square pixel)",
        value="50",
        help="Minimum area of a face to be considered, useful for reducing noise."
    )

    col1, col2 = st.columns(2)
    with col1:
        legacy = st.checkbox(
            "Use Legacy Algorithm",
            value=False,
            help="Use the old algorithm? Better in case of logos without text and non gradient images."
        )

    with col2:
        zFight = st.checkbox(
            "Fix Z-Fighting?(New Algorithm)",
            value=True,
            help="Tries to fix Z Fighting observed in the model. ONLY FOR NEW ALGORITHM!!!"
        )

    zIncrement = st.text_input(
        "Z-Increment",
        value="0.01",
        help="Increment value for z axis. This is separation in meters between each face to fix Z Fighting."
    )

    process_btn = st.button("Process Image", type="primary")

# Results section
with right_col:
    st.header("Results")

    if uploaded_file is not None and process_btn:
        with st.spinner("Processing image..."):
            result_img_path, surf_content, surf_data, face_count = process_image(
                uploaded_file,
                color_levels,
                epsilon,
                countour_area,
                legacy,
                zFight,
                zIncrement
            )

            if result_img_path:
                # Display results
                st.image(result_img_path)

                # Face count
                st.text_input(
                    "Face Count (for SRF), Count after importing into Blender may differ!",
                    value=str(face_count),
                    disabled=True
                )

                # SURF File content with expander
                with st.expander("SURF File Preview"):
                    st.text_area("SURF File Content", value=surf_content, height=300, disabled=True)

                # File download
                if surf_data:
                    # Create a download button for the SURF file
                    st.download_button(
                        label="Download SURF File",
                        data=surf_data,
                        file_name="patchouli_output.srf",
                        mime="application/octet-stream",
                    )

                # Clean up the temporary result image
                os.unlink(result_img_path)
    else:
        st.info("Upload an image and click 'Process Image' to see results")

streamlit_analytics.stop_tracking(unsafe_password=analytics_password)
