import xml.etree.ElementTree as ET
import gradio as gr
import PIL.Image as Image
import numpy as np
import cv2
from ultralytics import ASSETS, YOLOv10
from exiftool import ExifToolHelper
from geopy.distance import geodesic
import folium
import base64
import supervision as sv
import os

# Constants for image dimensions
IMAGE_WIDTH = 4000
IMAGE_HEIGHT = 3000


# Load YOLO model
model = YOLOv10("./weights/yolov10m-e100-b16-full-best.pt")

# Define the directory for saving uploaded images
UPLOAD_DIR = 'uploads'  # Or any other directory within your project
os.makedirs(UPLOAD_DIR, exist_ok=True)


# Function to calculate ground distance from pixel distance
def calculate_ground_distance(altitude, fov_deg, image_dimension, pixel_distance):
    fov_rad = np.radians(fov_deg)
    ground_distance = (2 * altitude * np.tan(fov_rad / 2)) * (pixel_distance / image_dimension)
    return ground_distance

# Function to get GPS coordinates from offsets
def get_gps_coordinates(lat, lon, north_offset, east_offset):
    new_location = geodesic(meters=north_offset).destination((lat, lon), 0)
    new_location = geodesic(meters=east_offset).destination(new_location, 90)
    return new_location.latitude, new_location.longitude

def extract_xmp_metadata(xmp_data):
    # Parse the XMP data as an XML tree
    root = ET.fromstring(xmp_data)

    # Define the namespace to use for querying elements
    ns = {
        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
        'drone-dji': 'http://www.dji.com/drone-dji/1.0/'
    }

    # Find the rdf:Description element
    rdf_description = root.find('.//rdf:Description', ns)

    # Extract the desired values
    relative_altitude = float(rdf_description.get('{http://www.dji.com/drone-dji/1.0/}RelativeAltitude', '0'))
    gimbal_yaw_degree = float(rdf_description.get('{http://www.dji.com/drone-dji/1.0/}GimbalYawDegree', '0'))
    gimbal_pitch_degree = float(rdf_description.get('{http://www.dji.com/drone-dji/1.0/}GimbalPitchDegree', '0'))

    return relative_altitude, gimbal_yaw_degree, gimbal_pitch_degree

def save_image_with_metadata(img, img_path):
    # Convert PIL Image to a format that retains EXIF
    img_format = img.format or 'JPEG'
    
    # Save image to a temporary file to preserve metadata
    img.save(img_path, format=img_format)


def predict_image(img, conf_threshold, iou_threshold):
    # Define the file path within the uploads directory
    img_path = os.path.join(UPLOAD_DIR, 'uploaded_image.jpg')

    # Save the image
    save_image_with_metadata(img, img_path)

    # Extract XMP data
    xmp_data = img.info.get("xmp")

    if xmp_data:
        relative_altitude, gimbal_yaw_degree, gimbal_pitch_degree = extract_xmp_metadata(xmp_data)
        # for debugging 
        print("Extracted XMP Metadata:")
        print(f"Relative Altitude: {relative_altitude}")
        print(f"Gimbal Yaw Degree: {gimbal_yaw_degree}")
        print(f"Gimbal Pitch Degree: {gimbal_pitch_degree}")
    else:
        print("XMP data not found in the image.")
    
    # Extract EXIF data
    exif_data = img.info.get("exif")
    try:
        xmp_data = img.info.get("xmp")
        #print(xmp_data)    
    except:
        print("error loading xmp data")    
    #print(exif_data)

    # Save the image with metadata
    if exif_data:
        img.save(img_path, exif=exif_data)  # Save the image with its EXIF data
    else:
        img.save(img_path)  # Save without EXIF data if not available
    
    # Convert PIL Image to OpenCV image
    img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Use ExifTool to extract metadata
    metadata = {}
    tag_list = [
        "Composite:FOV",
        "Composite:GPSLatitude",
        "Composite:GPSLongitude",
        "XMP:AbsoluteAltitude",
        "XMP:RelativeAltitude",
        "XMP:GimbalRollDegree",
        "XMP:GimbalYawDegree",
        "XMP:GimbalPitchDegree"
    ]

    #rel_path = img_path.lstrip("./")
    #print(rel_path)
    with ExifToolHelper() as et:
        for d in et.get_metadata(img_path):
            metadata.update({k: v for k, v in d.items() if k in tag_list})

    # Extract necessary metadata
    CAMERA_GPS = (metadata["Composite:GPSLatitude"], metadata["Composite:GPSLongitude"])
    RELATIVE_ALTITUDE = float(relative_altitude)
    GIMBAL_YAW_DEGREE = float(gimbal_yaw_degree)
    FOV_HORIZONTAL = float(metadata["Composite:FOV"])
    FOV_VERTICAL = FOV_HORIZONTAL * (IMAGE_HEIGHT / IMAGE_WIDTH)
    #GIMBAL_PITCH_DEGREE = float(gimbal_pitch_degree)
    
    # Convert degrees to radians
    yaw_rad = np.radians(GIMBAL_YAW_DEGREE)
    #pitch_rad = np.radians(GIMBAL_PITCH_DEGREE)

    # Perform prediction
    results = model.predict(
        source=img_cv2,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640,
    )

    detections = sv.Detections.from_ultralytics(results[0])
    
    # Annotate and display image
    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])

    # Process detections and calculate GPS coordinates
    building_locations = []
    for i, box in enumerate(detections.xyxy):  # Correct way to iterate through boxes
        # Extract bounding box coordinates and class
        #print(box)
        x_min, y_min, x_max, y_max = box  # Access the first (and only) box
        class_id = int(detections.class_id[i])  # Get class ID as an integer

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        pixel_distance_x = x_center - IMAGE_WIDTH / 2
        pixel_distance_y = IMAGE_HEIGHT / 2 - y_center

        ground_distance_x = calculate_ground_distance(RELATIVE_ALTITUDE, FOV_HORIZONTAL, IMAGE_WIDTH, pixel_distance_x)
        ground_distance_y = calculate_ground_distance(RELATIVE_ALTITUDE, FOV_VERTICAL, IMAGE_HEIGHT, pixel_distance_y)

        east_offset = ground_distance_x * np.cos(yaw_rad) - ground_distance_y * np.sin(yaw_rad)
        north_offset = ground_distance_x * np.sin(yaw_rad) + ground_distance_y * np.cos(yaw_rad)

        building_lat, building_lon = get_gps_coordinates(CAMERA_GPS[0], CAMERA_GPS[1], north_offset, east_offset)
        building_locations.append((building_lat, building_lon, class_id))

    # Create a Folium map centered at the camera's GPS position
    map_center = CAMERA_GPS
    m = folium.Map(
        location=map_center,
        zoom_start=18,
        tiles='Esri.WorldImagery'
    )

    # Add markers for each detected building
    for i, (building_lat, building_lon, class_id) in enumerate(building_locations):
        building_status = 'Damaged' if class_id == 1 else 'Undamaged'

        folium.Marker(
            location=(building_lat, building_lon),
            popup=f'Building {i+1}: {building_status}',
            icon=folium.Icon(color='red' if class_id == 1 else 'green', icon='home')
        ).add_to(m)

    # Save map to HTML and convert to display in Gradio
    m.save('temp_map.html')
    with open('temp_map.html', 'r') as f:
        folium_map_html = f.read()
    
    encoded_html = base64.b64encode(folium_map_html.encode()).decode('utf-8')
    data_url = f"data:text/html;base64,{encoded_html}"

    return im, f'<iframe src="{data_url}" width="100%" height="600" style="border:none;"></iframe>'

logo_url = "https://www.bewelltech.com.tr/_app/immutable/assets/bewell_logo.fda8f209.png"

description_with_logo = """
<img src="https://www.bewelltech.com.tr/_app/immutable/assets/bewell_logo.fda8f209.png" alt="Logo" style="width: 150px; margin-bottom: 10px;">
<p>Upload images for inference and view detected building locations on the map.</p>
"""

# Gradio Interface
iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
    ],
    outputs=[
        gr.Image(type="pil", label="Annotated Image"),
        gr.HTML(label="Map"),
    ],
    title="Custom trained Yolov10 Model on Rescuenet Dataset",
    description=description_with_logo,
)

if __name__ == "__main__":
    iface.launch()