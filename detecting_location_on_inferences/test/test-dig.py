import gradio as gr
import PIL.Image as Image
from libxmp.utils import file_to_dict


def predict_image(img):
    xmp_data = img.info.get("xmp")
    print(xmp_data)
    
    
    return 0

def read_xmp_data(image_path: Path):
    xmp_dict = file_to_dict(str(image_path))
    exif_dict = {}
    dji_data = {}

    # debug printout - helped me to find tag keywords in 'purl.org' 
    print(k)         

    for k in xmp_dict.keys():
        if 'drone-dji' in k:
            for element in xmp_dict[k]:
                dji_data[element[0].replace('drone-dji:', '')] = element[1]
        if 'exif' in k:
            for element in xmp_dict[k]:
                exif_dict[element[0].replace('exif:', '')] = element[1]
    return dji_data, exif_dict

read_xmp_data("image.jpg")



iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
    ],
    outputs=gr.Image(type="pil", label="Result"),
    title="Ultralytics Gradio",
    description="Upload images for inference. The Ultralytics YOLOv8n model is used by default.",
)

if __name__ == "__main__":
    iface.launch()

