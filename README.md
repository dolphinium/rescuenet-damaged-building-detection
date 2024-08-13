# RescueNet: A High Resolution UAV Semantic Segmentation Dataset for Natural Disaster Damage Assessment 
>https://www.kaggle.com/datasets/yaroslavchyrko/rescuenet

# Test the model on HF Spaces working with best weights found:
>https://huggingface.co/spaces/dolphinium/rescuenet-damaged-building-detection

# Check experiment results @cometML platform:
>https://www.comet.com/dolphinium/rescuenet-damaged-building-detection/view/new/panels

# Check documentation pdf: 
> [documentation](https://github.com/dolphinium/rescuenet-damaged-building-detection/blob/main/documentation/documentation.pdf)

# Model comparison table:
![model_comparison_table](figures/model_comparison_table.jpeg)


# Technologies and frameworks used on this project:
* Yolov5-8-10 
* CometML(For monitoring and maintaining models performance)
* Huggingface Spaces(For hosting and deploying models)
* Gradio(For building a web app)
* Folium(For mapping)
* PyEXIFTool(For extracting metadata from drone imagery)


# Known Issues:
* Problem with extracting metadata from images on HF Spaces platform. Image metadata extracting is working fine on local but problematic at host. Default parameter values are used for now. See issue at HF forums at following URL:
https://discuss.huggingface.co/t/image-lost-xmp-data-on-uploads/100954




## TODOS:
* Fine-tuning the model.
* Creating requirements.txt for the project.
* Editing the readme for a better documentation.
* Making the geolocations more precise.
* Cleaning up and creating a new repository for hosting.

