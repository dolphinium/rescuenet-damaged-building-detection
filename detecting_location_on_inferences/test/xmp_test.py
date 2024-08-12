from libxmp.utils import file_to_dict

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

read_xmp_data("15054.jpg")
