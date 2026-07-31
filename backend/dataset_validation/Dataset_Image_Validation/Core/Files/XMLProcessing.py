import os
import xml.etree.ElementTree as ET

def extract_sar_product_from_file(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        sar_product = root.find('.//SARProduct')
        return sar_product.text if sar_product is not None else None
    except (ET.ParseError, FileNotFoundError) as e:
        print(f"Error processing '{file_path}': {e}")
        return None

def collect_all_sar_products(directory_path):
    sar_products = []
    if directory_path.endswith("\\"):
        directory_path = directory_path[:-1]

    print(f"\t\t\t{directory_path}")

    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.xml'):
            file_path = os.path.join(directory_path, filename)
            result = extract_sar_product_from_file(file_path)
            if result:
                sar_products.append(result)
    return sar_products

# Example usage:
def fcn_XMLextract(XMLpath):
    all_sar_products = collect_all_sar_products(XMLpath)
    #print("SAR Products found:")
    #for product in all_sar_products:
    #    print(product)
    return all_sar_products