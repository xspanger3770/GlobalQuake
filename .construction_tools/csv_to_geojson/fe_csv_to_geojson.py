import argparse,os,csv,json

"""
"id","num","place","shape"
3,3,"Bering Sea","{""type"":""MultiPolygon"",""coordinates"":[[[[-172,64],[-172,63],[-171,63],[-170,63],[-170,62],[-169,62],[-169,63],[-168,63],[-167,63],[-166,63],[-166,62],[-167,62],[-167,61],[-168,61],[-168,60],[-168,59],[-167,59],[-166,59],[-165,59],[-164,59],[-163,59],[-163,58],[-163,57],[-163,56],[-164,56],[-165,56],[-166,56],[-167,56],[-167,57],[-167,58],[-168,58],[-169,58],[-170,58],[-171,58],[-172,58],[-172,57],[-172,56],[-171,56],[-170,56],[-169,56],[-169,55],[-170,55],[-171,55],[-172,55],[-172,54],[-173,54],[-174,54],[-175,54],[-176,54],[-177,54],[-178,54],[-179,54],[-180,54],[-180,55],[-180,56],[-180,57],[-180,58],[-180,59],[-180,60],[-180,61],[-180,62],[-180,63],[-180,64],[-179,64],[-178,64],[-177,64],[-176,64],[-175,64],[-174,64],[-173,64],[-172,64]]],[[[180,57],[180,56],[180,55],[180,54],[179,54],[178,54],[177,54],[176,54],[175,54],[175,55],[174,55],[173,55],[172,55],[171,55],[170,55],[170,56],[170,57],[169,57],[168,57],[167,57],[166,57],[165,57],[164,57],[164,58],[165,58],[165,59],[166,59],[167,59],[168,59],[169,59],[170,59],[171,59],[171,60],[172,60],[173,60],[173,61],[174,61],[175,61],[176,61],[176,62],[177,62],[178,62],[179,62],[180,62],[180,61],[180,60],[180,59],[180,58],[180,57]]]]}"
5,5,"Near Islands, Aleutian Islands, Alaska","{""type"":""Polygon"",""coordinates"":[[[171,51],[170,51],[170,52],[170,53],[170,54],[170,55],[171,55],[172,55],[173,55],[174,55],[175,55],[175,54],[175,53],[175,52],[175,51],[174,51],[173,51],[172,51],[171,51]]]}"
6,6,"Rat Islands, Aleutian Islands, Alaska","{""type"":""Polygon"",""coordinates"":[[[176,50],[175,50],[175,51],[175,52],[175,53],[175,54],[176,54],[177,54],[178,54],[179,54],[180,54],[180,53],[180,52],[180,51],[180,50],[179,50],[178,50],[177,50],[176,50]]]}"
"""

def parse_csv(input_file):
    #Open the CSV file
    with open(input_file, 'r') as csv_file:
        #Read the CSV file
        csv_reader = csv.DictReader(csv_file)
        #Create a list of dictionaries
        csv_list = []
        #Loop through each row in the CSV file
        for row in csv_reader:
            #Add the row to the list
            csv_list.append(row)
        #Return the list
        return csv_list

def convert_to_geojson(csv_list):
    #Create a GeoJSON dictionary
    geojson_dict = {"type": "GeometryCollection", "geometries": []}

    for geometry in csv_list:
        #Convert the geometry to a dictionary
        geometry_dict = json.loads(geometry['shape'])
        properties_dict = {'id': geometry['id'], 'num': geometry['num'], 'place': geometry['place']}

        geometry_dict['properties'] = properties_dict
        geojson_dict['geometries'].append(geometry_dict)

    return geojson_dict


def main(input_file, output_file):
    #Check if file exists
    if not os.path.isfile(input_file):
        print('File does not exist')
        return
    
    parsed_csv = parse_csv(input_file)
    geojson_dict = convert_to_geojson(parsed_csv)

    #Write the GeoJSON file
    with open(output_file, 'w+') as geojson_file:
        geojson_file.write(json.dumps(geojson_dict, indent=2))



def config_argparse():
    parser = argparse.ArgumentParser(description='Converts Flinn-Engdahl data from sciencebase.gov in CSV format to GeoJSON')
    parser.add_argument('input_file', help='Path to CSV file')
    parser.add_argument('output_file', help='Path to output GeoJSON file')
    return parser


if __name__ == '__main__':
    #Initialize argparse
    parser = config_argparse()
    #Access a dictionary of the arguments passed to the script
    args = parser.parse_args()
    #Call the main function with the given file
    main(args.input_file, args.output_file)