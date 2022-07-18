import xml.etree.ElementTree as ET
import os
import csv
from os import listdir
from os.path import isfile, join

save_path = '.'

def find_between(s, first, last):
    try:
        start = s.index(first)+len(first)
        end = s.index(last, start)
        return s[start:end]
    except:
        return ""

def hasExtension(file, ext_list):
    for ext in ext_list:
        if file.endswith(ext):
            return True
    return False

def getAllXmlFiles(path):
    path = '/data/AMPs/'+path
    onlyxml = [join(path,f) for f in listdir(path) if isfile(join(path, f)) and hasExtension(join(path, f), ['xml'])]
    return onlyxml

paths_to_load = ['first-round/tagged', 'second-round/tagged']

for path_to_load in paths_to_load:
    files_to_parse = getAllXmlFiles(path_to_load)

    with open(join(save_path, f'tagging_data_{os.path.split(path_to_load)[0]}.csv'), 'w', newline='') as save_file:
        field_names = ['subID','task', 'Video', 'Track', 'Timestamp', 'Attribute', 'Value', 'Comment']
        writer = csv.writer(save_file)
        writer.writerow(field_names)

        for file_to_parse in files_to_parse:
            tree = ET.parse(file_to_parse)
            root = tree.getroot()
            video = root.findall('.//video')
            video_name = 'n/a'
            if video:
                video_path = video[0].attrib['src']
                video_name = os.path.basename(video_path)
                if 'Subject' in video_name:
                    subID = 'P' + find_between(video_name, 'Subject', '_')
                else:
                    subID = find_between(video_name, '', '_')
                task = find_between(video_name, '_', '_')
            tracks = root.findall('.//track')
            for track in tracks:
                track_name = track.attrib['name']
                for el in track:
                    timestamp = el.attrib['time']
                    attributes = el.findall('attribute')
                    comments = el.findall('comment')

                    attrib_val = ''
                    for attribute in attributes:
                        # There is only one in this case - skill:score
                        attribute_name = attribute.attrib['name']
                        attrib_val = attribute.text

                    comment_val = ''
                    for comment in comments:
                        # There is only one in this case
                        comment_val = comment.text

                    writer.writerow([subID, task, video_name, track_name, timestamp, attribute_name, attrib_val, comment_val])