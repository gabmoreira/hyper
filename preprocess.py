"""
    preprocess.py
    Oct 15 2022
    Gabriel Moreira
"""
import os
import json
import torch

SPLIT           = 'val'
SPLIT_FILENAME  = '../ifetch/deepfashion/in_shop/' + SPLIT + '_listfile.json'
BBOX_FILENAME   = '../ifetch/deepfashion/in_shop/list_bbox_inshop.txt'
OUTPUT_FILENAME = './preprocessed/' + SPLIT + '_split.pt'
    
    
def gen_pseudo_label(base_labels, rule):
    pseudo_labels = []
    for label in base_labels:
        for group in rule.keys():
            if label in group:
                pseudo_labels.append(rule[group])
    return pseudo_labels


rule = {('Denim','Pants','Shorts','Skirts','Leggings') : 'Lower_Body', 
        ('Jackets_Vests','Sweaters','Tees_Tanks','Shirts_Polos','Graphic_Tees','Blouses_Shirts','Sweatshirts_Hoodies','Cardigans','Jackets_Coats') : 'Upper_Body',
        ('Dresses','Rompers_Jumpsuits','Suiting') : 'Full_Body'}


if __name__ == '__main__':
    """
        Creates dictionaries with bounding-boxes and poses for ALL data
        
        Dict-Keys:
            Filenames e.g. 'img/WOMEN/Blouses_Shirts/id_00000001/02_1_front.jpg'
            
        Extracts ALL bounding boxes from the BBOX_FILENAME file
        Extracts ALL poses from BBOX_FILENAME file
        
        Pose codes:
        Front-1, Side-2, Back-3, Full-4, Additional-5
        
        Creates dictionaries:
            bbox_dict = {'img/WOMEN/Blouses_Shirts/id_00000001/02_1_front.jpg': [50, 49, 208, 235], ... }
            pose_dict = {'img/WOMEN/Blouses_Shirts/id_00000001/02_1_front.jpg': 1, ... }
    """
    with open(BBOX_FILENAME) as file:
        lines     = file.readlines()
        lines     = [line.rstrip() for line in lines][2:]
        bbox_dict = {line.split()[0] : [int(i) for i in line.split()[3:]] for line in lines}
        pose_dict = {line.split()[0] : int(line.split()[2]) for line in lines}

        
    """
        Creates dictionary with all data from a split: train, val or test

        Dict-Keys: 
            'path', 'gender', 'cat', 'id', 'filename', 'bbox', 'pose'

        Examples:
            dict['path']     = ['img/WOMEN/Blouses_Shirts/id_00000001/02_1_front.jpg', ...]    
            dict['gender']   = ['WOMEN', 'WOMEN', ... ]
            dict['cat']      = ['Blouses_Shirts', 'Blouses_Shirts', 'Tees_Tanks', ...]
            dict['id']       = ['id_00000001', 'id_00000001', 'id_00000007', ...]
            dict['filename'] = ['02_1_front.jpg', '02_3_back.jpg', ...]
            dict['bbox']     = [[50, 40, 140, 180], [10, 55, 78, 180], ...]
            dict['pose']     = [1, 1, 2, 3, 1, ...]
    """
    with open(SPLIT_FILENAME) as f:
        split_data = json.load(f)['images']
    
    num_items = len(split_data)

    gender    = [item.split('/')[-4] for item in split_data]
    cat       = [item.split('/')[-3] for item in split_data]
    ids       = [item.split('/')[-2] for item in split_data]
    filename  = [item.split('/')[-1] for item in split_data]
    path      = [os.path.join('img', gender[i], cat[i], ids[i], filename[i]) for i in range(num_items)]
    bbox      = [bbox_dict[p] for p in path]
    pose      = [pose_dict[p] for p in path]
    body_part = gen_pseudo_label(cat, rule)

    
    data_dict = {'path'     : path,
                 'gender'   : gender,
                 'cat'      : cat,
                 'id'       : ids,
                 'filename' : filename,
                 'bbox'     : bbox,
                 'pose'     : pose,
                 'body_part': body_part}

    torch.save(data_dict, OUTPUT_FILENAME)
    print('Saved to {}'.format(OUTPUT_FILENAME))

