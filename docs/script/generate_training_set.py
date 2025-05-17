import os
import argparse
import numpy as np
from scipy import ndimage


def get_sub_path(path):
    sub_path = []
    if isinstance(path, list):
        for p in path:
            if os.path.isdir(p):
                for file in os.listdir(p):
                    sub_path.append(os.path.join(p, file))
            else:
                continue
    else:
        for file in os.listdir(path):
            sub_path.append(os.path.join(path, file))
    return sub_path

def resize(input):
    dimension = input.shape
    result = ndimage.zoom(input, (256 / dimension[0], 256 / dimension[1]), order=3)
    return result

def std(input):
    if input.max() == 0:
        return input
    else:
        result = (input-input.min()) / (input.max()-input.min())
        return result

def save_npy(out_list, save_path, name):
    output = np.array(out_list)
    output = np.transpose(output, (1, 2, 0))
    np.save(os.path.join(save_path, name), output)

def pack_data(args, name_list, read_feature_list, read_label_list, save_path):


    for name in name_list:

        for feature_name in read_feature_list:
            out_feature_list = []
            feature_save_path = os.path.join(args.save_path, args.task, name)
            os.system("mkdir -p %s " % (feature_save_path))
            name = os.path.basename(name)
            feature = np.load(os.path.join(args.data_path, feature_name, name))
            if args.task == 'IR_drop':   
                out_feature = feature*2.25
            else:
                raise ValueError('Task not implemented')

        save_npy(out_feature_list, feature_save_path, name)

        out_label_list = []
        for label_name in read_label_list:
            label_save_path = os.path.join(args.save_path, args.task, 'label')
            os.system("mkdir -p %s " % (label_save_path))
            name = os.path.basename(name)
            label = np.load(os.path.join(args.data_path, label_name, name))
            if args.task == 'congestion': 
                label = std(resize(label))
            elif args.task == 'DRC':
                label = np.clip(label, 0, 200)
                label = resize(label)/200
            elif args.task == 'IR_drop':
                label = np.squeeze(label)
                label = np.clip(label, 1e-6, label.max())
                label = resize(np.log10(label)/np.log10(50))
            else:
                raise ValueError('Task not implemented')
            out_label_list.append(label)

        save_npy(out_label_list, label_save_path, name)

def parse_args():
    description = "you should add those parameter" 
    parser = argparse.ArgumentParser(description=description)
                                                             
    parser.add_argument("--task", default = None, type=str, help = 'select from congestion, DRC and IR_drop' )
    parser.add_argument("--data_path", default = '../', type=str, help = 'path to the decompressed dataset')
    parser.add_argument("--save_path",  default = '../training_set', type=str, help = 'path to save training set')

    args = parser.parse_args()                                       
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.task == 'congestion':
        feature_list = ['routability_features_decompressed/macro_region', 'routability_features_decompressed/RUDY/RUDY', 
        'routability_features_decompressed/RUDY/RUDY_pin']
        label_list = ['routability_features_decompressed/congestion/congestion_global_routing/overflow_based/congestion_GR_horizontal_overflow', 
        'routability_features_decompressed/congestion/congestion_global_routing/overflow_based/congestion_GR_vertical_overflow']

    elif args.task == 'DRC':
        feature_list = ['routability_features_decompressed/macro_region', 'routability_features_decompressed/cell_density', 
        'routability_features_decompressed/RUDY/RUDY_long', 'routability_features_decompressed/RUDY/RUDY_short',
        'routability_features_decompressed/RUDY/RUDY_pin_long', 
        'routability_features_decompressed/congestion/congestion_early_global_routing/overflow_based/congestion_eGR_horizontal_overflow', 
        'routability_features_decompressed/congestion/congestion_early_global_routing/overflow_based/congestion_eGR_vertical_overflow', 
        'routability_features_decompressed/congestion/congestion_global_routing/overflow_based/congestion_GR_horizontal_overflow', 
        'routability_features_decompressed/congestion/congestion_global_routing/overflow_based/congestion_GR_vertical_overflow']
        label_list = ['routability_features_decompressed/DRC/DRC_all']

    elif args.task == 'IR_drop':
        feature_list = ['IR_drop_features_decompressed/power_i', 'IR_drop_features_decompressed/power_s', 
        'IR_drop_features_decompressed/power_sca', 'IR_drop_features_decompressed/power_all','IR_drop_features_decompressed/power_t']
        label_list = ['IR_drop_features_decompressed/IR_drop']
    else:
        raise ValueError('Please specify argument --task from congestion, DRC and IR_drop')

    name_list = get_sub_path(os.path.join(args.data_path, feature_list[0]))
    print('processing %s files' % len(name_list))
    save_path = os.path.join(args.save_path, args.task)
    os.system("mkdir -p %s " % (save_path))
    pack_data(args, name_list, feature_list, label_list, save_path)

    





