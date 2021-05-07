import argparse
import os
from tqdm import tqdm

import db_matching_images
import frame_matching
import colmap_feature_io


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_file', required=True)
    parser.add_argument('--feature_ext', required=True)
    parser.add_argument('--min_num_matches', type=int, default=15)
    parser.add_argument('--num_points_per_frame', type=int, default=10000)
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--feature_dir', required=True)

    # This argument lets us only look at the matches from a certain folder.
    # We want to avoid adding matches from other folders, e.g. query. This
    # filters images according to the prefix as stored in the db file.
    parser.add_argument('--image_prefix', type=str, default='')
    parser.add_argument('--match_list_path', required=True)
    parser.add_argument('--use_ratio_test', action='store_true')
    parser.add_argument('--ratio_test_values', type=str, default='0.85')
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    ratio_test_values = [float(v) for v in args.ratio_test_values.split(',')]
    print('Ratio test values to use:', ratio_test_values)
    outfiles = [open(args.match_list_path.format(x), 'w+')
                for x in [int(i * 100) for i in ratio_test_values]]

    print('Looking for matching image pairs...')
    matching_image_pairs = db_matching_images.get_matching_images(
        args.database_file, args.min_num_matches, args.image_prefix)
    print('Got', len(matching_image_pairs), 'matching image pairs.')

    feaext = args.feature_ext

    num_missing_images = 0
    num_missing_feas = 0
    missing_images = []
    missing_feas = []
    for (name1, name2) in tqdm(matching_image_pairs, unit='pairs'):
        # Get npz instead of image files.
        fea1 = os.path.join(args.feature_dir, os.path.splitext(name1)[0] + feaext)
        fea1_new = os.path.join(args.feature_dir, name1.strip() + feaext)
        fea2 = os.path.join(args.feature_dir, os.path.splitext(name2)[0] + feaext)
        fea2_new = os.path.join(args.feature_dir, name2.strip() + feaext)
        image1 = os.path.join(args.image_dir, name1)
        image2 = os.path.join(args.image_dir, name2)

        # Some images might be missing, e.g. in the Robotcar case.
        
        if not os.path.isfile(image1):
            if image1 not in missing_images:
                print("Missing ", image1)
                num_missing_images += 1
                missing_images.append(image1)
            continue
        if not os.path.isfile(image2): 
            if image2 not in missing_images:
                print("Missing ", image2)
                num_missing_images += 1
                missing_images.append(image2)
            continue
        if not os.path.isfile(fea1): 
            #print("try to find ", fea1_new)
            if os.path.isfile(fea1_new):
                fea1 = fea1_new 
            else:
                if fea1 not in missing_feas:
                    print("Missing ", fea1)
                    num_missing_feas += 1
                    missing_feas.append(fea1)
                continue
        if not os.path.isfile(fea2): 
            #print("try to find ", fea2_new)
            if os.path.isfile(fea2_new):
                fea2 = fea2_new 
            else:
                if fea2 not in missing_feas:
                    print("Missing ", fea2)
                    num_missing_feas += 1
                    missing_feas.append(fea2)
                continue
        # assert os.path.isfile(fea1), fea1
        # assert os.path.isfile(fea2), fea2

        num_points = args.num_points_per_frame
        matches_for_different_ratios = frame_matching.match_frames_colmap_fea(
            feaext, fea1, fea2, image1, image2, num_points,
            args.use_ratio_test, ratio_test_values, args.debug)

        if(args.use_ratio_test):
            assert len(matches_for_different_ratios) == len(ratio_test_values)

        for i, keypoint_matches in enumerate(matches_for_different_ratios):
            if len(keypoint_matches) > args.min_num_matches:
                outfiles[i].write(name1 + ' ' + name2 + '\n')
                for (match1, match2) in keypoint_matches:
                    outfiles[i].write(str(match1) + ' ' + str(match2) + '\n')
                outfiles[i].write('\n')

    for outfile in outfiles:
        outfile.close()

    print('Missing', num_missing_images, 'images skipped.')
    print('Missing', num_missing_feas, 'feas_file skipped.')

    print("Write to match file: ", args.match_list_path)

if __name__ == '__main__':
    main()
