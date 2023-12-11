import os


from libs.telemetry.libs.utils import read_json,extract_frames
from libs.telemetry.libs.telemetry import get_gps, save_gps_traj, save_gps_traj_density, save_frame_gps_traj
from libs.semantic import semantic_segmentation

from geographiclib.geodesic import Geodesic
import sys
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf, open_dict

# New
from libs.vegetation.libs.get_density_class import get_density
from libs.vegetation.libs.extract_vegetation import extract_vegetation
from libs.depth.libs.get_depth_density_class import get_depth_density

import cv2

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):

    geod = Geodesic.WGS84  # define the WGS84 ellipsoid

    video_name = cfg.video
    video_path = os.path.join("./src/", video_name)
    path_video_json = os.path.join(video_path, video_name + '-full-telemetry.json')
    path_video_avi = os.path.join(video_path, video_name + '.MP4')



    # telemetry FLAGS
    flag_extract_frames = False
    flag_save_frames_with_trajectory = False

    # semantic segmentation FLAGS
    flag_sem_run = False
    flag_save_sem_img = False
    flag_save_csv_classes = False
    flag_save_crops = False

    # Extract both interpolated telemetry and original as pandas dataframes
    #telemetry_interp, telemetry = get_gps(geod, path_video_json, path_video_avi)
    #telemetry_interp.to_csv(os.path.join(video_path, video_name + '-frame_gps_interp.csv'), index=False)

    # Save figures with the interpolated trajectory
    #save_gps_traj(telemetry, telemetry_interp, video_path, video_name)





    if flag_extract_frames == True:
        extract_frames(video_path, video_name)

    #Save figures with the RGB frame (left) and the trajectory (right)
    if flag_save_frames_with_trajectory == True:
        save_frame_gps_traj(telemetry_interp, video_path, video_name)

    crops = []
    if flag_sem_run:
        #Extract semantic segmentation
        semantic_segmentation.extract_semantic_segmentation(cfg,flag_save_sem_img,flag_save_csv_classes)

        #Save figure of the trajectory with the semantic segmentation of vegetation class
        semantic_csv_path = os.path.join(video_path, "semantic_classes.csv")
        semantic_segmentation.plot_trajectory_semantic(semantic_csv_path, telemetry_interp, video_path)

        path_sem_img = os.path.join(video_path, "img_semantic")
        crops = semantic_segmentation.extract_crops(path_sem_img, video_path,flag_save_crops)


    # vegetation FLAGS
    flag_extract_vegetation = True
    if flag_extract_vegetation == True:
        rank = 0
        data_dir = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/src/GX010052/img'
        out_dir = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/GX010052_vegetation'
        save_img = True
        world_size = 1
        sam = True
        ckpt_path = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/libs/vegetation/ssa/ckp/sam_vit_h_4b8939.pth'
        light_mode = True
        desired_classes = ["potted plant", "flower", "tree-merged", "grass-merged", "tree", "grass", "plant", "field",
                       "flower", "palm"]
        extract_vegetation(rank, data_dir, out_dir, save_img, world_size, sam, ckpt_path, light_mode, desired_classes)

    directory_path = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/GX010042_vegetation/'
    output_path = 'outputs/densities/GX010042/results_GX010042.csv'
    density = False
    if density == True:
        get_density(directory_path, output_path)


    save_density_map = False
    output_path_map = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/maps/video1_1/'
    densities_file_path = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/densities/video1/results.csv'

    if save_density_map == True:
        telemetry_interp, telemetry = get_gps(geod, path_video_json, path_video_avi)
        telemetry_interp.to_csv(os.path.join(video_path, video_name + '-frame_gps_interp.csv'), index=False)


        # Save figures with the interpolated trajectory
        save_gps_traj_density(telemetry, telemetry_interp, video_path, video_name, densities_file_path, output_path_map)

    segmentation_directory_path = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/video1_vegetation/'
    depth_directory_path = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/depth/video1/'
    density_segmentation_depth_output_path = 'outputs/depth_densities/video1/results_video1.csv'
    image_segmentation_depth_output_path = '/home/rlg/Desktop/rhome/rlg/PROYECTO/SEGA-CV/SEGA-CV/outputs/depth_segmentation_images/video1/'

    depth_density = False
    if depth_density == True:
        get_depth_density(segmentation_directory_path, depth_directory_path, density_segmentation_depth_output_path, image_segmentation_depth_output_path)
    #Devolver vector de indices, y crear mapa con cada una de las clases y total
    
    #TODO: crops_with_class = CLASSIFY_CROPS (crops)





if __name__ == "__main__":
    main()
