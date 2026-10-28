# MoDec-GS on Neural3DV (coffee_martini) for the S4GS comparison (docs/research-log.md 2026-10-28: beat MoDec-GS and ATGS on the
# same machine and protocol). Used by scripts/compare/run_modecgs.sh:
#   python train.py -s <converted scene> -m <model> --configs arguments/s4gs/neural3dv.py
# The scene is converted to the PanopticSports layout by scripts/compare/convert_hifi4g_panoptic.py --layout neural3dv and read by the
# same S4GS-patched PanopticSports reader as HiFi4G (branch s4gs); this branch (n3dv) needs no reader change.
#
# Base: arguments/s4gs/hifi4g.py, i.e. config_1.0.py of the authors' DyNeRF runs, iterations 20000 fine + 3000 coarse; every
# hyper-parameter below is identical to hifi4g.py (same iteration counts, same MODECGS_ITERATIONS / MODECGS_COARSE_ITERATIONS
# overrides, same upstream "anchor_dyamics_global_type" typo kept). No hyper-parameter was tuned. The differences are all in the data
# (the converted scene, not this file), because the config has no data keys:
#   - training and test images are the undistorted 1340x1003 images of our loader (scene/neural3DV_dataset.py), not resized
#     (hifi4g: training images resized to 969x541, test images at each camera's own size); K is the COLMAP K of that size;
#   - rasterizer background black ("bg" [0, 0, 0] in the meta files; hifi4g: white), the GT of Neural3DV is a real scene;
#     white_background stays False (the reader takes the background from the meta files);
#   - test camera: cam00 only (utils/camera_split.py), 17 training cameras x 300 frames; hifi4g: every 8th of 81 cameras;
#   - initial cloud: points3D.bin of frame000000 as it is, 3733 points (hifi4g: cropped to radius 2.5 by run_modecgs.sh); voxel_size
#     0.01 as upstream. The COLMAP units of coffee_martini are large (bounding box about 58 x 77 x 162), so the anchor count after
#     voxelisation is the number of points; MoDec-GS grows anchors during training (densify_grad_threshold unchanged).
import os

ModelParams = dict(
    n_offsets = 10,
    voxel_size = 0.01,
    feat_dim = 32,

    # anchor dynamics
    anchor_dynamics = True,
    anchor_dynamics_share = False,

    # TIA
    TIA_num_segments = 2,
    TIA_step_size = 0.01,
    TIA_threshold = 1.5,
)

ModelHiddenParams = dict(
    # global kplanes
    kplanes_config_global = {
        'grid_dimensions': 2,
        'input_coordinate_dim': 4,
        'output_coordinate_dim': 16,
        'resolution': [32, 32, 32, 10]
    },
    multires_global = [1, 2],
    # local kplanes
    kplanes_config_local = {
        'grid_dimensions': 2,
        'input_coordinate_dim': 4,
        'output_coordinate_dim': 16,
        'resolution': [64, 64, 64, 80]
    },
    multires_local = [1, 2, 4],

    anchor_deform = True,
    local_context_feature_deform = True,
    grid_offsets_deform = True,
    grid_scale_deform = True,

    # regularizer for global kplanes
    plane_tv_weight_global = 0.0001,
    time_smoothness_weight_global = 0.0005,
    l1_time_planes_global = 0.00005,
    # regularizer for local kplanes
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes = 0.0001,

    render_process = True,

    # dynamics
    anchor_dynamics_position = True,  # global dynamics
    anchor_dyamics_global_type = "mul",  # typo upstream: no effect, see above

    anchor_dynamics_local_context_feature = True,  # local dynamics
    anchor_dynamics_offset = True,
    anchor_dynamics_scaling = True,
    anchor_dynamics_local_type = "mul",  # mask or mul
)

OptimizationParams = dict(
    iterations = int(os.environ.get("MODECGS_ITERATIONS", 20000)),
    coarse_iterations = int(os.environ.get("MODECGS_COARSE_ITERATIONS", 3000)),
    batch_size = 2,

    dynamics_loss = None,
    lambda_dynamics = 0.0001,

    min_opacity = 0.01,
    success_threshold = 0.7,
    densify_grad_threshold = 0.0002,
)
