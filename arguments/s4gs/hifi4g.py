# MoDec-GS on HiFi4G for the S4GS comparison (docs/experiments/comparisons.md, review R-02 / A-02).
# Used by scripts/compare/run_modecgs.sh:  python train.py -s <converted scene> -m <model> --configs arguments/s4gs/hifi4g.py
# The scene is converted to the PanopticSports layout by scripts/compare/convert_hifi4g_panoptic.py and is read by
# the PanopticSports reader of scene/dataset_readers.py (S4GS patch: native test sizes, white background, uint8 images).
#
# Base: arguments/dynerf/config_1.0.py, the released setting for multi-view video (HiFi4G = 70 training cameras x
# 200 frames of one performer; Flute 300 frames, PianoFast 700). run_dynerf.sh copies config_1.0.py over
# arguments/dynerf/base.py before training, so config_1.0.py is what the authors' DyNeRF runs use; the committed
# base.py is a stale copy that differs only in multires_local ([1, 2] instead of [1, 2, 4]). The repository has no
# PanopticSports config, and the paper gives no setup for that dataset.
#
# Everything below is verbatim from config_1.0.py except:
#   iterations 14000 -> 20000 (fine stage; coarse_iterations stays 3000).
#       MoDec-GS paper, appendix B: "3,000 for the Global stage, and between 20,000 and 60,000 for the Local stage,
#       depending on the sequence length". 20,000 is the lower end of that range and the same budget as Mixamo
#       (arguments/s4gs/mixamo.py), so HiFi4G and Mixamo runs differ only by data. The paper gives no rule for
#       longer clips, so Flute and PianoFast are not scaled; use MODECGS_ITERATIONS for that.
#   The iteration counts can be overridden without editing this file: MODECGS_ITERATIONS, MODECGS_COARSE_ITERATIONS
#       (run_modecgs.sh sets them from --iterations / --coarse_iterations, because mmcv's merge_hparams lets the
#       config override the command line).
#
# Data-dependent choices that are NOT in this file (they are in the converted scene, see the converter docstring):
#   - training images are resized to 969x541 and the cameras' K are rescaled to match; test images stay at each
#     camera's original size (HiFi4G images differ in size from camera to camera, 934x525 .. 1057x554);
#   - the rasterizer background is white ("bg" in the meta files), matching the white-background GT;
#   - the initial point cloud is the COLMAP cloud of the scene (init_pt_cld.npz), voxel_size 0.01 as upstream.
# No hyper-parameter was tuned.
#
# Upstream quirk kept as is: config_1.0.py spells the global dynamics type "anchor_dyamics_global_type" (typo).
# merge_hparams ignores keys that are not arguments, so that line never had an effect and the argument keeps its
# default "mask" (arguments/__init__.py). Fixing the spelling would change the method, so it stays.
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
