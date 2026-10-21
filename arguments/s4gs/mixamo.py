# MoDec-GS on Mixamo for the S4GS comparison (docs/experiments/comparisons.md, review R-02 / A-02).
# Used by scripts/compare/run_modecgs.sh:  python train.py -s <converted scene> -m <model> --configs arguments/s4gs/mixamo.py
#
# Base: arguments/dynerf/config_1.0.py, the released setting for multi-view video (Mixamo = 17-19 training
# cameras x 120 frames of one moving character). run_dynerf.sh copies config_1.0.py over arguments/dynerf/base.py
# before training, so config_1.0.py is what the authors' DyNeRF runs use; the committed base.py is a stale copy that
# differs only in multires_local ([1, 2] instead of [1, 2, 4]).
#
# Everything below is verbatim from config_1.0.py except:
#   iterations 14000 -> 20000 (fine stage; coarse_iterations stays 3000).
#       Two sources agree: the 4DGS D-NeRF setting (arguments/dnerf/dnerf_default.py of hustvl/4DGaussians:
#       coarse 3000 + 20000 iterations; Mixamo is D-NeRF-like: synthetic, white background, 800x800 RGBA) and the
#       MoDec-GS paper, appendix B: "3,000 for the Global stage, and between 20,000 and 60,000 for the Local stage,
#       depending on the sequence length" (20,000 is the short-sequence end; these clips have 120 frames).
#   The iteration counts can be overridden without editing this file: MODECGS_ITERATIONS, MODECGS_COARSE_ITERATIONS
#       (run_modecgs.sh sets them from --iterations / --coarse_iterations, because mmcv's merge_hparams lets the
#       config override the command line).
#
# Resolution: 800x800, the Mixamo size. MoDec-GS's Blender reader (scene/dataset_readers.py) resizes every image
# to 800x800, composites RGBA on white (white_background=True is the default) and gives the same GT as our loader.
# No hyper-parameter was tuned. Voxel size 0.01 etc. are the authors' values.
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
