DEBUG = False
OUTPUT_ROOT = "output"
OUTPUT_DIR = "auto"
CUDNN_BENCHMARK = True
SEED = -1
MODEL = dict(
    NAME="SymNet",
    WEIGHTS="",
    BACKBONE=dict(
        ARCH="resnet",
        NUM_LAYERS=34,
        INPUT_CHANNEL=3,
        CONCAT=True,
        FREEZE=False,
        PRETRAINED="torchvision://resnet34",  # fixed
    ),
    GEOMETRY_NET=dict(
        ARCH="aspp",  # choose from ["aspp", "cdpn"]
        FREEZE=False,
        VISIB_MASK_LOSS_TYPE="L1",  # choose from ["BCE", "L1"]
        VISIB_MASK_LW=1,
        CODE_LOSS_TYPE="BCE",  # choose from ["BCE", "L1"]
        CODE_LW=3,
    ),
    PNP_NET=dict(
        ARCH="ConvPnPNet",    # choose from ["ConvPnPNet"]
        FREEZE=False,
        LR_MULT=1.0,
        # point matching loss
        PM_LW=1.0,
        PM_R_ONLY=True,
        PM_LOSS_TYPE="L1",  # choose from ["L1", "L2", "MSE", "SMOOTH_L1"]
        PM_NORM_BY_EXTENT=True,
        PM_NUM_POINTS=3000,
        PM_LOSS_SYM=True,

        SITE_XY_LW=1.0,
        SITE_XY_LOSS_TYPE="L1",  # choose from ["L1", "MSE"]

        SITE_Z_LW=1.0,
        SITE_Z_LOSS_TYPE="L1",  # choose from ["L1", "MSE"]


    )
)

SOLVER = dict(
    BASE_LR=1e-4,
    OPTIMIZER_CFG=dict(type="Ranger", lr=1e-4, weight_decay=0),
    AMP=dict(ENABLED=False),
    LR_SCHEDULER_NAME="flat_and_anneal",
    WARMUP_FACTOR=0.001,
    WARMUP_ITERS=1000,
    WARMUP_METHOD="linear",
    ANNEAL_METHOD="cosine",
    ANNEAL_POINT=0.72,
    REL_STEPS=(0.5, 0.75),
    CHECKPOINT_BY_EPOCH=True,
    CHECKPOINT_PERIOD=5,
    MAX_TO_KEEP=5,
)

DATASETS = dict(
    NAME="ycbv",
    TRAIN=("real",),  # tuple, which is the subset of ("pbr", "real",)
    TRAIN2=("pbr",),  # tuple, which is the subset of ("pbr", "real",)
    TRAIN2_RATIO=0.0,
    TEST="test",
    TEST_DETECTION_TYPE="type2",  # choose from ["type1", "type2", "type3"]
    TEST_DETECTION_PATH='datasets/detections/cosypose_maskrcnn_synt+real/challenge2022-642947_ycbv-test.json',
    TEST_SCORE_THR=0.5,
    TEST_TOP_K_PER_OBJ=1,
    RES_CROP=256,
    OBJ_IDS=[15, 18],  # OBJ_IDS=[15, 18],  # should be consistent with NUM_CLASSES
    NUM_CLASSES=2,  # NUM_CLASSES=2,  # should be consistent with OBJ_IDS
    COLOR_AUG_PROB=0.8,
    BG_AUG_PROB=0.5,
    BG_AUG_TYPE="VOC_table",  # choose from ["VOC", "VOC_table"]
)

TRAIN = dict(
    PRINT_FREQ=1000,
    NUM_WORKERS=4,
    BATCH_SIZE=2,
    TOTAL_BATCH_SIZE=2,  # should be BATCH_SIZE * num_gpu
    TOTAL_EPOCHS=160,
)

TEST = dict(
    DEBUG=True,
)
