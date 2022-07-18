## SET THESE PARAMETERS ##
DEVICE = 'varjo'
SUBJECT = 'P3'      # or ['PVL', 'CVL'] # feature importance tested on P3
CLASSIFIER = 'error' #['error', 'VL']
ERROR_TYPE = 'motor' #['motor', 'process']           

## Pupil Invisible Specs ##
if DEVICE == 'PI':
    SAMPLE_RATE = 66      #Hz
elif DEVICE == 'varjo':
    SAMPLE_RATE = 90
    
## Data processing parameters ##
WINDOW = 18  # number of events per sample
STEP = 2  # sliding step size

## Data parameters ##
ERROR_DICT = {'motor':1, 'process':2}
EVENT_DICT = {'fix':0, 'sac':1, 'smp':2, 'blink':3, 'other':4, 'loss':5}
LABELS = ['subID', 'VL', 'task', 'error']
FEATS = ['event',
         'duration',
         'amplitude',
         'dispersion',
         'avg_iss',
         'max_iss',
         'carpenter_error',
         'calculus_error',
         #'num_blinks',
         'prev_error',
         'next_error',
         'total_dur'
        ]
NUM_FEATS = len(FEATS)
IMAGE_SHAPE = (NUM_FEATS, WINDOW, 1) # The image shape should be of the format (height, width, channels)
NUM_CLASSES = 2  # binary (1=error, 0=no error)

## Network parameters (TWEAK THESE) ##
BATCH_SIZE = 20
HEIGHT, WIDTH, CHANNELS = IMAGE_SHAPE

OUT_CHANNELS = [16,8] # <-- Filters in your convolutional layer (large num allows layer to potentially learn more useful feats about the input data. size of CNN is a function of in_channels/out_channels)
KERNEL_SIZE = (3, 3) # size of 2d filter (height, width)
STRIDE = 1
PADDING = (0, 0) # (height, width)

LEARNING_RATE = 0.001 #5e-4
WEIGHT_DECAY = 1e-4
VALIDATION_SIZE = 0.1 # percent of train to allocate to validation
MAX_ITER = 3000

# Settings for training 
log_every = 200
eval_every = 100


