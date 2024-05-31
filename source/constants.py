BATCH_SIZE = 32
NUM_BATCHES = 4
TRAIN_DS_SIZE = NUM_BATCHES * BATCH_SIZE
EPOCHS = 2
LATENT_DIM = 128
LEARING_RATE = 0.00005
OPTIMIZER = 'adam'
LOSS = 'mse'


AUDIO_SAMPLE_RATE = 16000
N_FFT = 2048 // 4  # Define the FFT window size to reduce frequency bins
HOP_LENGTH = N_FFT // 4  # Define the hop length (adjust as needed)
TRACK_DURATION = 1 # seconds
MAX_AUDIO_LENGTH = AUDIO_SAMPLE_RATE * TRACK_DURATION
WINDOW = "hann"