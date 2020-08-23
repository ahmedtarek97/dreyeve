
from keras.optimizers import Adam

# --- GLOBAL --- #

dreyeve_dir = '/home/a_alwali96/GP-2020/'
dreyeve_train_seq =range(1,37+1)
dreyeve_test_seq = range(38,40+1)
n_sequences = 40
total_frames_each_run = 7499

# output directories
log_dir = '/home/a_alwali96/GP-2020/GP/logs'
ckp_dir = '/home/a_alwali96/GP-2020/GP/checkpoints'
prd_dir = '/home/a_alwali96/GP-2020/GP/predictions'
# --- TRAIN --- #
batchsize = 4
frames_per_seq = 16
h = 448
w = 448
train_frame_range = list(range(0, 3500 - frames_per_seq - 1)) + list(range(4000, total_frames_each_run - frames_per_seq - 1))
val_frame_range = range(3500, 4000 - frames_per_seq - 1)
test_frame_range = range(0, total_frames_each_run-frames_per_seq - 1)
frame_size_before_crop = (256, 256)
crop_type = 'central'  # choose among [`central`, `random`]

#512
#64

train_samples_per_epoch =512*batchsize
val_samples_per_epoch = 64*batchsize
nb_epochs = 10

force_sample_steering = False

# optimizer
full_frame_loss = 'kld'
crop_loss = 'kld'
w_loss_cropped = 1.0
w_loss_fine = 1.0
mse_beta = 0.1
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
simo_mode = False  # DVD: works only with full_frame_loss = 'simo'

# callbacks
callback_batchsize = 8
