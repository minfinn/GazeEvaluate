from .config_node import ConfigNode

config = ConfigNode()

#mode(ONLY FOR MPII)
config.mode = 'MPIIFaceGaze'

# model
config.model = ConfigNode()
config.model.name = 'face_res50'
config.model.checkpoint = './checkpoints/epoch_24_ckpt.pth.tar'
config.model.in_stride = 2

#dateset
config.dataset = ConfigNode()
config.dataset.name = "MPII"
config.dataset.data_dir = './data/MPIIGaze.h5'

#transform(MPII)
config.transform = ConfigNode()
config.transform.mpiifacegaze_face_size = 224
config.transform.mpiifacegaze_gray = False

#test
config.test = ConfigNode()
config.test.test_id = 0
config.test.use_gpu = True
config.test.output_dir = './test_output'
config.test.metric = 'angle_error'
config.test.batch_size = 64
config.test.dataloader = ConfigNode()
config.test.dataloader.num_workers = 4
config.test.dataloader.pin_memory = False



def get_default_config():
    return config.clone()
