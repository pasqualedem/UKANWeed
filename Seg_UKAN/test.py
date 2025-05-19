import archs

arch = 'UKAN'
num_classes = 3
input_channels = 3
deep_supervision = False
# input_list = [32, 40, 64]
input_list = [256,320,512]

model = archs.__dict__[arch](num_classes, input_channels, deep_supervision, embed_dims=input_list,no_kan=True)

print("Total params: ", sum(p.numel() for p in model.parameters()))