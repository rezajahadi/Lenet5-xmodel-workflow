import os
import sys
import argparse
import torch
import torchvision
from common import *
from pytorch_nndct.apis import torch_quantizer

DIVIDER = '-----------------------------------------'

def quantize(build_dir, quant_mode, batchsize):
    dset_dir = build_dir + '/dataset'
    float_model = build_dir + '/float_model'
    quant_model = build_dir + '/quant_model'

    if torch.cuda.is_available():
        print('You have', torch.cuda.device_count(), 'CUDA devices available')
        for i in range(torch.cuda.device_count()):
            print(' Device', str(i), ': ', torch.cuda.get_device_name(i))
        print('Selecting device 0..')
        device = torch.device('cuda:0')
    else:
        print('No CUDA devices available..selecting CPU')
        device = torch.device('cpu')

    # Load trained LeNet-5 model
    model = LeNet5().to(device)
    model.load_state_dict(torch.load(os.path.join(float_model, 'mnist_0.9869.pkl')))

    # Force to merge BN with CONV for better quantization accuracy
    optimize = 1

    # Override batchsize if in test mode
    if quant_mode == 'test':
        batchsize = 1

    rand_in = torch.randn([batchsize, 1, 28, 28])  # Use 28x28 input for LeNet-5
    quantizer = torch_quantizer(quant_mode, model, (rand_in), output_dir=quant_model)
    quantized_model = quantizer.quant_model

    # Data loader
    test_dataset = torchvision.datasets.MNIST(dset_dir, train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

    # Export config
    if quant_mode == 'calib':
        test_loss, accuracy = test(quantized_model, device, test_loader)
        quantizer.export_quant_config()
    if quant_mode == 'test':
        # Evaluate
        test_loss, accuracy = test(quantized_model, device, test_loader)
        print("Accuracy is {}, Loss is {}".format(accuracy, test_loss))
        quantizer.export_xmodel(deploy_check=False, output_dir=quant_model)

    return

def run_main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--build_dir', type=str, default='build', help='Path to build folder. Default is build')
    ap.add_argument('-q', '--quant_mode', type=str, default='calib', choices=['calib', 'test'],
                    help='Quantization mode (calib or test). Default is calib')
    ap.add_argument('-b', '--batchsize', type=int, default=100, help='Testing batchsize - must be an integer. Default is 100')
    args = ap.parse_args()

    print('\n' + DIVIDER)
    print('PyTorch version : ', torch.__version__)
    print(sys.version)
    print(DIVIDER)
    print(' Command line options:')
    print('--build_dir    : ', args.build_dir)
    print('--quant_mode   : ', args.quant_mode)
    print('--batchsize    : ', args.batchsize)
    print(DIVIDER)

    quantize(args.build_dir, args.quant_mode, args.batchsize)

    return

if __name__ == '__main__':
    run_main()
