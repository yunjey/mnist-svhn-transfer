## MNIST-to-SVHN and SVHN-to-MNIST





## Usage

#### Clone the repository

```bash
$ git clone https://github.com/yunjey/mnist-svhn-mnist.git
$ cd mnist-svhn-mnist/
```

#### Download the dataset
```bash
chmod +x download.sh
./download.sh
```

#### Resize MNIST dataset to 32x32

```bash
$ python prepro.py
```

#### Train the model

##### 1) CycleGAN

```bash
python main.py --use_labels=False --use_reconst_loss=True
```

##### 2) SGAN

```bash
python main.py --use_labels=True --use_reconst_loss=False
```
## Results

#### From SVHN to MNIST

##### 1) CycleGAN

##### 2) SGAN

#### From MNIST to SVHN

##### 1) CycleGAN

##### 2) SGAN
