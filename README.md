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

#### 1) CycleGAN
From SVHN to MNIST            |  From MNIST to SVHN
:-------------------------:|:-------------------------:
![alt text](gif/cycle-s-m.gif)  |  ![alt text](gif/cycle-m-s.gif)

#### 2) SGAN
From SVHN to MNIST            |  From MNIST to SVHN
:-------------------------:|:-------------------------:
![alt text](gif/sgan-s-m.gif)  |  ![alt text](gif/sgan-m-s.gif)



