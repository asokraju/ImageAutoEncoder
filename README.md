# Training and Tuning a Variational Autoencoder

In this repository, you'll find all the necessary scripts and code to train and tune a Variational Autoencoder (VAE). This Readme file guides you through three ways to use the code. It also includes a brief explanation of the arguments you can use when running the training script.

This project is the culmination of a three-part series. Check out the following articles for a thorough walk-through of the code and concepts:

1. [Variational Autoencoders Introduction and Implementation (Part 1)](https://asokraju.medium.com/part-1-variational-autoencoders-introduction-and-implementation-1ceb47a75cb5)
2. [Training Procedures for Variational Autoencoders (Part 2)](https://asokraju.medium.com/variational-autoencoders-training-procedures-part-2-203f760a9315)
3. [Hyperparameter Tuning with Docker and Bash Scripts (Part 3)](https://asokraju.medium.com/variational-autoencoders-hyperparameter-tuning-with-docker-and-bash-scripts-part-3-51ce05b92df7)

## Results
Example of reconstructed images 

<img src="https://github.com/asokraju/ImageAutoEncoder/blob/main/results.png" width="500" align="center">


In the image above, you can see the original post-compressed (56x56 pixel) image on the left and the predicted image on the right. The latent space for the image is of dimension 5.
## Parsing Arguments

The training script accepts several arguments to customize the training process:

- `--image-dir`: Path to the image data (default = 'Data')
- `--logs-dir`: Path to store logs (default = 'logs')
- `--output-image-shape`: Size to reshape the input images (default = 56)
- `--filters`: Number of convolutional filters in each layer (default = [32, 64])
- `--dense-layer-dim`: Dimension of the dense layer (default = 16)
- `--latent-dim`: Dimension of the latent space (default = 6)
- `--beta`: Beta parameter for the beta-VAE (default = 1.0)
- `--batch-size`: Batch size for training (default = 128)
- `--learning-rate`: Learning rate for the optimizer (default = 1e-4)
- `--patience`: Number of epochs to wait for improvement before stopping (default = 10)
- `--epochs`: Total number of epochs to train (default = 20)
- `--train-split`: Ratio of data to use for training (default = 0.8)

## Method 1: Running `train.py` Directly

You can train the VAE using specific hyperparameters by running the `train.py` script with your chosen arguments. For example:

```
python train.py --image-dir='my_data' --learning-rate=0.001 --latent-dim=10 --batch-size=128 --logs-dir='my_logs'
```

You can put your data inside the 'Data' folder and run the code without specifying `--image-dir` to get results with default settings.

## Method 2: Batch Experiments

To run a batch of experiments with different combinations of hyperparameters, you can use the `master.sh` script.

Before using this method, you need to specify the data directory in the `worker.sh` script. Open `worker.sh` and change the line `python train.py --image-dir='../train_data' ...` to match your data directory.

Then, to run the experiments, execute:

```bash
./scripts/master.sh
```
This will run the train.py script with various combinations of hyperparameters specified in the master.sh script.

## Method 3: Docker Setup

To ensure a consistent environment across different machines or platforms, you can use Docker. Docker allows you to build a container that has all the necessary dependencies pre-installed.

You can use the provided Dockerfile and docker-compose.yml files to build and run the Docker container. The Dockerfile specifies the instructions to build the Docker image, and the docker-compose.yml file defines the services that make up your application so they can be run together in an isolated environment.

Before starting, you may need to adjust the Dockerfile and docker-compose.yml file according to your specific needs.

In `docker-compose.yml` file, I have specified two volumes. The first volume maps the current directory (where your docker-compose.yml file is located) on your host machine to the `/autoencoders` directory in your Docker container.

The second volume is a bind mount, which binds a directory or file from your host machine to a directory or file in your Docker container. In this case, you are binding the `F:/train_data` directory on your host machine to the `/train_data` directory in your Docker container.

This line is significant because your training script (running inside the Docker container) expects to find your training data at `/train_data`. But since Docker containers are isolated from your host machine, you need a way to provide the training data to the script. The bind mount makes this possible by making the `F:/train_data` directory on your host machine available at `/train_data` in the Docker container.

However, not everyone who uses your scripts will have their training data at `F:/train_data`. That's why you need to instruct them to change this line according to where their training data is located. They can replace `F:/train_data` with the path to their training data. If their training data is located at `C:/Users/user123/data`, for example, they would need to change this line to:

```
source: C:/Users/user123/data
```

To build and run the Docker container, execute:
```
docker-compose up
```
This will run the `master.sh` script in the Docker container.

This will run the `master.sh` script in the Docker container.

## Troubleshooting

1. **Permission issues with the shell scripts:** If you receive a permission error when trying to execute the `master.sh` or `worker.sh` scripts, you may need to modify the scripts' permissions. You can do this with the `chmod` command: `chmod +x scripts/master.sh scripts/worker.sh`.

2. **Docker build issues:** If you have issues building the Docker image, make sure your Dockerfile is correctly written and that you have the correct permissions to access all the files and directories specified in the Dockerfile.

3. **Docker-compose issues:** If you have issues with the `docker-compose up` command, make sure your docker-compose.yml file is correctly written. If you're mapping volumes from your host machine to the Docker container, ensure the specified directories exist and you have the correct permissions to access them.

4. **Script arguments issues:** If your scripts fail due to incorrect arguments, make sure you're passing the correct types and values as specified in the `parse_arguments()` function in the `train.py` script.

5. **Line ending issues between Windows and Linux:** If you're developing on Windows and running the scripts on a Linux system (including the Docker container), you may encounter issues due to differences in how the two systems handle line endings. Windows uses a carriage return and line feed (CRLF) as a line ending, while Linux uses just a line feed (LF). This difference can cause "command not found" errors when running the scripts on Linux.

To resolve this issue, you can change the line endings from CRLF to LF in your text editor before saving the scripts. In Visual Studio Code, for example, you can change the line endings by clicking on "CRLF" in the status bar at the bottom and selecting "LF".

Alternatively, if you're using Git, you can configure it to automatically convert CRLF to LF on commit with the `core.autocrlf` setting. Run the following command in your Git bash or terminal:
