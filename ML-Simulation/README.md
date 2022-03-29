## Installation Steps:

1. Install Anaconda3.

2. Install Docker.

3. Clone the repository, navigate to the folder where the repository is located. 

4. Create and activate a conda environment to run the code. Run following commands in the environment to install packages.

```
pip install -r /path/to/requirements.txt
```
## Usage:
1. To build the docker image, we run the following command in the terminal:

```
docker build -t ml-simulation -f Dockerfile .
```
The -t flag names the docker image, -f tag is the name of the Dockerfile. The . at the end of the docker build command tells that Docker should look for the Dockerfile in the current directory.

2. To launch the web server, we will run our Docker container and frontend.py:
```
docker run -it -p 8080:8080 ml-simulation python api.py
```
The -p flag exposes port 8080 in the container to port 8080 on our host machine, -it flag allows us to see the logs from the container and we run python frontend.py in the ml-simulation image.

3. Now we can go to ‘0:0:0:0/8080' or IP of the terminal displayed, to check if our application is running or not.

