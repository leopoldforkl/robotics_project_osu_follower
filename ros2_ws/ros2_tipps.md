# ROS 2 Python Package Tips

## How to create a Python package (do inside the container!)

### 0. Container Setup and Management

#### Creating and Starting a ROS 2 Container

Create a new ROS 2 Galactic container with GUI support and workspace volume mounting:

```bash
docker build -t ros2-galactic-dev .
```

```bash
docker run -it \
    --name ros2-galactic-dev \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="/home/vcipl/Documents/robotics_project/robotics_project_osu_follower/ros2_ws:/root/workspace:rw" \
    ros2-galactic-dev
```

After creation, your container will show up as `ros2-galactic-dev` when you list containers:

```bash
docker ps -a
```

#### Container Management Commands

**Start an existing container:**
```bash
docker start -ai ros2-galactic-dev
```

**Stop a running container:**
```bash
docker stop ros2-galactic-dev
```

**Remove a container:**
```bash
docker rm ros2-galactic-dev
```

**Open additional shell in running container:**
```bash
docker exec -it ros2-galactic-dev bash
```

> **Note:** You might need to source the ROS 2 setup in new shells:
> ```bash
> source /opt/ros/galactic/setup.bash
> ```

> **Note:** You might need fix locks on host system:
> ```bash
> sudo chown -R $USER:$USER /home/vcipl/Documents/robotics_project/robotics_project_osu_follower/ros2_ws
> ```


### 1. Create a Python ROS 2 Package

Go to your workspace src folder (e.g., `~/ros2_ws/src`) and create the package:

```bash
cd ~/ros2_ws/src
ros2 pkg create my_python_pkg --build-type ament_python --dependencies rclpy
```

This will create a folder `my_python_pkg` with a minimal Python package structure.

### 2. Add a Node

Add a node like `my_node.py` in `src/<package_name>/<package_name>`

### 3. Update setup.py

Make sure your `setup.py` points to the node:

```python
entry_points={
    'console_scripts': [
        'my_node = my_python_pkg.my_node:main',
    ],
},
```

### 4. Build the Package

Go back to the root of your workspace (e.g., `~/ros2_ws`) and build with colcon:

```bash
cd ~/ros2_ws
colcon build
```

Make sure to source the local setup file after building:

```bash
source install/setup.bash
```

### 5. Run the Python Node

Run your node using `ros2 run`:

```bash
ros2 run my_python_pkg my_node
```

### 6. Rebuilding After Changes

If you change the node code, rebuild with:

```bash
colcon build --packages-select my_python_pkg
source install/setup.bash
```