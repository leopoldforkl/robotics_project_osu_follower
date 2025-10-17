# ROS 2 Python Package Tips

## How to create a Python package (do inside the container!)

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