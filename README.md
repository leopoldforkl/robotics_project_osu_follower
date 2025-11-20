# Rosmaster X3 Human Follower Robot

## Project Overview

This project implements an autonomous human-following robot system using the Yahboom Rosmaster X3 platform for gait analysis applications. Development was conducted in Fall 2025.

**Hardware Platform:** [Yahboom Rosmaster X3](https://www.yahboom.net/study/ROSMASTER-X3)

**Primary Objective:** Implement autonomous tracking of walking humans to facilitate enhanced gait analysis studies.

## Repository Structure

### Code Organization

- **`/code_cale`**: Contains implementation of an adapted Dynamic Window Approach (DWA) algorithm for robot control in simulation
  - Simulation environment: Python with Matplotlib visualization
  - Multiple algorithm versions (v1-v4) showing progression of implementation
  - Comparison utilities and parameter sweep tools in `/compare_dwa`
  
- **`/ros2_ws`**: ROS 2 workspace for full-system simulation and deployment
  - Dockerized environment replicating target hardware (ROS 2 Galactic)
  - Custom packages for robot simulation, control, and target tracking
  - Launch files and configuration parameters

## Hardware Specifications

### Target Platform: Rosmaster X3 with NVIDIA Jetson Orin NX

**Jetson Module:**
- Model: NVIDIA Jetson Orin NX Engineering Reference Developer Kit Super
- Module: Jetson Orin NX (16 GB RAM)
- Board: Generic
- P-Number: p3767-0000

**Operating System:**
- Ubuntu: 22.04.5 LTS (Jammy Jellyfish)
- Kernel: 5.15.148-tegra
- JetPack / L4T: JetPack 6.2 (L4T 36.4.3)
- Power Mode: MAXN_SUPER

**AI/ML Stack:**
- CUDA: 12.6.85
- cuDNN: 9.6.0.74
- TensorRT: 10.7.0.23
- VPI: 3.2.4
- OpenCV: 3.2.0 (without CUDA support)

### Development Workstation
- OS: Ubuntu 22.04 LTS

## Getting Started

### Running Standalone Python Simulations

For testing the DWA algorithm implementation without ROS 2:

1. **Create Python environment:**
   ```bash
   conda create -n follower_robot_env python=3.10
   conda activate follower_robot_env
   ```

2. **Install dependencies:**
   ```bash
   pip install -r code_cale/requirements.txt
   ```

3. **Run simulation:**
   ```bash
   python code_cale/v3/main.py
   ```

## Hardware Setup and Network Configuration

### Step 1: Establish PC-Robot Connection

The Rosmaster X3 creates a WiFi hotspot by default for direct connection.

1. **Connect to robot hotspot** from your development PC

2. **Identify robot IP address** on the hotspot network

3. **SSH into robot:**
   ```bash
   ssh jetson@<robot_ip_address>
   ```
   Example: `ssh jetson@192.168.1.11`

4. **Transfer files to robot via SCP:**
   ```bash
   scp -r /path/to/local/files/* jetson@<robot_ip>:/home/jetson/follower_ws/
   ```
   Example:
   ```bash
   scp -r /home/vcipl/Documents/robotics_project/robotics_project_osu_follower/share/* jetson@192.168.1.11:/home/jetson/follower_ws/share/
   ```

### Step 2: Remote Desktop Setup (NoMachine)

Install NoMachine for graphical remote access to the robot and PC:

1. Download and install [NoMachine](https://www.nomachine.com/) on both PC and robot
2. Configure connection using robot's IP address

#### Optional: Headless Display Configuration

If operating the robot without a physical display, configure a virtual display:

1. **Install dummy display driver:**
   ```bash
   sudo apt-get install xserver-xorg-video-dummy
   ```

2. **Create X11 configuration:**
   ```bash
   sudo mkdir -p /etc/X11/xorg.conf.d
   sudo vi /etc/X11/xorg.conf.d/10-virtual.conf
   ```

3. **Add the following configuration:**
   
   Basic vi commands:
   - Press `i` to enter insert mode
   - Paste the configuration below
   - Press `Esc` to exit insert mode
   - Type `:wq` and press Enter to save and quit
   
   ```
   Section "Device"
       Identifier "VirtualGPU"
       Driver "dummy"
   EndSection

   Section "Monitor"
       Identifier "VirtualMonitor"
       HorizSync 28-80
       VertRefresh 48-75
       Modeline "1920x1080" 148.5 1920 2008 2052 2200 1080 1084 1089 1125 +hsync +vsync
   EndSection

   Section "Screen"
       Identifier "VirtualScreen"
       Device "VirtualGPU"
       Monitor "VirtualMonitor"
       DefaultDepth 24
       SubSection "Display"
           Depth 24
           Modes "1920x1080"
       EndSubSection
   EndSection
   ```

4. **Reboot the robot:**
   ```bash
   sudo reboot
   ```

> **Note:** After this configuration, the physical display port will be disabled, but NoMachine will function with the virtual display.

> **Note:** If you want to undo it just remove the config file and reboot: sudo rm /etc/X11/xorg.conf.d/10-virtual.conf.

### Step 3: Internet Sharing (Optional)

Enable the robot to access the internet through the development PC's connection.

#### Substep 3.1: Enable IP Forwarding and NAT on PC

Execute the following commands on your development PC:

1. **Enable IP forwarding:**
   ```bash
   sudo sysctl -w net.ipv4.ip_forward=1
   ```

2. **Configure NAT routing:**
   ```bash
   # Set up NAT from PC's hotspot interface (wlo1) to Ethernet interface (enp5s0)
   # Adjust interface names based on your system configuration
   sudo iptables -t nat -A POSTROUTING -o enp5s0 -j MASQUERADE
   sudo iptables -A FORWARD -i enp5s0 -o wlo1 -m state --state RELATED,ESTABLISHED -j ACCEPT
   sudo iptables -A FORWARD -i wlo1 -o enp5s0 -j ACCEPT
   ```
   
   Where:
   - `wlo1` is the PC's hotspot interface (connected to Jetson)
   - `enp5s0` is the PC's internet-connected interface (Ethernet)

   This configuration allows traffic from the Jetson (connected via hotspot) to route through the PC's internet connection.

#### Substep 3.2: Configure Default Route on Jetson

Execute the following commands on the robot:

1. **Remove existing default route:**
   ```bash
   sudo ip route del default
   ```

2. **Add new default route through PC:**
   ```bash
   sudo ip route add default via <PC_IP_on_hotspot> dev wlP1p1s0
   ```
   Example: 
   ```bash
   sudo ip route add default via 192.168.1.154 dev wlP1p1s0
   ```
   
   Where `192.168.1.154` is your PC's IP address on the hotspot network. This routes all internet traffic through the PC.

#### Substep 3.3: Test Connectivity

Verify internet access on the Jetson:

```bash
ping 8.8.8.8       # Test raw internet connectivity
ping google.com    # Test DNS resolution
```

> **Troubleshooting:** If you need to identify the correct IP addresses and interface names, run `ip route` on both devices. You can provide this output to ChatGPT or another AI assistant for guidance.

> **Tip:** You can also use tailscale instead of the Hotspot connection, if you connect the robot to the internet via wifi.

## Docker-Based ROS 2 Development Environment

### Prerequisites

1. **Clone repository** on both PC and robot:
   ```bash
   git clone https://github.com/leopoldforkl/robotics_project_osu_follower.git
   ```

2. **Install Docker** on development PC (if not already installed)

### Container Setup and Management

#### Initial Container Creation

1. **Enable Docker GUI access** (may be required for visualization):
   ```bash
   xhost +local:docker
   ```

2. **Build Docker image:**
   ```bash
   docker build -t ros2-galactic-dev .
   ```

3. **Create and run container with GUI support:**
   ```bash
   docker run -it \
       --name ros2-galactic-dev \
       --env="DISPLAY" \
       --env="QT_X11_NO_MITSHM=1" \
       --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
       --volume="/home/vcipl/Documents/robotics_project/robotics_project_osu_follower/ros2_ws:/root/workspace:rw" \
       ros2-galactic-dev
   ```
   
   > **Note:** Adjust the volume mount path to match your repository location. The command mounts the `ros2_ws` directory for persistent development.

4. **Verify container creation:**
   ```bash
   docker ps -a
   ```
   The container should appear with the name `ros2-galactic-dev`.

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

> **Important:** Source the ROS 2 environment in new shells:
> ```bash
> source /opt/ros/galactic/setup.bash
> ```

#### Common Issues and Solutions

**File permission errors on host system:**

If you encounter permission issues when accessing files in the mounted workspace:
```bash
sudo chown -R $USER:$USER /home/vcipl/Documents/robotics_project/robotics_project_osu_follower/ros2_ws
```

Adjust the path to match your repository location.

## ROS 2 Workspace Development

The following instructions apply to both Docker container environments and direct robot deployment in the `ros2_ws` directory.

### Building and Running Existing Packages

This project includes pre-configured ROS 2 packages for robot simulation, control, and target tracking.

#### Initial Build

1. **Navigate to workspace root:**
   ```bash
   cd ~/ros2_ws  # or /root/workspace in Docker
   ```

2. **Build all packages:**
   ```bash
   colcon build
   ```

3. **Source the workspace:**
   ```bash
   source install/setup.bash
   ```
   
   > **Note:** You must source the setup file in every new terminal session before running ROS 2 commands.

#### Running Individual Nodes

**Robot Simulator:**
```bash
ros2 run robot_sim_pkg robot_simulator
```

**Target Movement (TF Publisher):**
```bash
ros2 run target_movement_pkg tf_publisher_node
```

**P-Controller:**
```bash
ros2 run robot_control_pkg robot_p_control_node
```

#### Using Launch Files

Launch files start multiple nodes with configured parameters:

**Robot Simulator with Configuration:**
```bash
ros2 launch robot_sim_pkg robot_simulator.launch.py
```

**Robot P-Control System:**
```bash
ros2 launch robot_control_pkg robot_p_control.launch.py
```

#### Rebuilding After Code Changes

When modifying package code, rebuild selectively for faster iteration:

```bash
# Rebuild specific package
colcon build --packages-select robot_control_pkg

# Re-source the workspace
source install/setup.bash
```

### Creating New ROS 2 Packages

If extending the project with new functionality:

#### 1. Create Package Structure

```bash
cd ~/ros2_ws/src
ros2 pkg create my_python_pkg --build-type ament_python --dependencies rclpy
```

This generates a Python package with the standard ROS 2 structure:
```
my_python_pkg/
├── my_python_pkg/
│   └── __init__.py
├── package.xml
├── setup.py
├── setup.cfg
└── resource/
```

#### 2. Add Node Implementation

Create your node file in `src/<package_name>/<package_name>/`:

```bash
touch src/my_python_pkg/my_python_pkg/my_node.py
```

Example node structure:
```python
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('my_node')
        self.get_logger().info('Node started')
    
def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### 3. Register Node Entry Point

Edit `setup.py` to register the node executable:

```python
entry_points={
    'console_scripts': [
        'my_node = my_python_pkg.my_node:main',
    ],
},
```

#### 4. Build and Test

```bash
cd ~/ros2_ws
colcon build --packages-select my_python_pkg
source install/setup.bash
ros2 run my_python_pkg my_node
```

### Project-Specific Package Overview

**`robot_sim_pkg`**: Simulates robot dynamics and publishes odometry
- Parameters: `config/robot_sim_params.yaml`
- Launch: `launch/robot_simulator.launch.py`

**`robot_control_pkg`**: Implements P-controller for robot following behavior
- Parameters: `config/p_controller_params.yaml`
- Launch: `launch/robot_p_control.launch.py`

**`target_movement_pkg`**: Publishes target transformation data for tracking

**`my_python_pkg`**: Example package template (can be removed or customized)

### Common ROS 2 Commands Reference

```bash
# List all nodes
ros2 node list

# Get node info
ros2 node info /node_name

# List topics
ros2 topic list

# Echo topic data
ros2 topic echo /topic_name

# View TF tree
ros2 run tf2_tools view_frames

# Launch RViz2 for visualization
rviz2
```
## Use Rosmaster for Skeleton Tracking

Bring up the camera:
```bash
ros2 launch yahboomcar_bringup yahboomcar_bringup_X3_launch.py
ros2 launch astra_camera astro_pro_plus.launch.xml
```

Then launch the tracker (in a terminal with display via Nomachine!)
```bash
cd /home/jetson/Github/ai-media-pipe
python HandTrackingROS.py
#or
python PosEstimationROS.py
#or
python skeleton_publisher.py
```

If you launch `skeleton_publisher.py` you can open `rviz2` with a second terminal and open the config
`/home/jetson/Github/robotics_project_osu_follower/ros2_ws/rviz2_configs/skeleton_vizualization.rviz`
you will then see the 3d sceleton.

> **Note:** Uses https://github.com/AnanthaKannan/ai-media-pipe which also has agreat article https://medium.com/@sreeananthakannan/full-body-tracking-c7c4cf68bb9d


## Nuitrack
Status: I am able to open the video stream of the both webcams:
ffplay /dev/video0 will open the Astra pro
ffplay /dev/video2 will open the default rosmaster camera

Nuitrack However does not detect the sensors on the rosmaster.


## Known Issues

### Log Files Filling Disk Space

**Issue:** `/var/log/uvcdynctrl-udev.log` can grow excessively (observed 23GB), filling system disk.

**Diagnosis:**
```bash
df -h                    # Check disk usage
sudo du -sh /var/log/*   # Identify large log files
```

**Fix:**
```bash
sudo truncate -s 0 /var/log/uvcdynctrl-udev.log
```

## License

[Specify your license here]

## Contact

[Specify contact information or project maintainer]

## Acknowledgments

- Yahboom Technology for the Rosmaster X3 platform
- NVIDIA for Jetson development tools
- ROS 2 community for framework support
