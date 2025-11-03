FROM osrf/ros:galactic-desktop

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Update apt and install required packages
RUN apt-get update && \
    apt-get install -y \
        software-properties-common \
        curl \
        ca-certificates \
        python3-pip \
    && add-apt-repository universe \
    && apt-get update \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install transforms3d

# Add ROS repository and install tf-transformations
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] https://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Update package list and install the package (with error handling)
RUN apt-get update || true && \
    apt-get install -y ros-galactic-tf-transformations

# Automatically source ROS on container start
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/galactic/setup.bash" >> /root/.bashrc

# Set default working directory
WORKDIR /root/workspace
