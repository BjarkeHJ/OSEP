# PX4-Autopilot in simulation envirmonet using ROS2

Virtuel enviroments can in general be an issue, so it's recomended to deactivate.
```
conda deactivate
```

<details>
<summary>Guide to install Isaac Sim v.4.2.0 </summary>

This guide is slightly modified from:
<https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html>

### Install Isaac Sim
Ensure system requirements as stated by the documentation...

Download Isaac Sim v.4.2.0 from <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html#isaac-sim-latest-release> and place in Downloads

```
mkdir ~/isaacsim
cd ~/Downloads
unzip "your-downloaded-file.zip" -d ~/isaacsim
cd ~/isaacsim
./omni.isaac.sim.post.install.run.sh
./isaac-sim.sh
```
Isaac Sim should now open...

Create a conda environment for Isaac Sim and Pegasus Simulator
```
conda create -n isaac_env python=3.10

# Update libstdcxx-ng to satisfy C++17 standards
conda install -c conda-forge libstdcxx-ng

# Install package dependencies
pip install catkin_pkg
pip install lark
```
Create alias in ~/.bashrc
```
alias isaac_env="conda activate isaac_env && source ~/isaacsim/setup_conda_env.sh"
```

</details>

<details>
<summary>Guide to install Pegasus Simulator for Isaac Sim v.4.2.0 </summary>

This guide is slightly modified from:
<https://pegasussimulator.github.io/PegasusSimulator/>

### Install Pegasus Simulator
```
isaac_env
cd
git clone git clone https://github.com/PegasusSimulator/PegasusSimulator.git
cd PegasusSimulator/extensions
python -m pip install --editable pegasus.simulator
```

</details>

<details>
<summary>Guide to install PX4-Autopilot </summary>

This guide is sligthly modified from:
<https://docs.px4.io/main/en/ros2/user_guide.html#install-px4>

### Install PX4
```
cd
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
bash ./PX4-Autopilot/Tools/setup/ubuntu.sh
cd PX4-Autopilot/
make px4_sitl
```
### Install ROS2

```
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update && sudo apt upgrade -y
sudo apt install ros-humble-desktop
sudo apt install ros-dev-tools
source /opt/ros/humble/setup.bash && echo "source /opt/ros/humble/setup.bash" >> .bashrc
```
### Install the Micro XRCE-DDS Agent
```
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade setuptools wheel twine check-wheel-contents
```

```
git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
cd Micro-XRCE-DDS-Agent
mkdir build
cd build
cmake ..
make
sudo make install
sudo ldconfig /usr/local/lib/
```
### Get the ROS2 msg for PX4
```
mkdir -p ~/ws_ros2/src/
cd ~/ws_ros2/src/
git clone https://github.com/PX4/px4_msgs.git
git clone https://github.com/PX4/px4_ros_com.git
cd ..
colcon build
```
#### Remember to source the ROS2 WS
```
echo source ~/ws_ros2/install/setup.bash >> ~/.bashrc 
```

</details>


## Running simulation

The Micro XRCE-DDS Agent can be run with the following command. The default port of the client is 8888

```
MicroXRCEAgent udp4 -p 8888
```

<details>
<summary> Gazebo Simulation </summary>

#### Dependencies
```
pip install --user -U empy==3.3.4 pyros-genmsg setuptools
```
#### Running the simulation
```
cd ~/PX4-Autopilot/
make px4_sitl gz_x500
```

</details>


<details>
<summary> Isaac sim - Pegasus</summary>

```
ISSACSIM_ENV
python3 PegasusSimulator/examples/1_px4_single_vehicle.py 
```
</details>
