
# Road Segmentation Network ROS Integration

This repository provides the integration of a road segmentation neural network with the ROS (Robot Operating System) framework, enabling real-time road segmentation in robotic applications.

---

## Setup and Usage

### Prerequisites
Ensure you have the following ready:
- A ROS workspace configured (`ws_road`)
- Python interpreter installed with the required machine learning dependencies (e.g., PyTorch, OpenCV, ROS Python bindings)
- Trained road segmentation model file

### Instructions

1. **Navigate to the ROS Workspace**  
   Use the following command to move into the ROS workspace:
   ```bash
   cd ws_road
   ```

2. **Build the Workspace**  
   Compile the workspace using `catkin_make`, specifying the Python interpreter installed with the necessary dependencies:
   ```bash
   catkin_make -DPYTHON_EXECUTABLE='path_to_your_python_interpreter'
   ```
   Replace `'path_to_your_python_interpreter'` with the actual path to your Python executable.

3. **Source the Setup File**  
   Activate the workspace by sourcing the setup file:
   ```bash
   source devel/setup.bash
   ```

4. **Configure the Script**  
   Go to the `road_seg_3d` script directory to configure paths and ROS topics:
   ```bash
   cd src/road_seg/scripts/road_seg_3d
   ```
   Edit the configuration:
   - Set the path to your trained road segmentation model.
   - Adjust input rostopics and other settings as needed.

5. **Run the Road Segmentation Node**  
   Start the road segmentation node with the following command:
   ```bash
   rosrun road_seg road_seg_3d
   ```

---

## Additional Notes

- **Environment Sourcing**: Ensure you source the workspace in each new terminal where you intend to run ROS nodes. This activates the necessary ROS paths and dependencies.
- **Troubleshooting**: If you encounter issues, verify that all required dependencies are installed and compatible with your ROS setup. Additionally, ensure that your Python environment includes all libraries needed for machine learning and ROS.

For further details, consult the documentation or reach out via the repository’s issue tracker.


