## This is the repo used to integrate road segmentation network into ROS SYSTEM

### Usage
  1. Go into ws_road workspace
  2. catkin_make -DPYTHON_EXECUTABLE='path to your python inturpreter installed with corresponding learning dependencies'
  3. source devel/setup.bash
  4. Goes to ws_road/src/road_seg/scripts/road_seg_3d to modify: the path to the trained model, the input rostopics, etc.
  5. rosrun road_seg road_seg_3d
