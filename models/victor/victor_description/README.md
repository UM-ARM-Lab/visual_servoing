If you see this error running openrave stuff

```[catkin_finder.cpp:183 getWorkspaces] Failed reading package source directories from the marker file '/home/victorrope/local/.catkin'. Resources in the source space of this workspace will not resolve.
[catkin_finder.cpp:245 find] Could not find file 'victor_description' under catkin package 'urdf/victor.urdf'!
[ros.urdf]: Could not open file [] for parsing.
[urdf_loader.cpp:865 loadURI] Failed loading URDF model: Failed to open URDF file.
```

you might need to run xacro on victor's URDF:

    ~/catkin_ws/src/kuka_iiwa_interface/victor_description/urdf$ rosrun xacro xacro.py victor.urdf.xacro > victor.urdf
