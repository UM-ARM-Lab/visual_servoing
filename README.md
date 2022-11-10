# Position Based Visual Servoing 

This package contains code to do position based visual servoing in the PyBullet simulator. 

The methods are explained in this report: https://drive.google.com/file/d/1N3Cuwtxr-NA0eG1iCCvZUsygbrnd9Mni/view

A demo of this in action can be seen here: https://youtu.be/48twyQt-Zg0

To try out PBVS on Val with ARUCO markers, run scripts/marker_pbvs_demo. For ICP PBVS, run scripts/evaluation.py. Trajectories generated by evaluation.py can also be played back in rviz via the code in playback.py. To do this, `roslaunch launch/rviz_victor.launch` then run playback.py and select the generated result file from evaluation.py.

Note that for the code to work the working directory must be the top level of this repoistory. vscode configurations are included.

Dependencies: 
- numpy 
- PyBullet
- OpenCV + OpenCV extra modules 
- rospy
- Tensorflow (working on removing this)

You will need to run: `rosdep install -y -r --from-paths . --ignore-src`

** Credits **
- Using transformation functions from PyTorch 3D in utils.py
