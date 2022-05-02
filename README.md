# Position Based Visual Servoing 

This package contains code to do position based visual servoing in the PyBullet simulator. 

The methods are explained in this report: https://drive.google.com/file/d/1N3Cuwtxr-NA0eG1iCCvZUsygbrnd9Mni/view

A demo of this in action can be seen here: https://youtu.be/48twyQt-Zg0

To try out PBVS in marker mode, see scripts/marker_pbvs_demo. For ICP mode, see scripts/evaluation.py. Trajectories generated by evaluation.py can also be played back in rviz via the code in playback.py.

Dependencies: 
- numpy 
- PyBullet
- OpenCV + OpenCV extra modules 
- rospy
- arm_robots
- arc_utilities 
- link_bot 
- Tensorflow (working on removing this)