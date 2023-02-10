import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import pybullet_data

from pybullet_envs.scene_abstract import Scene
import pybullet
import numpy as np

class SlopeScene(Scene):
    

    zero_at_running_strip_start_line = True  # if False, center of coordinates (0,0,0) will be at the middle of the stadium
    slope_halflen = 105 * 0.25  # FOOBALL_FIELD_HALFLEN
    slope_halfwidth = 50 * 0.25  # FOOBALL_FIELD_HALFWID
    slopeLoaded = 0  


    def episode_restart(self, bullet_client):
        self._p = bullet_client
        Scene.episode_restart(self, bullet_client)  # contains cpp_world.clean_everything()
        if (self.slopeLoaded == 0):
            self.slopeLoaded = 1

            # stadium_pose = cpp_household.Pose()
            # if self.zero_at_running_strip_start_line:
            #	 stadium_pose.set_xyz(27, 21, 0)  # see RUN_STARTLINE, RUN_RAD constants

            # procedurally generate a slope-like plane
            filename = os.path.join(pybullet_data.getDataPath(), "plane_stadium.sdf")
            self.ground_plane_mjcf = self._p.loadSDF(filename)
            # filename = os.path.join(pybullet_data.getDataPath(),"stadium_no_collision.sdf")
            # self.ground_plane_mjcf = self._p.loadSDF(filename)
            #


            # tilt the ground plane to 30 degrees, so that the running start line is not perfectly horizontal
            # disable reflection of the ground plane (useful if you use a plane instead of the stadium)

            # generate_random_plane(self, bullet_client)
            # self.generate_random_plane()

            # generate sine wave plane
            self.generate_sine_plane()
        
    def generate_sine_plane(self):

        # TODO: add to args
        heightPerturbationRange = 2
        numHeightfieldRows = 100
        numHeightfieldColumns = 400
        
        # import plane from .obj file
        # filename = os.path.join(pybullet_data.getDataPath(), "plane.obj")

        # add sine in z axel
        heightfieldData = np.zeros((numHeightfieldRows, numHeightfieldColumns))
        # randimized sin wave, skip every other wave with cosine
        for i in range(numHeightfieldRows):
            for j in range(numHeightfieldColumns):
              
                # get sine
                sine_phase = j/7
                noise = -(0.2*heightPerturbationRange*np.sin(sine_phase)-(j*0.2)+np.cos(sine_phase)*heightPerturbationRange*2) 
                
                # get derivative of sin wave
                cosine_phase = j/56
                cosine = np.cos(cosine_phase) * heightPerturbationRange
                # if cosine > 0 and noise > 0:
                #     noise = 0
                
                # noise = max(0, noise)
                heightfieldData[i, j] = noise
        
        heightfieldData
        col,row = heightfieldData.shape
        heightfieldData = heightfieldData.reshape(-1)
        
        # meshScaling 
        meshScale = [0.5, 0.1, 0.4]
        # terrain tilt
        tilt_shape = [0.3, 0, 1, 0]
        # terrain position
        terrain_pos = [55, 0, 50]


        # create collision shape
        terrainShape = self._p.createCollisionShape(shapeType=self._p.GEOM_HEIGHTFIELD, 
        meshScale=meshScale,         
        heightfieldData=heightfieldData, 
        numHeightfieldRows=row, 
        numHeightfieldColumns=col)
        terrain = self._p.createMultiBody(0, terrainShape)

        # set terrain friction
        self._p.changeDynamics(terrain, -1, lateralFriction=0.0, spinningFriction=0.01, rollingFriction=0.01)
        self._p.changeDynamics(terrain, -1, restitution=0.0)
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION, 0)
        # set terrain white
        self._p.changeVisualShape(terrain, -1, rgbaColor=[1, 1, 1, 1])
        # change robot friction
        self._p.changeDynamics(self.ground_plane_mjcf[0], -1, lateralFriction=0.0)
        # remove reflection of the robot
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION, 0)
        # move terrain edge to the center
        self._p.resetBasePositionAndOrientation(terrain, terrain_pos, tilt_shape)
        for i in self.ground_plane_mjcf:
            self._p.changeDynamics(i, -1, lateralFriction=0.05, restitution=0.5)
            self._p.changeVisualShape(i, -1, rgbaColor=[1, 1, 1, 0.8])
            self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION, 0)
            self._p.resetBasePositionAndOrientation(i, [0, 0, 0], tilt_shape)

        self.terrain_plane = terrain
class SinglePlayerSlopeScene(SlopeScene):
  "This scene created by environment, to work in a way as if there was no concept of scene visible to user."
  multiplayer = False