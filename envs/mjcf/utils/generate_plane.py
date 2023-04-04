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
    
    def episode_restart(self, bullet_client, difficulty=0):
        self.difficulty = difficulty
        self._p = bullet_client
        Scene.episode_restart(self, bullet_client)  # contains cpp_world.clean_everything()
        if (True):
            self.slopeLoaded = 1

            filename = os.path.join(pybullet_data.getDataPath(), "plane_stadium.sdf")
            self.ground_plane_mjcf = self._p.loadSDF(filename)
        
            # procedurally generate a slope-like plane
            self.generate_sine_plane()

    def generate_sine_plane(self):
        # clear previous planes
        if self.terrain_plane != None:
            # get id of plane
            self._p.removeBody(self.terrain_plane)
            self.terrain_plane = None
        print("generating sine plane")
        # TODO: add to args
        numHeightfieldRows = 100
        numHeightfieldColumns = 1000
        
        # import plane from .obj file
        # filename = os.path.join(pybullet_data.getDataPath(), "plane.obj")
        
        bin_index = self.difficulty

        # sample float steepness from bin range
        steepness_bins = [[0.1, 0.15],[0.15, 0.2],[0.2, 0.25],[0.25,0.3], [0.3, 0.35]]
        steepness = np.random.uniform(steepness_bins[4][0], steepness_bins[4][1])

        # frequency from 0 to 7
        frequency_bins = [[9, 8],[8, 7],[8, 6],[8, 5],[10, 6]]
        frequency = np.random.uniform(frequency_bins[bin_index][0], frequency_bins[bin_index][1])

        # sample float amplitude from bin range
        amplitude_bins = [[0.1, 0.5],[0.5, 1],[1, 1.5],[1.5, 2], [2, 3]]
        amplitude = np.random.uniform(amplitude_bins[bin_index][0], amplitude_bins[bin_index][1])
        print("steepness: ", steepness)
        print("frequency: ", frequency)
        print("amplitude: ", amplitude)
        # add sine in z axel
        heightfieldData = np.zeros((numHeightfieldRows, numHeightfieldColumns))
        # randimized sin wave, skip every other wave with cosine
        for i in range(numHeightfieldRows):
            for j in range(numHeightfieldColumns):
                # get sine
                sine_phase = j/frequency
                terrain_value = -np.sin(sine_phase) * amplitude
                
                heightfieldData[i, j] = terrain_value - j*steepness
        
        heightfieldData
        col,row = heightfieldData.shape
        heightfieldData = heightfieldData.reshape(-1)
        
        # meshScaling 
        meshScale = [0.3, 0.05, 1]
        # terrain tilt
        tilt_shape = [0, 0, 0, 1]
        # terrain position
        terrain_pos = [0, 0, 500]


        # create collision shape
        terrainShape = self._p.createCollisionShape(shapeType=self._p.GEOM_HEIGHTFIELD, 
        meshScale=meshScale,         
        heightfieldData=heightfieldData, 
        numHeightfieldRows=row, 
        numHeightfieldColumns=col)
        terrain = self._p.createMultiBody(0, terrainShape)
        # get slope angle from points
        slope_angle = np.arctan((heightfieldData[0] - heightfieldData[-1])/(numHeightfieldColumns*meshScale[0]))*180/np.pi
        print("slope angle: ", slope_angle)
        
        # set terrain friction
        self._p.changeDynamics(terrain, -1, lateralFriction=0.0, spinningFriction=0.00, rollingFriction=0.00)
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
            self._p.changeVisualShape(i, -1, rgbaColor=[1, 1, 1, 0.0])
            self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION, 0)
            # self._p.resetBasePositionAndOrientation(i, [0, 0, 0], tilt_shape)

        self.terrain_plane = terrain
class SinglePlayerSlopeScene(SlopeScene):
  "This scene created by environment, to work in a way as if there was no concept of scene visible to user."
  multiplayer = False
  terrain_plane = None