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
        if (self.slopeLoaded == 0):
            self.slopeLoaded = 1

            filename = os.path.join(pybullet_data.getDataPath(), "plane_stadium.sdf")
            self.ground_plane_mjcf = self._p.loadSDF(filename)

            # procedurally generate a slope-like plane

        # self.generate_sine_plane()
        # else:
        #     filename = os.path.join(pybullet_data.getDataPath(), "plane_stadium.sdf")
        #     self.ground_plane_mjcf = self._p.loadSDF(filename)
        #     self.regenerate_sine_plane()
        for i in self.ground_plane_mjcf:
            self._p.changeVisualShape(i, -1, rgbaColor=[1, 1, 1, 0.0])
            self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION, 0)
        
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS, 1)
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION, 1)
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        self._p.configureDebugVisualizer(pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
        
    def generate_sine_plane(self, steepness=0.3, amplitude=0.3, frequency=0.3):
        # clear previous planes
        if self.terrain_plane != None:
            self._p.removeBody(self.terrain_plane)
            # remove collision shape
            self._p.removeCollisionShape(self.terrainShape)
            self.terrain_plane = None
        # TODO: add to args
        numHeightfieldRows = 50
        numHeightfieldColumns = 1000
        
        # import plane from .obj file
        # filename = os.path.join(pybullet_data.getDataPath(), "plane.obj")
        
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
        
        terrain_pos = [0,0,0]
        
        # create collision shape
        self.terrainShape = self._p.createCollisionShape(shapeType=self._p.GEOM_HEIGHTFIELD, 
            meshScale=meshScale,         
            heightfieldData=heightfieldData, 
            numHeightfieldRows=row, 
            numHeightfieldColumns=col)
            
        # create visual shape from terrainShape
        # terrainVisualShape = self._p.createVisualShape(shapeType=self._p.GEOM_MESH, 
        #     rgbaColor=[1, 1, 1, 1],
        #     specularColor=[0.4, .4, 0],
        #     meshScale=meshScale)
        


        terrain = self._p.createMultiBody(0, 
            baseCollisionShapeIndex=self.terrainShape,
            # baseVisualShapeIndex=terrainVisualShape, 
            physicsClientId=self._p._client)
        # # get slope angle from points
        # slope_angle = np.arctan((heightfieldData[0] - heightfieldData[-1])/(numHeightfieldColumns*meshScale[0]))*180/np.pi
        # print("slope angle: ", slope_angle)
        
        # # set terrain friction
        self._p.changeDynamics(terrain, -1, lateralFriction=0.0, spinningFriction=0.0, rollingFriction=0.0)
        self._p.changeDynamics(terrain, -1, restitution=0.0)
        # self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION, 0)
        # # set terrain white
        # self._p.changeVisualShape(terrain, -1, rgbaColor=[1, 1, 1, 1])
        # # change robot friction
        # self._p.changeDynamics(self.ground_plane_mjcf[0], -1, lateralFriction=0.0)
        # # remove reflection of the robot
        # self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION, 0)

        self._p.resetBasePositionAndOrientation(terrain, terrain_pos, tilt_shape)
        for i in self.ground_plane_mjcf:
            self._p.changeDynamics(i, -1, lateralFriction=0.05, restitution=0.5)
            self._p.changeVisualShape(i, -1, rgbaColor=[1, 1, 1, 1])
            self._p.configureDebugVisualizer(pybullet.COV_ENABLE_PLANAR_REFLECTION, 0)
            self._p.resetBasePositionAndOrientation(i, [0, 0, -1000], tilt_shape)
        self.terrain_plane = terrain
        self._p.changeVisualShape(self.terrain_plane, -1, rgbaColor=[1,1,1, 1])
        # add "chess.png" texture to the plane from absolute path
        chess_path = os.path.join(os.path.dirname(__file__), "chess.png")
        textureId = self._p.loadTexture(chess_path)
        self._p.changeVisualShape(terrain, -1, textureUniqueId=textureId)
    
class SinglePlayerSlopeScene(SlopeScene):
  "This scene created by environment, to work in a way as if there was no concept of scene visible to user."
  multiplayer = False
  terrain_plane = None