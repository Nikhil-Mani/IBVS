import mujoco
import mujoco.viewer
import time
import vs
import numpy as np

model = mujoco.MjModel.from_xml_path("./assets/robotstudio_so101/scene_box.xml")
data = mujoco.MjData(model)

renderer_rgb = mujoco.Renderer(model, height=480, width=640)
renderer_depth = mujoco.Renderer(model, height=480, width=640)
renderer_depth.enable_depth_rendering()

mujoco.mj_resetData(model, data)

DESIRED_PTS = np.array([[270, 190], [370, 190], [370, 290], [270, 290]])
CONTROL_PARAMS = [500.0, [1.0, 0.05], model.opt.timestep, 2.0]

try:
    CAM_SITE_ID = model.site("wrist_cam").id
except KeyError:
    print("Warning: 'wrist_cam' site not found. Mappings will fail")
    CAM_SITE_ID = 0


def extract_features(rgb_image):
    # Placeholder return for now
    return np.array([260, 180], [380, 180], [380, 300], [260, 300])

# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        
        #Get rgb
        renderer_rgb.update_scene(data, "wrist_cam")
        rgb_image = renderer_rgb.render()

        # Get depth
        renderer_depth.update_scene(data, "wrist_cam")
        depth_map = renderer_depth.render()

        current_pts = extract_features(rgb_image)
        v_c = vs.ibvs_control(current_pts, depth_map, DESIRED_PTS, CONTROL_PARAMS)

        # cartesian -> joint
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        mujoco.mj_jacSite(model, data, jacp, jacr, CAM_SITE_ID)
        J = np.vstack((jacp, jacr))
        dq = np.linalg.pinv(J) @ v_c
        
        # integrate for position actuation:
        cur_pos = data.qpos[:model.nu]
        target_pos = cur_pos + (dq[:model.nu] * model.opt.timestamp)
        data.ctrl[:model.nu] = target_pos
        mujoco.mj_step(model, data)
        viewer.sync()
        remaining = model.opt.timestep - (time.time()-step_start)
        if(remaining > 0):
            time.sleep(remaining)
        