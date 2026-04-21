import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path("./assets/robotstudio_so101/so101.xml")
data = mujoco.MjData(model)

# Launch the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)