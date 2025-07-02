import os
import numpy as np

# Add to render correctly
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['figure.dpi'] = 100

import matplotlib.pyplot as plt
import gymnasium as gym
import mujoco
import cv2
import yaml
from skimage.transform import resize
import trimesh
import mimoEnv.utils as env_utils
import sys
sys.path.append(".")
sys.path.append("..")
from babybench.build_xml import build

ENVS = {
    'none': 'BabyBench', 
    'self_touch': 'BabyBench-SelfTouch',
    'hand_regard': 'BabyBench-HandRegard',
}

EPS = 1e-6

def make_env(config=None, training=True):
    scene_path = os.path.join(config['save_dir'],'scene.xml')
    if training is True:
        make_save_dirs(config['save_dir'])
        scene_xml = build(config, path_to_assets=os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../MIMo/mimoEnv/assets')))
        with open(scene_path, 'w') as f:
            print(scene_xml, file=f)
    else:
        assert os.path.exists(config['save_dir']), "Save directory does not exist, did you run training with this config?"
        assert os.path.exists(scene_path), "Scene file does not exist, did you run training with this config?"
    env = gym.make(
        ENVS[config['behavior']],
        model_path=os.path.abspath(scene_path),
        actuation_model=config['actuation_model'],
        max_episode_steps=config['max_episode_steps'],
        frame_skip=config['frame_skip'],
        width=config['render_size'],
        height=config['render_size'],
        config=config,
        training=training,
    )
    return env

def render(env, camera="corner"):
    img = env.mujoco_renderer.render(render_mode="rgb_array", camera_name=camera)
    return img.astype(np.uint8)

def evaluation_img(env, up='side2', down='top'):
    img = np.zeros((480,720,3))
    img_corner = render(env, "corner")
    img[:,:480,:] = img_corner
    # Down-right rendering
    if down in ['top', 'side1', 'side2', 'closeup']:
        img_down = render(env, down)
        img[240:,480:,:] = img_down[::2,::2,:]
    elif down == 'binocular':
        img[240:,480:,:] =  view_binocular(env)
    elif down == 'touches_with_hands':
        img[240:,480:,:] = view_touches(env, contact_with='hands')
    # Up-right rendering
    if up in ['top', 'side1', 'side2', 'closeup']:
        img_top = render(env, up)
        img[:240,480:,:] = img_top[::2,::2,:] 
    elif up == 'closeup':
        img_close = render(env, "closeup")
        img[:240,480:,:] = img_close[::2,::2,:]
    elif up == 'touches_with_hands':
        img[:240,480:,:] = view_touches(env, contact_with='hands')
    elif up == 'binocular':
        img[:240,480:,:] = view_binocular(env)
    return img.astype(np.uint8)

def evaluation_video(images, save_name=None, frame_rate=60, resolution=((720,480))):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_name, fourcc, frame_rate, resolution)
    for img in images:
        video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    video.release()

def view_binocular(env):
    img_left_eye = render(env, "eye_left")
    img_right_eye = render(env, "eye_right")
    stereo = np.zeros((240,240,3))
    stereo[:,:,0] = to_grayscale(img_left_eye[::2,::2,:])
    stereo[:,:,1] = to_grayscale(img_right_eye[::2,::2,:])
    stereo[:,:,2] = to_grayscale(img_right_eye[::2,::2,:])
    return stereo
    
def view_touches(env, focus_body='hip', contact_with=None):
    root_id = env_utils.get_body_id(env.model, body_name='mimo_location')
    points_no_contact = []
    points_contact = []
    contact_magnitudes = []
    # Go through all bodies and note their child bodies
    subtree = env_utils.get_child_bodies(env.model, root_id)
    for body_id in subtree:
        if (body_id in env.touch.sensor_positions) and (body_id in env.touch.sensor_outputs):
            sensor_points = env.touch.sensor_positions[body_id]
            force_vectors = env.touch.sensor_outputs[body_id]
            force_magnitude = np.linalg.norm(force_vectors, axis=-1, ord=2)
            no_touch_points = sensor_points[force_magnitude <= 1e-7]
            touch_points = sensor_points[force_magnitude > 1e-7]
            no_touch_points = env_utils.body_pos_to_world(env.data, position=no_touch_points, body_id=body_id)
            touch_points = env_utils.body_pos_to_world(env.data, position=touch_points, body_id=body_id)

            points_no_contact.append(no_touch_points)
            points_contact.append(touch_points)
            contact_magnitudes.append(force_magnitude[force_magnitude > 1e-7])

    points_gray = np.concatenate(points_no_contact)
    points_red = np.concatenate(points_contact)
    forces = np.concatenate(contact_magnitudes)
    if len(forces) > 0:
        size_min = 5
        size_max = 10
        sizes = forces / np.amax(forces) * (size_max - size_min) + size_min
        opacity_min = 0.4
        opacity_max = 0.5
        opacities = forces / np.amax(forces) * (opacity_max - opacity_min) + opacity_min
        # Opacities can't be set as an array, so must be set using color array
        red_colors = np.tile(np.array([1.0, 0, 0, 0]), (points_red.shape[0], 1))
        red_colors[:, 3] = opacities
    else:
        sizes = 5
        red_colors = [0.4,0,0]

    if focus_body:
        target_pos = env.data.body(focus_body).xpos
    else:
        target_pos = np.zeros((3,))

    # Subtract all by ball position to center on ball
    xs_gray = points_gray[:, 0] - target_pos[0]
    ys_gray = points_gray[:, 1] - target_pos[1]
    zs_gray = points_gray[:, 2] - target_pos[2]

    xs_red = points_red[:, 0] - target_pos[0]
    ys_red = points_red[:, 1] - target_pos[1]
    zs_red = points_red[:, 2] - target_pos[2]

    fig = plt.figure(figsize=(6,6), dpi=100)
    #print("fig dpi:", fig.dpi)
    #print("fig canvas width, height:", fig.canvas.get_width_height())
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=90, azim=0, roll=0)
    # Draw sensor points
    ax.scatter(xs_gray, ys_gray, zs_gray, color="k", s=10, depthshade=False, alpha=.15)
    ax.scatter(xs_red, ys_red, zs_red, color=red_colors, s=sizes, depthshade=False)
    ax.set_xlim([-0.75, 0.75])     
    ax.set_ylim([-0.75, 0.75])     
    ax.set_zlim([-0.75, 0.75])     
    ax.set_axis_off()

    # Draw contact points
    if contact_with is not None:
        if contact_with == 'hands':
            contact_checks = np.concatenate([env.right_hand_geoms, env.left_hand_geoms])

        contacts = env.data.contact
        for idx in range(len(contacts.geom1)):
            # Check if hands in contact
            if (contacts.geom1[idx] in contact_checks) and (contacts.geom2[idx] in env.mimo_geoms) \
            or (contacts.geom2[idx] in contact_checks) and (contacts.geom1[idx] in env.mimo_geoms):
                contact_position = contacts.pos[idx]
                ax.scatter(contact_position[0], contact_position[1], contact_position[2],
                           color="y", s=20, depthshade=True, alpha=0.8)
            
    fig.canvas.draw()
    plt.close()

    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image_from_plot = image_from_plot[160:160+240,195:195+240,:]
    #image_from_plot = (resize(image_from_plot, (240,240,3))*255).astype(np.uint8)
    return image_from_plot

def to_grayscale(x):
    return 0.2989*x[:,:,0] + 0.5870*x[:,:,1] + 0.1140*x[:,:,2]

def make_save_dirs(save_dir):
    make_dir(save_dir)
    dirs = ['logs','videos']
    for dir_name in dirs:
        make_dir(f'{save_dir}/{dir_name}')

def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)