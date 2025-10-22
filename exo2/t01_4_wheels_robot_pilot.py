# type: ignore
import mujoco
from mujoco import viewer
import numpy as np
import time
import xml.etree.ElementTree as ET

# Global viewer reference for keyboard callback
viewer_instance = None
display_camera_info = False
quit_requested = False
speed = 0.8


def extract_robot_from_xml(xml_file_path):
    """
    Extract robot components from existing XML file.
    This teaches students how to parse and reuse XML components.
    """
    print(f"Extracting robot components from {xml_file_path}...")

    # Parse the existing XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Extract useful components
    components = {
        'compiler': None,
        'option': None,
        'default': None,
        'asset': None,
        'robot_body': None,
        'actuators': None
    }

    # Find and extract each component
    for child in root:
        if child.tag == 'compiler':
            components['compiler'] = child
        elif child.tag == 'option':
            components['option'] = child
        elif child.tag == 'default':
            components['default'] = child
        elif child.tag == 'asset':
            components['asset'] = child
        elif child.tag == 'worldbody':
            # Extract the robot body from worldbody
            for body in child:
                if body.get('name') == 'robot':
                    components['robot_body'] = body
        elif child.tag == 'actuator':
            components['actuators'] = child

    return components

def create_floor_element(floor_type="standard"):
    """
    Create floor element programmatically.
    """
    floor_configs = {
        "standard": {
            "material": "mat_floor_normal",
            "friction": "1.0 0.005 0.0001",
            "rgba": "0.8 0.9 0.8 1",
            "description": "Standard floor with normal friction"
        },
        "ice": {
            "material": "mat_floor_ice",
            "friction": "0.1 0.001 0.0001",
            "rgba": "0.9 0.95 1.0 1",
            "description": "Icy floor with very low friction"
        },
        "sand": {
            "material": "mat_floor_sand",
            "friction": "2.0 0.1 0.01",
            "rgba": "0.9 0.8 0.6 1",
            "description": "Sandy floor with high friction"
        }
    }

    config = floor_configs.get(floor_type, floor_configs["standard"])
    print(f"Creating {config['description']}")

    # Create floor geometry element - simple plane without texture attributes
    floor_geom = ET.Element('geom')
    floor_geom.set('name', 'dynamic_floor')
    floor_geom.set('type', 'plane')
    floor_geom.set('size', '100 100 0.1')
    floor_geom.set('pos', '0 0 -0.1')
    floor_geom.set('material', config['material'])
    floor_geom.set('friction', config['friction'])

    return floor_geom, config

def create_enhanced_materials():
    """
    Create enhanced materials with textures and visual appeal.
    Returns a list of material elements.
    """
    materials = []

    # Enhanced floor materials with textures and proper scaling
    floor_materials = [
        {
            'name': 'mat_floor_normal',
            'texture': 'tex_grid',
            'rgba': '0.7 0.8 1.0 0.8',  # Faint blue with transparency
            'shininess': '0.1',
            'specular': '0.3',
            'texrepeat': '50 50'  # 2x2 meter grid cells
        },
        {
            'name': 'mat_floor_ice',
            'texture': 'tex_grid',
            'rgba': '0.6 0.7 1.0 0.7',  # Slightly more blue for ice
            'shininess': '0.8',
            'specular': '0.9',
            'texrepeat': '50 50'  # 2x2 meter grid cells
        },
        {
            'name': 'mat_floor_sand',
            'texture': 'tex_grid',
            'rgba': '0.9 0.8 0.6 0.8',  # Keep sand yellowish
            'shininess': '0.1',
            'specular': '0.1',
            'texrepeat': '50 50'  # 2x2 meter grid cells
        }
    ]

    # Enhanced robot materials
    robot_materials = [
        {
            'name': 'mat_chassis_beige',
            'rgba': '0.96 0.87 0.70 1',  # Nice beige color
            'shininess': '0.3',
            'specular': '0.5'
        },
        {
            'name': 'mat_wheel_black',
            'texture': 'tex_wheel_radius',
            'rgba': '0.1 0.1 0.1 1',
            'shininess': '0.6',
            'specular': '0.3'
        }
    ]

    # Create material elements
    all_materials = floor_materials + robot_materials
    for mat_config in all_materials:
        material = ET.Element('material')
        material.set('name', mat_config['name'])
        material.set('rgba', mat_config['rgba'])
        if 'texture' in mat_config:
            material.set('texture', mat_config['texture'])
        if 'shininess' in mat_config:
            material.set('shininess', mat_config['shininess'])
        if 'specular' in mat_config:
            material.set('specular', mat_config['specular'])
        if 'texrepeat' in mat_config:
            material.set('texrepeat', mat_config['texrepeat'])
        materials.append(material)

    return materials

def create_textures():
    """
    Create texture elements for enhanced visuals.
    Returns a list of texture elements.
    """
    textures = []

    # Grid texture for floor - 2x2 meter cells
    grid_texture = ET.Element('texture')
    grid_texture.set('name', 'tex_grid')
    grid_texture.set('type', '2d')
    grid_texture.set('builtin', 'checker')
    grid_texture.set('rgb1', '0.7 0.8 1.0')    # Faint blue base
    grid_texture.set('rgb2', '0.75 0.85 1.0')  # Slightly lighter blue alternate
    grid_texture.set('width', '100')           # Texture resolution
    grid_texture.set('height', '100')          # Texture resolution
    grid_texture.set('mark', 'edge')           # White grid lines
    grid_texture.set('markrgb', '1 1 1')       # Pure white grid lines
    textures.append(grid_texture)

    # Wheel radius texture to show rotation
    wheel_texture = ET.Element('texture')
    wheel_texture.set('name', 'tex_wheel_radius')
    wheel_texture.set('type', '2d')
    wheel_texture.set('builtin', 'checker')
    wheel_texture.set('rgb1', '0.8 0.2 0.2')  # Red spoke
    wheel_texture.set('rgb2', '0.1 0.1 0.1')  # Black tire
    wheel_texture.set('width', '32')
    wheel_texture.set('height', '32')
    wheel_texture.set('mark', 'random')
    wheel_texture.set('markrgb', '0.8 0.8 0.2')  # Yellow marks
    textures.append(wheel_texture)

    return textures

def enhance_robot_visuals(robot_body):
    """
    Enhance robot body with better materials and visual features.
    Also adjusts wheel positions to be outside the robot body.
    Modifies the robot body XML element in place.
    """
    if robot_body is None:
        return

    print("Enhancing robot visuals...")

    # Find and update chassis material
    for geom in robot_body.iter('geom'):
        if geom.get('name') == 'chassis':
            geom.set('material', 'mat_chassis_beige')
            print("  Updated chassis to beige color")

    # Find and update wheel materials (positions now defined in XML)
    wheel_count = 0

    for body in robot_body.iter('body'):
        body_name = body.get('name', '')
        if body_name.startswith('wheel_'):
            # Update wheel material
            for geom in body.iter('geom'):
                if geom.get('name', '').startswith('geom_'):
                    geom.set('material', 'mat_wheel_black')
                    wheel_count += 1

            # Log the wheel position (already set in XML)
            current_pos = body.get('pos', '0 0 0')
            print(f"  {body_name} at position: {current_pos}")

    print(f"  Updated {wheel_count} wheels with textured black material")
    print("  Wheel positions defined in XML (no runtime modification)")

## Ex. 01: Create combined model with enhanced visuals
### Add gravity

def build_combined_model(robot_components, floor_type="standard", robot_height=1.0):
    """
    Build a complete MuJoCo model by combining robot components with programmatic floor.
    Environment controls physics settings (gravity, timestep, etc).

    Args:
        robot_components: Extracted robot parts from XML
        floor_type: Type of floor to create ("standard", "ice", "sand")
        robot_height: Starting height of robot above floor (meters)
    """
    print("Building combined model...")
    print(f"Robot starting height: {robot_height}m above floor")

    # Create root mujoco element
    root = ET.Element('mujoco')
    root.set('model', 'robot_with_programmatic_floor')

    # Add compiler settings from robot XML
    if robot_components['compiler'] is not None:
        root.append(robot_components['compiler'])

    # CREATE ENVIRONMENT-CONTROLLED PHYSICS SETTINGS (override robot's settings)
    option = ET.Element('option')
    option.set('timestep', '0.01')
    ## Ex. 01: Add gravity
    ### TODO COMPLETE here
    option.set("gravity", "0 0 -9.0")
    option.set('solver', 'Newton')
    option.set('iterations', '50')
    root.append(option)
    print("  Environment physics: gravity enabled, timestep=0.01s")

    # Add size settings
    size = ET.Element('size')
    size.set('njmax', '1000')
    size.set('nconmax', '500')
    root.append(size)

    if robot_components['default'] is not None:
        root.append(robot_components['default'])

    # Create asset section with textures and enhanced materials
    asset = ET.Element('asset')

    # Add textures first
    textures = create_textures()
    for texture in textures:
        asset.append(texture)

    # Add enhanced materials (including floor and robot materials)
    enhanced_materials = create_enhanced_materials()
    for material in enhanced_materials:
        asset.append(material)

    # Also keep any original materials from robot XML if they exist
    if robot_components['asset'] is not None:
        for original_material in robot_components['asset']:
            # Only add if it's not already in our enhanced materials
            material_name = original_material.get('name', '')
            enhanced_names = [mat.get('name', '') for mat in enhanced_materials]
            if material_name not in enhanced_names:
                asset.append(original_material)

    root.append(asset)

    # Create worldbody with floor and robot
    worldbody = ET.Element('worldbody')

    # Add programmatic floor
    floor_geom, floor_config = create_floor_element(floor_type)
    ## Ex. 01: Add floor to worldbody
    worldbody.append( floor_geom )

    #
    print(f"DEBUG | robot_components = {robot_components}")

    # Add robot body with enhanced visuals and adjusted height
    if robot_components['robot_body'] is not None:

        # Enhance the robot's visual appearance
        ## Ex. 01: Enhance robot visuals
        ### TODO COMPLETE here
        enhance_robot_visuals(robot_components['robot_body'])

        # Adjust robot starting height (floor is at -0.1, so robot center should be at robot_height - 0.1)
        robot_z_position = robot_height - 0.1 - 10  # Floor offset
        current_pos = robot_components['robot_body'].get('pos', '0 0 0.2')
        pos_parts = current_pos.split()
        if len(pos_parts) == 3:
            new_pos = f"{pos_parts[0]} {pos_parts[1]} {robot_z_position}"
            robot_components['robot_body'].set('pos', new_pos)
            print(f"  Robot positioned at: {new_pos} (will fall {robot_height}m to floor)")

        ## Ex. 01: Add robot body to worldbody
        ### TODO COMPLETE here
        worldbody.append(robot_components['robot_body'])


    ## Ex. 01: Add worldbody to root
    ### TODO COMPLETE here
    root.append( worldbody )

    # Add actuators
    if robot_components['actuators'] is not None:
        root.append(robot_components['actuators'])

    #
    with open("b.xml", "w") as f:
        #
        f.write(ET.tostring(worldbody, encoding='unicode'))

    # Convert to XML string
    xml_string = ET.tostring(root, encoding='unicode')

    #
    with open("a.xml", "w") as f:
        #
        f.write(xml_string)

    # Create and return MuJoCo model
    ## Ex. 01: Create and return MuJoCo model
    ### TODO modify return statement here
    return mujoco.MjModel.from_xml_string(xml_string)


# LESSON: Show students how to compose models from existing components
print("=== LESSON: Enhanced Model Composition with Visuals ===")
print("1. Loading robot components from XML file...")
print("2. Creating textured floor programmatically...")
print("3. Adding visual enhancements (colors, textures)...")
print("4. Combining components into new model...")
print()

# Extract robot from existing XML file
robot_components = extract_robot_from_xml("four_wheels_robot.xml")

def demonstrate_visual_options():
    """
    Show students the different visual options available.
    """
    print("=== AVAILABLE VISUAL OPTIONS ===")
    print("Floor features:")
    print("  • Faint blue color with transparency")
    print("  • White grid lines with 2x2 meter cells")
    print("  • Professional appearance with realistic lighting")
    print()
    print("Floor types:")
    print("  'standard' - Faint blue grid with normal friction")
    print("  'ice'      - Slightly more blue with low friction (slippery)")
    print("  'sand'     - Yellow tinted grid with high friction")
    print()
    print("Robot features:")
    print("  • Beige chassis (warm, natural color)")
    print("  • Black wheels with rotation indicators")
    print("  • Wheels positioned OUTSIDE robot body for better visibility")
    print("  • 15cm wheel extension from body sides")
    print()
    print("To change floor type, modify the 'floor_type' variable above!")
    print("=" * 50)
    print()

# Choose floor type
floor_type = "standard"
## Ex. 03 play with different floor types
### TODO Change floor type and experiment here
# Try: "standard", "ice", "sand"

# Choose robot starting height (for gravity demonstration)
h = 2.0  # Robot will start 2 meters above floor and fall down

# Show visual options to students
demonstrate_visual_options()

print(f"=== PHYSICS DEMONSTRATION ===")
print(f"Robot will start {h}m above the floor")
print("Watch it fall due to gravity - this proves environment physics work!")
print()

# Build combined model with enhanced visuals and physics
m_combined = build_combined_model(robot_components, floor_type, robot_height=h)
d = mujoco.MjData(m_combined)

# For convenience, use shorter variable name in the rest of the code
m = m_combined

print("\n")
print("=== VISUAL ENHANCEMENTS COMPLETE ===")
print(f"✅ Created model with {floor_type} floor")
print("✅ Floor: Faint blue with white 2x2m grid cells")
print("✅ Robot chassis: Beige color")
print("✅ Robot wheels: Black with rotation indicators")
print("✅ Wheels positioned outside robot body (defined in XML)")
print("✅ No XML strings embedded in Python code!")
print()

def set_wheel_speeds(d, w_fl, w_fr, w_rl, w_rr):
    # actuator order matches XML order above
    # Clamp values to safe range to prevent instability
    max_speed = 1.0
    d.ctrl[:] = [
        np.clip(w_fl, -max_speed, max_speed),
        np.clip(w_fr, -max_speed, max_speed),
        np.clip(w_rl, -max_speed, max_speed),
        np.clip(w_rr, -max_speed, max_speed)
    ]

def check_wheel_spinning(d,m):
    # Add this in the simulation loop to monitor wheel speeds:
    wheel_joints = ['hinge_fl', 'hinge_fr', 'hinge_rl', 'hinge_rr']
    for i, joint_name in enumerate(wheel_joints):
        joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            # qvel index for hinge joints starts after the free joint (6 DOFs)
            qvel_addr = m.jnt_dofadr[joint_id]
            print(f"{joint_name}: vel={d.qvel[qvel_addr]:.3f} rad/s, ctrl={d.ctrl[i]:.3f}")


## Ex. 02: add robot controls (via keys):
### go straight: 'g' or 'G'
### go backward: 'b' or 'B'
### turn left: 't' or 'T'
### stop: ' ' (space)
### TODO COMPLETE here

def key_callback(keycode):
    global viewer_instance, display_camera_info, quit_requested, d
    # print (f"Key pressed: {keycode}")
    if keycode == ord('c') or keycode == ord('C'):
        print("\nCamera info requested:")
        display_camera_info = True
    elif keycode == ord('q') or keycode == ord('Q'):
        print("\nQuitting...")
        quit_requested = True
    elif keycode == 256:  # ESC key
        print("\nESC pressed - Quitting...")
        quit_requested = True
    ## Ex. 02: add robot controls (via keys)
    ### TODO COMPLETE here
    if keycode == ord('o') or keycode == ord('O'):
        set_wheel_speeds(d, +speed, +speed, +speed, +speed)
    ### TODO COMPLETE here
    if keycode == ord('k') or keycode == ord('K'):
        set_wheel_speeds(d, +(speed * 0.75), -(speed * 0.75), +(speed * 0.75), -(speed * 0.75))
    ### TODO COMPLETE here
    if keycode == ord('l') or keycode == ord('L'):
        set_wheel_speeds(d, -speed, -speed, -speed, -speed)
    ### TODO COMPLETE here
    if keycode == ord('j') or keycode == ord('J'):
        set_wheel_speeds(d, -(speed * 0.75), +(speed * 0.75), -(speed * 0.75), +(speed * 0.75))
    if keycode == 32: # Espace
        set_wheel_speeds(d, 0, 0, 0, 0)


# Initialize simulation with zero velocities for stability
d.qvel[:] = 0  # Zero all velocities
d.qacc[:] = 0  # Zero all accelerations

# See what DOF 0 represents:
print("\n=== MODEL DEBUG INFO ===")
print(f"Total DOFs: {m.nv}")
print(f"Joint names: {[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]}")
print(f"DOF names: {[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_DOF, i) for i in range(m.nv)]}")
print(f"First few qpos: {d.qpos[:7]}")  # First 7 DOFs (free joint = 7: 3 pos + 4 quat)
print(f"First few qvel: {d.qvel[:6]}")  # First 6 velocities (free joint = 6: 3 linear + 3 angular)

print("\n=== GRAVITY & PHYSICS DEBUG ===")
print(f"Gravity setting: {m.opt.gravity}")
print(f"Timestep: {m.opt.timestep}")
print(f"Solver: {m.opt.solver}")

print("\n=== GEOMETRY DEBUG ===")
print(f"Total geometries: {m.ngeom}")
for i in range(m.ngeom):
    geom_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
    geom_type = m.geom_type[i]
    geom_size = m.geom_size[i]
    geom_pos = m.geom_pos[i]
    print(f"  Geom {i}: {geom_name}, type={geom_type}, size={geom_size}, pos={geom_pos}")
print("\n")

# Check if wheels are touching the floor:

print("=== CONTACT DEBUG ===")
print(f"Number of contacts: {d.ncon}")
for i in range(d.ncon):
    contact = d.contact[i]
    geom1_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
    geom2_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
    print(f"Contact {i}: {geom1_name} <-> {geom2_name}")
print("\n")

set_wheel_speeds(d, 0, 0, 0, 0)

with viewer.launch_passive(m, d, key_callback=key_callback) as v:
    # Configure camera for a good view of the robot
    cam = v.cam
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.fixedcamid = -1
    cam.lookat[:] = [0.0, 0.0, 0.2]  # Look at robot height
    cam.azimuth = 135  # Angled view
    cam.elevation = -20  # Slightly above
    cam.distance = 8  # Good distance to see the robot

    print(f"\n=== SIMULATION START ===")
    print(f"Floor type: {floor_type}")
    print("Robot visualization started!")
    print("You should see the robot moving on the programmatically created floor...")
    print("Notice: The robot is loaded from XML, the floor is created in Python!")
    print("Press 'C' to display camera info, 'Q' or 'ESC' to quit.")
    print("Press 'G' to go straight, 'T' to turn left, or SPACE to stop.")
    print("Press 'B' to go backward.")
    print()

    start_time = time.time()
    step_count = 0
    while v.is_running():
        mujoco.mj_step(m, d)
        v.sync()
        step_count += 1

        # check_wheel_spinning(d,m)

        # Check for instability and break if detected
        if np.any(np.isnan(d.qacc)) or np.any(np.isinf(d.qacc)):
            print(f"Simulation became unstable at step {step_count}")
            break

        # Handle keyboard requests
        if quit_requested:
            print("Quit requested via keyboard.")
            break
        elif display_camera_info:
            print(f"Camera position: {cam.pos}, lookat: {cam.lookat}, distance: {cam.distance}, azimuth: {cam.azimuth}, elevation: {cam.elevation}")
            display_camera_info = False
        ## Ex. 02: add robot controls (via keys)
        ### TODO COMPLETE here

    if not v.is_running():
        exit()

print("Simulation completed successfully!")

