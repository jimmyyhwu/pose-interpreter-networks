from math import pi

import bpy

def clear_scene():
    for obj in bpy.data.objects:
        if obj.name == 'Camera':
            obj.select = False
        else:
            obj.select = True
    bpy.ops.object.delete()

def place_lamp():
    lamp_data = bpy.data.lamps.new(name='lamp', type='POINT')
    lamp = bpy.data.objects.new(name='lamp', object_data=lamp_data)
    bpy.context.scene.objects.link(lamp)
    lamp.location = (0, 0, 0)

def setup_camera(camera_parameters, camera_scale):
    bpy.data.objects['Camera'].location = (0, 0, 0)
    bpy.data.objects['Camera'].rotation_euler = (0, pi, pi)
    width = camera_scale * camera_parameters['width']
    height = camera_scale * camera_parameters['height']
    f = camera_scale * (camera_parameters['f_x'] + camera_parameters['f_y']) / 2.0
    p_x = camera_scale * camera_parameters['p_x']
    p_y = camera_scale * camera_parameters['p_y']
    camera = bpy.data.cameras['Camera']
    camera.lens = 1
    camera.sensor_width = width / f
    camera.shift_x = 0.5 - p_x / width
    camera.shift_y = (p_y - 0.5 * height) / width

def load_model(model_path):
    bpy.ops.import_mesh.stl(filepath=model_path)
    bpy.context.object.location = (0, 0, 0.5)
    bpy.context.object.rotation_mode = 'QUATERNION'
    bpy.context.object.rotation_quaternion = (1, 0, 0, 0)
    bpy.context.object.data.materials.append(None)

def create_object_material():
    object_material = bpy.data.materials.new(name='object')
    object_material.use_shadeless = False

def create_mask_material():
    mask_material = bpy.data.materials.new(name='mask')
    mask_material.use_shadeless = True
    mask_material.translucency = 1.0

def set_render_layers_output():
    tree = bpy.context.scene.node_tree
    render_layers = tree.nodes['Render Layers']
    composite = tree.nodes['Composite']
    tree.links.new(render_layers.outputs[0], composite.inputs[0])

def init(model_path, camera_parameters, camera_scale=0.5):
    clear_scene()
    bpy.context.scene.render.resolution_x = int(camera_scale * camera_parameters['width'])
    bpy.context.scene.render.resolution_y = int(camera_scale * camera_parameters['height'])
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.use_nodes = True
    bpy.data.worlds['World'].horizon_color = (0, 0, 0)
    place_lamp()
    setup_camera(camera_parameters, camera_scale)
    load_model(model_path)
    create_object_material()
    create_mask_material()
    set_render_layers_output()

def set_mode_object():
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGB'
    bpy.context.scene.render.image_settings.color_depth = '8'
    bpy.context.scene.render.use_antialiasing = True
    bpy.context.object.data.materials[0] = bpy.data.materials['object']

def set_mode_mask():
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'BW'
    bpy.context.scene.render.image_settings.color_depth = '8'
    bpy.context.scene.render.use_antialiasing = False
    bpy.context.object.data.materials[0] = bpy.data.materials['mask']

def set_object_pose(position, orientation):
    bpy.context.object.location = position
    bpy.context.object.rotation_quaternion = orientation

def render(output_path):
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
