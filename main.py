import numpy as np
import taichi as ti

ti.init(arch=ti.cuda)

class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur

dimension = 2
resolution = [512, 512]
v_pair = TexPair(ti.Vector.field(2, ti.f32, shape=resolution),
                 ti.Vector.field(2, ti.f32, shape=resolution))
color_pair = TexPair(ti.Vector.field(3, ti.f32, shape=resolution),
                     ti.Vector.field(3, ti.f32, shape=resolution))
f_temp_1 = ti.field(ti.f32, shape=resolution)
f_temp_2 = ti.field(ti.f32, shape=resolution)
f_temp_for_solving_laplacian_equation = ti.field(ti.f32, shape=resolution)
f_v_temp_1 = ti.Vector.field(2, ti.f32, shape=resolution)

domain = (ti.Vector([-1., -1.]), ti.Vector([1., 1.]))

fluid_background_velocity = ti.Vector([0., 0.])  # [2, 0]

box_position = ti.Vector([-0.3, 0., 0.])
box_shape = ti.Vector([0.2, 0.2])
box_velocity = ti.Vector([0., 0., 0.])
box_mass = 5.
box_I = box_shape.norm_sqr()*box_mass/3  # ∫[-a, a]×[-b, b] x^2+y^2 ρ dx dy
box_color = 0xffffff

dt = 0.02
laplacian_steps = 500

def paint_solid():
    cos = np.cos(box_position.z)
    sin = np.sin(box_position.z)
    box_points = (np.array([[(box_position.x+(-box_shape.x*cos+box_shape.y*sin)-domain[0].x)/(domain[1].x-domain[0].x),
                             (box_position.y+(-box_shape.x*sin-box_shape.y*cos)-domain[0].y)/(domain[1].y-domain[0].y)]]),
                  np.array([[(box_position.x+(+box_shape.x*cos+box_shape.y*sin)-domain[0].x)/(domain[1].x-domain[0].x),
                             (box_position.y+(+box_shape.x*sin-box_shape.y*cos)-domain[0].y)/(domain[1].y-domain[0].y)]]),
                  np.array([[(box_position.x+(+box_shape.x*cos-box_shape.y*sin)-domain[0].x)/(domain[1].x-domain[0].x),
                             (box_position.y+(+box_shape.x*sin+box_shape.y*cos)-domain[0].y)/(domain[1].y-domain[0].y)]]),
                  np.array([[(box_position.x+(-box_shape.x*cos-box_shape.y*sin)-domain[0].x)/(domain[1].x-domain[0].x),
                             (box_position.y+(-box_shape.x*sin+box_shape.y*cos)-domain[0].y)/(domain[1].y-domain[0].y)]]))
    gui.triangles(a=box_points[0],
                  b=box_points[1],
                  c=box_points[2],
                  color=box_color)
    gui.triangles(a=box_points[2],
                  b=box_points[3],
                  c=box_points[0],
                  color=box_color)
    
def solid_move():
    global box_position
    box_position += box_velocity*dt

@ti.func
def get_fluid_default_color(p):
    # brightness = ti.math.mod(ti.math.floor(p.x*20.)+ti.math.floor(p.y*20.), 2)
    brightness = .5
    return ti.Vector([brightness, brightness, brightness])

@ti.func
def indices_to_coordinates(i, j):
    return ti.Vector([domain[0].x+i*(domain[1].x-domain[0].x)/resolution[0],
                      domain[0].y+j*(domain[1].y-domain[0].y)/resolution[1]])

@ti.func
def coordinates_to_indices(x, y):
    return ti.Vector([(x-domain[0].x)/(domain[1].x-domain[0].x)*resolution[0],
                      (y-domain[0].y)/(domain[1].y-domain[0].y)*resolution[1]])


@ti.kernel
def init_fluid():
    for i, j in v_pair.cur:
        # if (i-128)*(i-128)+(j-128)*(j-128)<400:
        #     v_pair.cur[i, j] = ti.Vector([1., 0.])
        #     color_pair.cur[i, j] = ti.Vector([0., 0., 1.])
        # else:
            v_pair.cur[i, j] = fluid_background_velocity
            color_pair.cur[i, j] = get_fluid_default_color(indices_to_coordinates(i, j))

@ti.func
def lerp(a, b, f):
    return a+f*(b-a)

@ti.func
def bilerp(a, b, c, d, f, g):
    return lerp(lerp(a, b, f), lerp(c, d, f), g)

@ti.func
def sample_with_padding(v, i, j):  # Sample a field. i and j are floats.
    if i<0.:
        i = 0.
    if i>resolution[0]-1:
        i = resolution[0]-1
    if j<0.:
        j = 0.
    if j>resolution[1]-1:
        j = resolution[1]-1
    i_floor= int(i)
    j_floor= int(j)
    return bilerp(v[i_floor, j_floor],
                  v[i_floor+1, j_floor],
                  v[i_floor, j_floor+1],
                  v[i_floor+1, j_floor+1],
                  i-i_floor, j-j_floor)

@ti.func
def sample_with_default_value_vec2(v, i, j, default_value):  # Sample a field. i and j are floats.
    i_floor= int(i)
    j_floor= int(j)
    result = ti.Vector([0., 0.])
    if i_floor>=0 and i_floor<resolution[0]-1 and j_floor>=0 and j_floor<resolution[1]-1:
        result = bilerp(v[i_floor, j_floor],
                          v[i_floor+1, j_floor],
                          v[i_floor, j_floor+1],
                          v[i_floor+1, j_floor+1],
                          i-i_floor, j-j_floor)
    else:
        result = default_value
    return result

# @ti.func
# def sample_with_default_value_vec3(v, i, j, default_value):
#     i_floor= int(i)
#     j_floor= int(j)
#     result = ti.Vector([0., 0., 0.])
#     if i_floor>=0 and i_floor<resolution[0]-1 and j_floor>=0 and j_floor<resolution[1]-1:
#         result = bilerp(v[i_floor, j_floor],
#                           v[i_floor+1, j_floor],
#                           v[i_floor, j_floor+1],
#                           v[i_floor+1, j_floor+1],
#                           i-i_floor, j-j_floor)
#     else:
#         result = default_value
#     return result

@ti.kernel
def advect(v_cur: ti.template(), v_nxt: ti.template(), color_cur: ti.template(), color_nxt: ti.template()):
    # Advection
    for i, j in v_cur:
        p_cur = indices_to_coordinates(i, j)
        p_last = p_cur-v_cur[i, j]*dt
        ij_last = (p_last-domain[0])/(domain[1]-domain[0])*ti.Vector(resolution)
        v_nxt[i, j] = sample_with_padding(v_cur, ij_last.x, ij_last.y)
        color_nxt[i, j] = sample_with_padding(color_cur, ij_last.x, ij_last.y)

@ti.kernel
def apply_jetters(v: ti.template(), color: ti.template()):
    for i, j in v:
        p = indices_to_coordinates(i, j)
        if p.x<-0.9 and p.y>-0.3 and p.y<-0.2:
            v[i, j] = ti.Vector([2., 0])
            color[i, j] = ti.Vector([1., 0., 0.])
        if p.x>0.9 and p.y>0.2 and p.y<0.3:
            v[i, j] = ti.Vector([-2., 0])
            color[i, j] = ti.Vector([1., 1., 0.])

@ti.kernel
def set_fluid_velocity_same_to_solid(v: ti.template(), box_position: ti.types.vector(3, ti.f32), box_velocity: ti.types.vector(3, ti.f32)):
    for i, j in v:
        p = indices_to_coordinates(i, j)
        p -= box_position.xy
        cos = ti.math.cos(box_position.z)
        sin = ti.math.sin(box_position.z)
        p_rotated = ti.Vector([p.x*cos+p.y*sin,
                               -p.x*sin+p.y*cos])
        if p_rotated.x>-box_shape.x and p_rotated.x<box_shape.x and p_rotated.y>-box_shape.y and p_rotated.y<box_shape.y:
            v[i, j] = box_velocity.xy+p.yx*ti.Vector([-1., 1.])*box_velocity.z
            # v[i, j] = ti.Vector([1., 0.])

@ti.kernel
def calculate_divergence(f: ti.template(), result: ti.template()):
    for i, j in f:
        div = 0.
        
        if i==0:
            div += f[i+1, j].x-f[i, j].x
        elif i<resolution[0]-1:
            div += (f[i+1, j].x-f[i-1, j].x)*0.5
        else:
            div += f[i, j].x-f[i-1, j].x
            
        if j==0:
            div += f[i, j+1].y-f[i, j].y
        elif j<resolution[1]-1:
            div += (f[i, j+1].y-f[i, j-1].y)*0.5
        else:
            div += f[i, j].y-f[i, j-1].y
            
        result[i, j] = div
        
@ti.kernel
def assign_0(f: ti.template()):
    for i, j in f:
        f[i, j] = 0.

@ti.kernel
def solve_laplacian_equation_step(f: ti.template(), result_last: ti.template(), result: ti.template()):
    for i, j in f:
        neighbor_num = 0
        average = 0.
        if i>0:
            average += result_last[i-1, j]
            neighbor_num += 1
        if i<resolution[0]-1:
            average += result_last[i+1, j]
            neighbor_num += 1
        if j>0:
            average += result_last[i, j-1]
            neighbor_num += 1
        if j<resolution[1]-1:
            average += result_last[i, j+1]
            neighbor_num += 1
        average /= neighbor_num
        result[i, j] = average-f[i, j]*0.5/dimension+0.001

def solve_laplacian_equation(f, result, steps):
    if steps%2==0:
        assign_0(result)
    else:
        assign_0(f_temp_for_solving_laplacian_equation)
    while steps>0:
        if steps%2==0:
            solve_laplacian_equation_step(f, result, f_temp_for_solving_laplacian_equation)
        else:
            solve_laplacian_equation_step(f, f_temp_for_solving_laplacian_equation, result)
        steps -= 1

@ti.kernel
def assign_pressure_to_fluid(p: ti.template(), v: ti.template()):
    for i, j in v:
        if i==0:
            v[i, j].x -= p[i+1, j]-p[i, j]
        elif i<resolution[0]-1:
            v[i, j].x -= (p[i+1, j]-p[i-1, j])*0.5
        else:
            v[i, j].x -= p[i, j]-p[i-1, j]
            
        if j==0:
            v[i, j].y -= p[i, j+1]-p[i, j]
        elif j<resolution[1]-1:
            v[i, j].y -= (p[i, j+1]-p[i, j-1])*0.5
        else:
            v[i, j].y -= p[i, j]-p[i, j-1]

@ti.kernel
def calculate_grad(p: ti.template(), v: ti.template()):
    for i, j in v:
        v_x = 0.
        if i==0:
            v_x = p[i+1, j]-p[i, j]
        elif i<resolution[0]-1:
            v_x = (p[i+1, j]-p[i-1, j])*0.5
        else:
            v_x = p[i, j]-p[i-1, j]
        
        v_y = 0.
        if j==0:
            v_y = p[i, j+1]-p[i, j]
        elif j<resolution[1]-1:
            v_y = (p[i, j+1]-p[i, j-1])*0.5
        else:
            v_y = p[i, j]-p[i, j-1]
        
        v[i, j] = ti.Vector([v_x, v_y])

def eliminate_divergence():
    calculate_divergence(v_pair.nxt, f_temp_1)
    solve_laplacian_equation(f_temp_1, f_temp_2, laplacian_steps)
    assign_pressure_to_fluid(f_temp_2, v_pair.nxt)

@ti.kernel
def calculate_force(v: ti.template(), result: ti.template()):
    for i, j in v:
        p_cur = indices_to_coordinates(i, j)
        p_last = p_cur-v[i, j]*dt
        ij_last = (p_last-domain[0])/(domain[1]-domain[0])*ti.Vector(resolution)
        result[i, j] = v[i, j]-sample_with_default_value_vec2(v, ij_last.x, ij_last.y, v[i, j])

@ti.kernel
def calculate_force_on_line(p: ti.template(), p1: ti.types.vector(2, ti.f32), p2: ti.types.vector(2, ti.f32), sample_num:int)->ti.types.vector(3, ti.f32):
    v = p2-p1
    length = v.norm()
    n = v.yx*ti.Vector([1., -1.])
    n /= length
    pressure = 0.
    torque = 0.
    for i in range(sample_num):
        t = (i+0.5)/sample_num
        p_here = p1+v*t
        ij = coordinates_to_indices(p_here.x, p_here.y)
        pressure_here = -sample_with_padding(p, ij.x, ij.y)
        pressure += pressure_here
        torque += pressure_here*(t-0.5)
    n *= pressure*length/sample_num
    torque *= length/sample_num
    return ti.Vector([-n.x, -n.y, torque])/ti.Vector([box_mass, box_mass, box_I])

def assign_force_on_solid():  # Calculate fluid pressure and assign it on solid
    # Calculate fluid force
    calculate_force(v_pair.nxt, f_v_temp_1)
    calculate_divergence(f_v_temp_1, f_temp_1)
    solve_laplacian_equation(f_temp_1, f_temp_2, laplacian_steps)  # Calculate minus pressure
    # calculate_grad(f_temp_2, f_v_temp_1)  # Calculate fluid force
    
    # Assign fluid force on solid
    cos = np.cos(box_position.z)
    sin = np.sin(box_position.z)
    box_points = ([box_position.x+(-box_shape.x*cos+box_shape.y*sin),
                   box_position.y+(-box_shape.x*sin-box_shape.y*cos)],
                  [box_position.x+(+box_shape.x*cos+box_shape.y*sin),
                   box_position.y+(+box_shape.x*sin-box_shape.y*cos)],
                  [box_position.x+(+box_shape.x*cos-box_shape.y*sin),
                   box_position.y+(+box_shape.x*sin+box_shape.y*cos)],
                  [box_position.x+(-box_shape.x*cos-box_shape.y*sin),
                   box_position.y+(-box_shape.x*sin+box_shape.y*cos)])
    global box_velocity
    box_velocity += calculate_force_on_line(f_temp_2, ti.Vector(box_points[0]), ti.Vector(box_points[1]), 10)
    box_velocity += calculate_force_on_line(f_temp_2, ti.Vector(box_points[1]), ti.Vector(box_points[2]), 10)
    box_velocity += calculate_force_on_line(f_temp_2, ti.Vector(box_points[2]), ti.Vector(box_points[3]), 10)
    box_velocity += calculate_force_on_line(f_temp_2, ti.Vector(box_points[3]), ti.Vector(box_points[0]), 10)
    # print(box_velocity)

def run_one_step():
    solid_move()
    
    advect(v_pair.cur, v_pair.nxt, color_pair.cur, color_pair.nxt)
    apply_jetters(v_pair.nxt, color_pair.nxt)
    set_fluid_velocity_same_to_solid(v_pair.nxt, box_position, box_velocity)
    eliminate_divergence()
    
    assign_force_on_solid()
    
    v_pair.swap()
    color_pair.swap()

init_fluid()
apply_jetters(v_pair.cur, color_pair.cur)

gui = ti.GUI("Fluid & Solid 2D", res=(resolution[0], resolution[1]))
while gui.running:
    gui.set_image(color_pair.cur)  # Visualize fluid color
    # gui.set_image(v_pair.cur)  # Visualize fluid velocity
    # gui.set_image(f_temp_2)  # Visualize minus fluid pressure
    paint_solid()  # Visualize solid
    
    gui.show()
    
    run_one_step()