import math
from collections import namedtuple
sV = namedtuple("sV", "x y z")
sQ = namedtuple("sQ", "w x y z")

def q_norm(q):
    return math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);

def v_norm(v):
    return math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z);

def q_renorm(q):
    r = q_norm(q);
    w = 0
    x = 0
    y = 0
    z = 0
    if(r > 0):
        m = 1.0 / r;
        w = q.w*m
        x = q.x*m;	
        y = q.y*m;	
        z = q.z*m;	
    return sQ(w, x, y, z)

def v_renorm(v):
    r = v_norm(v);
    x = 0
    y = 0
    z = 0
    if(r > 0):
        m = 1.0 / r;
        x = v.x*m;	
        y = v.y*m;	
        z = v.z*m;	
    return sV(x, y, z)

def q_make_conj(q):
    rq = sQ(q.w, -q.x, -q.y, -q.z) 
    return rq

def q_mult(q1, q2):
    w = q1.w*q2.w - (q1.x*q2.x + q1.y*q2.y + q1.z*q2.z);
    x = q1.w*q2.x + q2.w*q1.x + q1.y*q2.z - q1.z*q2.y;
    y = q1.w*q2.y + q2.w*q1.y + q1.z*q2.x - q1.x*q2.z;
    z = q1.w*q2.z + q2.w*q1.z + q1.x*q2.y - q1.y*q2.x;
    return sQ(w, x, y, z);

def rotate_v(q, v):
    r = sQ(0, v.x, v.y, v.z) 
    qc = q_make_conj(q);
    qq = q_mult(r, qc);
    rq = q_mult(q, qq);
    return sV(rq.x, rq.y, rq.z);

def qv_mult(q1, q2):
    x = q1.y*q2.z - q1.z*q2.y;
    y = q1.z*q2.x - q1.x*q2.z;
    z = q1.x*q2.y - q1.y*q2.x;
    return sQ(0, x, y, z) ;

def v_mult(v1, v2):
    x = v1.y*v2.z - v1.z*v2.y;
    y = v1.z*v2.x - v1.x*v2.z;
    z = v1.x*v2.y - v1.y*v2.x;
    return sV(x, y, z);

def v_dot(v1, v2):
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;

def q_from_vectors(u, v):
    u_norm = v_renorm(u)
    v_norm = v_renorm(v)

    # Calculate dot product (cosine of the angle)
    dot_product = v_dot(u_norm, v_norm)

    # 2. Handle the special case: u and v are parallel (same direction)
    # The quaternion is the identity: [1, 0, 0, 0]
    if dot_product > 0.9999:
        return sQ(1.0, 0.0, 0.0, 0.0)

    # 3. Handle the special case: u and v are antiparallel (opposite direction)
    # Rotation is 180 degrees around an arbitrary perpendicular axis.
    if dot_product < -0.9999:
        # Find a perpendicular vector to use as the axis of rotation
        # Use cross product with the axis that is least parallel to u
        # This prevents the cross product from being near zero
        axis = v_mult(u_norm, sV(1, 0, 0))
        if v_norm(axis) < 0.0001:
            axis = v_mult(u_norm, sV(0, 1, 0))
        
        axis = v_renorm(axis)
        
        # 180 degree rotation: w=0, vector part is the axis
        return sQ(0.0, axis[0], axis[1], axis[2])

    # 4. General Case
    
    # Calculate the cross product (rotation axis direction)
    axis = v_mult(u_norm, v_norm)

    # The formula (unnormalized): q = (1 + cos(theta), axis * sin(theta))
    # A simplified, stable form: q = (w, x, y, z) where:
    # w = 1 + dot(u, v)
    # vector part = cross(u, v)
    
    w = 1.0 + dot_product
    
    q_unnormalized = sQ(w, axis[0], axis[1], axis[2])
    
    # 5. Normalize the resulting quaternion
    return q_renorm(q_unnormalized)

