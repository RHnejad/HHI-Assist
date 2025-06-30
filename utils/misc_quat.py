import torch

def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    conj = torch.cat((-a[:,:3], a[:, -1:]), dim=-1).view(shape)
    return conj

def quat_mul(a, b):
    assert a.shape == b.shape 
    shape = a.shape

    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz 
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    quat = torch.stack([x,y,z,w], dim=-1).view(shape)

    return quat 


    # HIPS 1 HIPS 2 RUL 1 RUL 2 
    # HIPS ( 1 2)
    # HIPS 1 RUL 1 HIPS 2 RUL 2
    # HIPS 

def quat_diff_rad(a,b):

    B, T, J, C = a.shape

    a2 = a.reshape(B,-1,4)
    b2 = b.reshape(B,-1,4)
    
    mul = a2 * b2

    sum_mul = torch.sum(mul, dim=-1)
    
    # l = torch.acos(torch.clamp((2 * (sum_mul**2)- 1), min = -1, max = 1.0))
    l = 1 - (sum_mul**2) #https://math.stackexchange.com/questions/90081/quaternion-distance

    #add breakpoint if nan :
    if torch.isnan(l).any():
        breakpoint()

    l = torch.mean(l)
    
    # print(">  l:", l)
    
    return l 
    
    
# def quat_diff_rad(a,b):
#     b_conj = quat_conjugate(b)
#     mul = quat_mul(a, b_conj)
#     # print("mu.shapel:", mul.shape, mul)
#     return 2.0 * torch.asin(
#         torch.clamp(
#             torch.norm(
#                 mul[:, :, :, 0:3],
#                 p = 2, dim = -1), max=1.0
#             )
#         )
    
