""" Forward and backward Monte-Carlo integral functions.
"""
import torch
import numpy as np
import torch.nn.functional as nnF

from .basics import EPSILON

def forward_integral(ray, ps, ks, pointc_ref=None, interpolate=False, param_list=None):
    """ Forward integral model, including PSF and vignetting

    Args:
        ray: Ray object. Shape of ray.o is [spp, N, 3].
        ps: pixel size
        ks: kernel size.
        pointc_ref: reference pointc, shape [2]
        center: whether to center the PSF.
        interpolate: whether to interpolate the PSF

    Returns:
        psf: point spread function, shape [N, ks, ks]
    """
    single_point = True if len(ray.o.shape) == 2 else False
    points = - ray.o[..., :2]       # shape [spp, N, 2] or [spp, 2]. flip points.
    psf_range = [(- ks / 2 + 0.5) * ps, (ks / 2 - 0.5) * ps]    # this ensures the pixel size doesnot change in assign_points_to_pixels function
    
    # ==> PSF center
    if pointc_ref is None:
        # Use RMS center
        pointc = (points * ray.ra.unsqueeze(-1)).sum(0) / ray.ra.unsqueeze(-1).sum(0).add(EPSILON)
        points_shift = points - pointc
    else:
        # Use manually given center (can be calculated by chief ray or perspective)
        points_shift = points - pointc_ref.to(points.device)
    
    # ==> Remove invalid points
    ra = ray.ra * (points_shift[...,0].abs() < (psf_range[1] - 0.01*ps)) * (points_shift[...,1].abs() < (psf_range[1] - 0.01*ps))   # shape [spp, N] or [spp].
    points_shift *= ra.unsqueeze(-1)


    # ==> Calculate PSF
    if single_point:
        obliq = ray.d[:, 2]**2
        psf = assign_points_to_pixels(points=points_shiftxx, ks=ks, x_range=psf_range, y_range=psf_range, ra=ra, obliq=obliq)

    else:
        psf = []
        for i in range(ray.o.shape[1]):
            points_shift_0 = points_shift[:, i, :]   # from [spp, N, 2] to [spp, 2]
            ra_0 = ra[:, i]                          # from [spp, N] to [spp]
            obliq = ray.d[:, i, 2]**2

            x_tan = (-ray.d[:, i, 0])/ray.d[:, i, 2] # flip d_x.
            if param_list is not None:
                lamda,h,f,leak,radius,v,direct=param_list
            else:
                radius = 0.5
            if radius <=0.5:
                psf_l, psf_r = assign_points_to_pixels_3d(points=points_shift_0, ks=ks, x_range=psf_range, y_range=psf_range, ra=ra_0, obliq=obliq, x_tan=x_tan, param_list=param_list)
            else:
                psf_l, psf_r = assign_points_to_pixels_3d_bigradius(points=points_shift_0, ks=ks, x_range=psf_range, y_range=psf_range, ra=ra_0, obliq=obliq, x_tan=x_tan, param_list=param_list)
            psf.append(psf_l)
            # psf.append(psf_r)
        psf = torch.stack(psf, dim=0)   # shape [N, ks, ks]
    
    return psf

def assign_points_to_pixels(points, ks, x_range, y_range, ra, interpolate=True, coherent=False, phase=None, d=None, obliq=None, wvln=0.589):
    """ Assign points to pixels, both coherent and incoherent. Use advanced indexing to increment the count for each corresponding pixel. This function can only compute single point source, single wvln. If you want to compute multiple point or muyltiple wvln, please call this function multiple times.
    
    Args:
        points: shape [spp, 1, 2]
        ks: kernel size
        x_range: [x_min, x_max]
        y_range: [y_min, y_max]
        ra: shape [spp, 1, 1]
        interpolate: whether to interpolate
        coherent: whether to consider coherence
        phase: shape [spp, 1, 1]

    Returns:
        psf: shape [ks, ks]
    """
    # ==> Parameters
    device = points.device
    x_min, x_max = x_range
    y_min, y_max = y_range
    ps = (x_max - x_min) / (ks - 1)

    # ==> Normalize points to the range [0, 1]
    points_normalized = torch.zeros_like(points)
    points_normalized[:, 0] = (points[:, 1] - y_max) / (y_min - y_max)
    points_normalized[:, 1] = (points[:, 0] - x_min) / (x_max - x_min)

    if interpolate:
        # ==> Weight. The trick here is to use (ks - 1) to compute normalized indices
        pixel_indices_float = points_normalized * (ks - 1)
        w_b = pixel_indices_float[..., 0] - pixel_indices_float[..., 0].floor()
        w_r = pixel_indices_float[..., 1] - pixel_indices_float[..., 1].floor()

        # ==> Pixel indices
        pixel_indices_tl = pixel_indices_float.floor().long()
        pixel_indices_tr = torch.stack((pixel_indices_float[:, 0], pixel_indices_float[:, 1]+1), dim=-1).floor().long()
        pixel_indices_bl = torch.stack((pixel_indices_float[:, 0]+1, pixel_indices_float[:, 1]), dim=-1).floor().long()
        pixel_indices_br = pixel_indices_tl + 1

        if coherent:
            # ==> Use advanced indexing to increment the count for each corresponding pixel
            grid = torch.zeros(ks, ks).to(device) + 0j
            grid.index_put_(tuple(pixel_indices_tl.t()), (1-w_b)*(1-w_r)*ra*torch.exp(1j*phase), accumulate=True)
            grid.index_put_(tuple(pixel_indices_tr.t()), (1-w_b)*w_r*ra*torch.exp(1j*phase), accumulate=True)
            grid.index_put_(tuple(pixel_indices_bl.t()), w_b*(1-w_r)*ra*torch.exp(1j*phase), accumulate=True)
            grid.index_put_(tuple(pixel_indices_br.t()), w_b*w_r*ra*torch.exp(1j*phase), accumulate=True)

        else:
            grid = torch.zeros(ks, ks).to(points.device)
            grid.index_put_(tuple(pixel_indices_tl.t()), (1-w_b)*(1-w_r)*ra, accumulate=True)
            grid.index_put_(tuple(pixel_indices_tr.t()), (1-w_b)*w_r*ra, accumulate=True)
            grid.index_put_(tuple(pixel_indices_bl.t()), w_b*(1-w_r)*ra, accumulate=True)
            grid.index_put_(tuple(pixel_indices_br.t()), w_b*w_r*ra, accumulate=True)

    else:
        pixel_indices_float = points_normalized * (ks - 1)
        pixel_indices_tl = pixel_indices_float.floor().long()

        grid = torch.zeros(ks, ks).to(points.device)
        grid.index_put_(tuple(pixel_indices_tl.t()), ra, accumulate=True)
        
    return grid

def assign_points_to_pixels_3d(points, ks, x_range, y_range, ra, interpolate=True, coherent=False, phase=None, d=None, obliq=None, wvln=0.589, x_tan=None, param_list=None):
    """ Assign points to pixels, both coherent and incoherent. Use advanced indexing to increment the count for each corresponding pixel. This function can only compute single point source, single wvln. If you want to compute multiple point or muyltiple wvln, please call this function multiple times.
    
    Args:
        points: shape [spp, 1, 2]
        ks: kernel size
        x_range: [x_min, x_max]
        y_range: [y_min, y_max]
        ra: shape [spp, 1, 1]
        interpolate: whether to interpolate
        coherent: whether to consider coherence
        phase: shape [spp, 1, 1]

    Returns:
        psf: shape [ks, ks]
    """
    # ==> Parameters
    device = points.device
    x_min, x_max = x_range
    y_min, y_max = y_range
    ps = (x_max - x_min) / (ks - 1)
    #2,0.76923,1.25,0.5,0.25
    if param_list is None:
        # lamda = f*h/(f-h)
        lamda = 1.7
        f = 1.4
        h = lamda*f/(lamda+f)
        leak = 0.6
        radius = 0.5
        direct = "l"
    else:
        lamda,h,f,leak,radius,v,direct=param_list

    # print("lamda,lamda/ps: ",lamda, h, f)
    dp_ps = 1/2*(leak)
    assert radius<=0.5
    r = torch.tensor(radius).to(points.device)

    # lights integral in micro lens
    xr = f/(f-h)*(dp_ps  -h*x_tan)
    xm = f/(f-h)*(0      -h*x_tan)
    xl = f/(f-h)*(-dp_ps -h*x_tan)
    xr = torch.clamp(xr,-r,r)
    xm = torch.clamp(xm,-r,r)
    xl = torch.clamp(xl,-r,r)
    ur = torch.arccos(xr/r)
    um = torch.arccos(xm/r)
    ul = torch.arccos(xl/r)
    # The definite integral of (r^2-x^2)^(1/2)dx in the interval [xm, x1], where x=r*cost, t belongs to the interval [0, pi]
    sr_ml = r*r*((um-1/2*torch.sin(2*um)) - (ur-1/2*torch.sin(2*ur)))
    sl_ml = r*r*((ul-1/2*torch.sin(2*ul)) - (um-1/2*torch.sin(2*um)))

    # lights integral in margin off lens
    xr = dp_ps -h*x_tan
    xm = 0     -h*x_tan
    xl = -dp_ps-h*x_tan
    xr = torch.clamp(xr,-0.5,0.5)
    xm = torch.clamp(xm,-0.5,0.5)
    xl = torch.clamp(xl,-0.5,0.5)
    xr_inplace = torch.clamp(xr,-r,r)
    xm_inplace = torch.clamp(xm,-r,r)
    xl_inplace = torch.clamp(xl,-r,r)
    ur = torch.arccos(xr_inplace/r)
    um = torch.arccos(xm_inplace/r)
    ul = torch.arccos(xl_inplace/r)
    sr_mg_inplace = r*r*((um-1/2*torch.sin(2*um)) - (ur-1/2*torch.sin(2*ur)))
    sl_mg_inplace = r*r*((ul-1/2*torch.sin(2*ul)) - (um-1/2*torch.sin(2*um)))
    sr_mg = (xr-xm)*1 - sr_mg_inplace
    sl_mg = (xm-xl)*1 - sl_mg_inplace
    sr = sr_ml + sr_mg
    sl = sl_ml + sl_mg

    d_l = sl
    d_r = sr
    
    # ==> Normalize points to the range [0, 1], img_coord is not ray_coord
    points_normalized = torch.zeros_like(points)
    points_normalized[:, 0] = (points[:, 1] - y_max) / (y_min - y_max)
    points_normalized[:, 1] = (points[:, 0] - x_min) / (x_max - x_min)

    if interpolate:
        # ==> Weight. The trick here is to use (ks - 1) to compute normalized indices
        pixel_indices_float = points_normalized * (ks - 1)
        w_b = pixel_indices_float[..., 0] - pixel_indices_float[..., 0].floor()
        w_r = pixel_indices_float[..., 1] - pixel_indices_float[..., 1].floor()

        # ==> Pixel indices
        pixel_indices_tl = pixel_indices_float.floor().long()
        pixel_indices_tr = torch.stack((pixel_indices_float[:, 0], pixel_indices_float[:, 1]+1), dim=-1).floor().long()
        pixel_indices_bl = torch.stack((pixel_indices_float[:, 0]+1, pixel_indices_float[:, 1]), dim=-1).floor().long()
        pixel_indices_br = pixel_indices_tl + 1

        l_grid = torch.zeros(ks, ks).to(points.device)
        l_grid.index_put_(tuple(pixel_indices_tl.t()), (1-w_b)*(1-w_r)*ra*d_l, accumulate=True)
        l_grid.index_put_(tuple(pixel_indices_tr.t()), (1-w_b)*w_r*ra*d_l, accumulate=True)
        l_grid.index_put_(tuple(pixel_indices_bl.t()), w_b*(1-w_r)*ra*d_l, accumulate=True)
        l_grid.index_put_(tuple(pixel_indices_br.t()), w_b*w_r*ra*d_l, accumulate=True)

        r_grid = torch.zeros(ks, ks).to(points.device)
        if param_list is not None:
            r_grid.index_put_(tuple(pixel_indices_tl.t()), (1-w_b)*(1-w_r)*ra*d_r, accumulate=True)
            r_grid.index_put_(tuple(pixel_indices_tr.t()), (1-w_b)*w_r*ra*d_r, accumulate=True)
            r_grid.index_put_(tuple(pixel_indices_bl.t()), w_b*(1-w_r)*ra*d_r, accumulate=True)
            r_grid.index_put_(tuple(pixel_indices_br.t()), w_b*w_r*ra*d_r, accumulate=True)

    else:
        # ==> Weight. The trick here is to use (ks - 1) to compute normalized indices
        pixel_indices_float = points_normalized * ks
        w_r = pixel_indices_float[..., 1] - pixel_indices_float[..., 1].floor()
        l_ra = w_r <= d0
        r_ra = w_r >= d0

        # ==> Pixel indices
        pixel_indices_tl = pixel_indices_float.floor().long()
        l_grid = torch.zeros(ks, ks).to(points.device)
        r_grid = torch.zeros(ks, ks).to(points.device)
        l_grid.index_put_(tuple(pixel_indices_tl.t()), ra*l_ra, accumulate=True)
        # r_grid.index_put_(tuple(pixel_indices_tl.t()), ra*r_ra, accumulate=True)
    
    if direct =='l':
        return l_grid,r_grid
    else:
        return r_grid,l_grid

def assign_points_to_pixels_3d_bigradius(points, ks, x_range, y_range, ra, interpolate=True, coherent=False, phase=None, d=None, obliq=None, wvln=0.589, x_tan=None, param_list=None):
    """ Assign points to pixels, both coherent and incoherent. Use advanced indexing to increment the count for each corresponding pixel. This function can only compute single point source, single wvln. If you want to compute multiple point or muyltiple wvln, please call this function multiple times.
    
    Args:
        points: shape [spp, 1, 2]
        ks: kernel size
        x_range: [x_min, x_max]
        y_range: [y_min, y_max]
        ra: shape [spp, 1, 1]
        interpolate: whether to interpolate
        coherent: whether to consider coherence
        phase: shape [spp, 1, 1]

    Returns:
        psf: shape [ks, ks]
    """
    # ==> Parameters
    device = points.device
    x_min, x_max = x_range
    y_min, y_max = y_range
    ps = (x_max - x_min) / (ks - 1)
    #2,0.76923,1.25,0.5,0.25
    if param_list is None:
        # lamda = f*h/(f-h)
        lamda = 1.7
        f = 1.4
        h = lamda*f/(lamda+f)
        leak = 0.6
        radius = 0.5
        direct = "l"
    else:
        lamda,h,f,leak,radius,v,direct=param_list

    # print("lamda,lamda/ps: ",lamda, h, f)
    dp_ps = 1/2*(leak)
    assert radius>=0.5
    r = torch.tensor(radius).to(points.device)
    tr = torch.asin(torch.tensor(0.5/r))
    tl = torch.pi - tr

    # lights integral in micro lens
    xr = f/(f-h)*(dp_ps  -h*x_tan)
    xm = f/(f-h)*(0      -h*x_tan)
    xl = f/(f-h)*(-dp_ps -h*x_tan)
    xr = torch.clamp(xr,-0.5,0.5)
    xm = torch.clamp(xm,-0.5,0.5)
    xl = torch.clamp(xl,-0.5,0.5)

    ur = torch.arccos(xr/r)
    um = torch.arccos(xm/r)
    ul = torch.arccos(xl/r)
    # The definite integral of (r^2-x^2)^(1/2)dx in the interval [xm, x1], where x=r*cost, t belongs to the interval [0, pi]
    sr_ml = r*r*((um-1/2*torch.sin(2*um)) - (ur-1/2*torch.sin(2*ur)))
    sl_ml = r*r*((ul-1/2*torch.sin(2*ul)) - (um-1/2*torch.sin(2*um)))

    ur_extend = torch.clamp(ur,tr,tl)
    um_extend = torch.clamp(um,tr,tl)
    ul_extend = torch.clamp(ul,tr,tl)
    xr_extend = torch.cos(ur_extend)*r
    xm_extend = torch.cos(um_extend)*r
    xl_extend = torch.cos(ul_extend)*r
    sr_ml_extend = (r*r*((um_extend-1/2*torch.sin(2*um_extend)) - (ur_extend-1/2*torch.sin(2*ur_extend))))-(xr_extend-xm_extend)
    sl_ml_extend = (r*r*((ul_extend-1/2*torch.sin(2*ul_extend)) - (um_extend-1/2*torch.sin(2*um_extend))))-(xm_extend-xl_extend)
    sr_ml = sr_ml - sr_ml_extend
    sl_ml = sl_ml - sl_ml_extend

    # lights integral in margin off lens
    xr = dp_ps -h*x_tan
    xm = 0     -h*x_tan
    xl = -dp_ps-h*x_tan
    xr = torch.clamp(xr,-0.5,0.5)
    xm = torch.clamp(xm,-0.5,0.5)
    xl = torch.clamp(xl,-0.5,0.5)

    ur = torch.arccos(xr/r)
    um = torch.arccos(xm/r)
    ul = torch.arccos(xl/r)
    # The definite integral of (r^2-x^2)^(1/2)dx in the interval [xm, x1], where x=r*cost, t belongs to the interval [0, pi]
    sr_mg_inplace = r*r*((um-1/2*torch.sin(2*um)) - (ur-1/2*torch.sin(2*ur)))
    sl_mg_inplace = r*r*((ul-1/2*torch.sin(2*ul)) - (um-1/2*torch.sin(2*um)))

    ur_extend = torch.clamp(ur,tr,tl)
    um_extend = torch.clamp(um,tr,tl)
    ul_extend = torch.clamp(ul,tr,tl)
    xr_extend = torch.cos(ur_extend)*r
    xm_extend = torch.cos(um_extend)*r
    xl_extend = torch.cos(ul_extend)*r
    sr_mg_extend = (r*r*((um_extend-1/2*torch.sin(2*um_extend)) - (ur_extend-1/2*torch.sin(2*ur_extend))))-(xr_extend-xm_extend)
    sl_mg_extend = (r*r*((ul_extend-1/2*torch.sin(2*ul_extend)) - (um_extend-1/2*torch.sin(2*um_extend))))-(xm_extend-xl_extend)
    sr_mg_inplace = sr_mg_inplace - sr_mg_extend
    sl_mg_inplace = sl_mg_inplace - sl_mg_extend

    sr_mg = (xr-xm)*1 - sr_mg_inplace
    sl_mg = (xm-xl)*1 - sl_mg_inplace
    sr = sr_ml + sr_mg
    sl = sl_ml + sl_mg

    d_l = sl
    d_r = sr
    
    # ==> Normalize points to the range [0, 1], img_coord is not ray_coord
    points_normalized = torch.zeros_like(points)
    points_normalized[:, 0] = (points[:, 1] - y_max) / (y_min - y_max)
    points_normalized[:, 1] = (points[:, 0] - x_min) / (x_max - x_min)

    if interpolate:
        # ==> Weight. The trick here is to use (ks - 1) to compute normalized indices
        pixel_indices_float = points_normalized * (ks - 1)
        w_b = pixel_indices_float[..., 0] - pixel_indices_float[..., 0].floor()
        w_r = pixel_indices_float[..., 1] - pixel_indices_float[..., 1].floor()

        # ==> Pixel indices
        pixel_indices_tl = pixel_indices_float.floor().long()
        pixel_indices_tr = torch.stack((pixel_indices_float[:, 0], pixel_indices_float[:, 1]+1), dim=-1).floor().long()
        pixel_indices_bl = torch.stack((pixel_indices_float[:, 0]+1, pixel_indices_float[:, 1]), dim=-1).floor().long()
        pixel_indices_br = pixel_indices_tl + 1

        l_grid = torch.zeros(ks, ks).to(points.device)
        l_grid.index_put_(tuple(pixel_indices_tl.t()), (1-w_b)*(1-w_r)*ra*d_l, accumulate=True)
        l_grid.index_put_(tuple(pixel_indices_tr.t()), (1-w_b)*w_r*ra*d_l, accumulate=True)
        l_grid.index_put_(tuple(pixel_indices_bl.t()), w_b*(1-w_r)*ra*d_l, accumulate=True)
        l_grid.index_put_(tuple(pixel_indices_br.t()), w_b*w_r*ra*d_l, accumulate=True)

        r_grid = torch.zeros(ks, ks).to(points.device)
        if param_list is not None:
            r_grid.index_put_(tuple(pixel_indices_tl.t()), (1-w_b)*(1-w_r)*ra*d_r, accumulate=True)
            r_grid.index_put_(tuple(pixel_indices_tr.t()), (1-w_b)*w_r*ra*d_r, accumulate=True)
            r_grid.index_put_(tuple(pixel_indices_bl.t()), w_b*(1-w_r)*ra*d_r, accumulate=True)
            r_grid.index_put_(tuple(pixel_indices_br.t()), w_b*w_r*ra*d_r, accumulate=True)

    else:
        # ==> Weight. The trick here is to use (ks - 1) to compute normalized indices
        pixel_indices_float = points_normalized * ks
        w_r = pixel_indices_float[..., 1] - pixel_indices_float[..., 1].floor()
        l_ra = w_r <= d0
        r_ra = w_r >= d0

        # ==> Pixel indices
        pixel_indices_tl = pixel_indices_float.floor().long()
        l_grid = torch.zeros(ks, ks).to(points.device)
        r_grid = torch.zeros(ks, ks).to(points.device)
        l_grid.index_put_(tuple(pixel_indices_tl.t()), ra*l_ra, accumulate=True)
        # r_grid.index_put_(tuple(pixel_indices_tl.t()), ra*r_ra, accumulate=True)
    
    if direct =='l':
        return l_grid,r_grid
    else:
        return r_grid,l_grid
