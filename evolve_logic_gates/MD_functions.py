import numpy as np

def ForceWall(N, x, y, D, Lx, Ly, k_list):
    # forces exerted on walls from particles
    Fw_x = np.zeros(N)
    Fw_y = np.zeros(N)

    # particle wall force
    for n in range(N):
        xp = x[n]
        yp = y[n]
        r = 0.5 * D[n]

        if xp < r:
            Fw_x[n] -= k_list[n] * (r - xp)
        elif xp > Lx - r:
            Fw_x[n] -= k_list[n] * (Lx - r - xp)
        if yp < r:
            Fw_y[n] -= k_list[n] * (r - yp)
        elif yp > Ly - r:
            Fw_y[n] -= k_list[n] * (Ly - r - yp)

    return Fw_x, Fw_y

def Force(N, x, y, D, Lx, Ly, k_list, Fx, Fy):
    Fx[:] = 0.0
    Fy[:] = 0.0

    # particle particle force
    for n in range(N - 1):
        for m in range(n+1, N):
            dy = y[m] - y[n]
            Dmn = 0.5 * (D[m] + D[n])
            if abs(dy) < Dmn:
                dx = x[m] - x[n]
                if abs(dx) < Dmn:
                    dmn = np.sqrt(dx**2 + dy**2)
                    if dmn < Dmn:
                        F = k_list[n] * k_list[m] / (k_list[n] + k_list[m]) * (1.0 - Dmn / dmn)
                        dFx = F * dx
                        dFy = F * dy
                        Fx[n] += dFx
                        Fx[m] -= dFx
                        Fy[n] += dFy
                        Fy[m] -= dFy

    # particle wall force
    for n in range(N):
        xp = x[n]
        yp = y[n]
        r = 0.5 * D[n]

        if xp < r:
            Fx[n] += k_list[n] * (r - xp)
        elif xp > Lx - r:
            Fx[n] += k_list[n] * (Lx - r - xp)
        if yp < r:
            Fy[n] += k_list[n] * (r - yp)
        elif yp > Ly - r:
            Fy[n] += k_list[n] * (Ly - r - yp)

    return


def Force_VL(N, x, y, D, Lx, Ly, k_list, Fx, Fy, VL_list, VL_counter):
    Fx[:] = 0.0
    Fy[:] = 0.0

    # particle particle force
    for vl_idx in range(VL_counter):
        n = VL_list[vl_idx][0]
        m = VL_list[vl_idx][1]
        dy = y[m] - y[n]
        Dmn = 0.5 * (D[m] + D[n])
        if abs(dy) < Dmn:
            dx = x[m] - x[n]
            if abs(dx) < Dmn:
                dmn = np.sqrt(dx**2 + dy**2)
                if dmn < Dmn:
                    k = k_list[n] * k_list[m] / (k_list[n] + k_list[m])
                    F = k * (1.0 - Dmn / dmn)
                    dFx = F * dx
                    dFy = F * dy
                    Fx[n] += dFx
                    Fx[m] -= dFx
                    Fy[n] += dFy
                    Fy[m] -= dFy

    # particle wall force
    for n in range(N):
        xp = x[n]
        yp = y[n]
        r = 0.5 * D[n]

        if xp < r:
            Fx[n] += k_list[n] * (r - xp)
        elif xp > Lx - r:
            Fx[n] += k_list[n] * (Lx - r - xp)
        if yp < r:
            Fy[n] += k_list[n] * (r - yp)
        elif yp > Ly - r:
            Fy[n] += k_list[n] * (Ly - r - yp)

    return


def VerletList(N, x, y, D, VL_list, VL_counter_old, x_save, y_save, first_call):    
    
    r_factor = 1.2
    r_cut = np.amax(D)
    r_list = r_factor * r_cut
    r_list_sq = r_list**2
    r_skin_sq = ((r_factor - 1.0) * r_cut)**2

    if first_call == 0:
        dr_sq_max = 0.0
        for n in range(N):
            dy = y[n] - y_save[n]
            dx = x[n] - x_save[n]
            dr_sq = dx**2 + dy**2
            if dr_sq > dr_sq_max:
                dr_sq_max = dr_sq
        if dr_sq_max < r_skin_sq:
            return VL_counter_old

    VL_counter = 0
    
    for n in range(N):
        for m in range(n+1, N):
            dy = y[m] - y[n]
            if abs(dy) < r_list:
                dx = x[m] - x[n]
                if abs(dx) < r_list:
                    dmn_sq = dx**2 + dy**2
                    if dmn_sq < r_list_sq:
                        VL_list[VL_counter][0] = n
                        VL_list[VL_counter][1] = m
                        VL_counter += 1

    for n in range(N):
        x_save[n] = x[n]
        y_save[n] = y[n]

    return VL_counter


def FIRE_VL(N, x, y, D, Lx, Ly, k_list):  
    # FIRE parameters
    Fthresh = 1e-14 * max(k_list)
    dt_md = 0.01 / np.sqrt(max(k_list))
    Nt = 1000000 # maximum fire md steps
    N_delay = 20
    N_pn_max = 2000
    f_inc = 1.1
    f_dec = 0.5
    a_start = 0.15
    f_a = 0.99
    dt_max = 10.0 * dt_md
    dt_min = 0.05 * dt_md
    initialdelay = 1
    
    vx = np.zeros(N)
    vy = np.zeros(N)  
    Fx = np.zeros(N)
    Fy = np.zeros(N)
    x_save = np.array(x)
    y_save = np.array(y)

    VL_list = np.zeros((N * 10, 2), dtype = int) 
    VL_counter = 0
    VL_counter = VerletList(N, x, y, D, VL_list, VL_counter, x_save, y_save, 1)
    Force_VL(N, x, y, D, Lx, Ly, k_list, Fx, Fy, VL_list, VL_counter)
    F_max = np.max([np.abs(Fx), np.abs(Fy)])
    # putting a threshold on total force
    if F_max < Fthresh:
        return
        
    a_fire = a_start
    delta_a_fire = 1.0 - a_fire
    dt = dt_md
    dt_half = dt / 2.0

    N_pp = 0 # number of P being positive
    N_pn = 0 # number of P being negative
    ## FIRE
    for nt in np.arange(Nt):
        # FIRE update
        P = np.dot(vx, Fx) + np.dot(vy, Fy)
        
        if P > 0.0:
            N_pp += 1
            N_pn = 0
            if N_pp > N_delay:
                dt = min(f_inc * dt, dt_max)
                dt_half = dt / 2.0
                a_fire = f_a * a_fire
                delta_a_fire = 1.0 - a_fire
        else:
            N_pp = 0
            N_pn += 1
            if N_pn > N_pn_max:
                break
            if (initialdelay < 0.5) or (nt >= N_delay):
                if f_dec * dt > dt_min:
                    dt = f_dec * dt
                    dt_half = dt / 2.0
                a_fire = a_start
                delta_a_fire = 1.0 - a_fire
                
            x -= vx * dt_half
            y -= vy * dt_half
            vx[:] = 0.0
            vy[:] = 0.0

        # MD using Verlet method
        vx += Fx * dt_half
        vy += Fy * dt_half
        rsc_fire = np.sqrt(np.sum(vx**2 + vy**2)) / np.sqrt(np.sum(Fx**2 + Fy**2))
        vx = delta_a_fire * vx + a_fire * rsc_fire * Fx
        vy = delta_a_fire * vy + a_fire * rsc_fire * Fy
        x += vx * dt
        y += vy * dt

        VL_counter = VerletList(N, x, y, D, VL_list, VL_counter, x_save, y_save, 0)
        Force_VL(N, x, y, D, Lx, Ly, k_list, Fx, Fy, VL_list, VL_counter)

        F_max = np.max([np.abs(Fx), np.abs(Fy)])
        if F_max < Fthresh:
            break

        vx += Fx * dt_half
        vy += Fy * dt_half

    return

def Force_FixedTopForce(N, x, y, D, Lx, Ly, k_list, F_top, Fx, Fy):
    Fx[:] = 0.0
    Fy[:] = 0.0
    Ft = F_top

    # particle particle force
    for n in range(N - 1):
        for m in range(n+1, N):
            dy = y[m] - y[n]
            Dmn = 0.5 * (D[m] + D[n])
            if abs(dy) < Dmn:
                dx = x[m] - x[n]
                if abs(dx) < Dmn:
                    dmn = np.sqrt(dx**2 + dy**2)
                    if dmn < Dmn:
                        F = k_list[n] * k_list[m] / (k_list[n] + k_list[m]) * (1.0 - Dmn / dmn)
                        dFx = F * dx
                        dFy = F * dy
                        Fx[n] += dFx
                        Fx[m] -= dFx
                        Fy[n] += dFy
                        Fy[m] -= dFy

    # particle wall force
    for n in range(N):
        xp = x[n]
        yp = y[n]
        r = 0.5 * D[n]

        if xp < r:
            Fx[n] += k_list[n] * (r - xp)
        elif xp > Lx - r:
            Fx[n] += k_list[n] * (Lx - r - xp)
        if yp < r:
            Fy[n] += k_list[n] * (r - yp)
        elif yp > Ly - r:
            dFy = k_list[n] * (Ly - r - yp)
            Fy[n] += dFy
            Ft -= dFy

    return Ft


def Force_FixedTopForce_VL(N, x, y, D, Lx, Ly, k_list, F_top, Fx, Fy, VL_list, VL_counter):
    Fx[:] = 0.0
    Fy[:] = 0.0
    Ft = F_top

    # particle particle force
    for vl_idx in range(VL_counter):
        n = VL_list[vl_idx][0]
        m = VL_list[vl_idx][1]
        dy = y[m] - y[n]
        Dmn = 0.5 * (D[m] + D[n])
        if abs(dy) < Dmn:
            dx = x[m] - x[n]
            if abs(dx) < Dmn:
                dmn = np.sqrt(dx**2 + dy**2)
                if dmn < Dmn:
                    k = k_list[n] * k_list[m] / (k_list[n] + k_list[m])
                    F = k * (1.0 - Dmn / dmn)
                    dFx = F * dx
                    dFy = F * dy
                    Fx[n] += dFx
                    Fx[m] -= dFx
                    Fy[n] += dFy
                    Fy[m] -= dFy

    # particle wall force
    for n in range(N):
        xp = x[n]
        yp = y[n]
        r = 0.5 * D[n]

        if xp < r:
            Fx[n] += k_list[n] * (r - xp)
        elif xp > Lx - r:
            Fx[n] += k_list[n] * (Lx - r - xp)
        if yp < r:
            Fy[n] += k_list[n] * (r - yp)
        elif yp > Ly - r:
            dFy = k_list[n] * (Ly - r - yp)
            Fy[n] += dFy
            Ft -= dFy

    return Ft


def FIRE_FixedTopForce_VL(N, x, y, D, Lx, Ly, k_list, FIRE_FixedTopForce_VL):  
    # FIRE parameters
    Fthresh = 1e-14 * max(k_list)
    dt_md = 0.01 / np.sqrt(max(k_list))
    Nt = 1000000 # maximum fire md steps
    N_delay = 20
    N_pn_max = 2000
    f_inc = 1.1
    f_dec = 0.5
    a_start = 0.15
    f_a = 0.99
    dt_max = 10.0 * dt_md
    dt_min = 0.05 * dt_md
    initialdelay = 1
    
    vx = np.zeros(N)
    vy = np.zeros(N)  
    v_w = 0.0
    m_w = np.round(Lx)
    Fx = np.zeros(N)
    Fy = np.zeros(N)
    x_save = np.array(x)
    y_save = np.array(y)

    VL_list = np.zeros((N * 10, 2), dtype = int) 
    VL_counter = 0
    VL_counter = VerletList(N, x, y, D, VL_list, VL_counter, x_save, y_save, 1)
    F_w = Force_FixedTopForce_VL(N, x, y, D, Lx, Ly, k_list, F_top, Fx, Fy, VL_list, VL_counter)
    F_max = np.max([np.max(np.abs(Fx)), np.max(np.abs(Fy)), np.abs(F_w)])
    # putting a threshold on total force
    if F_max < Fthresh:
        return
        
    a_fire = a_start
    delta_a_fire = 1.0 - a_fire
    dt = dt_md
    dt_half = dt / 2.0

    N_pp = 0 # number of P being positive
    N_pn = 0 # number of P being negative
    ## FIRE
    for nt in np.arange(Nt):
        # FIRE update
        P = np.dot(vx, Fx) + np.dot(vy, Fy) + v_w * F_w
        
        if P > 0.0:
            N_pp += 1
            N_pn = 0
            if N_pp > N_delay:
                dt = min(f_inc * dt, dt_max)
                dt_half = dt / 2.0
                a_fire = f_a * a_fire
                delta_a_fire = 1.0 - a_fire
        else:
            N_pp = 0
            N_pn += 1
            if N_pn > N_pn_max:
                break
            if (initialdelay < 0.5) or (nt >= N_delay):
                if f_dec * dt > dt_min:
                    dt = f_dec * dt
                    dt_half = dt / 2.0
                a_fire = a_start
                delta_a_fire = 1.0 - a_fire
                
            x -= vx * dt_half
            y -= vy * dt_half
            Ly -= v_w * dt_half
            vx[:] = 0.0
            vy[:] = 0.0
            v_w = 0.0

        # MD using Verlet method
        vx += Fx * dt_half
        vy += Fy * dt_half
        v_w += F_w * dt_half / m_w
        rsc_fire = np.sqrt(np.sum(vx**2 + vy**2 + v_w**2)) / np.sqrt(np.sum(Fx**2 + Fy**2 + F_w**2))
        vx = delta_a_fire * vx + a_fire * rsc_fire * Fx
        vy = delta_a_fire * vy + a_fire * rsc_fire * Fy
        v_w = delta_a_fire * v_w + a_fire * rsc_fire * F_w
        x += vx * dt
        y += vy * dt
        Ly += v_w * dt

        VL_counter = VerletList(N, x, y, D, VL_list, VL_counter, x_save, y_save, 0)
        F_w = Force_FixedTopForce_VL(N, x, y, D, Lx, Ly, k_list, F_top, Fx, Fy, VL_list, VL_counter)

        F_max = np.max([np.max(np.abs(Fx)), np.max(np.abs(Fy)), np.abs(F_w)])
        if F_max < Fthresh:
            break

        vx += Fx * dt_half
        vy += Fy * dt_half
        v_w += F_w * dt_half / m_w

    return Ly