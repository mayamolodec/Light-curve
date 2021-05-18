#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np
from numpy import sin,cos,pi,sqrt # makes the code more readable
from scipy.optimize import newton
import plotly.graph_objects as go
from sympy import Symbol
from sympy.solvers import solve
import matplotlib.pyplot as plt
import operator
import time
from astropy.io import fits


# # Functions

# In[ ]:


def roche(r, eta, phi, pot, q):
    
    # Roche potential function
    
    lam, nu = cos(phi) * sin(eta), cos(eta)  # ordinary direct cos
    
    return (pot - (1. / r + q * (1. / np.sqrt(1. - 2 * lam * r + r**2) - lam * r) + 0.5 * (q + 1) * r**2 * (1 - nu**2)))


def r_get(r_init, roche, eta, phi):
    
    # Function for finding r ( we know q and potential value)
    
    r1 = [newton(roche, r_init, args=(th, ph, pot1, q)) for th, ph in zip(eta.ravel(), phi.ravel())]
    r1 = np.array(r1)
    return r1


def pot_get(q, mu_1):
    
    # L1 dot coordinates -> (zeta*,0,0)
    
    x_coord = Symbol('x_coord')
    xi_all = solve(-1 / (x_coord**2) - q * (1 - 1 / ((x_coord - 1)**2)) + x_coord * (1 + q))
    
    xi = np.float([n for n in xi_all if n.is_real][0])  # Here you shold fin minimal real root
    

    pot_cr = 1 / xi + q * (1 / (1 - xi) - xi) + (xi**2) * (1 + q) / 2     # Here we getting critical potential value OMEGA*                                                
    
    lam_cr, nu_cr = 0, 1    # lam_cr, nu_cr, pot_cr = 0, 1, pot_cr
    
    #R0 finding
    
    r = Symbol('r')
    r_all = solve(pot_cr - (1. / r + q * (1. / (1. - 2 * lam_cr * r + r**2)**0.5 - lam_cr * r) + 0.5 * (q + 1) * r**2 * (1 - nu_cr**2)), r)   

    r_crit = np.float([n for n in r_all if n.is_real][0])
    r01 = mu_1 * r_crit
    
    # Findind real potential value OMEGA
    omega_1 = 1. / r01 + q * (1. / (1. - 2 * lam_cr * r01 + r01**2)**0.5 - lam_cr * r01) + 0.5 * (q + 1) * r01**2 * (1 - nu_cr**2)  
           
    return omega_1


def dir_cos(eta, phi):
    
    # Finding direct cos in cyllindrical system
    
    lam = cos(phi) * sin(eta)
    mu = sin(phi) * sin(eta)
    nu = cos(eta)
    return(lam, mu, nu)


def decart_get(r1, lam, mu, nu):
    
    #Finding Decat coorinates of surface dots
    
    x1 = r1 * lam
    y1 = r1 * mu
    z1 = r1 * nu
    return (x1, y1, z1)



def grad_xyz(x1, y1, z1, q):
    
    # Finding privat derivatives in Decart coordinate system
    
    r = np.sqrt(x1**2+y1**2+z1**2)
    
    grad_x = -x1/(r**3) + q*((1 - x1)/(1 - 2*x1 + r**2)**1.5 - 1) + (q + 1)*x1
    grad_y = -y1/(r**3) + q*(-y1/(1 - 2*x1 + r**2)**1.5) + (q + 1)*y1
    grad_z = -z1/(r**3) + q*(-z1/(1 - 2*x1 + r**2)**1.5)
    
    return grad_x, grad_y, grad_z


def surface_elem(r1, eta, cos_b, dots_eta_step, dots_phi_step):
    
    # Finding surface element square
    
    ds = ((r1**2)*sin(eta)*(np.pi/dots_eta_step)*(2*np.pi/dots_phi_step))/(cos_b)
    return ds


def T_get(grad_x, grad_y, grad_z, beta, ds):
    
    # Finding Temperature in every surface dot
    
    g_xyz = sqrt(grad_x ** 2 + grad_y**2 + grad_z**2)
    
    #g_0 = np.mean(g_xyz)
    g_0 = np.average(g_xyz, weights = ds)
    T = T_0 * pow((g_xyz / g_0), beta)
    T = np.round(T, 5) 
    return T


def normal_vector(grad_x, grad_y, grad_z):
    
    # Finding normal vectors in every surface dot
    
    g_xyz = np.sqrt(grad_x ** 2 + grad_y**2 + grad_z**2)
    n1 = np.array([-grad_x / g_xyz, -grad_y / g_xyz, -grad_z / g_xyz])
    
    return n1



def planck(lam, T):
    
    # Plank function equation 
    from scipy.constants import h,k,c
    
    lam = 1e-9 * lam  # from nm to m
    return 2*h*c**2 / (lam**5 * (np.exp(h*c / (lam*k*T)) - 1))



def model_spectra(Temper, mJ, mH, mK, mB, mV, T):
    
    # Finding Model fluxes in different filters - convolution
    
    j = np.interp(T, Temper, mJ)
    h = np.interp(T, Temper, mH)
    k = np.interp(T, Temper, mK)
    b = np.interp(T, Temper, mB)
    v = np.interp(T, Temper, mV)
    
    return (j, h, k, b, v)


def limb_darfening(I, mu, teff, T, n_eta, n_phi):
    
    # Creates an array ob limbdarks values
    
    mu_new = np.zeros((n_eta*n_phi, 78))
    I_new = np.zeros((n_eta*n_phi, 78))
    for i in range(n_eta*n_phi):
        mas = np.argmin(np.abs(teff - T[i]))
        mu_new[i, :] = mu[mas, :]
        I_new[i, :] = I[mas, :]
    return mu_new, I_new


def spectra(lam, T, n_eta, n_phi, filtr):
    
    # Finding Black Body fluxes in different filters - convolution
    
    spectrum = np.zeros((n_eta*n_phi))
    d_lam = 1 * 10**(-9)                  # 1 nanometer
    for i in range(n_eta*n_phi):
        spectrum[i] = np.sum(d_lam * filtr * planck(lam, T[i])
    return spectrum


def flux_count(n_eta, n_phi, n1, i, ds, spectrum, theta, mu_new, I_new):
    flux = np.zeros((len(theta)))
    a_N = np.zeros((len(theta), 3))
    I_dark = np.zeros((n_eta*n_phi))
    
    for index_a in range(len(theta)):
        a0 = np.array([sin(i) * cos(theta[index_a]), sin(i) * sin(theta[index_a]), cos(i)])
        a_N[index_a] = a0

        cos_gamma = a0[0] * n1[0, :] + a0[1] * n1[1, :] + a0[2] * n1[2, :]
        for item in range(n_eta*n_phi):
            I_dark[item] = I_new[item, np.argmin(np.abs(cos_gamma[item] - mu_new[item,:]))]

        spectrum_d = np.where(cos_gamma > 0, spectrum, 0) # If cos_gamma > 0 -> We can see that part of surface
        dI = spectrum_d * I_dark * cos_gamma * ds
        I = np.sum(dI)
        d_flux = -2.5 * np.log10(I)
        flux[index_a] = d_flux

    return(flux, (180 * theta / np.pi)/360-0.5)


# # Parameters

# In[ ]:


r_init = 0.01         # Initial value of r
q = 1                 # Masses ratio
i = 90 * np.pi / 180  # i angle in radians
i_deg = 90            # i angle in degrees
mu_1 = 1              # Omega/Omega_critical
T_0 = 4500            # K            
beta = 0.08          # Gravitational darkening degree


eta,phi = np.mgrid[0:np.pi:500j,0:2*np.pi:500j]  # Eta ond phi andles diapasones


theta = np.arange(180, 540, 3)*np.pi/180         # Angle of rotation 
dots_eta_step = 500                              # Angle eta steps 
dots_phi_step = 500                              # Angle phi steps 

# Filter passbands functions

lamJ, filter_J = np.loadtxt('F040JMKO.txt', skiprows=1, unpack=True)[:, ::-1]  # Wavelengh in nm
lamH, filter_H = np.loadtxt('F047HMKO.txt', skiprows=1, unpack=True)[:, ::-1]  # Wavelengh in nm     grid = 1 nm
lamK, filter_K = np.loadtxt('F049KMKO.txt', skiprows=1, unpack=True)[:, ::-1]  # Wavelengh in nm
lamV, filter_V = np.loadtxt('F003VBES.txt', skiprows=1, unpack=True)[:,::-1]   # Wavelengh in nm      
lamB, filter_B = np.loadtxt('F002BBES.txt', skiprows=1, unpack=True)[:,::-1]   # Wavelengh in nm

#filter converved model fluxes

Temper, mJ, mH, mK, mB, mV = np.loadtxt('Phoenix_flux_z0=0_logg=1.5.txt', delimiter=' ', unpack=True)

#Limb darkening coefficients
a = fits.open('LimbDark_full_stuck_logg=1.5.fits')

tempJ = np.array(a[1].data.field(0))
tempK = np.array(a[1].data.field(1))
tempV = np.array(a[1].data.field(2))
tempH = np.array(a[1].data.field(3))
tempB = np.array(a[1].data.field(4))
J = np.array(a[1].data.field(5))
H = np.array(a[1].data.field(6))
K = np.array(a[1].data.field(7)) 
B = np.array(a[1].data.field(8)) 
V = np.array(a[1].data.field(9)) 
muJ = np.array(a[1].data.field(10))
muH = np.array(a[1].data.field(11))
muK = np.array(a[1].data.field(12))
muB = np.array(a[1].data.field(13))
muV = np.array(a[1].data.field(14))


# # Code

# In[ ]:


pot1 = pot_get(q, mu_1)                   # Finding potential

r1_long = r_get(r_init, roche, eta, phi)   # Finding radius for every surface dot

eta_long = np.reshape(eta, dots_eta_step*dots_phi_step)
phi_long = np.reshape(phi, dots_eta_step*dots_phi_step)

lam, mu, nu = dir_cos(eta_long, phi_long)           # Finding dir cos
x1, y1, z1 = decart_get(r1_long, lam, mu, nu)       # Decart coordinates of surface dots

grad_x, grad_y, grad_z = grad_xyz(x1, y1, z1, q)    # Private derivatives

normal_v = normal_vector(grad_x, grad_y, grad_z)    # Finding normal vector in every surface dot

cos_b = lam*normal_v[0]+mu*normal_v[1]+nu*normal_v[2]  # Cos of an angle between normal and dir cos of surface dot

ds = surface_elem(r1_long, eta_long, cos_b, dots_eta_step, dots_phi_step)  # Finding sqare of every surface part
                  
T = T_get(grad_x, grad_y, grad_z, beta, ds)         # Temperature determination

mu_J, dark_J = limb_darfening(J, muJ, tempJ, T, dots_eta_step, dots_phi_step)       # LimbDark coefficients
mu_K, dark_K = limb_darfening(K, muK, tempK, T, dots_eta_step, dots_phi_step)  
mu_V, dark_V = limb_darfening(V, muV, tempV, T, dots_eta_step, dots_phi_step)
mu_H, dark_H = limb_darfening(H, muH, tempH, T, dots_eta_step, dots_phi_step)
mu_B, dark_B = limb_darfening(B, muB, tempB, T, dots_eta_step, dots_phi_step) 
 

Black_Body_J = spectra(lamJ, T, dots_eta_step, dots_phi_step, filter_J/100)          # BlackBody spectra
Black_Body_K = spectra(lamK, T, dots_eta_step, dots_phi_step, filter_K/100)
Black_Body_V = spectra(lamV, T, dots_eta_step, dots_phi_step, filter_V/100)
Black_Body_H = spectra(lamH, T, dots_eta_step, dots_phi_step, filter_H/100)
Black_Body_B = spectra(lamB, T, dots_eta_step, dots_phi_step, filter_B/100)

flux_BB_J, phase_BB_J = flux_count(dots_eta_step, dots_phi_step, normal_v, i, ds, Black_Body_J, theta, mu_J, dark_J)   # Light curve for black body model 
flux_BB_K, phase_BB_K = flux_count(dots_eta_step, dots_phi_step, normal_v, i, ds, Black_Body_K, theta, mu_K, dark_K)   
flux_BB_V, phase_BB_V = flux_count(dots_eta_step, dots_phi_step, normal_v, i, ds, Black_Body_V, theta, mu_V, dark_V)    
flux_BB_H, phase_BB_H = flux_count(dots_eta_step, dots_phi_step, normal_v, i, ds, Black_Body_H, theta, mu_H, dark_H)   
flux_BB_B, phase_BB_B = flux_count(dots_eta_step, dots_phi_step, normal_v, i, ds, Black_Body_B, theta, mu_B, dark_B)   

#Here we calculate lightcurve for syntetic spertra

J_filt, H_filt, K_filt, B_filt, V_filt = model_spectra(Temper, mJ, mH, mK, mB, mV, T)                    # Model spectra

fluxJ, phaseJ = flux_count(dots_eta_step, dots_phi_step, normal_v, i, ds, J_filt, theta, mu_J, dark_J)   # Light curve for syntetic atmosphere
fluxK, phaseK = flux_count(dots_eta_step, dots_phi_step, normal_v, i, ds, K_filt, theta, mu_K, dark_K)
fluxV, phaseV = flux_count(dots_eta_step, dots_phi_step, normal_v, i, ds, V_filt, theta, mu_V, dark_V)
fluxH, phaseH = flux_count(dots_eta_step, dots_phi_step, normal_v, i, ds, H_filt, theta, mu_H, dark_H)
fluxB, phaseB = flux_count(dots_eta_step, dots_phi_step, normal_v, i, ds, B_filt, theta, mu_B, dark_B)


# # Drawing

# In[ ]:


# Light curve drawing

fig = go.Figure()
fig = fig.add_trace(go.Scatter(name="BB_J", x=phase_BB_K, y=flux_BB_K, mode='markers'))
fig = fig.add_trace(go.Scatter(name="MODEL_J", x=phaseJ, y=fluxJ, mode='markers'))
fig.update_yaxes(autorange="reversed")
fig.show()


# In[ ]:


# Save in files 

text = np.vstack((phaseJ, fluxJ, fluxK, fluxV, fluxB, fluxH))
np.savetxt('Model_mu=%s_q=%s_i=%s_T=%s.txt' % (mu_1, q, i_deg, T_0), text.T, fmt='%.6f', header='Phase   ' +  'fluxJ    fluxK   fluxV   fluxB   fluxH')

text1 = np.vstack((phase_BB_J, flux_BB_J, flux_BB_K, flux_BB_V, flux_BB_B, flux_BB_H))
np.savetxt('BB_mu=%s_q=%s_i=%s_T=%s.txt' % (mu_1, q, i_deg, T_0), text1.T, fmt='%.6f', header='Phase   ' +  'fluxJ    fluxK   fluxV   fluxB   fluxH')


# In[ ]:


#Roche lobe shape. Tempetarure is a color.

fig3 = go.Figure(data=[go.Scatter3d(x=x1, y=y1, z=z1,
                                   mode='markers',
                                   marker=dict(
        size=5,
        color=T,                # set color to an array/list of desired values cos_gam_N, spectrum_d_N, dI_N
        colorscale='Viridis',
        colorbar = dict(thickness = 10),
        opacity=1
    ))])


fig3.update_layout(scene = dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'),
                    width=700,
                    margin=dict(r=20, b=10, l=10, t=10))
fig3.show()

