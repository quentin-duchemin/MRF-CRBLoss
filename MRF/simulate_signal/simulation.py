import numpy as np
import scipy as sc
import scipy.io

pi = np.pi

def load_data_6params():
  x = sc.io.loadmat('OCT_resutl_MT_100_3s_low_SAR.mat')['x']
  x[1,:] = x[1,:] * 1e-3 # convert ms to s
  return x

def decimale(a):
  return a - int(a)

def hann_interpolation(t, Tmax, x):
  Nb = len(x)-1
  x1 = np.copy(x[int(min(max(np.floor(t/Tmax*Nb),0),Nb))])
  x2 = np.copy(x[int(min(max(np.ceil (t/Tmax*Nb),0),Nb))])
  xi = (x1 * (1-np.cos(pi * (1 - decimale(t/Tmax*Nb)))) + x2 * (1-np.cos(pi *      decimale(t/Tmax*Nb))))/2
  return xi


def radial_MT_ODE_no_m0f(t,y, theta_t, TRF_t, TR, Tmax, m0s, T1f, T2f, R, T1s, T2s):
  theta = theta_t(t)
  TRF   = TRF_t(t)
  nbar = T2s / 1e3 / TRF / TR

  ct = np.cos(theta)
  st = np.sin(theta)

  A    = [[-ct**2/T1f - st**2/T2f - ct**2*R*m0s, R*ct*(1-m0s), (1-m0s)*ct/T1f ],[R*ct*m0s , - (1/T1s + R*(1-m0s) + nbar*4*theta**2), m0s/T1s],[ 0,0,0]]  
   
  dydt = np.dot(A, y)
  
  return dydt

def simulate_MT_ODE(x, TR, t, m0s, T1f, T2f, R, T1s, T2s):


  Tmax = t[-1]
  theta_t = lambda t: hann_interpolation(t, Tmax, x[0,:])
  TRF_t   = lambda t: hann_interpolation(t, Tmax, x[1,:])

  r0 = np.array([m0s-1, m0s, 1])
  r0 = r0.reshape(-1,1)
  f = lambda t,r: radial_MT_ODE_no_m0f(t,r , theta_t, TRF_t, TR, Tmax, m0s, T1f, T2f, R, T1s, T2s)
  
  ode = sc.integrate.ode(f).set_integrator('vode', atol=1e-12, rtol=1e-9)

  for ir in [1,2]:
      ode.set_initial_value(r0, t[0])
      b= np.zeros((len(t),r0.shape[0]))
      dt = t[1]-t[0]
      b[0,:] = r0.reshape(-1)
      count = 1
      while ode.successful() and ode.t < t[-1]:
        ode.integrate(t[count])
        b[count,:] = np.reshape(ode.y,-1)
        count += 1
      r0 = b[-1,:].reshape((-1,1))
      r0[0:-1:3] = -r0[0:-1:3] # anti periodic boundary conditions for the free pool
      r0[1:-1:3] =  r0[1:-1:3] * (1 - pi**2 * T2s/1e3/max(x[1,:])) # periodic boundary conditions for the semi-solid pool, attenuated by the inversion pulse
  s = b[:,0:-1:3]

  s = np.array([ s[i]*np.sin(theta_t(t[i])) for i in range(len(t))])

  return s, 3
