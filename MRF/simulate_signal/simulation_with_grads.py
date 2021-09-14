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

def hann_interpolation_with_grads(t, Tmax, x):
  Nb = len(x)-1
  x1 = np.copy(x[int(min(max(np.floor(t/Tmax*Nb),0),Nb))])
  x2 = np.copy(x[int(min(max(np.ceil (t/Tmax*Nb),0),Nb))])
  xi = (x1 * (1-np.cos(pi * (1 - decimale(t/Tmax*Nb)))) + x2 * (1-np.cos(pi *      decimale(t/Tmax*Nb))))/2
  return xi


def radial_MT_ODE_no_m0f_with_grads(t,y, theta_t, TRF_t, TR, Tmax, m0s, T1f, T2f, R, T1s, T2s):
  theta = theta_t(t)
  TRF   = TRF_t(t)
  nbar = T2s / 1e3 / TRF / TR

  ct = np.cos(theta)
  st = np.sin(theta)

  A    = [[-ct**2/T1f - st**2/T2f - ct**2*R*m0s, R*ct*(1-m0s), (1-m0s)*ct/T1f ],[R*ct*m0s , - (1/T1s + R*(1-m0s) + nbar*4*theta**2), m0s/T1s],[ 0,0,0]]
  

  dAdx = np.zeros((18,3))
  dAdx[0,:] = [-ct**2 *R,-R*ct,-ct/T1f]
  dAdx[1,:] = [R*ct,R,1/T1s]
  dAdx[2,:] = [0,0,0]
  dAdx[3,:] = [ct**2 / (T1f**2), 0, -(1-m0s)*ct/ (T1f**2)]
  dAdx[4,:] = [0, 0, 0]
  dAdx[5,:] = [0,0,0]
  dAdx[6,:] = [st**2/(T2f**2),        0        ,        0  ]
  dAdx[7,:] = [ 0     ,        0        ,        0]
  dAdx[8,:] = [0     ,        0        ,        0   ]
  dAdx[9,:] = [ -ct**2 *m0s ,   ct*(1-m0s)    ,        0 ]
  dAdx[10,:] = [ct*m0s   ,    -(1-m0s)     ,        0]
  dAdx[11,:] = [ 0     ,        0        ,        0  ]
  dAdx[12,:] = [ 0     ,        0        ,        0 ]
  dAdx[13,:] =  [0     ,     1/(T1s**2)     ,   -m0s/(T1s**2)]
  dAdx[14,:] = [ 0     ,        0        ,        0   ]
  dAdx[15,:] = [0     ,        0        ,        0  ]
  dAdx[16,:] = [0     , -4e-3*(theta**2)/TRF/TR,    0 ]
  dAdx[17,:] = [0     ,        0        ,        0  ]
  
  dydt = np.dot(A, y.reshape(3,7,order='F'))
  
  dydt = dydt.reshape(-1,1,order='F')
  yidx = np.zeros(len(y), dtype=bool)
  yidx[0:len(y):21] = True
  yidx[1:len(y):21] = True
  yidx[2:len(y):21] = True
  dydt[np.invert(yidx)] = dydt[np.invert(yidx)] + np.reshape( np.dot(dAdx, y[yidx].reshape(3, 1)), (-1, 1))
  return dydt

def simulate_MT_ODE_with_grads(x, TR, t, m0s, T1f, T2f, R, T1s, T2s):

  Tmax = t[-1]
  theta_t = lambda t: hann_interpolation_with_grads(t, Tmax, x[0,:])
  TRF_t   = lambda t: hann_interpolation_with_grads(t, Tmax, x[1,:])

  r0 = np.array([m0s-1, m0s, 1, 1, 1, 0])
  r0 = r0.reshape(-1,1)
  r0 = np.vstack((r0,np.zeros((15,1))))
  f = lambda t,r: radial_MT_ODE_no_m0f_with_grads(t,r , theta_t, TRF_t, TR, Tmax, m0s, T1f, T2f, R, T1s, T2s)
  
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
  ds = s[:,7:-1]
  s = s[:,0:7] # 1st is the signal, 2:8 are the derivatives wrt. T1 etc. 

  
  ls = []
  for i in range(len(t)):
    ls.append(np.sin(theta_t(t[i])))
  ls = np.array(ls)
  s =  s *np.tile(ls.reshape(-1,1),(1,7))

  return s, ds
