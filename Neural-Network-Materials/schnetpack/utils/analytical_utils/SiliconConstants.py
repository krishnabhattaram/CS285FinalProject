import numpy as np

__all__ = ['bohr', 'onsp_lambda', 'onsp_s', 'onsp_p',\
           'hp_sssigma', 'hp_spsigma', 'hp_ppsigma', 'hp_pppi',\
           'S_sssigma', 'S_spsigma', 'S_ppsigma', 'S_pppi', 'rc']
           
__all__ += ['cutoff', 'h_il', 'H_sssigma', 'H_spsigma', 'H_ppsigma', 'H_pppi',\
           'E_ss', 'E_sx', 'E_sy', 'E_sz', 'E_xx', 'E_yy', 'E_zz', 'E_xy', 'E_yz', 'E_xz',\
           'oS_sssigma', 'oS_spsigma', 'oS_ppsigma', 'oS_pppi',\
           'S_ss', 'S_sx', 'S_sy', 'S_sz', 'S_xx', 'S_yy', 'S_zz', 'S_xy', 'S_yz', 'S_xz']


########### Constants #############
bohr = 0.529177249
rc = 12.5


# On-site parameters
onsp_lambda = 1.1035662
onsp_s = {"alpha": -0.0532, "beta": -0.907642, "gamma":-8.30849, "chi":56.56613}
onsp_p = {"alpha":0.357859, "beta":0.303647, "gamma":7.092229, "chi": -77.47855}

# Hopping parameters
hp_sssigma = {"a": 219.5608, "b":-16.2132, "c":-15.5048, "d":1.264399}
hp_spsigma = {"a": 10.127, "b":-4.40368, "c":0.22667, "d":0.922671}
hp_ppsigma = {"a": -22.959, "b":1.72, "c":1.41913, "d":1.0313}
hp_pppi = {"a": 10.2654, "b":4.6718, "c":-2.2161, "d":1.1113}

# Overlap parameters
S_sssigma = {"p":5.15758, "q":0.66, "r":-0.0815, "s":1.1081}
S_spsigma = {"p":8.873, "q":-16.24, "r":5.182, "s":1.2406}
S_ppsigma = {"p":11.25, "q":-1.17, "r":-1.0591, "s":1.1376}
S_pppi = {"p":-692.18423, "q":396.1532, "r":-13.81721, "s":1.57248}


def cutoff(r, rc):
  l = 0.5
  return (1 - np.heaviside(r-rc,1))*1/(1 + np.exp((r - rc)/l))
  

def h_il(neighbors, rc):
  rho = 0
  for r in neighbors:
    rho = rho + np.exp(-onsp_lambda**2*r)*cutoff(r, rc)
  
  h_is = onsp_s["alpha"] + onsp_s["beta"]*rho**(2/3) + onsp_s["gamma"]*rho**(4/3) + onsp_s["chi"]*rho**2
  h_ip = onsp_p["alpha"] + onsp_p["beta"]*rho**(2/3) + onsp_p["gamma"]*rho**(4/3) + onsp_p["chi"]*rho**2
  return h_is, h_ip 
  
  

def H_sssigma(r, rc):
  return (hp_sssigma["a"] + hp_sssigma["b"]*r + hp_sssigma["c"]*r**2)*np.exp(-hp_sssigma["d"]**2*r)*cutoff(r,rc)
  
def H_spsigma(r, rc):
  return (hp_spsigma["a"] + hp_spsigma["b"]*r + hp_spsigma["c"]*r**2)*np.exp(-hp_spsigma["d"]**2*r)*cutoff(r,rc)
  
def H_ppsigma(r, rc):
  return (hp_ppsigma["a"] + hp_ppsigma["b"]*r + hp_ppsigma["c"]*r**2)*np.exp(-hp_ppsigma["d"]**2*r)*cutoff(r,rc)
  
def H_pppi(r, rc): 
  return (hp_pppi["a"] + hp_pppi["b"]*r + hp_pppi["c"]*r**2)*np.exp(-hp_pppi["d"]**2*r)*cutoff(r,rc)

def E_ss(l, m, n, rc):
  r = np.sqrt(l**2 + m**2 + n**2)
  if(r == 0):
    return 0
  
  return H_sssigma(r, rc)

def E_sx(l, m, n, rc):
  r = np.sqrt(l**2 + m**2 + n**2)  
  if(r == 0):
    return 0
  else:
    return l/r*H_spsigma(r, rc)
  
def E_sy(l, m, n, rc):
  r = np.sqrt(l**2 + m**2 + n**2) 
  if(r == 0):
    return 0
  else: 
    return m/r*H_spsigma(r, rc)

def E_sz(l, m, n, rc):
  r = np.sqrt(l**2 + m**2 + n**2)  
  if(r == 0):
    return 0
  else:
    return n/r*H_spsigma(r, rc)
  
def E_xx(l, m, n, rc):
  r = np.sqrt(l**2 + m**2 + n**2)  
  if(r == 0):
    return 0
  else:
    return (l/r)**2*H_ppsigma(r, rc) + (1 - (l/r)**2)*H_pppi(r, rc)
  
def E_yy(l, m, n, rc):
  r = np.sqrt(l**2 + m**2 + n**2)  
  if(r == 0):
    return 0
  else:
    return (m/r)**2*H_ppsigma(r, rc) + (1 - (m/r)**2)*H_pppi(r, rc)
  
def E_zz(l, m, n, rc):
  r = np.sqrt(l**2 + m**2 + n**2)  
  if(r == 0):
    return 0
  else:
    return (n/r)**2*H_ppsigma(r, rc) + (1 - (n/r)**2)*H_pppi(r, rc)
  
def E_xy(l, m, n, rc):
  r = np.sqrt(l**2 + m**2 + n**2)  
  if(r == 0):
    return 0
  else:
    return l*m/r**2*H_ppsigma(r, rc) - l*m/r**2*H_pppi(r, rc)  
  
def E_yz(l, m, n, rc):
  r = np.sqrt(l**2 + m**2 + n**2)
  if(r == 0):
    return 0
  else:  
    return m*n/r**2*H_ppsigma(r, rc) - m*n/r**2*H_pppi(r, rc)
  
def E_xz(l, m, n, rc):
  r = np.sqrt(l**2 + m**2 + n**2)
  if(r == 0):
    return 0
  else:  
    return l*n/r**2*H_ppsigma(r, rc) - l*n/r**2*H_pppi(r, rc)
  
def oS_sssigma(r, rc): 
  return (1 + S_sssigma["p"]*r + S_sssigma["q"]*r**2 + S_sssigma["r"]*r**3)*np.exp(-S_sssigma["s"]**2*r)*cutoff(r,rc)
  
def oS_spsigma(r, rc): 
  return (S_spsigma["p"]*r + S_spsigma["q"]*r**2 + S_spsigma["r"]*r**3)*np.exp(-S_spsigma["s"]**2*r)*cutoff(r,rc)
  
def oS_ppsigma(r, rc): 
  return (1 + S_ppsigma["p"]*r + S_ppsigma["q"]*r**2 + S_ppsigma["r"]*r**3)*np.exp(-S_ppsigma["s"]**2*r)*cutoff(r,rc)
  
def oS_pppi(r, rc): 
  return (1 + S_pppi["p"]*r + S_pppi["q"]*r**2 + S_pppi["r"]*r**3)*np.exp(-S_pppi["s"]**2*r)*cutoff(r,rc)



def S_ss(l, m, n, rc):
  r = np.sqrt(l**2 + m**2 + n**2)
  if(r == 0):
    return 1 
  else:
    return oS_sssigma(r, rc)

def S_sx(l, m, n, rc):
  r = np.sqrt(l**2 + m**2 + n**2)  
  if(r == 0):
    return 0
  else:
    return l/r*oS_spsigma(r, rc)
  
def S_sy(l, m, n, rc):
  r = np.sqrt(l**2 + m**2 + n**2) 
  if(r == 0):
    return 0
  else: 
    return m/r*oS_spsigma(r, rc)

def S_sz(l, m, n, rc):
  r = np.sqrt(l**2 + m**2 + n**2)  
  if(r == 0):
    return 0
  else:
    return n/r*oS_spsigma(r, rc)
  
def S_xx(l, m, n, rc):
  r = np.sqrt(l**2 + m**2 + n**2)  
  if(r == 0):
    return 1
  else:
    return (l/r)**2*oS_ppsigma(r, rc) + (1 - (l/r)**2)*oS_pppi(r, rc)
  
def S_yy(l, m, n, rc):
  r = np.sqrt(l**2 + m**2 + n**2)  
  if(r == 0):
    return 1
  else:
    return (m/r)**2*oS_ppsigma(r, rc) + (1 - (m/r)**2)*oS_pppi(r, rc)
  
def S_zz(l, m, n, rc):
  r = np.sqrt(l**2 + m**2 + n**2)  
  if(r == 0):
    return 1
  else:
    return (n/r)**2*oS_ppsigma(r, rc) + (1 - (n/r)**2)*oS_pppi(r, rc)
  
def S_xy(l, m, n, rc):
  r = np.sqrt(l**2 + m**2 + n**2)  
  if(r == 0):
    return 0
  else:
    return l*m/r**2*oS_ppsigma(r, rc) - l*m/r**2*oS_pppi(r, rc)  
  
def S_yz(l, m, n, rc):
  r = np.sqrt(l**2 + m**2 + n**2)
  if(r == 0):
    return 0
  else:  
    return m*n/r**2*oS_ppsigma(r, rc) - m*n/r**2*oS_pppi(r, rc)
  
def S_xz(l, m, n, rc):
  r = np.sqrt(l**2 + m**2 + n**2)
  if(r == 0):
    return 0
  else:  
    return l*n/r**2*oS_ppsigma(r, rc) - l*n/r**2*oS_pppi(r, rc)
