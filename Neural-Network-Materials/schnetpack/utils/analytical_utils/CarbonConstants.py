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
rc = 10.5


# On-site parameters
onsp_lambda = 1.599019
onsp_s = {"alpha": -0.1027899, "beta": -1.626, "gamma":-178.88, "chi":4516.113}
onsp_p = {"alpha":0.5426, "beta":2.7345, "gamma":-67.1397, "chi": 438.5288}
 
# Hopping parameters
hp_sssigma = {"a":  74.0837449667, "b":-18.3225697598, "c": -12.5253007169, "d":1.4110052180}
hp_spsigma = {"a": -7.9172955767, "b": 3.6163510241, "c": 1.0416715714, "d": 1.1687890843}
hp_ppsigma = {"a": -5.7016933899, "b": 1.0450894823, "c": 1.5062731505, "d": 1.1362744013}
hp_pppi = {"a": 24.9104111573, "b": -5.0603652530, "c": -3.6844386855, "d": 1.3654891930}

# Overlap parameters
S_sssigma = {"p": 0.18525064246, "q": 1.56010486948, "r": -0.308751658739, "s": 1.137005}
S_spsigma = {"p": 1.85250642463, "q": -2.50183774417, "r": 0.178540723033, "s": 1.129003}
S_ppsigma = {"p":-1.29666913067, "q": 0.28270660019, "r": -0.022234235553, "s": 0.761776}
S_pppi = {"p": 0.74092406925, "q": -0.07310263856, "r": 0.016694077196, "s": 1.021482}
  
  
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

