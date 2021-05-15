# -*- coding: utf-8 -*-
"""
2D Riemann Solver
Piecewise Parabolic Reconstruction
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as an

nx=101 #number of x cells
ny=11 #number of y cells
tstop=0.2 #end time
gamma=1.4 #heat capacity ratio for a diatomic gas
xmax=1 #max x value, min x value=0
ymax=1 #max y value, min y value=0
dx=xmax/(nx-1) #cell x width
dy=ymax/(ny-1) #cell y width
t=0 #initial time
step=0 #initial step

#initialise 1D arrays
x=np.linspace(0,xmax,nx)
y=np.linspace(0,ymax,ny)

#initialise 2D arrays
rho=np.zeros((ny,nx))
p=np.zeros((ny,nx))
u=np.zeros((ny,nx))
v=np.zeros((ny,nx))
totspeed=np.zeros((ny,nx))
c=np.zeros((ny,nx))
X, Y = np.meshgrid(x, y)

#initialise 3D arrays
U=np.zeros((ny,nx,4))
Uxminus=np.zeros((ny,nx,4))
Uyminus=np.zeros((ny,nx,4))
Uyplus=np.zeros((ny,nx,4))
Uxplus=np.zeros((ny,nx,4))
rx=np.zeros((ny,nx,4))
ry=np.zeros((ny,nx,4))
phix=np.zeros((ny,nx,4))
phiy=np.zeros((ny,nx,4))
fluxxtotplus=np.zeros((ny,nx,4))
fluxytotplus=np.zeros((ny,nx,4))
fluxxtotminus=np.zeros((ny,nx,4))
fluxytotminus=np.zeros((ny,nx,4))

#Cell flux in x direction
def fluxx(w):
    Fx=np.zeros((ny,nx,4))
    for i in range(0,nx):
        for j in range(0,ny):
            Fx[j][i][0]=w[j][i][1]
            Fx[j][i][1]=w[j][i][1]**2/w[j][i][0]+(gamma-1)*(w[j][i][3]-0.5*w[j][i][1]**2/w[j][i][0]-0.5*w[j][i][2]**2/w[j][i][0])
            Fx[j][i][2]=w[j][i][1]*w[j][i][2]/w[j][i][0]
            Fx[j][i][3]=w[j][i][1]/w[j][i][0]*(w[j][i][3]+(gamma-1)*(w[j][i][3]-0.5*w[j][i][1]**2/w[j][i][0]-0.5*w[j][i][2]**2/w[j][i][0]))
    return Fx

#Cell flux in y direction
def fluxy(w):
    Fy=np.zeros((ny,nx,4))
    for i in range(0,nx):
        for j in range(0,ny):
            Fy[j][i][0]=w[j][i][2]
            Fy[j][i][1]=w[j][i][1]*w[j][i][2]/w[j][i][0]
            Fy[j][i][2]=w[j][i][2]**2/w[j][i][0]+(gamma-1)*(w[j][i][3]-0.5*w[j][i][1]**2/w[j][i][0]-0.5*w[j][i][2]**2/w[j][i][0])
            Fy[j][i][3]=w[j][i][2]/w[j][i][0]*(w[j][i][3]+(gamma-1)*(w[j][i][3]-0.5*w[j][i][1]**2/w[j][i][0]-0.5*w[j][i][2]**2/w[j][i][0]))
    return Fy                                             
                                                 
#initial conditions
for j in range(0,ny):
    for i in range(0,nx):
        if x[i]<0.5:
            rho[j][i]=1.0
            u[j][i]=0.0
            v[j][i]=0.0
            p[j][i]=1.0
        else:
            rho[j][i]=0.125
            u[j][i]=0.0
            v[j][i]=0.0
            p[j][i]=0.1
        
        U[j][i][0]=rho[j][i]
        U[j][i][1]=rho[j][i]*u[j][i]
        U[j][i][2]=rho[j][i]*v[j][i]
        U[j][i][3]=p[j][i]/(gamma-1)+0.5*rho[j][i]*u[j][i]**2+0.5*rho[j][i]*v[j][i]**2

density=rho
pressure=p
xvelocity=u

while t<tstop:
    for i in range(0,nx):
        for j in range(0,ny):
            c[j][i]=np.sqrt(gamma*p[j][i]/rho[j][i])
    dt=0.3*min(dx/np.amax(np.absolute(u)+c),dy/np.amax(np.absolute(v)+c)) #calculates timestep from Courant number
    t+=dt
    for k in range(0,4):
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                if U[j+1][i][k]!=U[j][i][k]:
                    ry[j][i][k]=(U[j][i][k]-U[j-1][i][k])/(U[j+1][i][k]-U[j][i][k])
                else:
                    ry[j][i][k]=1000000
                if U[j][i+1][k]!=U[j][i][k]:
                    rx[j][i][k]=(U[j][i][k]-U[j][i-1][k])/(U[j][i+1][k]-U[j][i][k])
                else:
                    rx[j][i][k]=1000000
    for k in range(0,4):
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                phix[j][i][k]=(2*rx[j][i][k])/(1+rx[j][i][k]**2)
                phiy[j][i][k]=(2*ry[j][i][k])/(1+ry[j][i][k]**2)
    for k in range(0,4):
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                Uyplus[j][i][k]=U[j][i][k]+0.25*phiy[j][i][k]*(2/3*(U[j][i][k]-U[j-1][i][k])+4/3*(U[j+1][i][k]-U[j][i][k]))
                Uxplus[j][i][k]=U[j][i][k]+0.25*phix[j][i][k]*(2/3*(U[j][i][k]-U[j][i-1][k])+4/3*(U[j][i+1][k]-U[j][i][k]))
                Uyminus[j][i][k]=U[j][i][k]-0.25*phiy[j][i][k]*(2/3*(U[j+1][i][k]-U[j][i][k])+4/3*(U[j][i][k]-U[j-1][i][k]))
                Uxminus[j][i][k]=U[j][i][k]-0.25*phix[j][i][k]*(2/3*(U[j][i+1][k]-U[j][i][k])+4/3*(U[j][i][k]-U[j][i-1][k]))
        for i in range(0,nx):
            Uyplus[ny-1][i][k]=Uyplus[ny-2][i][k]
            Uyminus[ny-1][i][k]=Uyminus[ny-2][i][k]
            Uyplus[0][i][k]=Uyplus[1][i][k]
            Uyminus[0][i][k]=Uyminus[1][i][k]
        for j in range(0,ny):
            Uxplus[j][nx-1][k]=Uxplus[j][nx-2][k]
            Uxminus[j][nx-1][k]=Uxminus[j][nx-2][k]
            Uxplus[j][0][k]=Uxplus[j][1][k]
            Uxminus[j][0][k]=Uxminus[j][1][k]
    fluxyplus=fluxy(Uyplus)
    fluxxplus=fluxx(Uxplus)
    fluxyminus=fluxy(Uyminus)
    fluxxminus=fluxx(Uxminus)
    for k in range(0,4):
        for i in range(1,nx-1):
            for j in range(1,ny-1):
                fluxytotminus[j][i][k]=0.5*(fluxyminus[j][i][k]+fluxyplus[j-1][i][k]-max(c[j-1][i],c[j][i])*(Uyminus[j][i][k]-Uyplus[j-1][i][k]))
                fluxytotplus[j][i][k]=0.5*(fluxyminus[j+1][i][k]+fluxyplus[j][i][k]-max(c[j+1][i],c[j][i])*(Uyminus[j+1][i][k]-Uyplus[j][i][k]))
                fluxxtotminus[j][i][k]=0.5*(fluxxminus[j][i][k]+fluxxplus[j][i-1][k]-max(c[j][i-1],c[j][i])*(Uxminus[j][i][k]-Uxplus[j][i-1][k]))
                fluxxtotplus[j][i][k]=0.5*(fluxxminus[j][i+1][k]+fluxxplus[j][i][k]-max(c[j][i+1],c[j][i])*(Uxminus[j][i+1][k]-Uxplus[j][i][k]))
    Ucopy=U.copy()
    for k in range(0,4):
        for i in range(1,nx-1):
            for j in range(0,ny):
                U[j][i][k]=Ucopy[j][i][k]-dt/dx*(fluxxtotplus[j][i][k]-fluxxtotminus[j][i][k])-dt/dy*(fluxytotplus[j][i][k]-fluxytotminus[j][i][k])
            U[0][i][k]=U[1][i][k]
    for i in range(0,nx):
        for j in range(0,ny):
            p[j][i]=(gamma-1)*(U[j][i][3]-0.5*U[j][i][1]**2/U[j][i][0]-0.5*U[j][i][2]**2/U[j][i][0])
            u[j][i]=U[j][i][1]/U[j][i][0]
            v[j][i]=U[j][i][2]/U[j][i][0]
            rho[j][i]=U[j][i][0]
    step+=1
    mtot=0
    for i in range(0,nx):
        for j in range(0,ny):
            mtot+=rho[j][i]*dx*dy
    print('Step {:<5} Time-Step {:.3e}  Time {:.4e}  Mass {:.5e}'.format(step,dt,t,mtot))
    density=np.dstack((density,rho))
    pressure=np.dstack((pressure,p))
    xvelocity=np.dstack((xvelocity,u))
    
#Pseduocolour plot function
def varplot(variable,title):
    plt.figure()
    plt.pcolor(x,y,variable,cmap='RdBu')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()

#pseuodcolour plots
varplot(rho,'Density')
varplot(p,'Pressure')
varplot(u,'x-Velocity')
varplot(v,'y-Velocity')
plt.show()

#cross-section plots
densitycs=[]
pressurecs=[]
xvelocitycs=[]
for j in range(0,ny):
    for i in range(0,nx):
        if j==(ny-1)/2:
            densitycs.append(rho[j][i])
            pressurecs.append(p[j][i])
            xvelocitycs.append(u[j][i])
            
def csplot(variable,title):
    plt.figure()
    plt.plot(x,variable)
    plt.xlabel('x')
    plt.title(title)

csplot(densitycs,'Density')
csplot(pressurecs,'Pressure')
csplot(xvelocitycs,'x-Velocity')
            
#animations
densityani=plt.figure()
ims_density=[]
for add in np.arange(step):
    ims_density.append((plt.pcolor(x,y,density[:,:,add]),))
plt.colorbar()
plt.title('Density')

pressureani=plt.figure()
ims_pressure=[]
for add in np.arange(step):
    ims_pressure.append((plt.pcolor(x,y,pressure[:,:,add]),))   
plt.colorbar()
plt.title('Pressure')
 
xvelocityani=plt.figure()
ims_xvelocity=[]
for add in np.arange(step):
    ims_xvelocity.append((plt.pcolor(x,y,xvelocity[:,:,add]),))
plt.colorbar()
plt.title('x-Velocity')

ani=an.ArtistAnimation(densityani,ims_density,interval=200,repeat_delay=3000,blit=True)
plt.show()

ani=an.ArtistAnimation(pressureani,ims_pressure,interval=200,repeat_delay=3000,blit=True)
plt.show()

ani=an.ArtistAnimation(xvelocityani,ims_xvelocity,interval=200,repeat_delay=3000,blit=True)
plt.show()




            




