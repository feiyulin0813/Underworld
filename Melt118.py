#!/usr/bin/env python
# coding: utf-8

# # Slab Subduction
#
#
# This notebook gives a two dimensional thermo-mechanical subduction model that has 410 and 660 phase transitions, internal heat production, and complex (temperature, pressure, strain rate, composition, and deforming history dependent) visco-plastic rheology.
#
# uneven mesh is incorporated, note that the gradient of the grid gradient is also smooth in this notebook.
#
# The notebook incorporates essentially all the components in Yang et al., 2018, EPSL, but many improvements have been made to make it more readable and computational more efficiency, etc. See the attached NEAsiaReferenceModel.py if you want to reproduce the results in Yang et al., 2018
#
# Low resolution and small domain size are used to enable the users to run a complex subduction model on one cpu within one day.
#
# GMT plot of the model viscosity, strain rate, and accumulated plastic strain at 400 step is as below:
# <img src="img/MeshVariables400.png" alt="ThermoChemicalSlabSubduction" style="width: 800px;"/>
#
#
#
# **References**
#
# Ting Yang, Louis Moresi, Dapeng Zhao, Dan Sandiford, Joanne Whittaker. Cenozoic lithospheric deformation in Northeast Asia and the rapidly-aging Pacific Plate, EPSL (2018)
#


import underworld as uw
import math
from underworld import function as fn
import glucifer
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import operator
import mpi4py
import datetime
from Solidus_Liquidus_ import SolLiq

comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# **Setup parameters**

# File directory
# If restart from a certain time step, set RESTART=1 and assign the inputPath
RESTART=0
filename = "data118/"
inputPath = os.path.join(os.path.abspath("."),filename)
outputPath = os.path.join(os.path.abspath("."),filename)
outputPicturePath = os.path.join(os.path.abspath("."),filename+"Figure/")

# time evolution is put in time.txt
if uw.rank()==0:
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    if not os.path.exists(outputPicturePath):
        os.makedirs(outputPicturePath)
uw.barrier()

# record the time
if uw.rank()==0:
    print('Model setup begins at '+str(datetime.datetime.now()))

# step and time
time = 0.0  # Initial time
step = 0   # Initial timestep
maxSteps = 4001      # Maximum timesteps
steps_output = 50   # output every X timesteps

# physical constants
alpha = 3.0e-5
rho   = 3.3e3 # km/m^3
g     = 9.8
h     = 1000e3
kappa = 1e-6 # thermal diffusivity
eta0=1e21 # eta0 should be consistent with various Ra numbers
drhoOPC = -500.0 # Overiding plate crust density variation relative to reference value (mantle), km/m^3
drhoOPC1 = -300.0 # Overiding plate crust density variation relative to reference value (mantle), km/m^3
drhoOPM  = 15.0 #-0.0*rho # Overiding plate mantle density variation relative to reference value (mantle)
drhoSlabC=-400.0 # Oceanic crust
drhoSlabM=0.0 # Oceanic mantle
Cp=1200.0 # J/kg/K specific heat

# Temperature
dT    = 1350.0
TM = 550.0/dT #Moho Temperature
Tm= 1.0 # mantle temperature
Tsurf=273./dT
dTdz=0.0 # Adiabatic temperature gradient for viscosity calculation
tempMin = 0.0 # Set max & min temperautres
tempMax = 1.0

L = 6*1.0e5 # J/kg Latent Heat
HeatProductionM= 7.38e-12 # J/s/kg Mantle heat production rate
HeatProductionC= 3.69e-10 # J/s/kg Continental crust heat production rate

# The mesh grid space in x direction is heterogeneous if (xx[n]-xx[n-1])/(nx[x]-nx[n-1]) is different for each peice
xx = (0.0, 2.5, 3.5, 4.5) # mesh coordinate in x direction
nx = (0, 200, 320, 400) # the corresponding horizontal index
MeshSmoothWidthX=10 # the grid space varies continuously through 2*MeshSmoothWidthX grids
yy = (0.0, 1.-770e3/h, 1.-250e3/h, 1.025) # yy[-1] needs to fix at 1.0
ny = (0, 10, 62, 162)
MeshSmoothWidthY=10


# LargePenalty for direct solver
# Small Penalty for iterative solver
LargePenalty = 1

# Initial composition distribution
lowerPlateCrustThk = 10e3/h # oceanic crust
lowerPlateLithThk = 80e3/h  # oceanic lith

upperPlateCrustThk1 = 22e3/h # continental lower crust
upperPlateCrustThk = 35e3/h # continental crust
upperPlateLithThk = 90e3/h # continental lith
stepLithThk = 30e3/h 
# Weak zone decoupling the subducting and overriding plates is defined by a line of passive particles embeded lowerPlateCrustThk+dhBeneathWeakOC deep in the oceanic slab
dhBeneathWeakOC = 5e3/h

# phase boundaries
zref410=1.-410e3/h # reference height for 410 km phase boundary
Tref410=Tm # reference temperature
phaseWidth410=50e3/h # transitional width
clapeyronSlope410=3e6 #Clapeyron slope, MPa/K
Ra410=0.05*rho # density jump across the phase boundary

zref660=1.-660e3/h
Tref660=Tm
phaseWidth660=50e3/h
clapeyronSlope660=-1.5e6
Ra660=0.1*rho

# Viscosity
viscLayerZ0=[zref410,zref660] # interface of each viscosity layer, one element fewer than visc0
visc0=[2.0,2.0,30]  # Viscosity prefactor of each layer (upper mantle, transition zone, lower mantle)
viscE0=[15.0,15.0,1.0] # activation energy of each layer
viscT0=[1.0,1.0,1.0] # viscT0 for each layer
viscZ0=[0.0,0.0,0.0] # viscZ0 for each layer
viscN=[3.0,3.0,1.0] # exponent for dislocation rheology for each layer
RefStrainRate=2.0e-15 # reference strain rate, check Yang et al., 2018
RefStrainRateSlab=RefStrainRate # RefStrainRateSlab can be higher than RefStrainRate due to dehydration
min_viscosity=1e-2
max_viscosity=1.0e4

# Initial slab radius and depth
SlabRadius=350e3/h
SlabInitialDepth=350e3/h

PeriodInnerV = 20. # The imposed velocity change period, Myr


# **Deduced parameters**

myr2s=365.25*24*3600*1e6
tao=h**2/kappa/myr2s

PeriodInnerV = PeriodInnerV/tao

HeatProductionM=HeatProductionM*(h**2)/(kappa*Cp*dT)
HeatProductionC=HeatProductionC*(h**2)/(kappa*Cp*dT)

StressScale=h**2/eta0/kappa # From dimensional to non-dim

xRes = nx[-1]-nx[0]
yRes = ny[-1]-ny[0]
boxLength = xx[-1]-xx[0]
boxHeight = yy[-1]-yy[0]

# Scalings
DensityChange = 0.13
Ra   = rho*g*alpha*dT*h**3/kappa/eta0
RM = rho*g*h**3/kappa/eta0 # reference Mantle Rayleigh number, for lithostatic pressure rho*g*h^3/kapa/eta0, rho=3300 kg/m^3
RC = (rho-400)*g*h**3/kappa/eta0 # reference Mantle Rayleigh number, for lithostatic pressure rho*g*h^3/kapa/eta0, rho=3300 kg/m^3
RcOPC = drhoOPC*g*h**3/kappa/eta0 # Overiding plate crust rayleigh number drho*g*h^3/kapa/eta0, drho=500 kg/m^3
RcOPC1 = drhoOPC1*g*h**3/kappa/eta0 # Overiding plate crust rayleigh number drho*g*h^3/kapa/eta0, drho=500 kg/m^3

RcOPM  = drhoOPM*g*h**3/kappa/eta0 # Overiding plate crust rayleigh number drho*g*h^3/kapa/eta0, drho=1%
clapeyronSlope410=clapeyronSlope410*dT/rho/g/h
Ra410=Ra410*g*h**3/kappa/eta0
clapeyronSlope660=clapeyronSlope660*dT/rho/g/h
Ra660=Ra660*g*h**3/kappa/eta0
RcSlabC=drhoSlabC*g*h**3/kappa/eta0
RcSlabM=drhoSlabM*g*h**3/kappa/eta0

RefStrainRate=RefStrainRate*h**2/kappa
RefStrainRateSlab=RefStrainRateSlab*h**2/kappa

# Melting
XH_bulk = 0.19
M_ext = 0.04
M_max = 0.44


# **Define functions**

def StepWiseFun(xx, x0, width):
    zz = (xx - x0) / width
    return 0.5*(1.0+fn.math.tanh(zz))

def LayerValFun(LayerVal0, LayerZ0, coord, width):
    # Smooth discretized values of each layer to get a continuous function
    LayerValFun0=LayerVal0[0]
    for i1 in range(1,len(LayerVal0)):
        LayerValFun1=LayerValFun0+(LayerVal0[i1]-LayerVal0[i1-1])*StepWiseFun(1.-coord[1],1.-LayerZ0[i1-1],width)
        LayerValFun0=LayerValFun1
    return LayerValFun1

def refine_coord1D(xx,nx,MeshSmoothWidth0):
    # refine coordinate in one direction
    N=nx[-1]+1
    x=np.zeros(N)
    if isinstance(MeshSmoothWidth0,int):
        MeshSmoothWidth0=[MeshSmoothWidth0]*(len(xx)-2)
    elif len(MeshSmoothWidth0) == 1:
        MeshSmoothWidth0=MeshSmoothWidth0*(len(xx)-2)

    for k in range(1,len(nx)):
        k2=range(nx[k-1],nx[k]+1)
        x[k2]=np.linspace(xx[k-1],xx[k],len(k2))

    # Smooth the grid around nx[k]
    for k in range(1,len(nx)-1):
        if (k == 1):
            MeshSmoothWidth=min(MeshSmoothWidth0[k-1],nx[k]-nx[k-1]-1,(nx[k+1]-nx[k])/2-1)
        elif (k == len(nx)-2):
            MeshSmoothWidth=min(MeshSmoothWidth0[k-1],(nx[k]-nx[k-1])/2-1,nx[k+1]-nx[k]-1)
        else:
            MeshSmoothWidth=min(MeshSmoothWidth0[k-1],(nx[k]-nx[k-1])/2-1,(nx[k+1]-nx[k])/2-1)

        dx0=(xx[k]-xx[k-1])/((nx[k]-nx[k-1]))
        dx1=(xx[k+1]-xx[k])/((nx[k+1]-nx[k]))
        k2=np.arange(nx[k]-MeshSmoothWidth,nx[k]+MeshSmoothWidth+1)
        #dxGrad=(dx1-dx0)/(2*MeshSmoothWidth+1)
        #dx=dx0+dxGrad*np.arange(1,len(k2))
        phase=math.pi*(k2[1:]-k2[0])/(2*MeshSmoothWidth+1)-math.pi/2
        dx=dx0+(dx1-dx0)*(np.sin(phase)+1.0)/2.0
        x[k2[1:]]=x[k2[0]]+np.cumsum(dx)

    return x


def SlabShapeFn(xmin,xST,yST,xmax,SlabRadius,SlabInitialDepth,LeftExtend):
    # slab shape at the subduction zone (the curved part)
    theta=np.linspace(0,np.arccos((1.-SlabInitialDepth-yST)/SlabRadius),200)
    lowerPlateCrustTop=np.vstack((xST+SlabRadius*np.sin(theta),yST+SlabRadius*np.cos(theta)))
    lowerPlateCrustBot=np.vstack((xST+(SlabRadius-lowerPlateCrustThk)*np.sin(theta),yST+(SlabRadius-lowerPlateCrustThk)*np.cos(theta)))
    lowerPlateLithBot=np.vstack((xST+(SlabRadius-lowerPlateLithThk)*np.sin(theta),yST+(SlabRadius-lowerPlateLithThk)*np.cos(theta)))

    # lith and slab polygons in the whole domain from xmin to xmax
    lowerPlateCrustShape = np.vstack(([[xmin,1.],lowerPlateCrustTop.T,np.flipud(lowerPlateCrustBot.T),[xmin,1.-lowerPlateCrustThk]]))
    lowerPlateMantleShape = np.vstack(([[xmin,1.-lowerPlateCrustThk],lowerPlateCrustBot.T,np.flipud(lowerPlateLithBot.T),[xmin,1.-lowerPlateLithThk]]))
    upperPlateShape = np.vstack(([[xmax,1.],[xST,1.],lowerPlateCrustTop.T,[xmax,1.-upperPlateLithThk]]))
    upperPlateShape1 = np.vstack(([[xmax-0.6,1.-upperPlateLithThk],[xmax-0.6,1.-(upperPlateLithThk+stepLithThk)],[xmax-2.0+0.8,1.0-(upperPlateLithThk+stepLithThk)],[xmax-2.0+0.8, 1.0-upperPlateLithThk]]))
    upperPlateShape2 = np.vstack(([[xmax-0.6,1.-upperPlateLithThk],[xmax-0.6,1.-(upperPlateLithThk+stepLithThk)],[xmax,1.0-(upperPlateLithThk+stepLithThk)],[xmax, 1.0-upperPlateLithThk]]))

    lowerPlateCrust = fn.shape.Polygon( lowerPlateCrustShape )
    lowerPlateMantle = fn.shape.Polygon( lowerPlateMantleShape )
    upperPlate = fn.shape.Polygon( upperPlateShape )
    upperPlate1 = fn.shape.Polygon( upperPlateShape1 )#+fn.shape.Polygon(upperPlateShape1)
    upperPlate2 = fn.shape.Polygon( upperPlateShape2 )

    # low viscosity Stencil line
    n1=int((xST-xmin+LeftExtend)*h/5.0e3)
    n2=int((xST-xmin+LeftExtend+SlabInitialDepth*2)*h/5.0e3)
    lowViStenLine = np.zeros((n2,2))

    lowViStenLine[0:n1,0] = np.linspace(xmin-LeftExtend+0.001, xST-5.0e3/h/2, n1)
    lowViStenLine[0:n1,1] = 1.-lowerPlateCrustThk-dhBeneathWeakOC

    theta=np.linspace(0,np.arccos((1.-SlabInitialDepth-yST)/SlabRadius),n2-n1)
    lowViStenLine[n1:n2+1,0] = xST+(SlabRadius-lowerPlateCrustThk-dhBeneathWeakOC)*np.sin(theta)
    lowViStenLine[n1:n2+1,1] = yST+(SlabRadius-lowerPlateCrustThk-dhBeneathWeakOC)*np.cos(theta)

    # slab circle for temperature
    SlabCircleShape = np.vstack(([[xST,yST],lowerPlateCrustTop.T]))
    SlabCircle = fn.shape.Polygon( SlabCircleShape )

    return SlabCircle,lowerPlateCrust,lowerPlateMantle,upperPlate,upperPlate1,upperPlate2,lowViStenLine


def BackViscFormula(coord,viscEFn,viscZFn,RefStrainRate,nsFn):
    # Background (do not consider plastic yielding) viscosity
    tmp2=11.753*(1.0-(1.0-coord[1])*h/2890e3)-14.235*(1.0-(1.0-coord[1])*h/2890e3)**2
    viscositydif = visc0Fn*fn.math.exp((viscEFn+(1.0-coord[1])*viscZFn)/visT-(viscEFn+viscZFn)/visTm+tmp2) #2890 km
    viscositydis = fn.math.pow((strainRate_2ndInvariantFn+SRMin)/RefStrainRate,(1.0-nsFn)/nsFn)*fn.math.pow(viscositydif,1.0/nsFn)
    viscosity=1./(1./viscositydis+1./viscositydif)
    return viscosity

# Hlaf-space cooling temperature
def HalfSpaceCooling(Age,Tm,z,tao):
    # Age and tao are in myrs
    # z is depth
    T=tempMin+(Tm-tempMin)*math.erf(z/2.0/math.sqrt(Age/tao))
    return T

# The continental lithospheric temperature is defined by double linear segments
def DoubleLinearTemp(TM,Tm,z):
    upperPlateLithThkTemp= 1.5*upperPlateLithThk
    if z<upperPlateCrustThk:
        T=tempMin+(TM-tempMin)*z/upperPlateCrustThk
    elif z<upperPlateLithThkTemp:
        T=TM+(Tm-TM)*(z-upperPlateCrustThk)/(upperPlateLithThkTemp-upperPlateCrustThk)
    else:
        T=Tm
    return T

# The continental lithospheric temperature is defined by double linear segments
def DoubleLinearTemp1(TM,Tm,z):
    upperPlateLithThkTemp= 1.5*upperPlateLithThk
    if z<upperPlateCrustThk:
        T=tempMin+(TM-tempMin)*z/upperPlateCrustThk
    elif z<upperPlateLithThkTemp:
        T=TM+(Tm-TM)*(z-upperPlateCrustThk)/(upperPlateLithThkTemp-upperPlateCrustThk)
    else:
        T=Tm
    return T

# The continental lithospheric temperature is defined by double linear segments
def DoubleLinearTemp2(TM,Tm,z):
    upperPlateLithThkTemp= 1.5*(upperPlateLithThk+stepLithThk)
    if z<upperPlateCrustThk:
        T=tempMin+(TM-tempMin)*z/upperPlateCrustThk
    elif z<upperPlateLithThkTemp:
        T=TM+(Tm-TM)*(z-upperPlateCrustThk)/(upperPlateLithThkTemp-upperPlateCrustThk)
    else:
        T=Tm
    return T

# The continental lithospheric temperature is defined by double linear segments
def DoubleLinearTemp3(TM,Tm,z):
    upperPlateLithThkTemp= 1.5*(upperPlateLithThk+stepLithThk)
    if z<upperPlateCrustThk:
        T=tempMin+(TM-tempMin)*z/upperPlateCrustThk
    elif z<upperPlateLithThkTemp:
        T=TM+(Tm-TM)*(z-upperPlateCrustThk)/(upperPlateLithThkTemp-upperPlateCrustThk)
    else:
        T=Tm
    return T


def LowViStenShape(filename,filename2):
    # the low viscosity stencil polygon between the subducting and overriding plates
    f = h5py.File(filename, 'r')
    f2 = h5py.File(filename2, 'r')

    # List all groups
    #print("Keys: %s" % f.keys())
    a_group_key = f.keys()[0]
    # Get the data
    data = f[a_group_key][()]

    a_group_key = f2.keys()[0]
    # Get the data
    dataVar = f2[a_group_key][()]


    logic1 = np.greater_equal(data[:,1], 1.-30e3/h)
    logic2 = np.logical_and(np.logical_and(data[:,1] < 1.-30e3/h, data[:,1] > 1.-200e3/h), dataVar[:,0]==1)

    ind = np.arange(0,logic1.size)
    inx1 = ind[logic1]
    inx2 = ind[logic2]
    data1A=data[inx1];
    data2A=-data[inx2];

    data1=data1A[data1A[:,0].argsort()]
    data2=-data2A[data2A[:,1].argsort()]
    k1=max(len(data1)-20,0)
    #lowViLine0=np.vstack((data1[-15:,:],data2))
    lowViLine0=np.vstack((data1[k1:,:],data2))
    lowViLine1=lowViLine0.copy()
    lowViLine2=lowViLine0.copy()

    for index, coord in enumerate(lowViLine0):
        lowViLine1[index,0] += 3e3/h
        lowViLine2[index,0] += lowerPlateCrustThk+dhBeneathWeakOC+2e3/h
        lowViLine1[index,1] = min(1.0, lowViLine0[index,1]+3e3/h)
        lowViLine2[index,1] = min(1.0, lowViLine0[index,1]+lowerPlateCrustThk+dhBeneathWeakOC+2e3/h)

    lowViStenShapePolygon=np.vstack((lowViLine1,lowViLine2[::-1,:]))

    lowViSten=fn.shape.Polygon( lowViStenShapePolygon)
    return lowViSten

def LowViStenEval():
    # Viscosity in the weak zone polygon is reset as min_viscosity
    swarm2.save(outputPath+"swarmLowVi"+ str(step).zfill(4)+".h5")
    LowViVariable.save(outputPath+"LowViVariable"+ str(step).zfill(4)+".h5")
    uw.barrier()

    filename1 = outputPath+"swarmLowVi"+ str(step).zfill(4)+".h5"
    filename1A = outputPath+"LowViVariable"+ str(step).zfill(4)+".h5"
    lowViSten=LowViStenShape(filename1,filename1A)

    condition = [(lowViSten, min_viscosity),
                 (True,      max_viscosity)]

    return fn.branching.conditional(condition).evaluate(swarm)

def DiffuseTemp():
    # the temperature field is diffused for 2 myrs after the initial model setup cause the slab top temperature changes from tempMin to tempMax within one grid
    t_diff_max=2./tao #
    t_diff=0
    while t_diff<t_diff_max:
        dt = 0.9*advDiff.get_max_dt()
        t_diff += dt
        advDiff.integrate(dt)
        print(dt*tao)

    print(t_diff*tao,' Myr')

    return

def RidgeSideTemp():
    # The left boundary temperature is set as the ridge temperature is set as a mid-ocean ridge
    for index in mesh.specialSets["MinI_VertexSet"].data[0:-1]:
        if mesh.data[index,1]<1.:
            temperatureField.data[index] = tempMax
        else:
            temperatureField.data[index] = tempMin
    return

def InnerVeloBCVx(time):
    # Velocity varies from Vx0 to Vx1 at half period and then goes back to Vx0 at one period.
    Vx=InnerVeloBCVx0+(InnerVeloBCVx1-InnerVeloBCVx0)*(1.0+math.cos(2.*math.pi*time/PeriodInnerV+math.pi))/2.
    return Vx

def RecalibratePressure():
    #The pressure field is smoothed as in Citcom: from subMesh->Nodes->subMesh
    # First, subMesh->Nodes
    PresCell2Nodes.solve()
    # Second, Nodes->subMesh
    PresNodes2Cell.solve()

    #fix the horizontal average of the surface pressure field as zero
    (area,) = surfaceArea.evaluate()
    (p0,) = surfacePressureIntegral.evaluate()
    pressureField.data[:] -= p0 / area
    return

def MeltModifierFn(viscosityChangeX1,viscosityChangeX2,viscosityChange):
    change=(1.0 + (viscosityChange-1.0) /(viscosityChangeX2-viscosityChangeX1) * (swarmM0-viscosityChangeX1))
    conditions = [(swarmM0 < viscosityChangeX1, 1.0),
                  (swarmM0 > viscosityChangeX2, viscosityChange),
                  (True, change)]
    melt_modif.data[:]= fn.branching.conditional( conditions ).evaluate(swarm)

    return

def CalMeltFraction():
    T = temperatureField.evaluate(swarm.particleCoordinates.data)

    for index in range(len(materialVariable.data)):
        Melt = (T[index]*dT-Tsolidus.data[index])/(Tliquidus.data[index]-Tsolidus.data[index])
        if Melt > 0:
            swarmM0.data[index] = Melt
        else:
            swarmM0.data[index] = 0
    return

def SaveFig(step):
    figParticle.save(outputPicturePath +"FigParticle"+str(step).zfill(4))
    figMelt.save(outputPicturePath +"FigMelt"+str(step).zfill(4))
    figTemp.save(outputPicturePath+ "FigTemp"+str(step).zfill(4))
    figVisc.save(outputPicturePath+ "FigVisc"+str(step).zfill(4))
    figStress.save(outputPicturePath+ "FigStress"+str(step).zfill(4))


def SaveData(step):
    projectorVisc.solve()
    projectorFiniteStrain.solve()
    # Update buoyancy
    projectorBuoyancy.solve()
    projectorDevStress.solve()
    projectorDevStress2nd.solve()
    # Update phase field
    phase410Field.data[:] = fn_phase410.evaluate(mesh)
    phase660Field.data[:] = fn_phase660.evaluate(mesh)

    #surface and moho
    swarm3.save(outputPath+"surfaceSwarm1."+ str(step).zfill(4)+".h5")
    swarm4.save(outputPath+"surfaceSwarm2."+ str(step).zfill(4)+".h5")

    #melt
    swarmM0.save(outputPath+"swarmM0-"+ str(step).zfill(4)+".h5")
    Tsolidus.save(outputPath+"Tsolidus"+ str(step).zfill(4)+".h5")
    Tliquidus.save(outputPath+"Tliquidus"+ str(step).zfill(4)+".h5")

    mesh.save(outputPath+"mesh"+ str(step).zfill(4)+".h5")
    temperatureField.save(outputPath+"tempfield"+ str(step).zfill(4)+".h5")
    temperatureDotField.save(outputPath+"tempDotfield"+ str(step).zfill(4)+".h5")
    pressureField.save(outputPath+"presfield"+ str(step).zfill(4)+".h5")
    velocityField.save(outputPath+"velfield"+ str(step).zfill(4)+".h5")
    swarm.save(outputPath+"swarm"+ str(step).zfill(4)+".h5")
    finiteStrainField.save(outputPath+"finiteStrainField"+ str(step).zfill(4)+".h5")
    materialVariable.save(outputPath+"material"+ str(step).zfill(4)+".h5")
    viscField.save(outputPath+"viscfield"+ str(step).zfill(4)+".h5")
    phase410Field.save(outputPath+"phase410field"+ str(step).zfill(4)+".h5")
    phase660Field.save(outputPath+"phase660field"+ str(step).zfill(4)+".h5")
    buoyancyField.save(outputPath+"buoyancyfield"+ str(step).zfill(4)+".h5")
    DevStressField.save(outputPath+"DevStressfield"+ str(step).zfill(4)+".h5")
    DevStress2ndField.save(outputPath+"DevStress2ndfield"+ str(step).zfill(4)+".h5")
    return


# define an update function
def update():
    # Retrieve the maximum possible timestep for the advection system.
    uw.barrier()

    CalMeltFraction()
    # Advect
    dt1 = advector.get_max_dt()
    dt2 = advDiff.get_max_dt()
    dt = 0.9*min(dt1,dt2)
    # Advect using this timestep size.
    advector.integrate(dt,update_owners=True)
    advector2.integrate(dt,update_owners=True)

    advector3.integrate(dt,update_owners=True)
    advector4.integrate(dt,update_owners=True)

    advDiff.integrate(dt)

    MeltModifierFn(viscosityChangeX1 = 0.2,
                   viscosityChangeX2 = 0.3,
                   viscosityChange = 1.0e-2)

    # Fix the left boundary as mid-ocean ridge temperature
    #RidgeSideTemp()

    # particle population control
    #if (step%5 ==0):
    pop_control.repopulate()

    # update Weak Stencil
    if step%steps_output ==0:
        ViscMin.data[:]=LowViStenEval()

    # update plastic strain
    viscDiff = viscosityFn-backgroundViscosityFn
    YieldTemp = Tm-0.1 # only low temperature region yields
    YieldHeight = 1.-250e3/h # No yielding below 250 km depth
    conditional_viscDiff = fn.branching.conditional( ( (temperatureField > YieldTemp, 99999999.), (fn.input()[1] < YieldHeight, 99999999.), (True, viscDiff) ) )
    swarmYieldID = np.where(conditional_viscDiff.evaluate(swarm) < 0.0)[0]
    if len(swarmYieldID)>0:
        swarmStrainRateInv = strainRate_2ndInvariantFn.evaluate(swarm.particleCoordinates.data[swarmYieldID])
        plasticStrain.data[swarmYieldID] += dt*swarmStrainRateInv


    # Save data for post-process
    if (step%steps_output ==0):

        figParticle.save(outputPicturePath +"FigParticle"+str(step).zfill(4))
        figMelt.save(outputPicturePath +"FigMelt"+str(step).zfill(4))
        figTemp.save(outputPicturePath+ "FigTemp"+str(step).zfill(4))
        figVisc.save(outputPicturePath+ "FigVisc"+str(step).zfill(4))
        figStress.save(outputPicturePath+ "FigStress"+str(step).zfill(4))

        SaveData(step)

    Tm1=(temperatureField.evaluate(swarm.particleCoordinates.data)[:,0]+dTdz*(1.0-swarm.particleCoordinates.data[:,1]))*dT #dimension C
    Pm1= (1.0-swarm.particleCoordinates.data[:,1])*RM/StressScale/1.0e9  #dimensional pressure Gpa(lithostatic pressure)

    for index,coord in enumerate(swarm.particleCoordinates.data):
        if coord[0]>2.5:
            TT = SolLiq(Tm1[index],Pm1[index],XH_bulk)
            Tsolidus.data[index] = TT[0]
            Tliquidus.data[index] = TT[1]
        else:
            TT = SolLiq(Tm1[index],Pm1[index],0.0)
            Tsolidus.data[index] = TT[0]
        Tliquidus.data[index] = TT[1]

    return dt, time+dt, step+1


# **Create mesh**

mesh = uw.mesh.FeMesh_Cartesian( elementType = ("Q1/dQ0"),
                                 elementRes  = (xRes, yRes),
                                 minCoord    = (0., 0.),
                                 maxCoord    = (boxLength, boxHeight),
                                 periodic    = [False, False] )



# **Refine mesh**

if len(yy) > 2:
    y0=refine_coord1D(yy,ny,MeshSmoothWidthY)
if len(xx) > 2:
    x0=refine_coord1D(xx,nx,MeshSmoothWidthX)

uw.barrier()

with mesh.deform_mesh():
    for index, coord in enumerate(mesh.data):
        if len(xx) > 2:
            index_x=int(mesh.data_nodegId[index] % (xRes+1))
            mesh.data[index][0] = x0[index_x]
        if len(yy) > 2:
            index_y=int(mesh.data_nodegId[index]/(xRes+1))
            mesh.data[index][1] = y0[index_y]


# **Create mesh variables**

velocityField   = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=2 )
pressureField   = uw.mesh.MeshVariable( mesh=mesh.subMesh, nodeDofCount=1 )
temperatureField    = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 )
temperatureDotField = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 )
phase410Field    = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 )
phase660Field    = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 )
viscField = uw.mesh.MeshVariable( mesh=mesh, nodeDofCount=1 )
finiteStrainField = uw.mesh.MeshVariable( mesh=mesh, nodeDofCount=1 )
strainRateInvField = uw.mesh.MeshVariable( mesh=mesh, nodeDofCount=1 )
stressInvField = uw.mesh.MeshVariable( mesh=mesh, nodeDofCount=1 )
buoyancyField    = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=2 )
DevStressField    = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=3 )
DevStress2ndField    = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 )

velocityField.data[:] = [0.,0.]
pressureField.data[:] = 0.
temperatureField.data[:]    = 0.
temperatureDotField.data[:] = 0.
phase410Field.data[:]=0.
phase660Field.data[:]=0.
buoyancyField.data[:] = [0.,0.]
DevStressField.data[:] = [0.0, 0.0, 0.0]
DevStress2ndField.data[:] = 0.


# Load variables if restart
if (RESTART == 1):
    mesh.load(inputPath+"mesh"+ str(step).zfill(4)+".h5")
    temperatureField.load(inputPath+"tempfield"+ str(step).zfill(4)+".h5")
    if os.path.isfile(inputPath+"tempDotfield"+ str(step).zfill(4)+".h5"):
        temperatureDotField.load(inputPath+"tempDotfield"+ str(step).zfill(4)+".h5")
    pressureField.load(inputPath+"presfield"+ str(step).zfill(4)+".h5")
    velocityField.load(inputPath+"velfield"+ str(step).zfill(4)+".h5")



# **Create particle swarms and swarm variables**

swarm = uw.swarm.Swarm( mesh=mesh, particleEscape=True)
swarm2=uw.swarm.Swarm(mesh=mesh, particleEscape=True) # Low viscosity tracer, following Manea
swarm3=uw.swarm.Swarm(mesh=mesh, particleEscape=True) # Surface topographic points
swarm4=uw.swarm.Swarm(mesh=mesh, particleEscape=True) # Moho surface topographic point

# Melting and dehydration stiffen

swarmM0 = swarm.add_variable( dataType="double",  count=1 )
melt_modif = swarm.add_variable( dataType="double", count=1 )
Tsolidus = swarm.add_variable( dataType="double",  count=1 )
Tliquidus = swarm.add_variable( dataType="double",  count=1 )


# Viscosity control

materialVariable   = swarm.add_variable( dataType="int", count=1 )
LowViVariable = swarm2.add_variable( dataType="int", count=1 ) # 0 not counted for low Vi channel, 1 counted
plasticStrain  = swarm.add_variable( dataType="double",  count=1 )
ViscMin = swarm.add_variable( dataType="double",  count=1 )
viscVariable = swarm.add_variable( dataType="double", count=1 )

# Initialize the number of swarm particles
if (RESTART == 1):
    swarm.load(inputPath+"swarm"+ str(step).zfill(4)+".h5")
    materialVariable.load(inputPath+"material"+ str(step).zfill(4)+".h5")
    swarm2.load(inputPath+"swarmLowVi"+ str(step).zfill(4)+".h5")

else:
    swarmLayout = uw.swarm.layouts.PerCellSpaceFillerLayout( swarm=swarm, particlesPerCell=20 )
    #swarmLayout = uw.swarm.layouts.GlobalSpaceFillerLayout( swarm=swarm, particlesPerCell=20 )
    swarm.populate_using_layout( layout=swarmLayout )


# create pop control object
pop_control = uw.swarm.PopulationControl(swarm, deleteThreshold=0.025, splitThreshold=0.2)

# Surface topographic points
surfacePoints = np.zeros((2500*10*5,2))
surfacePoints[:, 0] = np.linspace(2.0, 4.5, 2500*10*5)
for index in range( len(surfacePoints[:,0]) ):
    surfacePoints[index,1] = 0.997#0.995
swarm3.add_particles_with_coordinates( surfacePoints )

# Moho surface topographic point
surfacePoints1 = np.zeros((2500*10*5,2))
surfacePoints1[:, 0] = np.linspace(2.0, 4.5, 2500*10*5)
for index in range( len(surfacePoints1[:,0]) ):
    surfacePoints1[index,1] = 0.970#0.995
swarm4.add_particles_with_coordinates( surfacePoints1 )



# **Allocate materials to particles and create initial slab geometry**
# Create initial slab geometry

xST1=boxLength-2.0 #center of the circle of the slab, xST is also the location of the trench
#xST1=1500e3/h
yST1=1.-SlabRadius
SlabCircle1,lowerPlateCrust1,lowerPlateMantle1,upperPlate1,upperPlate2,upperPlate3,lowViStenLine1 = SlabShapeFn(0e3/h,xST1,yST1,boxLength,SlabRadius, SlabInitialDepth,0.)


# Allocate materials to particles

MantleIndex = 0
upperPlateCrustIndex   = 1
upperPlateCrustIndex1 = 2
ContUpperMantleLithIndex   = 3
lowerPlateCrustIndex   = 4
lowerPlateMantleIndex   = 5
StickyairIndex   = 6

if (RESTART == 0):
    # initialise everying to be mantle material
    materialVariable.data[:] = MantleIndex

    # change matieral index if the particle is not mantle
    for index in range( len(swarm.particleCoordinates.data) ):
        coord = swarm.particleCoordinates.data[index][:]
        if coord[1] > 1.0:
            materialVariable.data[index]     = StickyairIndex
        elif upperPlate2.evaluate(tuple(coord)):
            materialVariable.data[index]     = ContUpperMantleLithIndex
        elif upperPlate3.evaluate(tuple(coord)):
            materialVariable.data[index]     = ContUpperMantleLithIndex

        elif upperPlate1.evaluate(tuple(coord)):
            if coord[1] > 1.-upperPlateCrustThk1:
                materialVariable.data[index]     = upperPlateCrustIndex
            elif coord[1] > 1.-upperPlateCrustThk:
                materialVariable.data[index]     = upperPlateCrustIndex1
            elif coord[1] > 1.-upperPlateLithThk:
                materialVariable.data[index]     = ContUpperMantleLithIndex
        elif lowerPlateCrust1.evaluate(tuple(coord)):
            materialVariable.data[index] = lowerPlateCrustIndex
        elif lowerPlateMantle1.evaluate(tuple(coord)):
            materialVariable.data[index] = lowerPlateMantleIndex


# **Initial temperature and thermal BC**

RidgeAge=0.01
OCAgeL1=150.0
OCAgeR1=50.0

if (RESTART == 0):
    for index in range( len(mesh.data) ):
        coord = mesh.data[index][:]
        if coord[1]<=1.0:
            if SlabCircle1.evaluate(tuple(coord)):
                Age=OCAgeR1
                tmpz=SlabRadius-math.hypot(coord[0]-xST1,coord[1]-yST1)
                temperatureField.data[index]=HalfSpaceCooling(Age,Tm,tmpz,tao)
            else:
                if coord[0]<=xST1:
                    Age=OCAgeL1+coord[0]/xST1*(OCAgeR1-OCAgeL1) # Myr
                    temperatureField.data[index]=HalfSpaceCooling(Age,Tm,1.-coord[1],tao)
                elif ((xST1+0.4)<coord[0])&(coord[0]< (xST1+0.45)):
                    temperatureField.data[index]=DoubleLinearTemp1(TM,Tm,1.-coord[1])
                elif ((boxLength-0.6)>coord[0])&(coord[0]>(boxLength-2.0+0.8)):
                    temperatureField.data[index]=DoubleLinearTemp2(TM,Tm,1.-coord[1])
                elif (coord[0]>=(boxLength-0.6)):
                    temperatureField.data[index]=DoubleLinearTemp3(TM,Tm,1.-coord[1])
                else:
                    temperatureField.data[index]=DoubleLinearTemp(TM,Tm,1.-coord[1])

        else:
            temperatureField.data[index] = tempMin


# **Set boundary conditions**
iWalls = mesh.specialSets["MinI_VertexSet"] + mesh.specialSets["MaxI_VertexSet"]
jWalls = mesh.specialSets["MinJ_VertexSet"] + mesh.specialSets["MaxJ_VertexSet"]
bottomWall = mesh.specialSets["MinJ_VertexSet"]
topWall = mesh.specialSets["MaxJ_VertexSet"]
leftWall =  mesh.specialSets["MinI_VertexSet"]

for index in mesh.specialSets["MinJ_VertexSet"]:
    temperatureField.data[index] = tempMax
for index in mesh.specialSets["MaxJ_VertexSet"]:
    temperatureField.data[index] = tempMin


# Temperature BC
tempBC     = uw.conditions.DirichletCondition( variable        = temperatureField,
                                               indexSetsPerDof = (jWalls+leftWall,) )


# Velocity BC
InnerVeloBCX = 4.2
InnerVeloBCY = 1.+1.0 # No InnerVeloBC
InnerVeloBCVx0=2.0e-2/(365.25*24*3600)*h/kappa
InnerVeloBCVx1=-1.0e-2/(365.25*24*3600)*h/kappa
InnerVeloNodes = mesh.specialSets["Empty"]
tmp = np.where(operator.and_(abs(mesh.data[:,0]-InnerVeloBCX)<7.0e3/h,abs(mesh.data[:,1]-InnerVeloBCY)<80e3/h))[0]
InnerVeloNodes.add(tmp)
if len(tmp)>0:
    velocityField.data[tmp,0] = InnerVeloBCVx(time)
    print('InnerVeloNodes: ',uw.rank(), InnerVeloNodes, mesh.data[InnerVeloNodes.data,0], mesh.data[InnerVeloNodes.data,1])
uw.barrier()


VelBC = uw.conditions.DirichletCondition( variable        = velocityField,
                                          indexSetsPerDof = ( iWalls, jWalls) )


# Heat Latent
Tm1=(temperatureField.evaluate(swarm.particleCoordinates.data)[:,0]+dTdz*(1.0-swarm.particleCoordinates.data[:,1]))*dT #dimension C
Pm1= (1.0-swarm.particleCoordinates.data[:,1])*RM/StressScale/1.0e9  #dimensional pressure Gpa(lithostatic pressure)

for index,coord in enumerate(swarm.particleCoordinates.data):
    if coord[0]>2.5 and coord[0]<4.3:
        TT = SolLiq(Tm1[index],Pm1[index],XH_bulk)
        Tsolidus.data[index] = TT[0]
        Tliquidus.data[index] = TT[1]
    else:
        TT = SolLiq(Tm1[index],Pm1[index],0.0)
        Tsolidus.data[index] = TT[0]
        Tliquidus.data[index] = TT[1]


# **Set advection-diffusion function**
# thermal diffusivity
air_diffusivity = 10.0 / (1.0 + L/(Tliquidus-Tsolidus)/Cp)
else_diffusivity = 1.0 / (1.0 + L/(Tliquidus-Tsolidus)/Cp)
diffusivityMap = {
    StickyairIndex: air_diffusivity,
    upperPlateCrustIndex:else_diffusivity,
    upperPlateCrustIndex1:else_diffusivity,
    ContUpperMantleLithIndex:else_diffusivity,
    lowerPlateCrustIndex:else_diffusivity,
    lowerPlateMantleIndex:else_diffusivity,
    MantleIndex: else_diffusivity}
diffusivityFn   =  fn.branching.map( fn_key = materialVariable,mapping = diffusivityMap )

# heat production
HeatProductionC = HeatProductionC / (1.0 + L/(Tliquidus-Tsolidus)/Cp)
HeatProductionM = HeatProductionM / (1.0 + L/(Tliquidus-Tsolidus)/Cp)
HeatProductionFnMap = { StickyairIndex: 0.0,
                        upperPlateCrustIndex:HeatProductionC,
                        upperPlateCrustIndex1:HeatProductionC,
                        ContUpperMantleLithIndex:HeatProductionM,
                        lowerPlateCrustIndex:HeatProductionM,
                        lowerPlateMantleIndex:HeatProductionM,
                        MantleIndex:HeatProductionM}
HeatProductionFn = fn.branching.map( fn_key=materialVariable, mapping=HeatProductionFnMap )


CalMeltFraction()

# **410 and 660 km phases**

fn_z = fn.input()[1]
fn_phase410 = 0.5*(1.0 + fn.math.tanh((zref410-fn_z-clapeyronSlope410*(temperatureField-Tref410))/phaseWidth410))
fn_phase660 = 0.5*(1.0 + fn.math.tanh((zref660-fn_z-clapeyronSlope660*(temperatureField-Tref660))/phaseWidth660))


# **Background Viscosity**
# no yielding viscosity

if (RESTART == 0):
    swarm2.add_particles_with_coordinates( lowViStenLine1 )
    LowViVariable.data[:] = 1
else:
    if ~os.path.isfile(inputPath+"LowViVariable"+ str(step).zfill(4)+".h5"):
        LowViVariable.data[:] = 1
    else:
        LowViVariable.load(inputPath+"LowViVariable"+ str(step).zfill(4)+".h5")


strainRateFn = fn.tensor.symmetric( velocityField.fn_gradient )
strainRate_2ndInvariantFn = fn.tensor.second_invariant( strainRateFn )

coord = fn.input()
visT=Tsurf+fn.misc.max(temperatureField,0.0)+fn.misc.max(dTdz*(1.-coord[1]),0.0)
visTm=Tsurf+Tm+fn.misc.max(dTdz*(1.-coord[1]),0.0)
# Refer to Huismans and Beaumont, 2007 for continental crust and Hedjazian et al., 2017 for others
SRMin=1.0e-19*(h**2/kappa)
nsFn=LayerValFun(viscN, viscLayerZ0, coord, 10e3/h)
viscEFn=LayerValFun(viscE0, viscLayerZ0, coord, 10e3/h)
viscZFn=LayerValFun(viscZ0, viscLayerZ0, coord, 10e3/h)
visc0Fn=LayerValFun(visc0, viscLayerZ0, coord, 10e3/h)
#tmp=223e3/8.31/dT
#tmp1=eta0*kappa/h**2*(h**2/kappa)**(1./4.0)
#viscosityUPC = tmp1*fn.math.pow(strainRate_2ndInvariantFn+SRMin,-3.0/4.0)*fn.math.exp(tmp/4.0/visT)
viscosityM=BackViscFormula(coord,viscEFn,viscZFn,RefStrainRate,nsFn)
#viscosityOM=BackViscFormula(coord,viscEFn,viscZFn,RefStrainRateSlab,nsFn)
viscosityOM=1.*viscosityM # the lith viscosity may be higher due to dehydration
viscosityOC=viscosityOM
viscosityUPM=viscosityOM
viscosityUPC=BackViscFormula(coord,viscEFn/3.0,viscZFn,RefStrainRate,nsFn)
#viscosityUPC = viscosityOM
viscosityW=min_viscosity

viscosityMap = { StickyairIndex: min_viscosity, upperPlateCrustIndex:viscosityUPC*10.0, upperPlateCrustIndex1:viscosityUPC*1.0e-2, ContUpperMantleLithIndex:fn.misc.min(fn.misc.max(viscosityUPM*5.0e-1, 2.0),20.0),
                 lowerPlateCrustIndex:viscosityOC*3.0, lowerPlateMantleIndex:viscosityOM*3.0,
                 MantleIndex:viscosityM*5.0e-1}


backgroundViscosityFn0  = fn.branching.map( fn_key = materialVariable,
                                            mapping = viscosityMap )

backgroundViscosityFn1  = melt_modif * backgroundViscosityFn0

# Viscosity multiply

ViscMin.data[:]=LowViStenEval()

backgroundViscosityFn = fn.misc.min(ViscMin,fn.misc.max(min_viscosity,backgroundViscosityFn1))


# Plastic Yielding

if RESTART == 1:
    finiteStrainField.load(inputPath+"finiteStrainField"+ str(step).zfill(4)+".h5")
    plasticStrain.data[:]=finiteStrainField.evaluate(swarm)
else:
    for index in range( len(swarm.particleCoordinates.data) ):
        coord = swarm.particleCoordinates.data[index][:]
        if (coord[1] > 1.-upperPlateCrustThk)&(coord[0]<(xST1+400e3/h))&(coord[0]>(xST1+200e3/h)):
            plasticStrain.data[index] = 0.2
        else:
            plasticStrain.data[index] = 0.0


referenceStrain = fn.misc.constant(0.1)

# Friction - in this form it could also be made to weaken with strain
frictionMap = {
    StickyairIndex: 0.0,
    upperPlateCrustIndex: 0.01+0.55 * fn.math.exp(-plasticStrain / referenceStrain),
    upperPlateCrustIndex1: 0.01+0.55 * fn.math.exp(-plasticStrain / referenceStrain),
    ContUpperMantleLithIndex:0.01+0.55 * fn.math.exp(-plasticStrain / referenceStrain),
    lowerPlateCrustIndex:0.1+0.55 * fn.math.exp(-plasticStrain / referenceStrain),
    lowerPlateMantleIndex:0.1+0.55 * fn.math.exp(-plasticStrain / referenceStrain),
    MantleIndex:0.01+0.55 * fn.math.exp(-plasticStrain / referenceStrain)}
frictionFn     =  fn.branching.map( fn_key = materialVariable,
                                    mapping = frictionMap )

# Cohesion - a function of swarm variables
cohesionMap = {
    StickyairIndex: 1000e6,
    upperPlateCrustIndex:5e6,
    upperPlateCrustIndex1:5e6,
    ContUpperMantleLithIndex:10e6,
    lowerPlateCrustIndex:10e6,
    lowerPlateMantleIndex:10e6,
    MantleIndex: 5e6}
cohesionFn     =  StressScale * fn.branching.map( fn_key = materialVariable,
                                                  mapping = cohesionMap )

# Drucker-Prager yield criterion

coord = fn.input()
yieldStressFn   = fn.misc.min(frictionFn * RM * fn.misc.max(0.0,1.-coord[1]),350e6*StressScale)+cohesionFn

# Viscosity is the minimum of ductile deformation viscosity and yielding viscosity

yieldingViscosityFn =  0.5 * yieldStressFn / (strainRate_2ndInvariantFn+SRMin)

viscosityFn = fn.misc.max(fn.misc.min(yieldingViscosityFn,
                                      backgroundViscosityFn),
                          min_viscosity)


# deviatoric stress
devStressFn = 2.0 * viscosityFn * strainRateFn
devStress2ndFn = 2.0 * viscosityFn * strainRate_2ndInvariantFn

vis = np.log10(viscosityFn.evaluate(swarm)*eta0)
for index,coord in enumerate(swarm.particleCoordinates.data):
    viscVariable.data[index] = vis[index]




# **Set the density function, vertical unit vector and Buoyancy Force function**

MantleDens = RM-Ra*temperatureField+Ra410*fn_phase410+Ra660*fn_phase660
densityMap = { StickyairIndex: 0.0, upperPlateCrustIndex:MantleDens+RcOPC-(RM+RcOPC)*(swarmM0*DensityChange),
               upperPlateCrustIndex1:MantleDens+RcOPC1-(RM+RcOPC1)*(swarmM0*DensityChange),
               ContUpperMantleLithIndex:MantleDens+RcOPM-(RM+RcOPM)*(swarmM0*DensityChange),
               lowerPlateCrustIndex:MantleDens+RcSlabC-(RM+RcSlabC)*(swarmM0*DensityChange),
               lowerPlateMantleIndex:MantleDens+RcSlabM-(RM+RcSlabM)*(swarmM0*DensityChange),
               MantleIndex:MantleDens}

densityFn = fn.branching.map( fn_key=materialVariable, mapping=densityMap )
# And the final buoyancy force function.
z_hat = ( 0.0, 1.0 )

buoyancyFn = -densityFn * z_hat



# **System and solver Setup**

# The slab-mantle wedge interface temperature jump is reduced by thermal diffusion
advDiff = uw.systems.AdvectionDiffusion( phiField       = temperatureField,
                                         phiDotField    = temperatureDotField,
                                         velocityField  = velocityField,
                                         fn_diffusivity = diffusivityFn,
                                         fn_sourceTerm  = HeatProductionFn,
                                         conditions     = [tempBC,] )

# diffuse the initial temperature

if (RESTART == 0):
    DiffuseTemp()
    RidgeSideTemp()

# Stokes
stokes = uw.systems.Stokes(    velocityField = velocityField,
                               pressureField = pressureField,
                               #voronoi_swarm = swarm,
                               conditions    = VelBC,
                               fn_viscosity  = viscosityFn,
                               fn_bodyforce  = buoyancyFn )
# Create solver & solve
solver = uw.systems.Solver(stokes)


# "mumps" is a good alternative for "lu" but
# not every petsc installation has mumps !
# It also works fine in parallel
if LargePenalty == 1:
    solver.set_penalty(1.0e6)
    solver.set_inner_method("mumps")
    # use "lu" direct solve and large penalty (if running in serial)
    if(uw.nProcs()==1):
        solver.set_inner_method("lu")
else:
    solver.set_penalty(1.0e2)
    solver.set_inner_method("mg")
    solver.options.mg.levels = 6


solver.options.scr.ksp_rtol=1.0e-8

advector = uw.systems.SwarmAdvector( swarm=swarm, velocityField=velocityField, order=2 )
advector2 = uw.systems.SwarmAdvector( swarm=swarm2, velocityField=velocityField, order=2 )

# 地形点
advector3 = uw.systems.SwarmAdvector( swarm=swarm3, velocityField=velocityField, order=2 )
advector4 = uw.systems.SwarmAdvector( swarm=swarm4, velocityField=velocityField, order=2 )

MeltModifierFn(viscosityChangeX1 = 0.2,
               viscosityChangeX2 = 0.3,
               viscosityChange = 1.0e-2)

# **Main simulation loop**

#---------------------------------Smooth the pressure---------------------------------#
NodePressureField = uw.mesh.MeshVariable(mesh, nodeDofCount=1)
PresCell2Nodes = uw.utils.MeshVariable_Projection (NodePressureField, pressureField,type=0)
PresNodes2Cell = uw.utils.MeshVariable_Projection (pressureField, NodePressureField,type=0)

surfaceArea = uw.utils.Integral(fn=1.0,mesh=mesh, integrationType='surface', surfaceIndexSet=topWall)
surfacePressureIntegral = uw.utils.Integral(fn=pressureField, mesh=mesh, integrationType='surface', surfaceIndexSet=topWall)
RecalibratePressure()

# project to mesh variables
projectorVisc = uw.utils.MeshVariable_Projection( viscField, viscosityFn, type=0 )
projectorFiniteStrain = uw.utils.MeshVariable_Projection( finiteStrainField, plasticStrain, type=0 )
projectorBuoyancy = uw.utils.MeshVariable_Projection( buoyancyField, buoyancyFn,type=0)
projectorBuoyancy.solve()
projectorDevStress = uw.utils.MeshVariable_Projection( DevStressField, devStressFn,type=0)
projectorDevStress2nd = uw.utils.MeshVariable_Projection( DevStress2ndField, devStress2ndFn,type=0)



# Figure
#-------------------------Particle swarm-----------------------------#
figParticle = glucifer.Figure( figsize=(1200,500),title = 'Particle',quality=2)
figParticle.append( glucifer.objects.Points(swarm, materialVariable, pointSize=2,
                                            colours='red yellow green purple blue white',
                                            discrete=True) )
figParticle.append( glucifer.objects.Contours(mesh, fn_phase410+fn_phase660,interval=1.0, limits=(0.5,1.5),
                                              linewidth=2,colourBar=False,colours='purple purple') )

#-------------------------Melt-------------------------------------#
figMelt = glucifer.Figure( figsize=(1200,500),title = 'Melt',quality=2)
surf = glucifer.objects.Points(swarm, swarmM0, pointSize=4.0 ,
                               colours='white plum LightSalmon Tomato FireBrick',
                               logScale=True, valueRange=(2e-3,0.2))
figMelt.append(surf)
surf.colourBar["ticks"] = 4
figMelt.append(glucifer.objects.Contours(mesh, temperatureField,interval=0.25,
                                         limits=(0.0,0.9),linewidth=2,
                                         colourBar=False,colours='black'))
figMelt.append( glucifer.objects.Contours(mesh, fn_phase410+fn_phase660,interval=1.0, limits=(0.5,1.5),
                                          linewidth=2,colourBar=False,colours='purple purple') )


#-------------------------TemperatureField------------------------------#
figTemp = glucifer.Figure( figsize = (1200,500),title = "Temperature (non-dimensional)",quality = 2 )
surf=glucifer.objects.Surface(mesh, temperatureField,
                              resolution=500,
                              colours='Blue RoyalBlue SteelBlue Moccasin Brown ')
figTemp.append(glucifer.objects.Contours(mesh, temperatureField,interval=0.3,
                                         limits=(0.6,0.9),linewidth=2,
                                         colourBar=False,colours='white'))
# set the number of extra tick marks (besides start/finish)
surf.colourBar["ticks"] = 6
# set values for extra ticks
surf.colourBar["tickvalues"] = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
figTemp.append(surf)
figTemp.append( glucifer.objects.Contours(mesh, fn_phase410+fn_phase660,interval=1.0, limits=(0.5,1.5),
                                          linewidth=2,colourBar=False,colours='purple purple') )

#-------------------------ViscosityField------------------------------#
figVisc = glucifer.Figure( figsize=(1200,500),title = 'Viscosity(log10)',quality=2)
figVisc.append( glucifer.objects.Points(swarm, viscosityFn, pointSize=2.0 ,
                                        colours='red yellow green blue', logScale=True, valueRange=(1e-2,1.0e4)) )
figVisc.append( glucifer.objects.Contours(mesh, temperatureField,interval=0.2, limits=(0.6,0.9),
                                          linewidth=2,colourBar=False,colours='white white') )
figVisc.append( glucifer.objects.Contours(mesh, fn_phase410+fn_phase660,interval=1.0, limits=(0.5,1.5),
                                          linewidth=2,colourBar=False,colours='purple purple') )
figVisc.append( glucifer.objects.VectorArrows(mesh, velocityField, scaling=1e-5,arrowHead=0.3) )


#-------------------------stress invariant Mpa dimensional-----------------------------#
figStress = glucifer.Figure( figsize=(1500,250) ,title = 'Stress',quality=2)
StressSurf =glucifer.objects.Points(swarm, devStress2ndFn/StressScale/1.0e6,
                                    pointSize=2, resolution=500,
                                    valueRange = (1.6508e-5, 29.436))
StressSurf.colourBar["ticks"] = 8
StressSurf.colourBar["tickvalues"] = [1.6508e-5,1,5,10,15,20,25,29.436]
figStress.append( StressSurf )
figStress.append(glucifer.objects.Contours(mesh, temperatureField,interval=0.3,
                                           limits=(0.6,0.9),linewidth=2,
                                           colourBar=False,colours='white'))


# main loop of stokes solver

if uw.rank()==0:
    print('Stokes solver begins at '+str(datetime.datetime.now()))

dtStep=0.0
while step < maxSteps:

    if uw.rank()==0:
        fw = open(outputPath + "time.txt","a")
        fw.write("%.4f \t %.4f \t %.4f \t %.4f \n" %(step,time,time*tao,dtStep*tao*1e3))
        fw.close()

        #string = "{0}, {1:.5e}, {2:.5e}, {3:.5e}, {4:.5e}\n".format(step,time,time*tao,dtStep*tao*1e3, Velocity_Changbai)
        #outfile.write(string)


    if uw.rank()==0:
        print('Stokes solver of step ' + str(step) +' finished at '+str(datetime.datetime.now()))


    # Solve non linear Stokes system
    #velocityField.data[index][0] = InnerVeloBCVx(time)
    #Velocity_Changbai = 0.2

    solver.solve(nonLinearIterate=True,nonLinearMaxIterations=200,nonLinearTolerance=0.003)
    solver.print_stats()
    RecalibratePressure()


    # update
    dtStep,time,step = update()


if uw.rank()==0:
    print('Model finished at at '+str(datetime.datetime.now()))












