import numpy as np
from scipy.signal import convolve2d,fftconvolve

import warnings
warnings.filterwarnings("ignore")

thetavals_extra=np.pi*np.linspace(-0.1,1.4,150,endpoint=False).astype('float') # only need the range 0,1 plus a bit extra for periodicity
thetavals=thetavals_extra[10:110] # only need the range 0,1 plus a bit extra for periodicity
distvals=np.linspace(-10,10,100,endpoint=False)
thetavals_fine=np.pi*np.linspace(-0.1,1.4,1001,endpoint=False).astype('float')

def hough_counts_fast(passx,passy,lwtheta=3,lwr=3):
    allcounts=[]
    for ii in range(len(passx)):

        rvals_fine=np.sin(thetavals_fine)*passx[ii]\
              -np.cos(thetavals_fine)*passy[ii] 

        counts,_,_=np.histogram2d(thetavals_fine,rvals_fine,bins=[thetavals_extra,distvals])
        testcounts=convolve2d(counts,np.ones((lwtheta,lwr)),mode='valid').astype('bool').astype('int')

        allcounts.append(testcounts)
    #allcounts=np.sum(np.array(allcounts),axis=0)
    allcounts=np.array(allcounts)
#    allcounts2=np.sum(allcounts,axis=0)
    
    return allcounts

def line_finder_fast_xy(passx,passy,lwtheta=3,lwr=5,bgwindow=7):
    
    houghcounts_all=hough_counts_fast(passx,passy,lwtheta=lwtheta,lwr=lwr)
    
    houghcounts=np.sum(houghcounts_all,axis=0)

    Rr=10
    areastrip=[]
    for i in range(houghcounts.shape[-1]):
        r1=distvals[i]/Rr
        r2=distvals[i+lwr]/Rr

        areastrip.append(Rr**2*(-r1*np.sqrt(1 - r1**2) + r2*np.sqrt(1 - r2**2) - np.arcsin(r1) + np.arcsin(r2)))
    areastrip=np.array(areastrip)        
    houghcounts=np.array([houghcounts[i]/areastrip for i in range(len(houghcounts))])

    houghfilter=np.zeros(((bgwindow-1)*lwtheta+1,(bgwindow-1)*lwr+1))
    for ii in range(bgwindow):
        for jj in range(bgwindow):
            if (ii>=(bgwindow-1)/2-1 and ii<=(bgwindow-1)/2+1) and (jj>=(bgwindow-1)/2-1 and jj<=(bgwindow-1)/2+1) :
                continue
            houghfilter[ii*lwtheta,jj*lwr]=1
    #houghfilter=houghfilter[:,::-1]
    bgleveltry=convolve2d(houghcounts,houghfilter,mode='same')
    nbinstry=convolve2d(np.ones(houghcounts.shape),houghfilter,mode='same')
    bgleveltry=bgleveltry/nbinstry # this gives the sideband counts averaged
    bgstdtry=convolve2d(houghcounts**2,houghfilter,mode='same')
    bgstdtry=np.sqrt(bgstdtry/nbinstry-bgleveltry**2)
    
    bgerrtry=bgstdtry/np.sqrt(nbinstry) # the error on the mean is the std / sqrt(N)
    bgleveltry=bgleveltry*areastrip
    bgerrtry=bgerrtry*areastrip
    bgleveltry=bgleveltry[10:110]
    bgerrtry=bgerrtry[10:110]
    sigleveltry=houghcounts[10:110]*areastrip    
    significance=(sigleveltry-bgleveltry)/np.sqrt(bgleveltry+bgerrtry**2+1)
    
    indx,indy=np.unravel_index(np.argsort(significance.reshape((-1)))[::-1],significance.shape)
    maxind=[indx[(indy>1) & (indy<significance.shape[-1]-1)][0],\
            indy[(indy>1) & (indy<significance.shape[-1]-1)][0]]# exclude lines on the extreme edge of the patch
    bglevel=bgleveltry[maxind[0],maxind[1]]
    siglevel=sigleveltry[maxind[0],maxind[1]]
    linesigma=significance[maxind[0],maxind[1]]
    
    houghcounts_all=houghcounts_all[:,10:110,:]
    linestars=houghcounts_all[:,maxind[0],maxind[1]].astype('bool')
    if sum(linestars)!=round(siglevel):
        print('Error in linefit!',sum(linestars),round(siglevel))

        
    return maxind,bglevel,siglevel,linesigma,linestars,(bgleveltry,sigleveltry,bgerrtry)

def line_finder_fast(passstars,lwtheta=3,lwr=5,bgwindow=7):
    
    passx=passstars[:,6]
    passy=passstars[:,7]

    return line_finder_fast_xy(passx,passy,lwtheta=lwtheta,lwr=lwr,bgwindow=bgwindow)


def check_crossing(x,y,thetamin,thetamax,rmin,rmax):

    # x and y can be vectors
    
    crossing=np.array([False for i in range(len(x))])
    count=0
    phi=np.arctan2(y,x) # range [-pi,pi]
    rpoint=np.sqrt(x**2+y**2)
    
# UPPER EDGE
    ruse=rmax/rpoint

    thetasol=np.arcsin(ruse)+phi # range [-3pi/2,3pi/2]
    crossing = crossing | (~np.isnan(thetasol) & (thetasol>thetamin) & (thetasol<thetamax))
    
    thetasol=np.pi-np.arcsin(ruse)+phi
    crossing = crossing | (~np.isnan(thetasol) & (thetasol>thetamin) & (thetasol<thetamax))
    
    thetasol=np.arcsin(ruse)+phi-2*np.pi 
    crossing = crossing | (~np.isnan(thetasol) & (thetasol>thetamin) & (thetasol<thetamax))

    thetasol=np.pi-np.arcsin(ruse)+phi-2*np.pi
    crossing = crossing | (~np.isnan(thetasol) & (thetasol>thetamin) & (thetasol<thetamax))

    thetasol=np.arcsin(ruse)+phi+2*np.pi 
    crossing = crossing | (~np.isnan(thetasol) & (thetasol>thetamin) & (thetasol<thetamax))

    thetasol=np.pi-np.arcsin(ruse)+phi+2*np.pi
    crossing = crossing | (~np.isnan(thetasol) & (thetasol>thetamin) & (thetasol<thetamax))
    
    
#LOWER EDGE
    ruse=rmin/rpoint
    
    thetasol=np.arcsin(ruse)+phi # range [-3pi/2,3pi/2]
    crossing = crossing | (~np.isnan(thetasol) & (thetasol>thetamin) & (thetasol<thetamax))
    
    thetasol=np.pi-np.arcsin(ruse)+phi
    crossing = crossing | (~np.isnan(thetasol) & (thetasol>thetamin) & (thetasol<thetamax))
    
    thetasol=np.arcsin(ruse)+phi-2*np.pi 
    crossing = crossing | (~np.isnan(thetasol) & (thetasol>thetamin) & (thetasol<thetamax))

    thetasol=np.pi-np.arcsin(ruse)+phi-2*np.pi
    crossing = crossing | (~np.isnan(thetasol) & (thetasol>thetamin) & (thetasol<thetamax))

    thetasol=np.arcsin(ruse)+phi+2*np.pi 
    crossing = crossing | (~np.isnan(thetasol) & (thetasol>thetamin) & (thetasol<thetamax))

    thetasol=np.pi-np.arcsin(ruse)+phi+2*np.pi
    crossing = crossing | (~np.isnan(thetasol) & (thetasol>thetamin) & (thetasol<thetamax))

# Left edge
    rval=np.sin(thetamin)*x-np.cos(thetamin)*y
    crossing = crossing | ((rval>rmin) & (rval<rmax))
        
# Right edge
    rval=np.sin(thetamax)*x-np.cos(thetamax)*y
    crossing = crossing | ((rval>rmin) & (rval<rmax))

    return crossing






def line_stars(maxind,passx,passy,lwtheta=3,lwr=3):
    # maxind is taken from thetavals, not thetavals_extra
    linestars=check_crossing(passx,passy,\
        thetavals_extra[maxind[0]+10],thetavals_extra[maxind[0]+10+lwtheta],\
                             distvals[maxind[1]],distvals[maxind[1]+lwr])
            
            
#    thetavalstouse=0.5*(thetavals_extra[10+maxind[0]:10+maxind[0]+lwtheta]+thetavals_extra[10+maxind[0]+1:10+maxind[0]+1+lwtheta])
#    rvals=np.array([np.sin(thetavalstouse)*passx[i]\
#          -np.cos(thetavalstouse)*passy[i] for i in range(len(passx))])
#    linestars=((rvals>distvals[maxind[1]]) & (rvals<distvals[maxind[1]+lwr])).any(axis=1)
    return linestars


def hough_counts(passx,passy,lwtheta=3,lwr=3):
    houghcounts=np.zeros((len(thetavals_extra)-lwtheta,len(distvals)-lwr))
    for i in range(len(thetavals_extra)-lwtheta):
        for j in range(len(distvals)-lwr):

            crosslist=check_crossing(passx,passy,\
                    thetavals_extra[i],thetavals_extra[i+lwtheta],distvals[j],distvals[j+lwr])
            houghcounts[i,j]+=len(np.argwhere(crosslist))
    
    return houghcounts


def line_finder(passstars,lwtheta=3,lwr=3):
    
    passx=passstars[:,6]
    passy=passstars[:,7]

    houghcounts=hough_counts(passx,passy,lwtheta=lwtheta,lwr=lwr)
    Rr=10
    areastrip=[]
    for i in range(houghcounts.shape[-1]):
        r1=distvals[i]/Rr
        r2=distvals[i+lwr]/Rr

        areastrip.append(Rr**2*(-r1*np.sqrt(1 - r1**2) + r2*np.sqrt(1 - r2**2) - np.arcsin(r1) + np.arcsin(r2)))
    areastrip=np.array(areastrip)    
    houghcounts=np.array([houghcounts[i]/areastrip for i in range(len(houghcounts))])
    
    houghcounts3=[]
    for i in range(len(thetavals)):
        for j in range(houghcounts.shape[1]):

    #            houghcounts2=houghcounts[max(0,i-2*lwtheta):min(i+3*lwtheta,houghcounts.shape[0]):lwtheta,\
    #                                     max(0,j-2*lwr):min(j+3*lwr,houghcounts.shape[1]):lwr]        
            houghcounts2=houghcounts[max(0,10+i-2*lwtheta):min(10+i+3*lwtheta,houghcounts.shape[0]):lwtheta,\
                                     max(0,j-2*lwr):min(j+3*lwr,houghcounts.shape[1]):lwr]        
            bglevel=(np.sum(houghcounts2)-houghcounts[10+i,j])/(len(houghcounts2.reshape((-1)))-1)*areastrip[j]
            if bglevel==0:
                bglevel=1
            siglevel=houghcounts[10+i,j]*areastrip[j]

            houghcounts3.append([i,j,bglevel,siglevel,(siglevel-bglevel)/np.sqrt(bglevel)])
    houghcounts3=np.array(houghcounts3)
    houghcounts4=houghcounts3[houghcounts3[:,-2]>0]


    maxind=houghcounts4[np.argmax(houghcounts4[:,-1])][:2].astype('int')
    bglevel,siglevel,linesigma=tuple(houghcounts4[np.argmax(houghcounts4[:,-1])][2:])
    
    linestars=line_stars(maxind,passx,passy,lwtheta=lwtheta,lwr=lwr)
#    linestars=passstars[linestars]

    if sum(linestars)!=round(siglevel):
        print('Error in linefit!',sum(linestars),round(siglevel))

        
    return maxind,bglevel,siglevel,linesigma,linestars,houghcounts3

