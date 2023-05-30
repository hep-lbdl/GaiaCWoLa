import numpy as np

from utils import recenter_coords,load_gaiadata,mean_angles,cluster_cluster_distance2
#def recenter_coords(x,y,vx,vy,newcenter):
from astropy import units as u
from astropy.coordinates import SkyOffsetFrame, ICRS, SkyCoord, Galactic,Galactocentric
import astropy.coordinates as coords
from line_finder_v2 import line_finder_fast_xy
from astropy.coordinates import Angle
from astropy.stats import circmean
from utils import linefit_chi2

VMdir='./'

def stream_significance(protostreams,thresh=0):
    siglist=[]
    for protostream in protostreams:
        if protostream.line_sigma<thresh:
            continue
        siglist.append([protostream.ra,\
                        protostream.dec,\
                        protostream.line_sigma])
    siglist=np.array(siglist)
    radeclist=np.unique(siglist[:,:2],axis=0)
    significance=0
    for ra,dec in radeclist:
        significance+=max(siglist[(siglist[:,0]==ra) & (siglist[:,1]==dec)][:,2])**2
    significance=np.sqrt(significance)
    
    return significance

class Stream:
    def __init__(self, protostreams):
        self.protostreams=protostreams
        self.significance=stream_significance(protostreams)

        
thetavals_extra=np.pi*np.linspace(-0.1,1.4,150,endpoint=False).astype('float') # only need the range 0,1 plus a bit extra for periodicity
thetavals=thetavals_extra[10:110] # only need the range 0,1 plus a bit extra for periodicity
distvals=np.linspace(-10,10,100,endpoint=False)
thetavals_fine=np.pi*np.linspace(-0.1,1.4,1001,endpoint=False).astype('float')

dataset_dict=np.load(VMdir+'dataset_dict.npy',allow_pickle=True).item()
allradeclist=list(dataset_dict.keys())
# We have found that a number of patches which are extremely occluded by dust likely lead to spurious VM stream
# detections. So we have decided to cut these out of the analysis entirely.
# version 2 of this list is derived from the SFD dustmap, using a cut on A_V tuned to visibly dusty patches
tocutlist=np.load(VMdir+'dusty_patches_tocut.npy') # [ra,dec,l,b]
tocutlist2=np.array([[339.2,-3.7,63.0,-50.2]]) # this patch gave a lot of spurious stream detections so we cut it out
tocutlist3=np.concatenate((tocutlist,tocutlist2))
allradeclist=[x for x in allradeclist if list(x) not in tocutlist3[:,:2].tolist()]
#print(len(allradeclist))

def find_overlapping_patches(raseed,decseed):

    overlap_patches=[]
    for ra,dec in allradeclist:
        if ra==raseed and dec==decseed:
            continue

        # generate random points in 10 degree circle centered around ra,dec
        xy=20*np.random.uniform(size=(10000,2))-10
        xy=xy[np.sum(xy**2,axis=1)<10**2]
        center = ICRS(ra=ra*u.deg, dec=dec*u.deg)
        coordinates_list=SkyOffsetFrame(origin=center,lon=xy[:,0]*u.deg,lat=xy[:,1]*u.deg) 

        ranew=coordinates_list.transform_to(ICRS).ra.value
        decnew=coordinates_list.transform_to(ICRS).dec.value
        newx,newy,_,_=recenter_coords(ranew,decnew,np.zeros(len(ranew)),np.zeros(len(decnew)),newcenter=[raseed,decseed])
        if np.sum(newx**2+newy**2<10**2)>0:

            overlap_patches.append([ra,dec])
    
    return overlap_patches

def pc_equal(pc1,pc2):
    pcdata1=[[roi1.ra,roi1.dec,roi1.pmlat,roi1.pmlon,roi1.pmuse] for roi1 in pc1.ROIlist]
    pcdata2=[[roi2.ra,roi2.dec,roi2.pmlat,roi2.pmlon,roi2.pmuse] for roi2 in pc2.ROIlist]
    return pcdata1==pcdata2

def overlap_region(x1,x2):
    Amin=np.min(x1)
    Amax=np.max(x1)
    Bmin=np.min(x2)
    Bmax=np.max(x2)
    if Amax<Bmin or Bmax<Amin:
        overlap=(999,-999)
    else:
        overlap=max(Amin,Bmin),min(Amax,Bmax)

    return overlap

overlapping_patches_dict=np.load(VMdir+'overlapping_patches_dict.npy',allow_pickle=True).item()

def clusterQ(ps1,ps2,pmdist=1.5,deltatheta=5,deltar=5,fraccut=0.4,lwr=5,lwtheta=3):
# streamlist is a list of protostreams
    mergeQ=False

    if [ps1.ra,ps1.dec] in overlapping_patches_dict[ps2.ra,ps2.dec]:
        ra1=ps1.ra
        dec1=ps1.dec
        ra2=ps2.ra
        dec2=ps2.dec

        midpoint=[mean_angles(ra1,ra2),0.5*(dec1+dec2)]

    
    
# we include every pc in the ps
        for pc1 in ps1.pclist:
#        pc=ps.pclist[0]
    #    print(pc.ra,pc.dec)
#            stream_linestars=np.concatenate((stream_linestars,pc.highRstars[pc.line_stars]))
    #    plt.scatter(pc.highRstars[pc.line_stars][:,3],pc.highRstars[pc.line_stars][:,2],color='r')
    
    
            linestars_seed=pc1.highRstars[pc1.line_stars]
            newx,newy,_,_=recenter_coords(linestars_seed[:,3],linestars_seed[:,2],\
                                          np.zeros(len(linestars_seed)),np.zeros(len(linestars_seed)),\
                                          newcenter=midpoint)
            r=np.sqrt(newx**2+newy**2)
            xkeep1=newx[r<10]
            ykeep1=newy[r<10]
            linestars_keep1=linestars_seed[r<10]

        # we try every pc in the protostream, if any match with the stream then we merge
            for pc2 in ps2.pclist:
            #linestars_seed=pc.highRstars[pc.line_stars]
                linestars_try=pc2.highRstars[pc2.line_stars]


                newx,newy,_,_=recenter_coords(linestars_try[:,3],linestars_try[:,2],\
                                              np.zeros(len(linestars_try)),np.zeros(len(linestars_try)),\
                                              newcenter=midpoint)
                r=np.sqrt(newx**2+newy**2)
                xkeep2=newx[r<10]
                ykeep2=newy[r<10]
                linestars_keep2=linestars_try[r<10]
                #            if pc_equal(pctry,pcwant):
                #                print(cluster_cluster_distance2(linestars_keep1[:,[0,1]],linestars_keep2[:,[0,1]]))
                if cluster_cluster_distance2(linestars_keep1[:,[0,1]],linestars_keep2[:,[0,1]])<pmdist:
                #    continue

                #        plt.scatter(ykeep1,xkeep1,s=10)
                #        plt.scatter(ykeep2,xkeep2,s=10)
                #        plt.xlim(-15,15)
                #        plt.ylim(-15,15)
                #        plt.gca().set_aspect('equal')

                    if not (len(ykeep1)<2 or len(ykeep2)<2):


                        maxind1,_,_,_,_,_=line_finder_fast_xy(ykeep1,xkeep1,lwr=lwr)
                        maxind2,_,_,_,_,_=line_finder_fast_xy(ykeep2,xkeep2,lwr=lwr)
                        maxind12,_,_,_,_,_=line_finder_fast_xy(np.concatenate((ykeep1,\
                                                    ykeep2)),\
                                                np.concatenate((xkeep1,\
                                                    xkeep2)),lwr=lwr)

                        theta12=thetavals[maxind12[0]]
                        r12=distvals[maxind12[1]]

                        # rotate to canonical coordinates of higher significance stream
                        x_rot1=np.cos(theta12)*ykeep1+np.sin(theta12)*xkeep1
                        y_rot1=-np.sin(theta12)*ykeep1+np.cos(theta12)*xkeep1
                        x_rot2=np.cos(theta12)*ykeep2+np.sin(theta12)*xkeep2
                        y_rot2=-np.sin(theta12)*ykeep2+np.cos(theta12)*xkeep2

                        xmin,xmax=overlap_region(x_rot1,x_rot2)
                        linestars_keep1_overlap=linestars_keep1[(x_rot1>xmin) & (x_rot1<xmax)]
                        linestars_keep2_overlap=linestars_keep2[(x_rot2>xmin) & (x_rot2<xmax)]

                        # this is how many stars from the stream candidate are in the protocluster, 
                        # counting duplicates
                        overlapfrac=0
                        if len(linestars_keep1_overlap)==0 or len(linestars_keep2_overlap)==0:
                            continue                            
                        ii,jj=np.where((linestars_keep1_overlap[:,:6]==(linestars_keep2_overlap[:,:6])[:,None]).sum(axis=2)==6)
                        jj=np.unique(jj)
                        ii=np.unique(ii)
                        overlapfrac=max(len(jj)/len(linestars_keep1_overlap),\
                                        len(ii)/len(linestars_keep2_overlap) )
                        print(xmax-xmin,overlapfrac,[np.abs(maxind2[0]-maxind1[0]),np.abs(maxind2[1]-maxind1[1])],\
                        [np.abs(maxind2[0]-100-maxind1[0]), np.abs((100-lwr-maxind2[1])-maxind1[1])],\
                        [np.abs(maxind2[0]+100-maxind1[0]), np.abs((100-lwr-maxind2[1])-maxind1[1])])

                        if overlapfrac>fraccut:
                            if ((np.abs(maxind2[0]-maxind1[0])<=deltatheta\
                                 and np.abs(maxind2[1]-maxind1[1])<=deltar)\
                             or (np.abs(maxind2[0]-100-maxind1[0])<=deltatheta \
                                 and np.abs((100-lwr-maxind2[1])-maxind1[1])<=deltar)\
                             or (np.abs(maxind2[0]+100-maxind1[0])<=deltatheta\
                                 and np.abs((100-lwr-maxind2[1])-maxind1[1])<=deltar)):
                                mergeQ=True
                                break
                                
            if mergeQ:
                break
                
    return mergeQ

def cluster_mergelist(ps1,ps2,pmdist=1.5,deltatheta=5,deltar=5,fraccut=0.4,lwr=5,lwtheta=3):
# streamlist is a list of protostreams
    mergelist=[]

    if [ps1.ra,ps1.dec] in overlapping_patches_dict[ps2.ra,ps2.dec]:
#        ra1=ps1.ra
#        dec1=ps1.dec
#        ra2=ps2.ra
#        dec2=ps2.dec
#        midpoint=[mean_angles(ra1,ra2),0.5*(dec1+dec2)]

    
    
# we include every pc in the ps
        for ipc1,pc1 in enumerate(ps1.pclist):
#        pc=ps.pclist[0]
    #    print(pc.ra,pc.dec)
#            stream_linestars=np.concatenate((stream_linestars,pc.highRstars[pc.line_stars]))
    #    plt.scatter(pc.highRstars[pc.line_stars][:,3],pc.highRstars[pc.line_stars][:,2],color='r')
    
    
            linestars1=pc1.highRstars[pc1.line_stars]

        # we try every pc in the protostream, if any match with the stream then we merge
            for ipc2,pc2 in enumerate(ps2.pclist):
                
                linestars2=pc2.highRstars[pc2.line_stars]
                raavg=Angle(circmean(np.concatenate((linestars1,linestars2))[:,3]*u.degree)).wrap_at(360*u.degree).degree
                decavg=np.mean(np.concatenate((linestars1,linestars2))[:,2])
                midpoint=[raavg,decavg]

                newx1,newy1,_,_=recenter_coords(linestars1[:,3],linestars1[:,2],\
                                          np.zeros(len(linestars1)),np.zeros(len(linestars1)),\
                                          newcenter=midpoint)
#            r=np.sqrt(newx**2+newy**2)
#            xkeep1=newx[r<10]
#            ykeep1=newy[r<10]
#            linestars_keep1=linestars_seed[r<10]


                newx2,newy2,_,_=recenter_coords(linestars2[:,3],linestars2[:,2],\
                                              np.zeros(len(linestars2)),np.zeros(len(linestars2)),\
                                              newcenter=midpoint)
#                r=np.sqrt(newx**2+newy**2)
#                xkeep2=newx[r<10]
#                ykeep2=newy[r<10]
#                linestars_keep2=linestars_try[r<10]
                #            if pc_equal(pctry,pcwant):
                #                print(cluster_cluster_distance2(linestars_keep1[:,[0,1]],linestars_keep2[:,[0,1]]))
#               We don't need the proper motion requirement if we are checking the overlap fraction!
#                if cluster_cluster_distance2(linestars1[:,[0,1]],linestars2[:,[0,1]])<pmdist:
                if True:
                #    continue

                #        plt.scatter(ykeep1,xkeep1,s=10)
                #        plt.scatter(ykeep2,xkeep2,s=10)
                #        plt.xlim(-15,15)
                #        plt.ylim(-15,15)
                #        plt.gca().set_aspect('equal')

#                    if not (len(ykeep1)<2 or len(ykeep2)<2):


#                    maxind12,_,_,_,_,_=line_finder_fast_xy(np.concatenate((newx1,\
#                                                newx2)),\
#                                            np.concatenate((newy1,\
#                                                newy2)),lwr=lwr)
#
#                    theta12=thetavals[maxind12[0]]
#                    r12=distvals[maxind12[1]]
                    _,theta12,r12=linefit_chi2(np.concatenate((newx1,newx2)),\
                                            np.concatenate((newy1,newy2)))

                    # rotate to canonical coordinates of higher significance stream
                    x_rot1=np.cos(theta12)*newx1+np.sin(theta12)*newy1
                    y_rot1=-np.sin(theta12)*newx1+np.cos(theta12)*newy1+r12
                    x_rot2=np.cos(theta12)*newx2+np.sin(theta12)*newy2
                    y_rot2=-np.sin(theta12)*newx2+np.cos(theta12)*newy2+r12
                    
#                    maxind1,_,_,_,_,_=line_finder_fast_xy(x_rot1,y_rot1,lwr=lwr)
#                    maxind2,_,_,_,_,_=line_finder_fast_xy(x_rot2,y_rot2,lwr=lwr)

                    xmin,xmax=overlap_region(x_rot1,x_rot2)
                    linestars_keep1_overlap=linestars1[(x_rot1>xmin) & (x_rot1<xmax)]
                    linestars_keep2_overlap=linestars2[(x_rot2>xmin) & (x_rot2<xmax)]

                    # this is how many stars from the stream candidate are in the protocluster, 
                    # counting duplicates
                    overlapfrac=0
                    if len(linestars_keep1_overlap)==0 or len(linestars_keep2_overlap)==0:
                        continue   

#                        _,indtemp=np.unique(linestars_keep1_overlap[:,:6],axis=0,return_index=True) 
#                        linestars_keep1_overlap=linestars_keep1_overlap[indtemp]
#                        _,indtemp=np.unique(linestars_keep2_overlap[:,:6],axis=0,return_index=True) 
#                        linestars_keep2_overlap=linestars_keep2_overlap[indtemp]

                    ii,jj=np.where((linestars_keep1_overlap[:,:6]==(linestars_keep2_overlap[:,:6])[:,None]).sum(axis=2)==6)
                    jj=np.unique(jj)
                    ii=np.unique(ii)
                    overlapfrac=max(len(jj)/len(linestars_keep1_overlap),\
                                    len(ii)/len(linestars_keep2_overlap) )
                    print(ipc1,pc1.line_sigma,ipc2,pc2.line_sigma)
#                    print(ipc1,ipc2,xmax-xmin,overlapfrac,[np.abs(maxind2[0]-maxind1[0]),np.abs(maxind2[1]-maxind1[1])],\
#                    [np.abs(maxind2[0]-100-maxind1[0]), np.abs((100-lwr-maxind2[1])-maxind1[1])],\
#                    [np.abs(maxind2[0]+100-maxind1[0]), np.abs((100-lwr-maxind2[1])-maxind1[1])])

#                        plt.scatter(x_rot1,y_rot1,s=1)
#                        plt.scatter(x_rot2,y_rot2,s=1)
#                        plt.xlim(-10,10)
#                        plt.ylim(-10,10)
#                        plt.gca().set_aspect('equal')
#                        plt.show()

                    if overlapfrac>fraccut:
        
        
                        _,theta1,r1=linefit_chi2(x_rot1,y_rot1)
                        _,theta2,r2=linefit_chi2(x_rot2,y_rot2)
                        dtheta=theta1-theta2
                        dr=r1-r2
#                        lineQ=False
#                        if (np.abs(maxind2[0]-maxind1[0])<=deltatheta\
#                             and np.abs(maxind2[1]-maxind1[1])<=deltar):
#                            dtheta=np.abs(maxind2[0]-maxind1[0])
#                            dr=np.abs(maxind2[1]-maxind1[1])
#                            lineQ=True
#                        elif(np.abs(maxind2[0]-100-maxind1[0])<=deltatheta \
#                             and np.abs((100-lwr-maxind2[1])-maxind1[1])<=deltar):
#                            dtheta=np.abs(maxind2[0]-100-maxind1[0])
#                            dr=np.abs((100-lwr-maxind2[1])-maxind1[1])
#                            lineQ=True
#                        elif(np.abs(maxind2[0]+100-maxind1[0])<=deltatheta\
#                             and np.abs((100-lwr-maxind2[1])-maxind1[1])<=deltar):
#                            dtheta=np.abs(maxind2[0]+100-maxind1[0])
#                            dr=np.abs((100-lwr-maxind2[1])-maxind1[1])
#                            lineQ=True
#                        if lineQ:
                        mergelist.append([ipc1,ipc2,xmax-xmin,overlapfrac,dtheta,dr])
#                                break
                                
#            if mergeQ:
#                break
                
    return mergelist

def cluster_one_step(streamlist,tomergelist,pmdist=1.5,\
                     deltatheta=5,deltar=5,\
                     lwr=5,lwtheta=3,fraccut=0.4):

    print('Finding overlapping patches...')
    overlap_patches=[]
    for ps in streamlist:
        overlap_patches+=find_overlapping_patches(ps.ra,ps.dec)
    overlap_patches=np.unique(np.array(overlap_patches),axis=0).tolist()
    print('Done finding overlapping patches...')
    print(overlap_patches)
  
    streamlist_OUT=streamlist[:]
    tomergelist_OUT=tomergelist[:]
    
    for ps in tomergelist_OUT:
        if [ps.ra,ps.dec] not in overlap_patches:
            continue
        if clusterQ(streamlist_OUT,ps,\
                    pmdist=pmdist,deltatheta=deltatheta,\
                    deltar=deltar,fraccut=fraccut,\
                    lwr=lwr,lwtheta=lwtheta):
            streamlist_OUT.append(ps)
            tomergelist_OUT.remove(ps)
            break
    print(len(streamlist),len(tomergelist))
    print(len(streamlist_OUT),len(tomergelist_OUT))
    return streamlist_OUT,tomergelist_OUT


def cluster_one_seed(protostream_SEED,allprotostreams,\
                     deltatheta=5,deltar=5,pmdist=1.5,\
                     fraccut=0.4,lwr=5,lwtheta=3):
    streamlist_OUT=[protostream_SEED]
    tomergelist_OUT=allprotostreams
    streamlist=[]
    tomergelist=[]
    while streamlist_OUT!=streamlist:
        streamlist=streamlist_OUT
        tomergelist=tomergelist_OUT
        streamlist_OUT,tomergelist_OUT=cluster_one_step(streamlist,tomergelist,\
                                            pmdist=pmdist,deltatheta=deltatheta,\
                                            deltar=deltar,fraccut=fraccut,\
                                            lwr=lwr,lwtheta=lwtheta)
    
    
    return streamlist_OUT,tomergelist_OUT