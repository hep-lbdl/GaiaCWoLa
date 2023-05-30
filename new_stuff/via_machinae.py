import numpy as np
import glob

from astropy import units as u
from astropy.coordinates import SkyOffsetFrame, ICRS, SkyCoord, Galactic
import astropy.coordinates as coords
from utils import angular_distance, houghpar_distance, transform_linepars, mean_linepars, mean_angles, recenter_coords
from utils import cluster_cluster_distance, cluster_cluster_distance2, linefit_chi2
import time
from line_finder_v2 import line_finder
from sklearn.cluster import DBSCAN

# format: ra,dec,pmra,pmdec
allgd1stars=np.load('./gd1_stars.npy')
defaultROIdir='./ROIs/'
defaultROIdir_pmlon='./ROIs_pmlon/'
VMdir='./'



def FilterGD1(stars):
    gd1stars=np.zeros(len(stars))
    for x in allgd1stars:
        ra=x[0]
        dec=x[1]
        pmra=x[2]
        pmdec=x[3]
    #    print(ra,dec,pmra,pmdec)
    
        foundlist=angular_distance(np.dstack((stars[:,3],stars[:,2]))[0],np.array([[ra,dec]]))
        foundlist=np.sqrt(foundlist**2+(stars[:,0]-pmdec)**2+(stars[:,1]-pmra)**2)                                                                  
        foundlist=foundlist<.0001
        if len(np.argwhere(foundlist))>1:
            print(foundlist)
        if len(np.argwhere(foundlist))==1:
#            ngd1+=1
            gd1stars+=foundlist
#    print(ngd1)
    gd1stars=gd1stars.astype('bool')
    return gd1stars,stars[gd1stars]



class SignalRegion:
    def __init__(self, ra,dec,pm,pmuse='lat',highRstars=np.empty((0,10)),gcresults=[]): 
        
        self.ra=ra
        self.dec=dec
        self.pm=pm
        self.pmuse=pmuse
        self.highRstars=highRstars
        self.linestars=np.zeros(len(highRstars)).astype('bool')
        self.pmcluster=np.zeros(len(highRstars)).astype('bool')
        self.linesigma=0
        self.gcresults=gcresults
    def linefit(self,lwr=5,lwtheta=3,eps=2):
        
        clustering=DBSCAN(eps=eps).fit(self.highRstars[:,[8,9]])
        if max(clustering.labels_)>=0:
            maxlabel=np.argmax([np.sum(clustering.labels_==ii) for ii in range(max(clustering.labels_)+1)])
            self.pmcluster=(clustering.labels_==maxlabel)

            highRstars=self.highRstars[self.pmcluster]
            maxind_temp,bglevel_temp,siglevel_temp,linesigma_temp,linestars_temp,_=\
                line_finder(highRstars,lwr=lwr,lwtheta=lwtheta)
            self.linestars=linestars_temp
            self.linesigma=linesigma_temp

class StreamClusterSR:
    def __init__(self, SRlist):
        self.SRlist=SRlist
        self.line_sigma=np.array([x.linesigma for x in SRlist])
        self.significance=np.sqrt(np.sum(self.line_sigma**2))
        self.streamstars=np.empty((0,SRlist[0].highRstars.shape[-1]))
        for SR in SRlist:
            self.streamstars=np.concatenate((self.streamstars,SR.highRstars[SR.pmcluster][SR.linestars]))

            
class ROI:
    def __init__(self, ra,dec,pmlat,pmlon,lwr=3,lwtheta=3,pmuse='lat',ROIdir=defaultROIdir,dataset=''): 
        
        if dataset=='':
            dataset_dict=np.load(VMdir+'dataset_dict.npy',allow_pickle=True).item()
            self.dataset=dataset_dict[ra,dec]
        else:
            self.dataset=dataset
        self.ra=ra
        self.dec=dec
        self.pmlat=pmlat
        self.pmlon=pmlon
        self.pmuse=pmuse
        templist=self.dataset.split('_')
        self.l=float(templist[-4].strip('l'))
        self.b=float(templist[-3].strip('b'))
        
        pmmin=pmlat
        pmmax=pmlat+6

        if ROIdir!='':
            ROIstars=np.load(ROIdir+self.dataset+'_ROIstars.npy',allow_pickle=True).item()
            ROIRvals=np.load(ROIdir+self.dataset+'_ROIRvals.npy',allow_pickle=True).item()
            ROIcounts=np.load(ROIdir+self.dataset+'_ROIcounts.npy',allow_pickle=True).item()
            ROIsourceids=np.load(ROIdir+self.dataset+'_ROIsourceids.npy',allow_pickle=True).item()
            fitresults=np.load(ROIdir+self.dataset+'_fitresults_lwr'+str(lwr)+'_lwtheta'+str(lwtheta)+'.npy',\
                           allow_pickle=True).item()   
            linestars=np.load(ROIdir+self.dataset+'_linestars_lwr'+str(lwr)+'_lwtheta'+str(lwtheta)+'.npy',\
                           allow_pickle=True).item() 
            
            if pmuse=='lat':
                try:
                    gcresults=np.load(ROIdir+self.dataset+'_lat_gcresults.npy',allow_pickle=True).item()
                except:
                    gcresults={}
                    gcresults[pmlat]=[]
            else:
                try:
                    gcresults=np.load(ROIdir+self.dataset+'_lon_gcresults.npy',allow_pickle=True).item()
                except:
                    gcresults={}
                    gcresults[pmlon]=[]
                    
            if pmuse=='lat':

                try:
                    ROIstars=ROIstars[pmlat,pmlon]
                    ROIRvals=ROIRvals[pmlat,pmlon]
                    ROIcounts=ROIcounts[pmlat,pmlon]
                    ROIsourceids=ROIsourceids[pmlat,pmlon]
                    fitresults=fitresults[pmlat,pmlon]
                    linestars=linestars[pmlat,pmlon]
                    line_theta=thetavals[int(fitresults[0])]
                    line_r=distvals[int(fitresults[1])]
                    line_sigma=fitresults[-1]
                    try:
                        gcresults=gcresults[pmlat] #gcresults[gcresults[:,2]==pmlat][0]
                    except:
                        gcresults=[]
                except:
                    ROIstars=np.empty((0,13))
                    ROIRvals=np.empty((0))
                    ROIcounts=np.empty((0))
                    ROIsourceids=np.empty((0))
                    fitresults=np.empty((0))
                    linestars=np.empty((0))
                    line_theta=-999
                    line_r=-999
                    line_sigma=-999
                    gcresults=[]

            else:
                try:
                    ROIstars=ROIstars[pmlon,pmlat]   # be careful to reverse order of pmlon and pmlat here
                    ROIRvals=ROIRvals[pmlon,pmlat]   # be careful to reverse order of pmlon and pmlat here
                    ROIcounts=ROIcounts[pmlon,pmlat]   # be careful to reverse order of pmlon and pmlat here
                    ROIsourceids=ROIsourceids[pmlon,pmlat]   # be careful to reverse order of pmlon and pmlat here
                    fitresults=fitresults[pmlon,pmlat]
                    linestars=linestars[pmlon,pmlat]
                    line_theta=thetavals[int(fitresults[0])]
                    line_r=distvals[int(fitresults[1])]
                    line_sigma=fitresults[-1]
                    try:
                        gcresults=gcresults[pmlon] #gcresults[gcresults[:,2]==pmlat][0]
                    except:
                        gcresults=[]
                except:
                    ROIstars=np.empty((0,13))
                    ROIRvals=np.empty((0))
                    ROIcounts=np.empty((0))
                    ROIsourceids=np.empty((0))
                    fitresults=np.empty((0))
                    linestars=np.empty((0))
                    line_theta=-999
                    line_r=-999
                    line_sigma=-999
                    gcresults=[]


            self.highRstars=ROIstars
            self.Rvals=ROIRvals
            self.counts=ROIcounts
            self.sourceids=ROIsourceids
            self.linestars=linestars
            self.fitresults=fitresults
            self.line_theta=line_theta
            self.line_r=line_r
            self.line_sigma=line_sigma
            self.gcresults=gcresults
            gd1stars_mask,_=FilterGD1(ROIstars)
            self.gd1stars_mask=gd1stars_mask



        knownstreams_dict=np.load(VMdir+'knownstreams.npz')
        knownstreams=knownstreams_dict.files

        self.knownstreams={}
        for name in knownstreams:
            lat_stream,lon_stream,pmlat_stream,pmlon_stream=\
                recenter_coords(knownstreams_dict[name][:,2],knownstreams_dict[name][:,3],\
                                knownstreams_dict[name][:,4],knownstreams_dict[name][:,5],[ra,dec])
            stars_stream=knownstreams_dict[name][(lat_stream**2+lon_stream**2<10**2) & \
                                    (pmlat_stream>pmlat) & (pmlat_stream<pmlat+6) & \
                                    (pmlon_stream>pmlon) & (pmlon_stream<pmlon+6)]
            if(len(stars_stream)>0):
                self.knownstreams[name]=stars_stream

            

class Protocluster:
    def __init__(self, ROIlist):
        
        ralist=np.array([ROI.ra for ROI in ROIlist])
        declist=np.array([ROI.dec for ROI in ROIlist])
        pmlist=np.array([[ROI.pmlon,ROI.pmlat] for ROI in ROIlist])
#        pmlatlist=[ROI.pmlat for ROI in ROIlist]
        line_theta=np.array([ROI.line_theta for ROI in ROIlist])
        line_r=np.array([ROI.line_r for ROI in ROIlist])
        line_sigma=np.array([ROI.line_sigma for ROI in ROIlist])        
        
            
        if len(np.unique(ralist)==1 and np.unique(declist)==1): # and np.unique(pmlonlist)==1): try 3/1
            self.ra=ralist[0]
            self.dec=declist[0]
#            self.pmlon=pmlonlist
#            self.pmlat=pmlatlist
            self.pm=pmlist
#            self.pmlat=pmlatlist
            self.highRstars=[]
            self.fitresults=[]
            self.linestars=[]
            for ROI in ROIlist:
                self.highRstars+=[list(ROI.highRstars)]
                self.fitresults+=[list(ROI.fitresults)]
                self.linestars+=[list(ROI.linestars)]
                    
            self.highRstars=np.array(self.highRstars)
            self.fitresults=np.array(self.fitresults)
            self.linestars=np.array(self.linestars)
            self.line_theta=line_theta
            self.line_r=line_r
            self.line_sigma=line_sigma
            self.ROIlist=ROIlist
            ROIlist_pmlat=np.array([roi for roi in ROIlist if roi.pmuse=='lat'])
            ROIlist_pmlon=np.array([roi for roi in ROIlist if roi.pmuse=='lon'])
            

            # Calculate line significance of protocluster using independent ROIs
            # ROIs can be independent either by using different SR windows (pmlat for pmlat ROIs and pmlon for pmlon ROIs)
            # or by coming from different pm scans (pmlat and pmlon)
            line_sigma_tot=0
            ROIlist_maxsig=[]
            
            pmlattemp=np.array([roi.pmlat for roi in ROIlist_pmlat])
            pmlatlist=np.unique(pmlattemp)
            linesigmatemp=np.array([roi.line_sigma for roi in ROIlist_pmlat])            
            for pmlat in pmlatlist:
                line_sigma_tot=np.sqrt(line_sigma_tot**2+np.max(linesigmatemp[pmlattemp==pmlat])**2)
                ROIlist_maxsig.append(ROIlist_pmlat[pmlattemp==pmlat][np.argmax(linesigmatemp[pmlattemp==pmlat])])  # FIXED BUG 3/14/22
                
            pmlontemp=np.array([roi.pmlon for roi in ROIlist_pmlon])
            pmlonlist=np.unique(pmlontemp)
            linesigmatemp=np.array([roi.line_sigma for roi in ROIlist_pmlon])            
            for pmlon in pmlonlist:
                line_sigma_tot=np.sqrt(line_sigma_tot**2+np.max(linesigmatemp[pmlontemp==pmlon])**2)
                ROIlist_maxsig.append(ROIlist_pmlon[pmlontemp==pmlon][np.argmax(linesigmatemp[pmlontemp==pmlon])]) # FIXED BUG 3/14/22
          
            self.line_sigma_tot=line_sigma_tot
            self.ROIlist_maxsig=ROIlist_maxsig
            
        else:
            print('Error in Protocluster class!')

thetavals_extra=np.pi*np.linspace(-0.1,1.4,150,endpoint=False).astype('float') # only need the range 0,1 plus a bit extra for periodicity
thetavals=thetavals_extra[10:110] # only need the range 0,1 plus a bit extra for periodicity
distvals=np.linspace(-10,10,100,endpoint=False)




#def houghpar_distance_cluster(pars1_cl,pars2_cl):
#    distlist=[[i,j,houghpar_distance(pars1_cl[i],pars2_cl[j])] for i in range(len(pars1_cl)) for j in range(len(pars2_cl))]
#    return distlist

def merge_lineparsQ(pars1,pars2,theta_cut=np.pi/9.9,r_cut=2.1):
    #inputs are [theta1,r1] and [theta2,r2]
    dist=houghpar_distance(pars1,pars2)
#    distlist2=np.array([x[-1] for x in distlist])
    if (np.abs(dist[0])<theta_cut) & (np.abs(dist[1])<r_cut):
        mergeQ=True
    else:
        mergeQ=False
    return mergeQ

from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix


def merge_ROIs(ROIlist,r_cut=1.5,pmuse='lat'):
#    create adjacency matrix
    adj=[]
    for ROI1 in ROIlist:
        adjrow=[]
        for ROI2 in ROIlist:
            flag=0
#            if np.abs(ROI1.pmlat-ROI2.pmlat)<1.1 and np.abs(ROI1.pmlon-ROI2.pmlon)<1.1: #TRY 3/1
            if (np.abs(ROI1.pmlat-ROI2.pmlat)<1.1 and ROI1.pmlon==ROI2.pmlon) or \
               (np.abs(ROI1.pmlon-ROI2.pmlon)<1.1 and ROI1.pmlat==ROI2.pmlat): #TRY 3/1 # April 3: exclude diagonals
                if(pmuse=='lat'):
                    flag=merge_lineparsQ([ROI1.line_theta,ROI1.line_r],[ROI2.line_theta,ROI2.line_r],\
                                        r_cut=r_cut) # we used 2.1 for lwr=10 but i find 1.5 is good for lwr=5
                else:
                    flag=merge_lineparsQ([ROI1.line_theta_pmlon,ROI1.line_r_pmlon],[ROI2.line_theta_pmlon,ROI2.line_r_pmlon],\
                                        r_cut=r_cut) # we used 2.1 for lwr=10 but i find 1.5 is good for lwr=5
                   
            adjrow.append(flag)
        adj.append(adjrow)
#    adj=[[merge_ROIsQ(x,y) for x in parlist] for y in parlist]
#    print(adj)
    newarr = csr_matrix(np.array(adj))
    nc,labels=connected_components(newarr)
    return nc,labels


#def MakeProtoclusters(ra,dec,pmlon):
def MakeProtoclusters(ra,dec,fullROIlist,r_cut=1.5,fitresults_dir='',pmuse='lat'): #Try 3/1
    ROIlist=[]
    for roi in fullROIlist:
        if roi.ra==ra and roi.dec==dec:
            ROIlist.append(roi)
    
#    print(cluster)
 #   parlist=[[x.pmlat,x.line_theta,x.line_r] for x in cluster]
    cluster2=[]
    if(len(ROIlist)>0):
#        print(parlist)
        nc,labels=merge_ROIs(ROIlist,r_cut=r_cut,pmuse=pmuse) 
#        print(labels)

        cluster2=[
            Protocluster(np.array(ROIlist)[labels==ii],pmuse=pmuse)
            for ii in range(nc)]
        
    return cluster2

class StreamCluster:
    def __init__(self, protocluster_list):
        self.pclist=protocluster_list
        self.line_sigma=[x.line_sigma_tot for x in protocluster_list]
        self.significance=np.sqrt(np.sum([x.line_sigma_tot**2 for x in protocluster_list]))
        self.nsr=np.sum([len(x.ROIlist_maxsig) for x in protocluster_list])
        self.highRstars=np.empty((0,100,14))
        self.linestars=np.empty((0,100)).astype('bool')
        self.streamstars=np.empty((0,14))
        self.ROIlist=[]
        for pc in protocluster_list:
            if pc.highRstars.shape[-1]==12:
                #pc.highRstars shape is (NROIs,100,10)
                pc.highRstars=pc.highRstars.reshape((-1,12))
                pc.highRstars=np.hstack((pc.highRstars,np.zeros((pc.highRstars.shape[0],2))))
                pc.highRstars=pc.highRstars.reshape((-1,100,14))
                        
            
            self.highRstars=np.concatenate((self.highRstars,pc.highRstars))     
            self.linestars=np.concatenate((self.linestars,pc.linestars))     
            _,streamstars=np.unique(pc.highRstars[pc.linestars][:,[0,1,2,3,4,5]],axis=0,return_index=True)
            streamstars=pc.highRstars[pc.linestars][streamstars]
            self.streamstars=np.concatenate((self.streamstars,streamstars))
            self.ROIlist+=list(pc.ROIlist)
        
        
def merge_streamsQ(protocluster1,protocluster2):
    radec1=np.array([protocluster1.ra,protocluster1.dec])
    pm1=np.array(protocluster1.pm)
    radec2=np.array([protocluster2.ra,protocluster2.dec])
    pm2=np.array(protocluster2.pm)
    
    radec_avg=np.array([mean_angles(radec1[0],radec2[0]),0.5*(radec1[1]+radec2[1])])

    output=False
    
    pmpass=False
    for i in range(len(pm1)):
        for j in range(len(pm2)):
            if np.abs(pm1[i]-pm2[j])[0]<8 and np.abs(pm1[i]-pm2[j])[1]<8:
                pmpass=True
                break
                
    if angular_distance(np.array([radec1]),np.array([radec2]))[0]<20 and pmpass:
        lparray1=np.dstack((protocluster1.line_theta,protocluster1.line_r))[0]
        lparray2=np.dstack((protocluster2.line_theta,protocluster2.line_r))[0]        
#        print(lparray1)
#        print(lparray2)
        newlparray1=np.empty((0,2))
        newlparray2=np.empty((0,2))
        for theta,r in lparray1:
            newr,newtheta=transform_linepars(r,theta,radec1,radec_avg)
#            print(thetavals2,newtheta,rvals2,newr)
            newlparray1=np.concatenate((newlparray1,[[newtheta,newr]]))
#        print(newlparray1)
        for theta,r in lparray2:
            newr,newtheta=transform_linepars(r,theta,radec2,radec_avg)
#            print(thetavals2,newtheta,rvals2,newr)
            newlparray2=np.concatenate((newlparray2,[[newtheta,newr]]))
#        print(newlparray2)
#        print(mean_linepars(newlparray1),mean_linepars(newlparray2))
#        print(newlparray1,newlparray2)
        output=merge_lineparsQ(mean_linepars(newlparray1),\
                               mean_linepars(newlparray2),theta_cut=0.2,r_cut=1.5) # use 1.5 for lwr=5
    

        # Another way of deciding whether to merge two protoclusters: if enough of their line stars overlap, we say they are the same
        # Based on tests with Gaia3/Ylgr and Jhelum where the above criteria were not merging protoclusters that both corresponded to
        # the same known stream, we set this threshold at 0.4
        noverlap=0
        for x in np.unique(protocluster1.highRstars[protocluster1.linestars][:,[0,1,2,3,4,5]],axis=0):
            if x in protocluster2.highRstars[protocluster2.linestars][:,[0,1,2,3,4,5]]:
                noverlap+=1
        frac1=noverlap/len(np.unique(protocluster1.highRstars[protocluster1.linestars],axis=0))

        noverlap=0
        for x in np.unique(protocluster2.highRstars[protocluster2.linestars][:,[0,1,2,3,4,5]],axis=0):
            if x in protocluster1.highRstars[protocluster1.linestars][:,[0,1,2,3,4,5]]:
                noverlap+=1
        frac2=noverlap/len(np.unique(protocluster2.highRstars[protocluster2.linestars],axis=0))


        output2=(frac1>0.3) or (frac2>0.3)
    
#        output=output or output2
    
    return output


def merge_streams(protocluster_list):
#    create adjacency matrix
    t0=time.time()
    print("Calculating adjacency matrix...")
    niter=0
    Ntot=len(protocluster_list)*len(protocluster_list)
    adj=[]
    for x in protocluster_list:
        adj_row=[]
        for y in protocluster_list:
            niter+=1
            if(niter%1000==0):
                t1=time.time()
                print(niter,Ntot,t1-t0)
                t0=t1        
            adj_row.append(merge_streamsQ(x,y))
        adj.append(adj_row)
    print("Finished calculating adjacency matrix")
    newarr = csr_matrix(np.array(adj))
    print("Calculating connected components...")
    nc,labels=connected_components(newarr)
    return nc,labels


def MakeStreams(protocluster_list):
    nc,labels=merge_streams(protocluster_list)
    clustertemp=np.array(protocluster_list)
    cluster2=[StreamCluster(clustertemp[labels==ii]) for ii in range(nc)]
        
    return cluster2

def merge_streamsQ_pre(protocluster1,protocluster2,pmthresh=1.5,distthres=20):
    radec1=np.array([protocluster1.ra,protocluster1.dec])
#    pm1=np.array(protocluster1.pm)
    radec2=np.array([protocluster2.ra,protocluster2.dec])
#    pm2=np.array(protocluster2.pm)
    
    radec_avg=np.array([mean_angles(radec1[0],radec2[0]),0.5*(radec1[1]+radec2[1])])

    output=False
#    print(angular_distance(np.array([radec1]),np.array([radec2]))[0])
    if angular_distance(np.array([radec1]),np.array([radec2]))[0]<distthres:
    
        pc1_linestars=np.unique(protocluster1.highRstars[protocluster1.linestars][:,[0,1,2,3,4,5]],axis=0)
        pc2_linestars=np.unique(protocluster2.highRstars[protocluster2.linestars][:,[0,1,2,3,4,5]],axis=0)
#        pmpass=False
#        for i in range(len(pm1)):
#            for j in range(len(pm2)):
#                if np.abs(pm1[i]-pm2[j])[0]<8 and np.abs(pm1[i]-pm2[j])[1]<8:
#                    pmpass=True
#                    break
#        pc1_pmavg=np.mean(pc1_linestars[:,[0,1]],axis=0)
#        pc2_pmavg=np.mean(pc2_linestars[:,[0,1]],axis=0)
#        print(np.sqrt(np.sum((pc1_pmavg-pc2_pmavg)**2))<6)
#        if np.sqrt(np.sum((pc1_pmavg-pc2_pmavg)**2))<6:
        if cluster_cluster_distance2(pc1_linestars[:,[0,1]],pc2_linestars[:,[0,1]])<pmthresh:
            output=True
    return output

def pc_pc_distance_try(pc1,pc2):
    
    _,pc1stars=np.unique(pc1.highRstars[pc1.linestars][:,[0,1,2,3,4,5]],axis=0,return_index=True)
    _,pc2stars=np.unique(pc2.highRstars[pc2.linestars][:,[0,1,2,3,4,5]],axis=0,return_index=True)
    pc1stars=pc1.highRstars[pc1.linestars][pc1stars]
    pc2stars=pc2.highRstars[pc2.linestars][pc2stars]
    
    if pc1.ra!=pc2.ra or pc1.dec!=pc2.dec:
        radec_avg=np.array([mean_angles(pc1.ra,pc2.ra),0.5*(pc1.dec+pc2.dec)])

        center = ICRS(ra=pc1.ra*u.deg, dec=pc1.dec*u.deg)
        coordinates_list=SkyOffsetFrame(origin=center,lon=pc1stars[:,6]*u.deg,lat=pc1stars[:,7]*u.deg) # careful: my line finder uses lon as x and lat as y   
        ra_line=coordinates_list.transform_to(ICRS).ra.value
        dec_line=coordinates_list.transform_to(ICRS).dec.value
        center = ICRS(ra=radec_avg[0]*u.deg, dec=radec_avg[1]*u.deg)
        coordinates_list= ICRS(ra=ra_line*u.deg, dec=dec_line*u.deg)
        coordinates_list = coordinates_list.transform_to(SkyOffsetFrame(origin=center))
        x_line1=coordinates_list.lon.value
        y_line1=coordinates_list.lat.value


        center = ICRS(ra=pc2.ra*u.deg, dec=pc2.dec*u.deg)
        coordinates_list=SkyOffsetFrame(origin=center,lon=pc2stars[:,6]*u.deg,lat=pc2stars[:,7]*u.deg) # careful: my line finder uses lon as x and lat as y   
        ra_line=coordinates_list.transform_to(ICRS).ra.value
        dec_line=coordinates_list.transform_to(ICRS).dec.value
        center = ICRS(ra=radec_avg[0]*u.deg, dec=radec_avg[1]*u.deg)
        coordinates_list= ICRS(ra=ra_line*u.deg, dec=dec_line*u.deg)
        coordinates_list = coordinates_list.transform_to(SkyOffsetFrame(origin=center))
        x_line2=coordinates_list.lon.value
        y_line2=coordinates_list.lat.value
    else:
        x_line1=pc1stars[:,6]
        y_line1=pc1stars[:,7]
        x_line2=pc2stars[:,6]
        y_line2=pc2stars[:,7]

    numer,theta12,r12=linefit_chi2(np.concatenate((x_line1,x_line2)),np.concatenate((y_line1,y_line2)))
    denom1,theta1,r1=linefit_chi2(x_line1,y_line1)
    denom2,theta2,r2=linefit_chi2(x_line2,y_line2)
    numer1,_,_=linefit_chi2(x_line1,y_line1,thetause=theta12,ruse=r12)
    numer2,_,_=linefit_chi2(x_line2,y_line2,thetause=theta12,ruse=r12)

#    deltaLL=linefit_LL(np.concatenate((x_line1,x_line2)),np.concatenate((y_line1,y_line2)))\
#        -linefit_LL(x_line1,y_line1)-linefit_LL(x_line2,y_line2)
    
#    return numer,numer1,numer2,denom1,denom2,len(x_line1),len(x_line2)  #numer/(denom1+denom2),numer1/denom1,numer2/denom2,deltaLL
    return numer1/denom1,numer2/denom2

def calculate_pc_pc_distance_matrix(pclist,mode='avg',pmthresh=3):
    dist_list=[]
    niter=0
    print('Calculating distances...')
    for ii in range(len(pclist)):
        for jj in range(ii):
            niter+=1
            if(niter%1000==0):
                print(niter,len(pclist)*(len(pclist)-1)//2)
            pc1=pclist[ii]
            pc2=pclist[jj]
            dist=9999999999
            if merge_streamsQ_pre(pc1,pc2,pmthresh=pmthresh):
                dist1,dist2=pc_pc_distance_try(pc1,pc2)
                if mode=='avg':
                    dist=0.5*(dist1+dist2)
                elif mode=='min':
                    dist=min(dist1,dist2)
                elif mode=='max':
                    dist=max(dist1,dist2)    
    
            dist_list.append([ii,jj,dist])
        
    dist_mat=np.zeros((len(pclist),len(pclist)))
    for kk in range(len(dist_list)):
        dist_mat[dist_list[kk][0],dist_list[kk][1]]=dist_list[kk][2]
        dist_mat[dist_list[kk][1],dist_list[kk][0]]=dist_list[kk][2]
    dist_mat=np.array(dist_mat)
    return dist_mat

def calculate_pc_pc_distance_row(pc0,pclist,mode='avg',pmthresh=3):
    dist_row=[]
    niter=0
    for ii in range(len(pclist)):
        niter+=1
        if(niter%100==0):
            print(niter,len(pclist))
        pc1=pclist[ii]
        dist=9999999999
        if merge_streamsQ_pre(pc0,pc1,pmthresh=pmthresh):
            dist1,dist2=pc_pc_distance_try(pc0,pc1)
            if mode=='avg':
                dist=0.5*(dist1+dist2)
            elif mode=='min':
                dist=min(dist1,dist2)
            elif mode=='max':
                dist=max(dist1,dist2)    
    
        dist_row.append(dist)
    dist_row=np.array(dist_row)
    return dist_row

def cluster_one_step(pclist,distmat,mode='avg',thresh=1.65,pmthresh=3):
    lslist=np.array([pc.line_sigma_tot for pc in pclist])
    lslist_indsort=np.argsort(lslist)[::-1]
    mindist=99999999
    for ii in range(len(lslist)):
        for jj in range(ii+1,len(lslist)):
            iind=lslist_indsort[ii]
            jind=lslist_indsort[jj]
            if distmat[iind,jind]<thresh:
                iikeep=iind
                jjkeep=jind
                mindist=distmat[iind,jind]
                break
        if mindist<thresh:
            break
    
    pclist3=pclist
    distmat3=distmat
    if mindist<thresh:    
        ii=iikeep
        jj=jjkeep
        roilist1=pclist[ii].ROIlist
        roilist2=pclist[jj].ROIlist
        pclist2=np.delete(pclist,[ii,jj])
        pcnew=Protocluster(np.concatenate((roilist1,roilist2)))

        # delete the old ii and jj elements from the distance matrix, replace with new one
        distmat2=np.delete(distmat,[ii,jj],axis=0)
        distmat2=np.delete(distmat2,[ii,jj],axis=1)
        distrow=calculate_pc_pc_distance_row(pcnew,pclist2,mode=mode,pmthresh=pmthresh)
        distmat3=np.zeros((len(pclist2)+1,len(pclist2)+1))
        distmat3[:-1,:-1]=distmat2
        distmat3[:-1,-1]=distrow
        distmat3[-1,:-1]=distrow
        pclist3=np.concatenate((pclist2,[pcnew]))

        
    return pclist3,distmat3

def cluster_rois(roilist,verbose=False,thresh=1.65,pmthresh=3,mode='avg'):
    pcliststart=[Protocluster([roi]) for roi in roilist]
    pclistnew=[]
    distmatstart=calculate_pc_pc_distance_matrix(pcliststart,mode=mode,pmthresh=pmthresh)
    niter=0
    while list(pclistnew)!=list(pcliststart):
        if verbose:
            print(niter,len(pcliststart),len(pclistnew))
        niter+=1
        pclistnew=pcliststart
        distmatnew=distmatstart
        pcliststart,distmatstart=cluster_one_step(pclistnew,distmatnew,mode=mode,thresh=thresh,pmthresh=pmthresh)

    pclistnew=np.array(pclistnew)
    pclistfinal=pclistnew[np.argsort([x.line_sigma_tot for x in pclistnew])[::-1]]
    
    return pclistfinal
