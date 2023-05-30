# this requires three inputs
# obj_dict[i]=obj is a dictionary of objects whose keys are integers i=1,2,3,4,... indexing the objects
# neighbor_dict[i]=[j1,j2,j3,...] is a dictionary of neighbors for object i 
# dist_dict[i,j]=d(i,j) is a dictionary of distances between object i and j
#
# when object i and j are clustered together, their list of neighbors and distances to them are recalculated

import numpy as np
from utils import cluster_cluster_distance2
from line_finder_v2 import line_finder_fast


def check_neighbors(pc1,pc2,pmdistcut=1,require_independent=True):
    
    neighbors=False
    
    # require roilist1 and roilist2 to be fully pairwise independent
    independent=True
    if require_independent:
        for roi1 in pc1.ROIlist:
            for roi2 in pc2.ROIlist:
                if not ROI_independence(roi1,roi2):
                    independent=False
    
    if independent:
        highRstars1=pc1.highRstars
        highRstars2=pc2.highRstars
        pmdist=cluster_cluster_distance2(highRstars1[:,[0,1]],highRstars2[:,[0,1]])
        if pmdist<pmdistcut:
            neighbors=True
            
    return neighbors
            
            
def distance_func(pc1,pc2,pmdistcut=1):
    dist=999999.
    if pc1.lwr!=pc2.lwr or pc1.lwtheta!=pc2.lwtheta or pc1.bgwindow!=pc2.bgwindow:
        print('Error! pc lw pars dont agree')
        
    lwr=pc1.lwr
    lwtheta=pc1.lwtheta
    bgwindow=pc1.bgwindow
    if check_neighbors(pc1,pc2,pmdistcut=pmdistcut):

        pc12=ProtoclusterV2(pc1.ROIlist+pc2.ROIlist,lwr=lwr,lwtheta=lwtheta,bgwindow=bgwindow)
            
        if pc12.line_sigma>pc1.line_sigma and pc12.line_sigma>pc2.line_sigma:
            dist=1/pc12.line_sigma
    
    return dist

def ROI_independence(roi1,roi2):
    independent=False
    if (roi1.pmuse!=roi2.pmuse) or (roi1.pmuse=='lat' and roi1.pmlat!=roi2.pmlat) or\
            (roi1.pmuse=='lon' and roi1.pmlon!=roi2.pmlon):
        independent=True
    return independent

#class Protocluster_NEW:
#    def __init__(self, ROIlist):
#        
#        
#        self.ROIlist=ROIlist
#        self.highRstars=np.array([roi.highRstars for roi in ROIlist])
#
#        
#        if len(ROIlist)==1:
#            self.line_sigma=ROIlist[0].line_sigma
#            self.linestars=ROIlist[0].linestars
#            self.line_ind=[ROIlist[0].fitresults[0],ROIlist[0].fitresults[1]]
#        else:
#            maxind,_,_,linesigmatemp,linestarstemp,_=line_finder_fast(\
#                                            self.highRstars.reshape((-1,self.highRstars.shape[-1])),lwr=5)
#
#            self.line_sigma=linesigmatemp
#            self.linestars=linestarstemp
#            self.line_ind=maxind


class ProtoclusterV2:
    def __init__(self, ROIlist,line_sigma=None,line_stars=[],line_ind=None,lwr=5,lwtheta=3,bgwindow=7):
        
        
        self.ROIlist=ROIlist
        self.highRstars=np.array([roi.highRstars for roi in ROIlist])
        self.highRstars=self.highRstars.reshape((-1,self.highRstars.shape[-1]))
        self.ra=ROIlist[0].ra
        self.dec=ROIlist[0].dec
        self.lwr=lwr
        self.lwtheta=lwtheta
        self.bgwindow=bgwindow
        
        if len(ROIlist)==1:
            self.line_sigma=ROIlist[0].line_sigma
            self.line_stars=ROIlist[0].linestars
            self.line_ind=[ROIlist[0].fitresults[0],ROIlist[0].fitresults[1]]
        elif line_sigma!=None and len(line_stars)>0 and line_ind!=None:
            self.line_sigma=line_sigma
            self.line_stars=line_stars
            self.line_ind=line_ind
        else:
            # This step is time consuming so we allow the user to supply the line finding results externally
            maxind,_,_,linesigmatemp,linestarstemp,_=line_finder_fast(self.highRstars,lwr=self.lwr,lwtheta=self.lwtheta,bgwindow=self.bgwindow)

            self.line_sigma=linesigmatemp
            self.line_stars=linestarstemp
            self.line_ind=maxind

class ProtoclusteringObject:
    def __init__(self,obj_dict,neighbor_dict,dist_dict): 
        self.obj_dict=obj_dict.copy()
        self.neighbor_dict=neighbor_dict.copy()
        self.dist_dict=dist_dict.copy()          
            
        self.index_list=self.obj_dict.keys()
        
    def nobj(self):
        return len(self.obj_dict.keys())
        
    def delete_index(self,ind):
        del self.obj_dict[ind]

        keystodelete=[]
        for key in self.neighbor_dict.keys():
            if ind in self.neighbor_dict[key]:
                self.neighbor_dict[key].remove(ind)            
        del self.neighbor_dict[ind]                
        
        keystodelete=[]
        for key in self.dist_dict.keys():
            if ind in key:
                keystodelete.append(key)
        for key in keystodelete:
            del self.dist_dict[key]
            
    def add_index(self,ind,obj_list,neighbor_list,dist_list):
        self.obj_dict[ind]=obj_list
        self.neighbor_dict[ind]=neighbor_list
        for i in neighbor_list:
            self.neighbor_dict[i].append(ind)
        for i,jj in enumerate(neighbor_list):
            self.dist_dict[ind,jj]=dist_list[i]
            self.dist_dict[jj,ind]=dist_list[i]
        
    def merge(self,i,j,dfunc):
        newind=max(list(self.obj_dict.keys()))+1
        lwr=self.obj_dict[i].lwr
        lwtheta=self.obj_dict[i].lwtheta
        bgwindow=self.obj_dict[i].bgwindow
        new_obj_list=ProtoclusterV2(self.obj_dict[i].ROIlist+self.obj_dict[j].ROIlist,lwr=lwr,lwtheta=lwtheta,bgwindow=bgwindow)
        new_neighbor_list=list(set(self.neighbor_dict[i]+self.neighbor_dict[j]))
        new_neighbor_list.remove(i)
        new_neighbor_list.remove(j)
        new_dist_list=[dfunc(new_obj_list,self.obj_dict[aa]) for aa in new_neighbor_list]
        
        self.delete_index(i)
        self.delete_index(j)
        self.add_index(newind,new_obj_list,new_neighbor_list,new_dist_list)
     
        return newind,new_obj_list,new_neighbor_list,new_dist_list
    
    def min_distance(self):
        distvals=np.array(list(self.dist_dict.values()))
        mindist=9999999
        indout=[9999999,9999999]
        if len(distvals)>0:
            mindist=np.min(distvals)
            indout=np.unique([sorted(key) for key in list(self.dist_dict.keys()) if self.dist_dict[key]==mindist],axis=0)[0]
            
        return mindist,indout    
    
    
class Protostream:
    def __init__(self, pclist):
        self.pclist=pclist
        
        self.ra=self.pclist[0].ra
        self.dec=self.pclist[0].dec
        self.line_sigma=self.pclist[0].line_sigma
            
