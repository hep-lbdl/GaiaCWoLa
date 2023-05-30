import numpy as np

from astropy import units as u
from astropy.coordinates import SkyOffsetFrame, ICRS, SkyCoord, Galactic,Galactocentric
import astropy.coordinates as coords
from sklearn.linear_model import LinearRegression
from astropy.io import fits


datadir_default='/het/p4/mbuckley/density_estimation/Gaia_data/Gaia_data_redownloaded_merged/'
fitsdir_default='/het/p4/mbuckley/density_estimation/Gaia_data/Gaia_redownload/'
#ROIdir_default='/het/p4/dshih/jet_images-deep_learning/density_estimation/ConditionalMAF/ViaMachinae/data/ROIs/'

# This function returns 15 degree patch specified by "dataset", as well as mask corresponding to fiducial region
# May 11, 2022: included additional fiducial cut, removing stars with well-measured parallaxes that are definitively in the thick disk
def load_gaiadata(dataset,datadir=datadir_default,fitsdir=fitsdir_default,datatype='gaiadr2',zcut=2,zsigma=2,parallax_quality=0.2,cuttype='zcut'):
    
    if datatype=='gaiadr2' or datatype=='galaxia':

    #pmdec, pmra, dec, ra, color, mag, lon, lat, pm_lon_coslat, and pm_lat, parallax, parallax_error
        gaiadata=np.load(datadir+dataset+'.npy',allow_pickle=True).astype('float32') 


        if datatype=='gaiadr2':    
            file='gaiaredownload_'+dataset.strip('gaiascan_')
            hdul = fits.open(fitsdir+file+'.fits')
            sourceids=hdul[1].data['source_id']
        else:
            sourceids=np.zeros(len(gaiadata))
            
            
        nanmask=np.sum(np.isnan(gaiadata),axis=1)==0

        gaiadata = gaiadata[nanmask]
        sourceids=sourceids[nanmask]

        # Just use radius 15 circle
        center_dec=0.5*(np.max(gaiadata[:,6])+np.min(gaiadata[:,6]))
        center_ra=0.5*(np.max(gaiadata[:,7])+np.min(gaiadata[:,7]))
        radius=np.sqrt((gaiadata[:,6]-center_dec)**2+(gaiadata[:,7]-center_ra)**2)

        gaiadata=gaiadata[radius<15]
        sourceids=sourceids[radius<15]

        radius=radius[radius<15]
        magnitude=gaiadata[radius<15][:,5]
        
    elif datatype=='gaiaedr3':
        with fits.open(datadir+dataset) as hdul:
            lat=hdul[1].data['lat']
            lon=hdul[1].data['lon']
            pmlat=hdul[1].data['pmlat']
            pmlon=hdul[1].data['pmlon']
            color=hdul[1].data['bp_rp']
            mag=hdul[1].data['phot_g_mean_mag']
            ra=hdul[1].data['ra']
            dec=hdul[1].data['dec']
            pmra=hdul[1].data['pmra']
            pmdec=hdul[1].data['pmdec']
            parallax=hdul[1].data['parallax']
            parallax_error=hdul[1].data['parallax_error']
            sourceids=hdul[1].data['source_id']
            
            ruwecut=hdul[1].data['ruwe']<1.4
    #pmdec, pmra, dec, ra, color, mag, lon, lat, pm_lon_coslat, and pm_lat,parallax,parallax_error

            gaiadata=np.dstack((pmdec,pmra,dec,ra,color,mag,lon,lat,pmlon,pmlat,parallax,parallax_error))[0][ruwecut]
            nanmask=np.sum(np.isnan(gaiadata),axis=1)==0
            gaiadata = gaiadata[nanmask].astype('float32')
            
            sourceids=sourceids[ruwecut]
            sourceids=sourceids[nanmask]
        
            radius=np.sqrt((gaiadata[:,6])**2+(gaiadata[:,7])**2)
            magnitude=gaiadata[:,5]
        
    else:
        print('Error in load_gaiadata! Data type not recognized')

        
    # check shape:
    if datatype=='gaiadr2' or datatype=='gaiaedr3':
        if gaiadata.shape[-1]!=12:
            raise ValueError("Incorrect data format")
    else:
        if gaiadata.shape[-1]!=14:
            raise ValueError("Incorrect data format")
            

    
    
    pmzero_mask= (np.abs(gaiadata[:,8])>2) | (np.abs(gaiadata[:,9])>2)    
    colormask=(gaiadata[:,4]>0.5) & (gaiadata[:,4]<1) 

    fidmask_new=True
    if cuttype=='diagcut':
        # this is the diagonal cut in color/magnitude we were using to eliminate disk stars previously
        fidmask_new=gaiadata[:,5]>8*(gaiadata[:,4]-0.5)+14.5   # This is magcut2
    elif cuttype=='zcut':
# bug fixed Jan 24 2023: datatype=='gaiadr2' or 'gaiaedr3' previously
        if datatype=='gaiadr2' or datatype=='gaiaedr3':   
            parallax_quality_cut=(gaiadata[:,-2]>0) & (gaiadata[:,-1]/(np.abs(gaiadata[:,-2]))<parallax_quality)
            distance=np.where(gaiadata[:,-2]<0,1000,1/(gaiadata[:,-2]+zsigma*np.abs(gaiadata[:,-1])))
        else:
            parallax_quality_cut=(gaiadata[:,-3]>0) & (gaiadata[:,-1]/(np.abs(gaiadata[:,-3]))<parallax_quality)
            distance=np.where(gaiadata[:,-3]<0,1000,1/(gaiadata[:,-3]+zsigma*np.abs(gaiadata[:,-1])))

        c1_g = SkyCoord(ra=gaiadata[:,3]*u.degree, \
                      dec=gaiadata[:,2]*u.degree,
                        distance=distance*u.kpc,
                            pm_ra_cosdec=gaiadata[:,1]*u.mas/u.yr,
                            pm_dec=gaiadata[:,0]*u.mas/u.yr,
                            radial_velocity=np.zeros(len(gaiadata))*u.km/u.s,
                            frame='icrs')    

    # numbers for LSR taken from Galaxia paper
        c2_g=c1_g.transform_to(Galactocentric(galcen_distance=8.00*u.kpc,\
                                             galcen_v_sun=(11.1, 239.08, 7.25)*u.km/u.s,\
                                              z_sun=0.015*u.kpc))
    # bug fixed 23 June 2022 -- added the np.abs(...) here
        zcut=np.abs(c2_g.z.value)>zcut  

        fidmask_new=(~parallax_quality_cut | (parallax_quality_cut & zcut))    
    
    fidmask=(radius<10) & (magnitude<20.2) & pmzero_mask & colormask & fidmask_new
    
    if len(sourceids)!=len(gaiadata):
        print('Error sourceids dont match')
        
    return gaiadata,fidmask,sourceids


def clean_ROIs(ROIlist):
    slopelist=[]
    for roi in ROIlist:
        if np.abs(roi.line_r)>9.5:
#            print('Edge case found ',roi.ra,roi.dec,roi.pmlat,roi.pmlon)
            stream_coord = SkyCoord(ra=roi.highRstars[roi.linestars][:,3]*u.deg, \
                                dec=roi.highRstars[roi.linestars][:,2]*u.deg, frame='icrs')
            ltry=stream_coord.galactic.l.value
            btry=stream_coord.galactic.b.value
            lineangle=-10000
            meanb=-10000
            meanl=-10000
            lineloc=-10000
            if len(ltry)>5:
                if(np.max(ltry)-np.min(ltry)>100):
#                    print('before ',ltry)
                    ltry=anglewrap(ltry,180)
#                    print('after ',ltry)
                reg = LinearRegression().fit(ltry.reshape((-1,1)), btry)
                slope=reg.coef_[0]
                meanb=np.mean(btry)
                meanl=np.mean(ltry)
                lineloc=np.arctan2(np.sign(roi.b)*(meanb-roi.b),\
                   deltaphi(np.array([meanl]),np.array([roi.l]))[0])
                lineangle=np.sign(np.abs(roi.b)-np.abs(meanb))*np.abs(np.arctan(slope))
                
#                print('slope ',slope)
            slopelist.append([roi,roi.line_sigma,roi.l,roi.b,roi.pmlat,roi.pmlon,lineangle,meanb,meanl,lineloc])
    return slopelist

def anglewrap(anglearray,angle=180):
    return np.where(anglearray>angle,anglearray-360,anglearray)

def deltaphi(angle1,angle2):
    ind1=np.argmin(np.dstack((np.abs(angle1-angle2+360),np.abs(angle1-angle2),\
                          np.abs(angle1-angle2-360)))[0],axis=1)
    delta=np.dstack((angle1-angle2+360,angle1-angle2,\
                          angle1-angle2-360))[0][range(len(ind1)),ind1]
    return delta

def angular_distance(angle1,angle2):
    # inputs are np arrays of [ra,dec]
    deltara=np.minimum(np.minimum(np.abs(angle1[:,0]-angle2[:,0]+360),np.abs(angle1[:,0]-angle2[:,0])),\
                          np.abs(angle1[:,0]-angle2[:,0]-360))
    deltadec=np.abs(angle1[:,1]-angle2[:,1])
    return np.sqrt(deltara**2+deltadec**2)


def recenter_coords(x,y,vx,vy,newcenter):
    center = ICRS(ra=newcenter[0]*u.deg, dec=newcenter[1]*u.deg)

    coordinates_list= ICRS(ra=x*u.deg, dec=y*u.deg,\
                           pm_ra_cosdec=vx*u.mas/u.yr, pm_dec=vy*u.mas/u.yr)
    coordinates_list = coordinates_list.transform_to(SkyOffsetFrame(origin=center))
    newx=coordinates_list.lat.value
    newy=coordinates_list.lon.wrap_at('180d').value
    newvx=coordinates_list.pm_lat.value
    newvy=coordinates_list.pm_lon_coslat.value

    return newx,newy,newvx,newvy

##################################################
#
# helper functions for line finding




def houghpar_distance(pars1,pars2):
    #pars [theta,r]
    pars2_image1=np.array([pars2[0]+np.pi,-pars2[1]])
    pars2_image2=np.array([pars2[0]-np.pi,-pars2[1]])
    delta=np.array([np.array(pars1)-pars2,np.array(pars1)-pars2_image1,np.array(pars1)-pars2_image2])
    return delta[np.argmin(np.abs(delta[:,0]))]


            
def transform_linepars(r,theta,oldcenter,newcenter):
# use the two points where the line intersects the circle of radius 15 (using radius 10 leads to nans when r=10)
    Rmax=15
    x0=-np.sqrt(Rmax**2-r**2)*np.cos(theta)+r*np.sin(theta)
    y0=-np.sqrt(Rmax**2-r**2)*np.sin(theta)-r*np.cos(theta)
    
    x1=np.sqrt(Rmax**2-r**2)*np.cos(theta)+r*np.sin(theta)
    y1=np.sqrt(Rmax**2-r**2)*np.sin(theta)-r*np.cos(theta)
    
    center = ICRS(ra=oldcenter[0]*u.deg, dec=oldcenter[1]*u.deg)
    coordinates_list=SkyOffsetFrame(origin=center,lon=[x0,x1]*u.deg,lat=[y0,y1]*u.deg) # careful: my line finder uses lon as x and lat as y
    
    
    ra_line=coordinates_list.transform_to(ICRS).ra.value
    dec_line=coordinates_list.transform_to(ICRS).dec.value

    center = ICRS(ra=newcenter[0]*u.deg, dec=newcenter[1]*u.deg)
    coordinates_list= ICRS(ra=ra_line*u.deg, dec=dec_line*u.deg)
    coordinates_list = coordinates_list.transform_to(SkyOffsetFrame(origin=center))

    x_line=coordinates_list.lon.wrap_at('180d').value
    y_line=coordinates_list.lat.value

    x0=x_line[0]
    x1=x_line[1]
    y0=y_line[0]
    y1=y_line[1]
    
    newr=(x0*y1-x1*y0)/np.sqrt((x0-x1)**2+(y0-y1)**2+1e-10)
    newtheta=np.arctan2(\
                        (y1-y0)/np.sqrt((x0-x1)**2+(y0-y1)**2+1e-10),\
                       (x1-x0)/np.sqrt((x0-x1)**2+(y0-y1)**2+1e-10)\
                       )
    if newtheta<0:
        newr=-(x0*y1-x1*y0)/np.sqrt((x0-x1)**2+(y0-y1)**2+1e-10)
        newtheta=np.arctan2(\
                        -(y1-y0)/np.sqrt((x0-x1)**2+(y0-y1)**2+1e-10),\
                       -(x1-x0)/np.sqrt((x0-x1)**2+(y0-y1)**2+1e-10)\
                       )
        
    
    return newr,newtheta


def mean_linepars(lparray):
    # theta,r
    if len(lparray)==1:
        output=lparray[0]
    else:
        seedlp=lparray[0]
#        print(seedlp)
        lparray_imaged=seedlp.reshape((1,2))
        for lp in lparray[1:]:
            lpimages=np.array([[lp[0],lp[1]],[lp[0]-np.pi,-lp[1]],[lp[0]+np.pi,-lp[1]]])
#            print(lpimages)
            ii=np.argmin(np.abs((lpimages-seedlp)[:,0]))
#            print(ii)
            lparray_imaged=np.concatenate((lparray_imaged,[lpimages[ii]]))
#        print(lparray_imaged)
        output=np.mean(lparray_imaged,axis=0)
    return output

def mean_angles(ra1,ra2):
    ra2_images=np.array([ra2,ra2-360,ra2+360])
    ra2_use=ra2_images[np.argmin(np.abs(ra2_images-ra1))]
    return 0.5*(ra1+ra2_use)

def point_cluster_distance(point,cluster):
    return np.min(np.sum((cluster-point)**2,axis=1))
def cluster_cluster_distance(cluster1,cluster2):
    dlist1=[point_cluster_distance(point,cluster2) for point in cluster1]
    dlist2=[point_cluster_distance(point,cluster1) for point in cluster2]
#    print(dlist1,dlist2)
    d1=np.mean(np.sqrt(dlist1))
    d2=np.mean(np.sqrt(dlist2))
    return min(d1,d2)


def cluster_cluster_distance2(cluster1,cluster2):
    dist=np.sum((np.mean(cluster1,axis=0)-np.mean(cluster2,axis=0))**2/(np.std(cluster1,axis=0)**2+np.std(cluster2,axis=0)**2))
    return dist

def point_line_distance(pointsx,pointsy,theta,r):
    return np.abs(pointsx*np.sin(theta)-pointsy*np.cos(theta)-r)


def line_line_distance(x1,y1,theta1,r1,x2,y2,theta2,r2):
    return min(np.mean(point_line_distance(x1,y1,theta2,r2)),\
                  np.mean(point_line_distance(x2,y2,theta1,r1)))

def line_line_distance_try(x_line1,y_line1,x_line2,y_line2):
    
    numer,theta12,r12=linefit_chi2(np.concatenate((x_line1,x_line2)),np.concatenate((y_line1,y_line2)))
    denom1,theta1,r1=linefit_chi2(x_line1,y_line1)
    denom2,theta2,r2=linefit_chi2(x_line2,y_line2)
    numer1,_,_=linefit_chi2(x_line1,y_line1,thetause=theta12,ruse=r12)
    numer2,_,_=linefit_chi2(x_line2,y_line2,thetause=theta12,ruse=r12)

#    deltaLL=linefit_LL(np.concatenate((x_line1,x_line2)),np.concatenate((y_line1,y_line2)))\
#        -linefit_LL(x_line1,y_line1)-linefit_LL(x_line2,y_line2)
    
#    return numer,numer1,numer2,denom1,denom2,len(x_line1),len(x_line2)  #numer/(denom1+denom2),numer1/denom1,numer2/denom2,deltaLL
    return numer1/denom1,numer2/denom2


def linefit_chi2(xtry,ytry,thetause=None,ruse=None):
    # this fits x,y to a line using the angle, impact parameter parametrization 
    # and returns the goodness of fit
    # r=x sin theta- y cos theta
    if thetause==None:
        thetause=0.5*np.arctan2(2*(len(xtry)*np.sum(xtry*ytry)-np.sum(xtry)*np.sum(ytry)),\
            len(xtry)*np.sum(xtry**2-ytry**2)-np.sum(xtry)**2+np.sum(ytry)**2)
    if ruse==None:
        ruse=np.mean(xtry*np.sin(thetause)-ytry*np.cos(thetause))
    
    return np.sum((xtry*np.sin(thetause)-ytry*np.cos(thetause)-ruse)**2),thetause,ruse

def linefit_2LL(xtry,ytry,thetause=None,ruse=None):
    # this includes the width in a log likelihood calculation
    # 2 LL = chi^2/s^2+2Nlog s
    # so s^2=chi^2/N
    # and 2 LL = N+Nlog(chi^2/N)
    chi2,_,_=linefit_chi2(xtry,ytry,thetause,ruse)
    
    return len(xtry)*(1+np.log(chi2/len(xtry)))
