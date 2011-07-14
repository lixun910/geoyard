import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import math

import plfit


def plot_pl_roi(dataset,alpha_set,xmin_set,error_set):        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('truncated power law of regions-of-interest')
    ax.set_xlabel('x (noramlized popularity)')
    ax.set_ylabel('P(x)')
   
    count = 0
    colors = ['k','g','r','b']
    marks = ['s','^','o','d']
    for i,data in enumerate(dataset):
        if i==4 or i==8 or i==12 or i==16:
            alpha = alpha_set[i]
            xmin = xmin_set[i]
            x = np.sort(data)
            n = len(x)
            xcdf = np.arange(n, 0, -1, dtype='float') / float(n)
            q = x[x>=xmin]
            fcdf = (q/xmin)**(1-alpha)
            nc = xcdf[np.argmax(x>=xmin)]
            
            plotx = np.linspace(q.min(),q.max(),1000)
            ploty = (plotx/xmin)**(1-alpha) * nc
            prefactor = rate / ((xmin*rate)**exponent )
            C = prefactor / 
            
            ax.loglog(x,xcdf,ls='None', alpha='0.7',marker=marks[count],mec=colors[count],mfc='#cccccc',mew='1')
            #if i== 15:
            ax.loglog(plotx,ploty,ls='--',lw=2, alpha='1.0', c=colors[count])
            count += 1
        
    leg = plt.legend(('n=%d, level=4'% len(dataset[4]),r'$\alpha$=%.4f +/- %.4f'%(alpha_set[4],error_set[4]),
                      'n=%d, level=8'% len(dataset[8]),r'$\alpha$=%.4f +/- %.4f'%(alpha_set[8],error_set[8]),
                      'n=%d, level=12'% len(dataset[12]),r'$\alpha$=%.4f +/- %.4f'%(alpha_set[12],error_set[12]),
                      'n=%d, level=16'% len(dataset[16]),r'$\alpha$=%.4f +/- %.4f'%(alpha_set[16],error_set[16]),
                      ),
                     loc='lower left',shadow=True)
    frame  = leg.get_frame()  
    #frame.set_facecolor('0.80')    # set the frame face color to light gray
    for t in leg.get_texts():
        t.set_fontsize('small')    # the legend text fontsize
    for l in leg.get_lines():
        l.set_linewidth(1.5)  # the legend line width
        
    plt.show()
        
def plot_alpha(alpha_set,xmin_set,error_set):   
    alpha_set = alpha_set[:16]
    error_set = error_set[:16]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('power law scales at different geographical scales')
    ax.set_xlabel('hierachical level')
    ax.set_ylabel(r'$\alpha$')
    ax.set_xlim(0,18)
    ax.grid(True)
   
    n = len(alpha_set)
    x = np.arange(1,n+1,1)
    ax.plot(x, alpha_set,'o--')
    ax.errorbar(x, alpha_set,yerr=error_set, fmt='k.',label='_nolegend_')
    
    # linear regression 
    (ar,br,r,tt,stderr) = stats.linregress(x,alpha_set)
    yr=np.polyval([ar,br],x)
    
    ax.plot(x,yr,'r-',label='_nolegend_')
         
    ax.text(4,1.9,r'regression parameters: $\alpha$=%.2f $\beta$=%.2f std error= %.3f'%(ar,br,stderr), {'color':'r'})
    plt.show()
    
def plot_pl_distance(dataset,alpha_set,xmin_set,error_set):        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('power law of travel distance')
    ax.set_xlabel('x (noramlized distance)')
    ax.set_ylabel('P(x)')
   
    count = 0
    colors = ['k','g','r','b']
    marks = ['s','^','o','d']
    line_colors = ['y','#999999','b']
    
    plotx_set = []
    ploty_set = []
    for i,data in enumerate(dataset):
        alpha = alpha_set[i]
        xmin = xmin_set[i]
        x = np.sort(data)
        n = len(x)
        xcdf = np.arange(n, 0, -1, dtype='float') / float(n)
        q = x[x>=xmin]
        fcdf = (q/xmin)**(1-alpha)
        nc = xcdf[np.argmax(x>=xmin)]
        
        plotx = np.linspace(q.min(),q.max(),1000)
        ploty = (plotx/xmin)**(1-alpha) * nc
        
        plotx_set.append(plotx)
        ploty_set.append(ploty)
        
        ax.loglog(x,xcdf,ls='None', alpha='0.6',marker=marks[count],mec=colors[count],mfc='#cccccc',mew='1')
        count += 1
       
    for i in range(len(dataset)):
        ax.loglog(plotx_set[i],ploty_set[i],ls='--',lw=2, alpha='1.0', c=line_colors[i])
        
    leg = plt.legend(('1-day travel, n=%d'% len(dataset[0]),
                      '2-day travel, n=%d'% len(dataset[1]),
                      '3-day travel, n=%d'% len(dataset[2]),
                      r'$\alpha$=%.4f +/- %.4f'%(alpha_set[0],error_set[0]),
                      r'$\alpha$=%.4f +/- %.4f'%(alpha_set[1],error_set[1]),
                      r'$\alpha$=%.4f +/- %.4f'%(alpha_set[2],error_set[2]),
                      ),
                     loc='lower left',shadow=True)
    frame  = leg.get_frame()  
    #frame.set_facecolor('0.80')    # set the frame face color to light gray
    for t in leg.get_texts():
        t.set_fontsize('small')    # the legend text fontsize
    for l in leg.get_lines():
        l.set_linewidth(1.5)  # the legend line width
       
	plt.save 
    plt.show()
        
def get_roi_pop_dataset():
    dataset = []
    f = open('output.txt')
    line = f.readline()
    while len(line) > 0:
        data = line.strip()[line.index(',')+1:]
        data = data[1:-1].split(',')
        data = [float(j) for j in data]
        data = np.sort(data)
        data = data/np.linalg.norm(data)
        dataset.append(data)
        line = f.readline()
    f.close()
    return dataset

def get_distance_dataset():
    dataset = []
    f = open('travel_distance.txt')
    line = f.readline() # travel interval
    line = f.readline()
    while len(line) > 0:
        data = line.strip()
        data = data[1:-1].split(',')
        data = [float(j) for j in data]
        data = np.sort(data)
        data = data/np.linalg.norm(data)
        dataset.append(data)
        line = f.readline()
    f.close()
    return dataset
    
if __name__ == "__main__":
    alpha_set = []
    xmin_set = []
    error_set = []
    
    dataset = get_roi_pop_dataset()
    #dataset = get_distance_dataset()
    
    for i,data in enumerate(dataset):
        print i
        pl = plfit.plfit(data,quiet=False)
        alpha_set.append(pl._alpha)
        xmin_set.append(pl._xmin)
        error_set.append(pl._alphaerr)
        
    plot_pl_roi(dataset, alpha_set, xmin_set,error_set)
    #plot_alpha(alpha_set,xmin_set,error_set)
    #plot_pl_distance(dataset,alpha_set,xmin_set,error_set)
