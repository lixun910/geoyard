import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import gammainc as uigf
from scipy import stats
import math
import plfit

def plot_heavytails(fname,lvl):
    f = open(fname)
    line = f.readline()
    
    colors = ['b','b','b','b']
    marks = ['s','^','o','d']
    
    while len(line.strip()) >0:
        # real data
        data = line.strip().split(',')
        level = int(data[0])
        if level == 3:
            #count = lvl/4 -1
            count = level
            x = [float(i) for i in data[1:]]
            n = len(x)
            xcdf = np.arange(n, 0, -1, dtype='float') / float(n)
            # sample data 
            data = f.readline().strip().split(',')
            plotx = [float(i) for i in data[1:]]
            
            def get_plot_xy(plotx, data):
                if data.count('NA') > 0:
                    start_pos = len(data) - data[::-1].index('NA')
                    ploty = [float(i) for i in data[start_pos:]]
                    plotx = plotx[start_pos:]
                else:
                    ploty = [float(i) for i in data[1:]]
                return plotx,ploty
                    
            # power-law curve 
            data = f.readline().strip().split(',')
            plotx_pareto, ploty_pareto = get_plot_xy(plotx,data)
            
            # exponential  curve
            data = f.readline().strip().split(',')
            plotx_exp, ploty_exp = get_plot_xy(plotx,data)
            
            # skip orig one
            line = f.readline()
            
            # truc power-law curve
            data = f.readline().strip().split(',')
            plotx_powerexp, ploty_powerexp = get_plot_xy(plotx,data)
                        
            # others
            data = f.readline().strip().split(',')[1:]
            #n,x_min,powerlaw_exponent,powerlaw_AIC,exp_rate,AIC, powerexp_exponent,AIC,powerexp_rate,AIC 
            others = [float(i) for i in data]
            
            # loglog plot: real data/power-law/exp/trunc power-law
            fig = plt.figure()
            ax = fig.add_subplot(111)
            #ax.set_title('Long tails of Region-of-Interest popularity')
            ax.set_title('Long tails of travel distance')
            ax.set_xlabel('loglog(x)')
            ax.set_ylabel('loglog(P(x))')
            ax.set_xlim(auto=True)
            ax.set_ylim(auto=True)
            
            ax.loglog(x,xcdf,ls='None', alpha='0.5',marker=marks[count],mec=colors[count],mfc='#cccccc',mew='1')
            ax.loglog(plotx_pareto,ploty_pareto,ls='-',lw=1,  c='k')
            ax.loglog(plotx_exp,ploty_exp,ls='-',lw=1, c='r')
            ax.loglog(plotx_powerexp,ploty_powerexp,ls='-',lw=1, c='g')
            
            leg = plt.legend(('n=%d,lower cutoff=%.2f, %d-day-tourist'%(others[0],others[1],level),
                              r'power-law: $\alpha$=%.2g, AIC=%.2f' %(others[2],others[3]),
                              r'exponential: $\lambda$=%.2g, AIC=%.2f'%(others[4],others[5]),
                              r'truncated power-law: $\alpha$=%.2g, $\lambda$=%.2g, AIC=%.2f' %
                              (others[6],others[7],others[8]),)
                             ,loc='lower left',shadow=True)
            frame  = leg.get_frame()  
            for t in leg.get_texts():
                t.set_fontsize('small')
            for l in leg.get_lines():
                l.set_linewidth(1.5)  
                
            plt.show()
            count += 1
        line = f.readline()
    f.close()
    
plot_heavytails("data.powerlaw/TPLdata.Distance.minx.txt",4)
    
def plot_ranksize_roi(dataset):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Rank-Size plot of Regions-of-Interest')
    ax.set_xlabel('RoI (id)')
    ax.set_ylabel('popularity')
    ax.set_xlim(auto=True)
    ax.set_ylim(auto=True)
    
    count = 0
    colors = ['b','g','r','b']
    marks = ['d','^','o','d']
    legends = []
    for i,data in enumerate(dataset):
        if i==16:# or i==8 or i==12 or i==16:
            y = data[::-1]
            n = len(y)
            x = np.arange(1,n+1)

            legends.append(ax.plot(x,y,ls='None', alpha='0.7',marker=marks[count],mec=colors[count],mfc='#cccccc',mew='1'))
            ax.plot(x,y,ls='-',lw=1, alpha='1.0', c=colors[count])
            count += 1

    leg = plt.legend(legends,(#'n=%d, level=4'% len(dataset[4]),
                      #'n=%d, level=8'% len(dataset[8]),
                      #'n=%d, level=12'% len(dataset[12]),
                      'n=%d, level=16'% len(dataset[16]),
                      ),
                     loc='top right',shadow=True)
    frame  = leg.get_frame()  
    #frame.set_facecolor('0.80')    # set the frame face color to light gray
    for t in leg.get_texts():
        t.set_fontsize('small')    # the legend text fontsize
    for l in leg.get_lines():
        l.set_linewidth(1.5)  # the legend line width

    plt.show()
    
def plot_ranksize_traveldist(dataset):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Rank-Size plot of travel distance')
    ax.set_xlabel('displacement')
    ax.set_ylabel('distance')
    ax.set_xlim(auto=True)
    ax.set_ylim(auto=True)
    
    count = 0
    colors = ['k','g','r','b']
    marks = ['s','^','o','d']
    legends = []
    for i,data in enumerate(dataset):
        if i!=2: continue
        y = data[::-1]
        n = len(y)
        x = np.arange(1,n+1)

        legends.append(ax.plot(x,y,ls='None', alpha='0.7',marker=marks[i],mec=colors[i],mfc='#cccccc',mew='1'))
        ax.plot(x,y,ls='-',lw=1, alpha='1.0', c=colors[i])
        count += 1

    leg = plt.legend(legends,(#'1-day travel: n=%d'% len(dataset[0]),
                      #'2-day travel: n=%d'% len(dataset[1]),
                      '3-day travel: n=%d'% len(dataset[2]),
                      ),
                     loc='top right',shadow=True)
    frame  = leg.get_frame()  
    for t in leg.get_texts():
        t.set_fontsize('small')    # the legend text fontsize
    for l in leg.get_lines():
        l.set_linewidth(1.5)  # the legend line width

    plt.show()
    
def plot_exp_trunc_pl_roi(dataset, exponents, rates,xmins, loglikes,AICs):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Exponential truncated power law of Regions-of-Interest')
    ax.set_xlabel('loglog(x) (RoI popularity)')
    ax.set_ylabel('loglog(P(x))')
    ax.set_xlim(auto=True)
    ax.set_ylim(auto=True)

    count = 0
    colors = ['b','g','r','b']
    marks = ['d','^','o','d']
    
    for i,data in enumerate(dataset):
        if i==16:# or i==8 or i==12 or i==16:
            x,y,curve_x,curve_y = data
            start_pos = np.where(np.array(x)==xmins[i])[0][0]
            x,y,curve_x,curve_y = x[start_pos:],y[start_pos:],curve_x[start_pos:],curve_y[start_pos:]
            ax.loglog(x,y,ls='None', alpha='0.7',marker=marks[count],mec=colors[count],mfc='#cccccc',mew='1')
            ax.loglog(curve_x,curve_y,ls='--',lw=2, alpha='1.0', c=colors[count])
            count += 1

    leg = plt.legend((#'n=%d, level=4'% len(dataset[4]),r'$x_0=%d\hspace{0.4} \alpha=%.4f\hspace{0.4} \lambda$=%.4f'%(xmins[4],exponents[4],rates[4]),
                      #'n=%d, level=8'% len(dataset[8]),r'$x_0=%d\hspace{0.4} \alpha=%.4f\hspace{0.4} \lambda$=%.4f'%(xmins[8],exponents[8],rates[8]),
                      #'n=%d, level=12'% len(dataset[12]),r'$x_0=%d \hspace{0.4}\alpha=%.4f \hspace{0.4}\lambda$=%.4f'%(xmins[12],exponents[12],rates[12]),
                      'n=%d, level=16'% len(dataset[16]),r'$x_0=%d  \hspace{0.4}  \alpha=%.4f \hspace{0.4}\lambda$=%.4f'%(xmins[16],exponents[16],rates[16]),
                      ),
                     loc='lower left',shadow=True)
    frame  = leg.get_frame()  
    for t in leg.get_texts():
        t.set_fontsize('small')    # the legend text fontsize
    for l in leg.get_lines():
        l.set_linewidth(1.5)  # the legend line width

    plt.show()
        
def plot_exp_trunc_pl_distance(dataset, exponents, rates,xmins, loglikes,AICs):        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Exponential truncated power law of travel distance')
    ax.set_xlabel('loglog(x) (RoI distance)')
    ax.set_ylabel('loglog(P(x))')
    ax.set_xlim(auto=True)
    ax.set_ylim(auto=True)
    
    count = 0
    colors = ['k','r','b','g']
    marks = ['s','^','o','d']
    line_colors = ['#333333','pink','g']

    plotx_set = []
    ploty_set = []
    for i,data in enumerate(dataset):
        count = i
        if i!=2: continue
        x,y,curve_x,curve_y = data 
        #start_pos = np.where(np.array(x)==xmins[i])[0][0]
        #x,y,curve_x,curve_y = x[start_pos:],y[start_pos:],curve_x[start_pos:],curve_y[start_pos:]
        ax.loglog(x,y,ls='None', alpha='0.7',marker=marks[count],mec=colors[count],mfc='#cccccc',mew='1')

    for i,data in enumerate(dataset):
        if i!=2: continue
        x,y,curve_x,curve_y = data 
        #start_pos = np.where(np.array(x)==xmins[i])[0][0]
        #x,y,curve_x,curve_y = x[start_pos:],y[start_pos:],curve_x[start_pos:],curve_y[start_pos:]
        ax.loglog(curve_x,curve_y,ls='--',lw=2, alpha='1.0', c=line_colors[i]) 

    leg = plt.legend((#'1-day travel, n=%d'% len(dataset[0][0]),
                      #'2-day travel, n=%d'% len(dataset[1][0]),
                      '3-day travel, n=%d'% len(dataset[2][0]),
                      #r'$x_0=%d  \hspace{0.4}  \alpha=%.4f \hspace{0.4}\lambda$=%.4f'%(xmins[0],exponents[0],rates[0]),
                      #r'$x_0=%d  \hspace{0.4}  \alpha=%.4f \hspace{0.4}\lambda$=%.4f'%(xmins[1],exponents[1],rates[1]),
                      r'$x_0=%d  \hspace{0.4}  \alpha=%.4f \hspace{0.4}\lambda$=%.4f'%(xmins[2],exponents[2],rates[2]),
                      ),
                     loc='lower left',shadow=True)
    frame  = leg.get_frame()  
    #frame.set_facecolor('0.80')    # set the frame face color to light gray
    for t in leg.get_texts():
        t.set_fontsize('small')    # the legend text fontsize
    for l in leg.get_lines():
        l.set_linewidth(1.5)  # the legend line width

    plt.show()
        
def plot_pl_roi(dataset,alpha_set,xmin_set,error_set):        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('power law of regions-of-interest')
    ax.set_xlabel('x (popularity)')
    ax.set_ylabel('P(x)')
    ax.set_xlim(auto=True)
    ax.set_ylim(auto=True)


    count = 0
    colors = ['b','g','r','b']
    marks = ['d','^','o','d']
    for i,data in enumerate(dataset):
        if i==16:# or i==8 or i==12 or i==16:
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

            ax.loglog(x,xcdf,ls='None', alpha='0.7',marker=marks[count],mec=colors[count],mfc='#cccccc',mew='1')
            #if i== 15:
            ax.loglog(plotx,ploty,ls='--',lw=2, alpha='1.0', c=colors[count])
            count += 1

    leg = plt.legend((#'n=%d, level=4'% len(dataset[4]),r'$\alpha$=%.4f +/- %.4f'%(alpha_set[4],error_set[4]),
                      #'n=%d, level=8'% len(dataset[8]),r'$\alpha$=%.4f +/- %.4f'%(alpha_set[8],error_set[8]),
                      #'n=%d, level=12'% len(dataset[12]),r'$\alpha$=%.4f +/- %.4f'%(alpha_set[12],error_set[12]),
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
    ax.set_xlim(auto=True)
    ax.set_ylim(auto=True)

    colors = ['k','r','b','g']
    marks = ['s','^','o','d']
    line_colors = ['#999999','pink','y']
    count = 0
    plotx_set = []
    ploty_set = []
    for i,data in enumerate(dataset):
        if i!=2: continue
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

        ax.loglog(x,xcdf,ls='None', alpha='0.6',marker=marks[i],mec=colors[i],mfc='#cccccc',mew='1')
        count += 1

    for i in range(len(dataset)):
        if i!=2: continue
        ax.loglog(plotx_set[0],ploty_set[0],ls='--',lw=2, alpha='1.0', c=line_colors[i])

    leg = plt.legend((#'1-day travel, n=%d'% len(dataset[0]),
                      #'2-day travel, n=%d'% len(dataset[1]),
                      '3-day travel, n=%d'% len(dataset[2]),
                      #r'$\alpha$=%.4f +/- %.4f'%(alpha_set[0],error_set[0]),
                      #r'$\alpha$=%.4f +/- %.4f'%(alpha_set[1],error_set[1]),
                      r'$\alpha$=%.4f +/- %.4f'%(alpha_set[2],error_set[2]),
                      ),
                     loc='lower left',shadow=True)
    frame  = leg.get_frame()  
    #frame.set_facecolor('0.80')    # set the frame face color to light gray
    for t in leg.get_texts():
        t.set_fontsize('small')    # the legend text fontsize
    for l in leg.get_lines():
        l.set_linewidth(1.5)  # the legend line width

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
        #data = data/np.linalg.norm(data)
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
        #data = data/np.linalg.norm(data)
        dataset.append(data)
        line = f.readline()
    f.close()
    return dataset

def export_to_R_data(dataset,filename):
    n = len(dataset)
    o = open(filename,'w')
    for i in range(n):
        o.write("%s\n"%(','.join(str(i) for i in dataset[i])))
    o.close()
    
def import_R_results(filename):
    def parse_line(_f):
        line = _f.readline().strip()
        tmp = line.split(' ')
        for item in tmp:
            if len(item) > 0 and item.find('[') == -1:
                return float(item)
        
    def parse_lines(_f):
        x = []
        line = _f.readline().strip()
        while(len(line)>0):
            tmp = line.split(' ')
            for item in tmp:
                if len(item) > 0 and item.find('[') == -1:
                    x.append(float(item))
            line = _f.readline().strip()
        return x
    
    dataset = []
    exponents =[]
    rates = []
    minxs = []
    loglikes = []
    AICs = []
    f=open(filename)
    line = f.readline()
    while(len(line)>0):
        line = line.strip()
        if line == "$x":
            x = parse_lines(f)
        elif line == "$y":
            y =parse_lines(f) 
        elif line == "$samplex":
            samplex = parse_lines(f)
        elif line == "$sampley":
            sampley = parse_lines(f)
        elif line == "$exponent":
            exponent = parse_line(f)
        elif line == "$rate":
            rate = parse_line(f)
        elif line == "$minx":
            minx = parse_line(f)
        elif line == "$loglike":
            loglike = parse_line(f)
        elif line == "$AIC":
            AIC = parse_line(f)
            data = [x,y,samplex,sampley]
            exponents.append(exponent)
            rates.append(rate)
            minxs.append(minx)
            loglikes.append(loglike)
            AICs.append(AIC)
            dataset.append(data)
        line = f.readline()
    f.close()
    return dataset, exponents, rates,minxs, loglikes,AICs


"""
if __name__ == "__main__":
    
    alpha_set = []
    xmin_set = []
    error_set = []

    dataset = get_roi_pop_dataset()
    #export_to_R_data(dataset, 'roi.R.txt')
    #dataset = get_distance_dataset()
    #export_to_R_data(dataset, 'distance.R.txt')

    plot_ranksize_roi(dataset)
    #plot_ranksize_traveldist(dataset)
    
    #dataset, exponents, rates,minxs, loglikes,AICs = import_R_results("roi.results.txt")
    #plot_exp_trunc_pl_roi(dataset, exponents, rates,minxs, loglikes,AICs)
    
    #dataset, exponents, rates,minxs, loglikes,AICs = import_R_results("distance.results.txt")
    #plot_exp_trunc_pl_distance(dataset, exponents, rates,minxs, loglikes,AICs)
    
    for i,data in enumerate(dataset):
        pl = plfit.plfit(data,quiet=False)
        alpha_set.append(pl._alpha)
        xmin_set.append(pl._xmin)
        error_set.append(pl._alphaerr)

    #plot_pl_roi(dataset, alpha_set, xmin_set,error_set)
    #plot_alpha(alpha_set,xmin_set,error_set)
    #plot_pl_distance(dataset,alpha_set,xmin_set,error_set)
"""