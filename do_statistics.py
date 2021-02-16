import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
from scipy.stats import binned_statistic, entropy
import subprocess
from scipy.spatial.distance import jensenshannon
plt.rcParams.update({'font.size': 15})

def do_plots(data, name, title,  vmin=None, vmax=None):
    
    x = np.linspace(0,40,data.shape[0])
    y = np.linspace(0,22.5, data.shape[1])
    X,Y = np.meshgrid(x,y)
    plt.pcolormesh(X,Y, data.T, cmap='hot', vmin=vmin, vmax=vmax, shading='gouraud')
    plt.colorbar()
    plt.xlabel('X/D')
    plt.ylabel('Y/D')
    plt.title(title)
    plt.savefig(name, dpi=300)
    plt.close()

def do_conditional_mean(x,values, bins, Vrange=None):
    y, x, _ = binned_statistic(x,values, bins=bins, range=Vrange, statistic='std')
    return y, x[0:-1]

def do_pdf(x, bins, Vrange=None):
    y, x = np.histogram(x, bins=bins, range=Vrange, density=True)
    return y, x[0:-1]

def get_error_statistics(dataC, dataU):
    eps = np.linalg.norm(np.abs(dataC-dataU))/np.linalg.norm(dataU)
    err = np.abs(dataC-dataU)/(np.abs(dataU)+1e-6*np.max(dataU))
    maxE = np.max(err)
    meanE = np.mean(err)
    medianE = np.median(err)
    stats = {'normE': eps, 'maxE': maxE, 'medianE':medianE, 'meanE':meanE}
    return stats

def read_data(fname,prec):
    fid = open(fname,'rb')
    data = np.fromfile(fname,dtype=prec)
    return data

def get_unique_files(stats):
        allfiles = []
        for stat in stats:
            ifile = stat['file'].split('.'+args.compressor)[0]
            allfiles.append(ifile)
        uniFiles = list(set(allfiles))
        print(uniFiles)
        indices = []
        for stat in stats:
            ifile = stat['file'].split('.'+args.compressor)[0]
            indices.append(uniFiles.index(ifile))
        return uniFiles, indices

def read_stat_files(compressor):
        cfile = compressor+'_comp_stats.pkl'
        dfile = compressor+'_decomp_stats_update.pkl'
        cdata = pickle.load(open(cfile,'rb'))
        ddata = pickle.load(open(dfile,'rb'))
        return cdata, ddata

def get_stats(var, cdata, ddata):

    eNorm = []
    maxE = []
    meanE = []
    medianE = []
    CR = []
    tcomp = []
    tdecomp = []
    for stat in cdata:
        if(stat['file'] == var):
            tcomp.append(stat['tcomp'])
            for dstat in ddata:
                if(dstat['file'] == stat['ofile']):
                    CR.append(dstat['CR'])
                    eNorm.append(dstat['normE'])
                    maxE.append(dstat['maxE'])
                    meanE.append(dstat['meanE'])
                    medianE.append(dstat['medianE'])
                    tdecomp.append(dstat['tdecomp'])
                    break

    return {'tcomp':tcomp, 'tdecomp':tdecomp,'CR':CR, 'eNorm':eNorm, 'maxE':maxE, 'meanE':meanE, 'medianE':medianE}

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Do stats.')
    parser.add_argument('--compressor_stats',type=bool, default=False)
    parser.add_argument('--compressor',type=str, default='sz')
    parser.add_argument('--plots',type=bool, default=False)
    parser.add_argument('--stats_plots',type=bool, default=False)
    parser.add_argument('--conditional_plots',type=bool, default=False)
    parser.add_argument('--conditional_var',type=str, default=None)
    parser.add_argument('--infiles',type=str, nargs='+', default=None)
    parser.add_argument('--var',type=str,  default=None)

    args = parser.parse_args()
    if(args.conditional_plots):
        dataC = read_data(args.conditional_var,'single')
        fid = open(args.compressor+'_decomp_stats_update.pkl','rb')
        stats = pickle.load(fid)
        legend = ['uncompressed']
        linestyles = ['-','--','-.',':','.',',']
        P = np.zeros(500)
        for i, fname in enumerate(args.infiles):
            data = read_data(fname, 'single')
            y, x = do_conditional_mean(dataC,data,100,(0,1))
            #y,x = do_pdf(np.log10(np.abs(data)+1e-9),500)
            #if(i==0):
            #    P=y
            #print(entropy(P,y))
            #print(jensenshannon(P,y))
            print(fname)
            for stat in stats:
                if(stat['file']+'.out' == fname):
                    
                    legend.append('CR = %.2f'%stat['CR'] + ' MaxE = %.3f '% (stat['maxE']*100))

            plt.plot(x,y,linestyles[i],linewidth=3)
            #np.savetxt('pdf_'+str(i)+'.txt',y)
        plt.legend(legend,prop={'size': 10})
        plt.grid(alpha=0.2)
        plt.title(args.compressor)
        plt.xlabel('mixture fraction')
        plt.ylabel('Y('+args.var+')'+' std-dev')
        #plt.ylabel(args.var+' std-dev')
        #plt.yscale('log')
        plt.tight_layout() 
        plt.savefig(args.var+'_'+args.compressor+'_std.png', dpi=300)
        plt.close()
    if(args.plots):
        stats = pickle.load(open(args.compressor+'_decomp_stats_update.pkl','rb'))
        uniFiles, indices = get_unique_files(stats)
        #stats = pickle.load(open(args.compressor+'_decomp_stats_update.pkl','rb'))
        vmin = np.zeros(len(uniFiles))
        vmax = np.zeros(len(uniFiles))
        for i, fname in enumerate(uniFiles):
            data = read_data(fname,'single')
            vmin[i] = np.min(data)
            vmax[i] = np.max(data)
        
        for i, stat in enumerate(stats):
            fname = stat['file'] + '.out'
            data = read_data(fname,'single')
            data = np.reshape(data,[1280, 896, 576], order='F')
            data_slice = data[:,:,288]
            name = stat['file']+'.png'
            title = args.compressor + ' CR: %.2f'%stat['CR']
            do_plots(data_slice, name, title, vmin=vmin[indices[i]], vmax=vmax[indices[i]])
        
        for i, fname in enumerate(uniFiles):
            data = read_data(fname,'single')
            data = np.reshape(data,[1280, 896, 576], order='F')
            data_slice = data[:,:,288]
            name = fname+'.png'
            title = ' CR: 1'
            do_plots(data_slice, name, title, vmin=vmin[i], vmax=vmax[i])

    if(args.compressor_stats):
        sfile = args.compressor+'_decomp_stats.pkl'
        stats = pickle.load(open(sfile,'rb'))
        uniFiles, indices = get_unique_files(stats)
        dataU = np.zeros([1280*896*576,len(uniFiles)],dtype='single')
        szi = []
        for i in range(len(uniFiles)):
            dataU[:,i] = read_data(uniFiles[i],'single')
            sz = subprocess.run('du ' + uniFiles[i], shell=True, capture_output=True, text=True)
            szi.append(int(sz.stdout.split()[0]))

        Estats = []
        for i, stat in enumerate(stats):
            dataC = read_data(stat['file']+'.out','single')
            szo = subprocess.run('du ' + stat['file'], shell=True, capture_output=True, text=True)
            errstats = get_error_statistics(dataC, dataU[:,indices[i]])
            errstats.update({'CR': 1.0*szi[indices[i]]/int(szo.stdout.split()[0])})
            Estats.append(errstats)

        for i, estat in enumerate(Estats):
            stats[i].update(estat)

        fid = open(args.compressor+'_decomp_stats_update.pkl','wb')
        pickle.dump(stats, fid)


    if(args.stats_plots):
        csz, dsz = read_stat_files('sz')
        #print(csz)
        czfp, dzfp = read_stat_files('zfp')
        ctucker, dtucker = read_stat_files('tucker')
        varNames = ["OH_fortran.dat","NO2_fortran.dat","NO_fortran.dat","HO2_fortran.dat",\
                "temp_JICF_fortran.dat","mixture_fraction_fortran.dat"]

        for vname in varNames:
            szstats = get_stats("/scratch/w47/dkd561/compression/data/"+vname, csz, dsz)        
            zfpstats = get_stats("/scratch/w47/dkd561/compression/data/"+vname, czfp, dzfp)        
            tstats = get_stats("/scratch/w47/dkd561/compression/data/"+vname, ctucker, dtucker)        
            var  = "maxE"
            plt.figure()
            vmin = min(szstats['tcomp'])
            vmin = min(vmin,min(zfpstats['tcomp']),min(tstats['tcomp']))
            vmax = max(szstats['tcomp'])
            vmax = max(vmax,max(zfpstats['tcomp']),max(tstats['tcomp']))
            print(vmin,vmax)
            #mins = np.min(np.min(szstats['tcomp']),np.min(zfpstats['tcomp']),np.min(tstats['tcomp']))
            #print(szstats['tcomp'])
            plt.plot(np.sort(szstats[var])*100,np.asarray(szstats['CR'])[np.argsort(szstats[var])],'-o')
            #plt.scatter(np.sort(szstats[var])*100,np.asarray(szstats['CR'])[np.argsort(szstats[var])],c=np.asarray(szstats['tcomp'])[np.argsort(szstats[var])],vmin=vmin,vmax=vmax)
            plt.plot(np.sort(zfpstats[var])*100,np.asarray(zfpstats['CR'])[np.argsort(zfpstats[var])],'-o')
            #plt.scatter(np.sort(zfpstats[var])*100,np.asarray(zfpstats['CR'])[np.argsort(zfpstats[var])],c=np.asarray(zfpstats['tcomp'])[np.argsort(zfpstats[var])],vmin=vmin,vmax=vmax)
            plt.plot(np.sort(tstats[var])*100,np.asarray(tstats['CR'])[np.argsort(tstats[var])],'-o')
            #plt.scatter(np.sort(tstats[var])*100,np.asarray(tstats['CR'])[np.argsort(tstats[var])],c=np.asarray(tstats['tcomp'])[np.argsort(tstats[var])],vmin=vmin,vmax=vmax)
            #plt.colorbar()
            #plt.clim(vmin=vmin,vmax=vmax)
            plt.xscale('log')
            plt.yscale('log')
            plt.ylim([1,1e4])
            plt.xlabel('% Max error')
            plt.ylabel('Compression ratio')
            plt.title(vname.split('_')[0])
            plt.grid()
            plt.legend(['SZ','ZFP','TUCKERMPI'])
            plt.savefig(vname+'_'+var+'_stats.png',dpi=300)
            #plt.savefig(vname+'_test_fig.png',dpi=200)
            plt.close()
        #print(szstats)




