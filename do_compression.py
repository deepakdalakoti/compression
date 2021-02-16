import numpy as np
import subprocess
import pickle
import argparse
from multiprocessing import Pool
import time as timer
import os 

SZ_loc = '/scratch/w47/dkd561/packages/bin/sz'
ZFP_loc ='/scratch/w47/dkd561/compression/zfp/build/bin/zfp'
ISB_comp_loc = '/scratch/w47/dkd561/compression/ISABELA-compress-0.2.1/apps/example_file/file_compress'
ISB_decomp_loc = '/scratch/w47/dkd561/compression/ISABELA-compress-0.2.1/apps/example_file/file_decompress'
TUCKER_comp_loc = '/scratch/w47/dkd561/compression/TuckerMPI/build/serial/drivers/bin/Tucker_sthosvd'
TUCKER_decomp_loc = '/scratch/w47/dkd561/compression/TuckerMPI/build/serial/drivers/bin/Tucker_reconstruct'

def SZ(loc, mode, ifile, ofile, prec, errMode, err, Ndims, dims):
    if(mode=='c'):
        eMode = '-P'
        if(errMode=='ABS'):
            eMode = '-A'
        elif(errMode=='REL'):
            eMode = '-R'
        comm = loc + ' -i ' + ifile + ' -z ' + ofile + ' ' + prec + ' -M ' + errMode + ' ' + eMode \
                + ' ' + err + ' -' + Ndims + ' ' + dims + ' -c sz.config'  
        print(comm)
        t1 = timer.time()
        out=subprocess.run(comm, shell=True, capture_output=True)
        tcomp = timer.time()-t1
        timer.sleep(5.0)
        szi = subprocess.run('du ' + ifile, shell=True, capture_output=True, text=True)
        szo = subprocess.run('du ' + ofile, shell=True, capture_output=True, text=True)
        CR = int(szi.stdout.split()[0])/int(szo.stdout.split()[0])

        stats = {'Compressor':'sz', 'CR': CR, 'tcomp': tcomp, 'file': ifile, \
                'errMode': errMode, 'err': err, 'ofile':ofile}
        return stats

    if(mode=='d'):
        eMode = '-P'
        if(errMode=='ABS'):
            eMode = '-A'
        elif(errMode=='REL'):
            eMode = '-R'

        comm = loc + ' -s ' + ifile + ' -x ' + ofile + ' ' + prec + ' ' + eMode \
                + ' ' + err + ' -' + Ndims + ' ' + dims + ' -c sz.config'
        print(comm)
        t1 = timer.time()
        subprocess.run(comm, shell=True, capture_output=True)
        tdecomp = timer.time()-t1

        stats = {'Compressor':'sz', 'tdecomp': tdecomp, 'file': ifile, 'err': err}

        return stats

def ZFP(loc, mode, ifile, ofile, prec, err, Ndims, dims):
    if(mode=='c'):
        comm = loc + ' -i ' + ifile + ' -z ' + ofile + ' ' + prec +  \
                 ' -a ' + err + ' -' + Ndims + ' ' + dims
        print(comm)
        t1 = timer.time()
        out=subprocess.run(comm, shell=True, capture_output=True)
        tcomp = timer.time()-t1
        timer.sleep(5.0)
        szi = subprocess.run('du ' + ifile, shell=True, capture_output=True, text=True)
        szo = subprocess.run('du ' + ofile, shell=True, capture_output=True, text=True)
        CR = int(szi.stdout.split()[0])/int(szo.stdout.split()[0])

        stats = {'Compressor':'zfp', 'CR': CR, 'tcomp': tcomp, 'file': ifile, 'err': err, 'ofile':ofile}
        return stats

    if(mode=='d'):

        comm = loc + ' -z ' + ifile + ' -o ' + ofile + ' ' + prec + ' -a ' + err + \
                ' -' + Ndims + ' ' + dims
        print(comm)
        t1 = timer.time()
        subprocess.run(comm, shell=True, capture_output=True)
        tdecomp = timer.time()-t1

        stats = {'Compressor':'zfp', 'tdecomp': tdecomp, 'file': ifile, 'err': err}

        return stats

def TUCKER(loc, mode, ifile, ofile, err, Ndims, dims):
    if(mode=='c'):
        pid = os.getpid()

        ADDS = ["Global dims = " + dims, 
                "SV Threshold = " + err,
                "STHOSVD directory = " + ofile,
                "Input file list = " + "raw_"+str(pid)+".txt"]

        subprocess.run('echo ' + ifile + '> raw_'+str(pid)+'.txt', shell=True)
        subprocess.run('cat param_base.txt > params_'+str(pid)+'.txt', shell=True)
        subprocess.run('mkdir ' + ofile, shell=True)
        for add in ADDS:
            subprocess.run('echo ' + add + ' >> params_'+str(pid)+'.txt', shell=True)

        comm = loc + ' --parameter-file ' + 'params_'+str(pid)+'.txt'
        print(comm)
        t1 = timer.time()
        out=subprocess.run(comm, shell=True, capture_output=True)
        tcomp = timer.time()-t1
        timer.sleep(5.0)
        szi = subprocess.run('du ' + ifile, shell=True, capture_output=True, text=True)
        szo = subprocess.run('du ' + ofile, shell=True, capture_output=True, text=True)
        CR = int(szi.stdout.split()[0])/int(szo.stdout.split()[0])

        stats = {'Compressor':'tucker', 'CR': CR, 'tcomp': tcomp, 'file': ifile, 'err': err, 'ofile':ofile}
        return stats

    if(mode=='d'):
        Esub = ' '.join([str(int(x)-1) for x in dims.split()])
        pid = os.getpid()
        ADDS  = ["Global dims = " + dims, \
                 "Beginning subscripts = 0 0 0 ", \
                 "Ending subscripts = " + Esub, \
                 "Output file list = rec_"+str(pid)+'.txt',\
                 "STHOSVD directory = " + ifile]
        subprocess.run('echo ' + ofile + '> rec_'+str(pid)+'.txt', shell=True)
        subprocess.run('rm ' + 'recons_'+str(pid)+'.txt',shell=True)
        for add in ADDS:
            subprocess.run('echo ' + add + ' >> recons_'+str(pid)+'.txt', shell=True)

        comm = loc + ' --parameter-file ' + 'recons_'+str(pid)+'.txt'
        print(comm)
        t1 = timer.time()
        subprocess.run(comm, shell=True, capture_output=True)
        tdecomp = timer.time()-t1

        stats = {'Compressor':'tucker', 'tdecomp': tdecomp, 'file': ifile, 'err': err}

        return stats

def ISB(loc, mode, ifile, ofile, prec, win_size, Ncoeff, err):
    if(mode=='c'):
        comm = loc + ' ' + ifile + ' ' + ofile + ' ' + prec + ' '\
                + win_size + ' ' + Ncoeff + ' ' + err
        print(comm)
        t1 = timer.time()
        out=subprocess.run(comm, shell=True, capture_output=True)
        tcomp = timer.time()-t1
        timer.sleep(5.0)
        szi = subprocess.run('du ' + ifile, shell=True, capture_output=True, text=True)
        szo = subprocess.run('du ' + ofile, shell=True, capture_output=True, text=True)
        CR = int(szi.stdout.split()[0])/int(szo.stdout.split()[0])

        stats = {'Compressor':'isb', 'CR': CR, 'tcomp': tcomp, 'file': ifile, 'err': err, \
                'win_size': win_size, 'Ncoeff': Ncoeff, 'ofile': ofile}
        return stats

    if(mode=='d'):

        comm = loc + ' ' + ifile + ' ' + ofile + ' ' + prec + ' '\
                + win_size + ' ' + Ncoeff + ' ' + err
   
        print(comm)
        t1 = timer.time()
        subprocess.run(comm, shell=True, capture_output=True)
        tdecomp = timer.time()-t1

        stats = {'Compressor':'isb', 'tdecomp': tdecomp, 'file': ifile, 'err': err, \
                'win_size': win_size, 'Ncoeff': Ncoeff}

        return stats



def log_results(res):
    all_results.append(res)

def print_error(err):
    print("Fucked ", err)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Do compression.')
    parser.add_argument('--compressor',type=str, default='sz')
    parser.add_argument('--infiles',type=str, nargs='+', default=None, required=True)
    parser.add_argument('--mode',type=str, default='c', required=False)
    parser.add_argument('--precision',type=str, default='float', required=False)
    parser.add_argument('--errMode',type=str, default=None, required=False)
    parser.add_argument('--err',type=str, nargs='+', default=None, required=True)
    parser.add_argument('--Ndims',type=str,  default='3', required=False)
    parser.add_argument('--dims',type=str, nargs='+', default=None, required=True)
    parser.add_argument('--win_size',type=str, nargs='+', default=None, required=False)
    parser.add_argument('--Ncoeff',type=str, nargs='+', default=None, required=False)

    args = parser.parse_args()
    all_results = []
    p = Pool(10)

    if(args.compressor == 'sz'):
        prec = '-f'
        if(args.precision=='double'):
            prec='d'

        for ifile in args.infiles:
            for err in args.err:
                infile = ifile
                outfile = ifile  + '.sz.err.' + err
                if(args.mode=='d'):
                    infile = ifile+'.sz.err.'+err
                    outfile = ifile+'.sz.err.'+err+'.out'
                p.apply_async(SZ,args=(SZ_loc, args.mode, infile, outfile,  \
                        prec, args.errMode, err, args.Ndims, ' '.join(args.dims),),
                        callback = log_results, error_callback=print_error)

        p.close()
        p.join()
        if(args.mode=='c'):
            fid = open('sz_comp_stats.pkl','wb')
        elif(args.mode=='d'):
            fid = open('sz_decomp_stats.pkl','wb')
        pickle.dump(all_results,fid)

    if(args.compressor == 'zfp'):
        prec = '-f'
        if(args.precision=='double'):
            prec='d'

        for ifile in args.infiles:
            for err in args.err:
                infile = ifile
                outfile = ifile  + '.zfp.err.' + err
                if(args.mode=='d'):
                    infile = ifile+'.zfp.err.'+err
                    outfile = ifile+'.zfp.err.'+err+'.out'
                p.apply_async(ZFP,args=(ZFP_loc, args.mode, infile, outfile,  \
                        prec, err, args.Ndims, ' '.join(args.dims),),
                        callback = log_results, error_callback=print_error)

        p.close()
        p.join()
        if(args.mode=='c'):
            fid = open('zfp_comp_stats.pkl','wb')
        elif(args.mode=='d'):
            fid = open('zfp_decomp_stats.pkl','wb')
        pickle.dump(all_results,fid)

    if(args.compressor == 'tucker'):
        for ifile in args.infiles:
            for err in args.err:
                infile = ifile
                outfile = ifile  + '.tucker.err.' + err
                loc = TUCKER_comp_loc
                if(args.mode=='d'):
                    infile = ifile+'.tucker.err.'+err
                    outfile = ifile+'.tucker.err.'+err+'.out'
                    loc = TUCKER_decomp_loc
                p.apply_async(TUCKER,args=(loc, args.mode, infile, outfile,  \
                        err, args.Ndims, ' '.join(args.dims),),
                        callback = log_results, error_callback=print_error)

        p.close()
        p.join()
        if(args.mode=='c'):
            fid = open('tucker_comp_stats.pkl','wb')
        elif(args.mode=='d'):
            fid = open('tucker_decomp_stats.pkl','wb')
        pickle.dump(all_results,fid)

    if(args.compressor == 'isb'):
        prec = '4'
        if(args.precision=='double'):
            prec='8'
        for ifile in args.infiles:
            for err in args.err:
                for win in args.win_size:
                    for ncoeff in args.Ncoeff:
                        infile = ifile
                        outfile = ifile  + '.isb.err.' + err + '.win_size.' + win + '.ncoeff.' + ncoeff
                        loc = ISB_comp_loc
                        if(args.mode=='d'):
                            infile = ifile+'.isb.err.'+err+ '.win_size.' + win + '.ncoeff.' + ncoeff
                            outfile = ifile+'.isb.err.'+err + '.win_size.' + win + '.ncoeff.' + ncoeff+'.out'
                            loc = ISB_decomp_loc
                        p.apply_async(ISB,args=(loc, args.mode, infile, outfile,  \
                            prec, win, ncoeff, err,),callback = log_results, error_callback=print_error)

        p.close()
        p.join()
        if(args.mode=='c'):
            fid = open('isb_comp_stats.pkl','wb')
        elif(args.mode=='d'):
            fid = open('isb_decomp_stats.pkl','wb')
        pickle.dump(all_results,fid)


