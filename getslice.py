import numpy as np
import matplotlib.pyplot as plt

#specs = {'OH': 4, 'HO2': 6, 'CH4': 9, 'NO': 20, 'NO2': 21}


fid = open('/scratch/w47/dkd561/compression/data/OH_fortran.dat','rb')
data = np.fromfile(fid,dtype='single')
#data = np.reshape(data,[1280,896,576],order='F')
fid.close()

fid = open('/scratch/w47/dkd561/compression/data/OH_fortran.dat.tucker.err.0.01.out','rb')
dataC = np.fromfile(fid,dtype='single')
#dataC = np.reshape(dataC,[1280,896,576],order='C')
fid.close()

err = np.abs(dataC-data)/(np.abs(data)+1e-6*np.max(data))
highE = err[err>1]

print(np.min(highE),np.max(highE))
print(np.shape(highE))
plt.scatter(highE*100,np.abs(data[err>1]),marker='.')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('% error')
plt.ylabel('Y(OH)')
plt.grid(alpha=0.2)
plt.title('Max E = 3e6 %')
plt.savefig('testfig.png',dpi=300)
#ind=np.unravel_index(np.argmax(data, axis=None), data.shape)
#print(ind)
#data[data<0]=0.0
#print(np.min(data))
##data_slice = data[:,:,288]
#fid = open('OH_slice_test.dat','wb')
#data_slice.tofile(fid)
#fid = open('mixture_fraction_fortran.dat','wb')
#data.T.tofile(fid)
#for key, value in specs.items():
#    fid = open(key+'_fortran.dat','wb')
#    data[:,:,:,value].T.tofile(fid)
