from multiprocessing import Pool
n_process = 12

x_naxis, y_naxis = 319, 315

gas_names = ['Hbeta', 'Halpha', '[SII]6716', '[SII]6731', '[OIII]4960',
             '[OIII]5008', '[OI]6300_d', '[NII]6549', '[NII]6585', 'Balmer_v', 'Forbidden_v']
multimap = dict(zip(gas_names, np.zeros((len(gas_names), 2, y_naxis, x_naxis))))

def fit_pixel(x, y):
    #print("Coordinate: [{}, {}]".format(x,y))
    pixel_flux = np.ma.masked_invalid(pipe_data[:, y, x])
    pixel_err = np.sqrt(np.ma.masked_invalid(pipe_data_err[:, y, x]))
    pixel_flux = pixel_flux.filled(0.0)
    pixel_err = pixel_err.filled(1e-4)
    if np.ma.median(pixel_flux) / np.ma.median(pixel_err) < 1:
        return 0
    try:
        pp = ppxf_example_population_gas_sdss(wavelength, pixel_flux, pixel_err, z, quiet=True)
    except:
        print(">>>>>>> skip ", [x, y])
        
    dwave = np.roll(pp.lam, -1) - pp.lam
    dwave[-1] = dwave[-2] # fix the bad point introduced by roll
    flux = dwave @ pp.matrix * pp.weights* pp.flux_scale
    flux_err = dwave @ pp.matrix \
               * capfit.cov_err(pp.matrix / pp.noise[:, None])[1]* pp.flux_scale

    gas_flux = dict(zip(pp.gas_names, flux[-9:]))
    gas_flux_err = dict(zip(pp.gas_names, flux_err[-9:]))
    #v, sigma = np.transpose(np.array(pp.sol)[pp.component[-9:].tolist()])
    #rel_v = dict(zip(pp.gas_names, v - 299792.485 * np.log(1+pp.z)))
    #sigma = dict(zip(pp.gas_names, sigma))
    #for name in pp.gas_names:
        #print(name, gas_flux[name], gas_flux_err[name])
        #multimap[name][x, y] = gas_flux[name], gas_flux_err[name]
    balmer_v = pp.sol[1][0] - 299792.485 * np.log(1+z)
    balmer_sigma = pp.sol[1][1]
    forbidden_v = pp.sol[2][0] - 299792.485 * np.log(1+z)
    forbidden_sigma = pp.sol[2][1]
    gas_kinematics = [balmer_v, balmer_sigma, forbidden_v, forbidden_sigma]
    return (x,y), gas_flux, gas_flux_err, gas_kinematics

        
from multiprocessing import Process, Queue
task_queue = Queue()


def worker(input, output):
    for func, args in iter(input.get, 'STOP'):
        result = func(*args)
        output.put(result)
if True:
    
    #TASKS = [(fit_pixel, (i, 7)) for i in range(20)]
    #Tasks = []
    #for x in range(0, x_naxis):
    #    for y in range(0, y_naxis):
    #        Tasks.append((fit_pixel, (x, y)))
    
    Tasks = [(fit_pixel, cc) for cc in spiral(x_naxis-1,y_naxis-1,X0=158,Y0=157)]

    
    # Create queues
    task_queue = Queue()
    done_queue = Queue()


    # Submit tasks
    n = 0
    for m in range(10000, len(Tasks), 10000):

        Tasks_tmp = Tasks[n:m]
        
        for task in Tasks_tmp:
            task_queue.put(task)

        for i in range(n_process):
            Process(target=worker, args=(task_queue, done_queue)).start()

        for i in range(len(Tasks_tmp)):
            #print('\t', done_queue.get())
            fitted_pixel = done_queue.get()
            if fitted_pixel == 0:
                continue
            x, y = fitted_pixel[0]
            gas_flux = fitted_pixel[1]
            gas_flux_err = fitted_pixel[2]
            gas_kinematics = fitted_pixel[3]
            for name in gas_names[:-2]:
                multimap[name][:, y, x] = gas_flux[name], gas_flux_err[name]
            multimap['Balmer_v'][:, y, x] = gas_kinematics[0], gas_kinematics[1]
            multimap['Forbidden_v'][:, y, x] = gas_kinematics[2], gas_kinematics[3]

        for i in range(n_process):
            task_queue.put('STOP')
            
        n = m
