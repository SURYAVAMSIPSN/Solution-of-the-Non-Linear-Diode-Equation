# Name: SIVANAGA SURYA VAMSI POPURI
# ASU ID: 1217319207
# Project 3- NonLinear Diode. 
################################################################################

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

DiodeIV = [line.rstrip('\n').split() for line in open('DiodeIV.txt')]
for i in range(len(DiodeIV)):
    DiodeIV[i] = list(map(float, DiodeIV[i])) # Reads the diode IV stuff. Converts everything to float. 
DiodeIV = np.array(DiodeIV)

meas_i = DiodeIV[:, 1]


# I = (V - Vd)/R ---- (1)
# I = Is*exp(q*Vd/nkT - 1) ---- (2)
# Need to guess a Vdd and find I. With this, need to iterate for different values of Vdd such that I in (1) is
# the same as I in (2)

    
def diodeI(Vd, A, phi_value, ide_value, temp):
    Vt = ide_value * KB * temp/Q
    is_value = A * temp * temp * np.exp(-phi_value * Q / ( KB * temp ) )
    return is_value * np.exp((Vd/Vt) - 1)

def compute_diode_current(Vd, ide_value, temp, is_value):
    return is_value * np.exp((Q*Vd)/(ide_value*KB*temp) - 1) # Eqn 2

def solve_diode_v(Vd, src_v, r_value, ide_value, temp, is_value):
    error = ((Vd - src_v)/r_value) + compute_diode_current(Vd, ide_value, temp, is_value) # Error term (Eqn 1)
    # The above function calls the compute_diode_current function. Obtained thru nodal analysis. 
    return error

# Computes the voltage across the diode and current in the whole circuit. 
def diode_vi_optimize(r_value,is_value, ide_value, temp, src_v):
    est_v   = np.zeros_like(src_v)       # an array to hold the diode voltages
    diode_i = np.zeros_like(src_v)       # an array to hold the diode currents
    prev_v = 0.1                 # an initial guess for the voltage

    for index in range(len(src_v)):
        prev_v = optimize.fsolve(solve_diode_v,prev_v,
				(src_v[index],r_value,ide_value,temp,is_value),
                                xtol=1e-12)[0]
        est_v[index] = prev_v            
    # compute the diode current
    diode_i = compute_diode_current(est_v,ide_value,temp,is_value)
    return est_v, diode_i
    
# gives the residual error for resistance 
def opt_r(r_value, ide_value, phi_value, area, temp, src_v, meas_i):
    est_v   = np.zeros_like(src_v)       # an array to hold the diode voltages
    diode_i = np.zeros_like(src_v)       # an array to hold the diode currents
    prev_v = 0.1                 # an initial guess for the voltage

    # need to compute the reverse bias saturation current for this phi!
    is_value = area * temp * temp * np.exp(-phi_value * Q / ( KB * temp ) )

    for index in range(len(src_v)):
        prev_v = optimize.fsolve(solve_diode_v,prev_v,
				(src_v[index],r_value,ide_value,temp,is_value),
                                xtol=1e-12)[0]
        est_v[index] = prev_v            # store for error analysis

    # compute the diode current
    diode_i = compute_diode_current(est_v,ide_value,temp,is_value)
    err = meas_i - diode_i
    return err

# gives the residual error for ideality 
def opt_ide(ide_value, r_value, phi_value, area, temp, src_v, meas_i):
    est_v   = np.zeros_like(src_v)       # an array to hold the diode voltages
    diode_i = np.zeros_like(src_v)       # an array to hold the diode currents
    prev_v = 0.1                 # an initial guess for the voltage

    # need to compute the reverse bias saturation current for this phi!
    is_value = area * temp * temp * np.exp(-phi_value * Q / ( KB * temp ) )

    for index in range(len(src_v)):
        prev_v = optimize.fsolve(solve_diode_v,prev_v,
				(src_v[index],r_value,ide_value,temp,is_value),
                                xtol=1e-12)[0]
        est_v[index] = prev_v            # store for error analysis

    # compute the diode current
    diode_i = compute_diode_current(est_v,ide_value,temp,is_value)
    err_norm = (meas_i - diode_i)/(meas_i + diode_i + 1e-15)
    return err_norm

# gives the residual error for phi
def opt_phi(phi_value, r_value, area, ide_value, temp, src_v, meas_i):
    est_v   = np.zeros_like(src_v)       # an array to hold the diode voltages
    diode_i = np.zeros_like(src_v)       # an array to hold the diode currents
    prev_v = 0.1                 # an initial guess for the voltage

    # need to compute the reverse bias saturation current for this phi!
    is_value = area * temp * temp * np.exp(-phi_value * Q / ( KB * temp ) )

    for index in range(len(src_v)):
        prev_v = optimize.fsolve(solve_diode_v,prev_v,
				(src_v[index],r_value,ide_value,temp,is_value),
                                xtol=1e-12)[0]
        est_v[index] = prev_v            # store for error analysis

    # compute the diode current
    diode_i = compute_diode_current(est_v,ide_value,temp,is_value)
    err_norm = (meas_i - diode_i)/(meas_i + diode_i + 1e-15)
    return err_norm

if __name__ == '__main__':
    KB = 1.380648e-23 
    temp = 350
    Q = 1.6021766208e-19
    src_v = np.arange(0.0, 2.6, 0.1) # Range of source voltages V from 0.1 to 2.5
    r_value = 11000 # Resistor in series
    ide_value = 1.7
    is_value = 1e-9 # Diode current
    area = 1e-8 # Area of cross section
    
    # RESULTS OF PART 1 (PROBLEM 1) 
    est_v, diode_i = diode_vi_optimize(r_value, is_value, ide_value, temp, src_v) 
    plt.figure()
    plt.plot(est_v, np.log(diode_i))
    plt.plot(src_v, np.log(diode_i))
    plt.xlabel('Voltages in Volts')
    plt.ylabel('Diode current in Log scale (Amps)')
    plt.legend(('Diode voltage vs Diode current', 'Source Voltage vs diode current'))
    plt.show()
    
    
    # RESULTS OF PART 2 (PROBLEM 2) 
    src_v = np.arange(0.0, 5.1, 0.1)
    phi_value = 0.8
    r_value = 10000
    ide_value = 1.5 # Guess values. 
    new_temp = 375
    new_area = 1e-8
    tolerance = 1e-5 # random tolerance value.
    number_iterations = 100
    
    for i in range(number_iterations): # Optimization done here. 
        print("Iteration: ", i)
        r_value = optimize.leastsq(opt_r, r_value, args = (ide_value, phi_value, new_area, new_temp, src_v, meas_i))  
        #r_val = r_val_opt[0][0]
        r_value = r_value[0][0] # to be used by the subsequent optimizers.  
        print('r_value: ', r_value)
        
        ide_value = optimize.leastsq(opt_ide, ide_value, args = (r_value, phi_value, new_area, new_temp, src_v, meas_i))  
        #ide_val = ide_val_opt[0][0]
        ide_value = ide_value[0][0]
        print('ide_value: ', ide_value)
        
        phi_value = optimize.leastsq(opt_phi, phi_value, args = (r_value, new_area, ide_value, new_temp, src_v, meas_i))  
        #phi_val = phi_val_opt[0][0]
        phi_value = phi_value[0][0]
        print('phi_value: ', phi_value)
        
        print("\n")
        
    print("THe optimized values are: \n")
    print("r_value: %0.4f\nide_value: %0.4f\nphi_value: %0.4f\n"%(r_value, ide_value, phi_value))
    # now we have r_value, ide_value and phi_value. 
    
    is_value = new_area * new_temp * new_temp * np.exp(-phi_value * Q / ( KB * new_temp ) ) # calculate Is with r, phi and ide
    est_v, diode_i = diode_vi_optimize(r_value,is_value, ide_value, temp, src_v) # obtain current throughout the circuit
    
    # Let's plot. 
    plt.figure()
    plt.plot(src_v, np.log(diode_i)) # actual values
    plt.plot(DiodeIV[:,0], np.log(meas_i)) # estimated values 
    plt.xlabel('Voltages in volts')
    plt.ylabel('Diode currents in log scale (Amps)')
    plt.legend(('Voltage vs Actual current', 'Voltage vs Measured current'))
    plt.show()