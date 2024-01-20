import matplotlib.pyplot as plt
import taichi as ti
ti.init()
import numpy as np

import sys
sys.path.append('/home/eleven/work/tiDEMPM_v0.2')

eps1 = 0.08 # total axial strain to be loaded
nsub = int(eps1/0.0005) # load 0.0005 per step
sig1 = ti.Vector([306., 306., 306., 0., 0., 0.])
sig2 = ti.Vector([98., 98., 98., 0., 0., 0.])
sig3 = ti.Vector([33., 33., 33., 0., 0., 0.])

nu = 0.25 # Poisson's ratio ν
lam = 0.12 # slope of NCL λ
kappa = 0.023 # slope of URL κ
M = 1.02 # critical state stress ratio
p0 = 392. # preconsolidation pressure
N = 2.7 # specific volume in NCL at unit pressure
k = ti.Vector([1., 1., 1., 0., 0., 0.]) # Kronecker tensor δᵢⱼ
FTOL = 1.e-9 # yield function tolerance
STOL = 1.e-6 # stress tolerance
LTOL = 1.e-6 # detecting tolerance
MAXITS = 3
NSUB = 10
dTmin = 1e-4
EPS = 1.e-16 # machine error
ITS = 10 # maximum iteration number

def vegiot_tensor_dot(vec1, vec2):
    return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2] + \
           2 * (vec1[3] * vec2[3] + vec1[4] * vec2[4] + vec1[5] * vec2[5])

def invariant_j2(stress):
    return ((stress[0] - stress[1]) * (stress[0] - stress[1]) \
            + (stress[1] - stress[2]) * (stress[1] - stress[2]) \
            + (stress[0] - stress[2]) * (stress[0] - stress[2])) / 6. \
            + stress[3] * stress[3] + stress[4] * stress[4] + stress[5] * stress[5]

def mean_stress(stress):
    return (stress[0] + stress[1] + stress[2]) / 3.

def deviatoric_stress(stress):
    p = mean_stress(stress)
    return stress - p * k

def equivalent_stress(stress):
    J2 = invariant_j2(stress)
    return ti.sqrt(3 * J2)

def delastic_step(de, shear_modulus, bulk_modulus):
    mde = (de[0] + de[1] + de[2]) / 3.
 
    dstress = ti.Vector([0., 0., 0., 0., 0., 0.])
    dstress[0] = 3. * bulk_modulus * mde + 2. * shear_modulus * (de[0] - mde)
    dstress[1] = 3. * bulk_modulus * mde + 2. * shear_modulus * (de[1] - mde)
    dstress[2] = 3. * bulk_modulus * mde + 2. * shear_modulus * (de[2] - mde)
    dstress[3] = shear_modulus * de[3]
    dstress[4] = shear_modulus * de[4]
    dstress[5] = shear_modulus * de[5]

    return dstress

def RegulaFalsi(pi,qi,p0,pe,qe,its,alpha0,alpha1):
    Fsave = M**2*(pi**2-p0*pi)+qi**2 # f = M²(p² - pₒp) + q²
    F0 = M**2*((pi+alpha0*pe)**2-p0*(pi+alpha0*pe))+(qi+alpha0*qe)**2
    F1 = M**2*((pi+alpha1*pe)**2-p0*(pi+alpha1*pe))+(qi+alpha1*qe)**2
    for i in range(its):
        alpha = alpha1 - (alpha1-alpha0)*F1/(F1-F0)
        Fnew = M**2*((pi+alpha*pe)**2-p0*(pi+alpha*pe))+(qi+alpha*qe)**2
        if abs(Fnew) <= FTOL:
            return alpha
        if Fnew*F0 < 0: # Fnew and F0 have opposite signs
            alpha1 = alpha
            F1 = Fnew
            if Fnew*Fsave > 0: # Fnew and Fsave have same signs
                F0 = F0/2.
        else:
            alpha0 = alpha
            F0 = Fnew
            if Fnew*Fsave > 0: # Fnew and Fsave have same signs
                F1 = F1/2.
        Fsave = Fnew
    raise RuntimeError('Convergence not achieved after maximum iteration, exit!')

def RegulaFalsiNegativePlasticMultiplier(sig,p0,dsig_e,MAXITS,NSUB,alpha0,alpha1):
    pi = mean_stress(sig)
    qi = equivalent_stress(sig)
    Fsave = M**2*(pi**2-p0*pi)+qi**2 # f = M²(p² - pₒp) + q²
    F0 = Fsave
    for i in range(MAXITS):
        dalpha = (alpha1 - alpha0) / NSUB
        for j in range(NSUB):
            alpha = alpha0 + dalpha
            sig1 = sig + alpha * dsig_e
            p1 = mean_stress(sig1) 
            q1 = equivalent_stress(sig1)
            f = M**2*(p1**2-p0*p1)+q1**2
            if f > FTOL:
                alpha1 = alpha
                if F0 < -FTOL:
                    F1 = f
                    return alpha0, alpha1
                else:
                    alpha0 = 0
                    F0 = Fsave
                    break
            else:
                alpha0 = alpha
                F0 = f
    raise RuntimeError('Convergence not achieved after maximum iteration, exit!')

def DriftCorrect(sig,p0,its):
    for i in range(its):
        p = mean_stress(sig) # mean stress
        devsig = deviatoric_stress(sig)
        q = equivalent_stress(sig)
        e0 = N - lam*np.log(p0) + kappa*np.log(p0/p) - 1. # initial void ratio
        bulk_modulus = (1. + e0) / kappa * p
        shear_modulus = 3. * bulk_modulus * (1 - 2 * nu) / (2 * (1 + nu))
        
        F0 = M**2*(p**2-p0*p)+q**2 # f = M²(p² - pₒp) + q²
        Kp0 = M**4*p*p0*(2*p-p0)*(1+e0)/(lam-kappa) # Kp = M4ppₒ(2p-pₒ)*(1+eₒ)/(λ-κ)
        h0 = M**2*(2*p-p0)
        A = M**2*(2*p-p0)/3.*k + 3.*devsig # ∂f/∂σᵢⱼ
        tempMat = delastic_step(A, shear_modulus, bulk_modulus)
        ADB = vegiot_tensor_dot(A,  tempMat) # (aᵀᵢⱼ Cᵢⱼₒᵣ aₒᵣ)
        dlam = F0/(ADB+Kp0)
        dkap = dlam*h0
        sig_new = sig - dlam*tempMat
        p_new = mean_stress(sig_new)
        q_new = equivalent_stress(sig_new)
        p0_new = p0 + (1+e0)*p0/(lam-kappa)*dkap
        Fnew = M**2*(p_new**2-p0_new*p_new)+q_new**2
        if abs(Fnew) > abs(F0):
            dlam = F0/vegiot_tensor_dot(A,  A)
            sig_new = sig - dlam*A
            p0_new = p0
            p_new = mean_stress(sig_new)
            q_new = equivalent_stress(sig_new)
            Fnew = M**2*(p_new**2-p0_new*p_new)+q_new**2
        if abs(Fnew) <= FTOL:
            return sig_new,p0_new
        sig = sig_new
        p0 = p0_new
    raise RuntimeError('Convergence not achieved after maximum iteration, exit!')

########################################################################################
def Triaxial(sig, p0, eps1, nsub):
    res = np.zeros((nsub+1,5))
    p = mean_stress(sig)
    q = equivalent_stress(sig)

    res[0] = [0, p, q, p0, 0]
    deps1 = eps1 / nsub
    deps = ti.Vector([deps1, -.5*deps1, -.5*deps1, 0., 0., 0.])
    
    for i in range(nsub):
        sig,p0 = RuttaKunga(sig,p0,deps,ITS)
        p = mean_stress(sig)
        q = equivalent_stress(sig)
        u = q/3. - p + res[0][1]
        
        res[i+1] = [deps1*(i+1), p, q, p0, u]
        
    return res


def RuttaKunga(sig,p0,deps,its):

    ############################## STEP2 ##############################
    p = mean_stress(sig) # mean stress
    devsig = deviatoric_stress(sig) # deviatoric stress tensor sᵢⱼ = σᵢⱼ - pδᵢⱼ
    q = equivalent_stress(sig)

    a0 = M**2*(2*p-p0)/3*k+3*devsig
    F0 = M**2*(p**2-p0*p)+q**2
    
    e0 = N - lam*ti.log(p0) + kappa*ti.log(p0/p) - 1. # initial void ratio
    bulk_modulus = (1. + e0) / kappa * p
    shear_modulus = 3. * bulk_modulus * (1 - 2 * nu) / (2 * (1 + nu))
    dsig_e = delastic_step(deps, shear_modulus, bulk_modulus)
    sig_e = sig + dsig_e
    pe = mean_stress(sig_e)
    qe = equivalent_stress(sig_e)
    F = M**2*(pe**2-p0*pe)+qe**2
    if F <= FTOL:
        return sig_e, p0
    ############################## STEP2 ##############################
    
    
    
    
    alpha = 0
    if F0 < -FTOL and F > FTOL:
        ############################## STEP3 ##############################
        dpe = mean_stress(dsig_e)
        dqe = equivalent_stress(dsig_e)
        alpha = RegulaFalsi(p, q, p0, dpe, dqe, ITS, 0., 1.)
        ############################## STEP2 ##############################
        
    
    
    
    elif F0 <= FTOL and F > FTOL:
        ############################## STEP4 ##############################
        theta = ti.arccos(vegiot_tensor_dot(a0,  dsig_e) / a0.norm() / dsig_e.norm())
        if theta >= LTOL:
            alpha = 1.
        else:
            alpha0, alpha1 = RegulaFalsiNegativePlasticMultiplier(sig,p0,dsig_e,MAXITS,NSUB,0.,1.)
            alpha = RegulaFalsi(p,q,p0,dpe,dqe,ITS,alpha0, alpha1)
        ############################## STEP4 ##############################
    
    
    ############################## STEP5 ##############################
    sig = sig + alpha * dsig_e
    dsig_e = (1. - alpha) * dsig_e
    ############################## STEP5 ##############################
    
    
    
    
    ############################## STEP6 ##############################
    T = 0.
    dT = 1.
    ############################## STEP6 ##############################
    
    
    
    ############################## STEP7 ##############################
    while(T < 1):
    ############################## STEP7 ##############################


        ############################## STEP8 ##############################
        # Substep1 #
        sig1 = sig
        p1 = mean_stress(sig1)
        dev1 = deviatoric_stress(sig1)
        p01 = p0
        bulk_modulus = (1. + e0) / kappa * p
        shear_modulus = 3. * bulk_modulus * (1 - 2 * nu) / (2 * (1 + nu))
        A1 = M**2*(2*p1-p01)/3*k+3*dev1 # ∂f/∂σ
        Kp1 = M**4*p1*p01*(2*p1-p01)*(1+e0)/(lam-kappa)
        tempMat = delastic_step(A1, shear_modulus, bulk_modulus)
        ADB1 = vegiot_tensor_dot(A1,  tempMat)
        dlam1 = dT*vegiot_tensor_dot(A1,  dsig_e)/(ADB1 + Kp1)
        dlam1 = max(dlam1,0)
        dsig1 = dT*dsig_e - dlam1*tempMat
        dkap1 = dlam1*M**2*(2*p1-p01)
        dp01 = p01*(1+e0)/(lam-kappa)*dkap1
        # Substep1 #
        
        # Substep2 #
        sig2 = sig + 0.2*dsig1
        p02 = p0 + 0.2*dp01
        p2 = mean_stress(sig2)
        dev2 = deviatoric_stress(sig2)
        bulk_modulus = (1. + e0) / kappa * p
        shear_modulus = 3. * bulk_modulus * (1 - 2 * nu) / (2 * (1 + nu))
        A2 = M**2*(2*p2-p02)/3*k+3*dev2 # ∂f/∂σ
        Kp2 = M**4*p2*p02*(2*p2-p02)*(1+e0)/(lam-kappa)
        tempMat = delastic_step(A2, shear_modulus, bulk_modulus)
        ADB2 = vegiot_tensor_dot(A2,  tempMat)
        dlam2 = dT*vegiot_tensor_dot(A2,  dsig_e)/(ADB2 + Kp2)
        dlam2 = max(dlam2,0)
        dsig2 = dT*dsig_e - dlam2*tempMat
        dkap2 = dlam2*M**2*(2*p2-p02)
        dp02 = p02*(1+e0)/(lam-kappa)*dkap2
        # Substep2 #
        
        # Substep3 #
        sig3 = sig + 3./40. * dsig1 + 9./40.*dsig2
        p03 = p0 + 3./40. * dp01 + 9./40.*dp02
        p3 = mean_stress(sig3)
        dev3 = deviatoric_stress(sig3)
        bulk_modulus = (1. + e0) / kappa * p
        shear_modulus = 3. * bulk_modulus * (1 - 2 * nu) / (2 * (1 + nu))
        A3 = M**2*(2*p3-p03)/3*k+3*dev3 # ∂f/∂σ
        Kp3 = M**4*p3*p03*(2*p3-p03)*(1+e0)/(lam-kappa)
        tempMat = delastic_step(A3, shear_modulus, bulk_modulus)
        ADB3 = vegiot_tensor_dot(A3,  tempMat)
        dlam3 = dT*vegiot_tensor_dot(A3,  dsig_e)/(ADB3 + Kp3)
        dlam3 = max(dlam3,0)
        dsig3 = dT*dsig_e - dlam3*tempMat
        dkap3 = dlam3*M**2*(2*p3-p03)
        dp03 = p03*(1+e0)/(lam-kappa)*dkap3
        # Substep3 #
        
        # Substep4 #
        sig4 = sig + 0.3 * dsig1 - 0.9 * dsig2 + 6./5. * dsig3
        p04 = p0 + 0.3 * dp01 - 0.9 * dp02 + 6./5. * dp03
        p4 = mean_stress(sig4)
        dev4 = deviatoric_stress(sig4)
        bulk_modulus = (1. + e0) / kappa * p
        shear_modulus = 3. * bulk_modulus * (1 - 2 * nu) / (2 * (1 + nu))
        A4 = M**2*(2*p4-p04)/3*k+3*dev4 # ∂f/∂σ
        Kp4 = M**4*p4*p04*(2*p4-p04)*(1+e0)/(lam-kappa)
        tempMat = delastic_step(A4, shear_modulus, bulk_modulus)
        ADB4 = vegiot_tensor_dot(A4,  tempMat)
        dlam4 = dT*vegiot_tensor_dot(A4,  dsig_e)/(ADB4+ Kp4)
        dlam4 = max(dlam4,0)
        dsig4 = dT*dsig_e - dlam4*tempMat
        dkap4 = dlam4*M**2*(2*p4-p04)
        dp04 = p04*(1+e0)/(lam-kappa)*dkap4
        # Substep4 #
        
        # Substep5 #
        sig5 = sig + 226./729. * dsig1 - 25./27. * dsig2 + 880./729. * dsig3 + 55./729. * dsig4 
        p05 = p0 + 226./729. * dp01 - 25./27. * dp02 + 880./729. * dp03 + 55./729. * dp04 
        p5 = mean_stress(sig5)
        dev5 = deviatoric_stress(sig5)
        bulk_modulus = (1. + e0) / kappa * p
        shear_modulus = 3. * bulk_modulus * (1 - 2 * nu) / (2 * (1 + nu))
        A5 = M**2*(2*p5-p05)/3*k+3*dev5 # ∂f/∂σ
        Kp5 = M**4*p5*p05*(2*p5-p05)*(1+e0)/(lam-kappa)
        tempMat = delastic_step(A5, shear_modulus, bulk_modulus)
        ADB5 = vegiot_tensor_dot(A5,  tempMat)
        dlam5 = dT*vegiot_tensor_dot(A5,  dsig_e)/(ADB5 + Kp5)
        dlam5 = max(dlam5,0)
        dsig5 = dT*dsig_e - dlam5*tempMat
        dkap5 = dlam5*M**2*(2*p5-p05)
        dp05 = p05*(1+e0)/(lam-kappa)*dkap5
        # Substep5 #
        
        # Substep6 #
        sig6 = sig - 181./270. * dsig1 + 5./2. * dsig2 - 226./297. * dsig3 - 91./27. * dsig4 + 189./55. * dsig5
        p06 = p0 - 181./270. * dp01 + 5./2. * dp02 - 226./297. * dp03 - 91./27. * dp04 + 189./55. * dp05 
        p6 = mean_stress(sig6)
        dev6 = deviatoric_stress(sig6)
        bulk_modulus = (1. + e0) / kappa * p
        shear_modulus = 3. * bulk_modulus * (1 - 2 * nu) / (2 * (1 + nu))
        A6 = M**2*(2*p6-p06)/3*k+3*dev6 # ∂f/∂σ
        Kp6 = M**4*p6*p06*(2*p6-p06)*(1+e0)/(lam-kappa)
        tempMat = delastic_step(A6, shear_modulus, bulk_modulus)
        ADB6 = vegiot_tensor_dot(A6,  tempMat)
        dlam6 = dT*vegiot_tensor_dot(A6,  dsig_e)/(ADB6 + Kp6)
        dlam6 = max(dlam6,0)
        dsig6= dT*dsig_e - dlam6*tempMat
        dkap6 = dlam6*M**2*(2*p6-p06)
        dp06 = p06*(1+e0)/(lam-kappa)*dkap6
        # Substep6 #
        ############################## STEP8 ##############################
        
        
        
        ############################## STEP9 ##############################
        sigTemp = sig + 19./216. * dsig1 + 1000./2079. * dsig3 - 125./216. * dsig4 + 81./88. * dsig5 + 5./56. * dsig6
        p0Temp = p0 + 19./216. * dp01 + 1000./2079. * dp03 - 125./216. * dp04 + 81./88. * dp05 + 5./56. * dp06
        ############################## STEP9 ##############################

        
        
        ############################## STEP10 ##############################
        E_sigma = 11./360. * dsig1 - 10./63. * dsig3 + 55./72. * dsig4 - 27./40. * dsig5 + 11./280. * dsig6
        E_p0 = 11./360. * dp01 - 10./63. * dp03 + 55./72. * dp04 - 27./40. * dp05 + 11./280. * dp06
        err = ti.max(ti.max(E_sigma.norm()/sigTemp.norm(),ti.abs(E_p0)/ti.abs(p0Temp)),EPS)
        ############################## STEP10 ##############################
        
        
        
        ############################## STEP11 ##############################
        if err > STOL:
            Q = ti.max(.9*ti.sqrt(STOL/err)**0.2,.1)
            dT = Q*dT
            continue
        ############################## STEP11 ##############################
        
        
        
        ############################## STEP12 ##############################
        sig = sigTemp
        p0 = p0Temp
        ############################## STEP12 ##############################
        
        
        ############################## STEP13 ##############################
        p = mean_stress(sig) # mean stress
        q = equivalent_stress(sig)
        F = M**2*(p**2-p0*p)+q**2
        if F > FTOL:
            sig,p0 = DriftCorrect(sig,p0,ITS)
        ############################## STEP13 ##############################
        
        
        
        ############################## STEP14 ##############################
        Q = max(.9*ti.sqrt(STOL/err),1.)
        dT = ti.max(dTmin, ti.min(Q*dT,1.-T))
        T += dT
        
        ############################## STEP14 ##############################       
    assert(T==1)
    return sig,p0

########################################################################################


res1 = Triaxial(sig1,p0,eps1,nsub)            # triaxial undrained test
res2 = Triaxial(sig2,p0,eps1,nsub)
res3 = Triaxial(sig3,p0,eps1,nsub)

yield_p = np.linspace(0,p0,200)
yield_q = np.sqrt(M**2*yield_p*(p0-yield_p))
CSL = M*yield_p

dat1 = np.array([[0,306,0],[0.001,305,80],[0.002,295,124],[0.009,261,176],[0.024,228,200],[.06,212,210],[.08,206,208]])
dat2 = np.array([[0,98,0],[0.002,103,37],[0.003,110,75],[0.01,130,100],[0.016,142,150],[.02,149,165],[.03,151,174],[.08,176,180]])
dat3 = np.array([[0,33,0],[0.004,36,23],[0.01,48,54],[0.015,55,76],[0.02,71,100],[.03,92,129],[.04,118,148],[.06,130,152],[.08,145,151]])

fig = plt.figure(figsize=(27,6))
fig.suptitle("Triaxial Undrained Tests", fontsize=24)
ax1 = plt.subplot(1,3,1)
ax1.plot(res1[:,0],res1[:,2],'b-',label='Stress-strain relation')
ax1.plot(dat1[:,0],dat1[:,2],'bo--')
ax1.plot(res2[:,0],res2[:,2],'g-')
ax1.plot(dat2[:,0],dat2[:,2],'go--')
ax1.plot(res3[:,0],res3[:,2],'y-')
ax1.plot(dat3[:,0],dat3[:,2],'yo--')
ax1.text(0.03,90,'Dashed - Experimental data',fontsize=12)
ax1.text(0.03,80,'Solid - Numerical prediction',fontsize=12)
ax1.set_xlabel(r'$\varepsilon_1$',fontsize=22)
ax1.set_ylabel(r'$q$ (kPa)',fontsize=22)
ax1.legend(loc='lower right')

ax2 = plt.subplot(1,3,2)
ax2.plot(yield_p,yield_q,'k-',lw=2,label='Initial yield surface')
ax2.plot(yield_p,CSL,'r-',lw=3,label='Critical state line')
ax2.plot(res1[:,1],res1[:,2],'b-',label='Loading path')
ax2.plot(dat1[:,1],dat1[:,2],'bo--')
ax2.plot(res2[:,1],res2[:,2],'g-')
ax2.plot(dat2[:,1],dat2[:,2],'go--')
ax2.plot(res3[:,1],res3[:,2],'y-')
ax2.plot(dat3[:,1],dat3[:,2],'yo--')
ax2.set_xlabel(r"$p'$ (kPa)",fontsize=22)
ax2.set_ylabel(r'$q$ (kPa)',fontsize=22)
ax2.legend(loc='upper left') 

ax3 = plt.subplot(1,3,3)
ax3.plot(res1[:,0],res1[:,4],'b-',label='Excess pore pressure')
ax3.plot(dat1[:,0],dat1[:,2]/3.-dat1[:,1]+dat1[0,1],'bo--')
ax3.plot(res2[:,0],res2[:,4],'g-')
ax3.plot(dat2[:,0],dat2[:,2]/3.-dat2[:,1]+dat2[0,1],'go--')
ax3.plot(res3[:,0],res3[:,4],'y-')
ax3.plot(dat3[:,0],dat3[:,2]/3.-dat3[:,1]+dat3[0,1],'yo--')
ax3.set_xlabel(r'$\varepsilon_1$',fontsize=22)
ax3.set_ylabel(r'$\Delta u$ (kPa)',fontsize=22)
ax3.legend(loc='lower left')
plt.show()

