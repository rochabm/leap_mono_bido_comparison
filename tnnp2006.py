"""
Standalone test of the vectorized TNNP-2006 port (ported from Cardiax
ten_tusscher2004.cpp -- which is actually the 19-variable 2006 model).

Rush-Larsen on gates 1..12, forward Euler on the rest, exactly as in the C++.
SAC term OFF (default) -- kept as a switch.

This test runs a single cell (N=1) with a periodic stimulus and prints APD90
and peak voltage to confirm the port is correct before wiring into FEniCSx.
"""
import numpy as np

# indices: 0 V,1 Xr1,2 Xr2,3 Xs,4 m,5 h,6 j,7 d,8 f,9 f2,10 fCass,
#          11 s,12 r,13 Ca_i,14 Ca_SR,15 Ca_ss,16 R_prime,17 Na_i,18 K_i
NVAR = 19
GATE_IDX = list(range(1, 13))   # RL-integrated
CONC_IDX = [13, 14, 15, 16, 17, 18]  # forward-Euler

def init_state(N, celltype="EPI"):
    y = np.zeros((NVAR, N))
    v0 = [-85.23,0.00621,0.4712,0.0095,0.00172,0.7444,0.7045,3.373e-5,
          0.7888,0.9755,0.9953,0.999998,2.42e-8,0.000126,3.64,0.00036,
          0.9073,8.604,136.89]
    for i in range(NVAR):
        y[i, :] = v0[i]
    return y

def rhs(y, i_stim, celltype="EPI", use_sac=False, dt_rl=0.05):
    """
    Returns (dV_dt array, gate_inf dict, gate_tau dict, dconc dict).
    We separate RL gates from Euler concentrations so the integrator can
    apply the exact-exponential update for gates and Euler for the rest,
    matching the Cardiax scheme.
    """
    V   = y[0]; Xr1=y[1]; Xr2=y[2]; Xs=y[3]; m=y[4]; h=y[5]; j=y[6]
    d   = y[7]; f=y[8]; f2=y[9]; fCass=y[10]; s=y[11]; r=y[12]
    Ca_i=y[13]; Ca_SR=y[14]; Ca_ss=y[15]; R_prime=y[16]; Na_i=y[17]; K_i=y[18]

    R=8314.472; T=310.0; F=96485.3415; Cm=0.185
    V_c=0.016404
    RToF=(R*T)/F; VcF=V_c*F; VFoRT=(V*F)/(R*T)

    Ko=5.4; Nao=140.0; Cao=2.0; P_kna=0.03
    Km_Nai=87.5; K_mk=1.0; P_NaK=2.724; K_mNa=40.0
    V_sr=0.001094; V_ss=5.468e-05; V_rel=0.102
    k1_prime=0.15; max_sr=2.5; min_sr=1.0; EC=1.5; k3=0.06
    k2_prime2=0.045; k4=0.005
    Buf_c=0.2; K_buf_c=0.001; K_buf_sr=0.3; Buf_sr=10.0
    Buf_ss=0.4; K_buf_ss=0.00025
    Vmax_up=0.006375; K_up=0.00025; V_leak=0.00036; V_xfer=0.0038

    EK  = RToF*np.log(Ko/K_i)
    EKs = RToF*np.log((Ko+P_kna*Nao)/(K_i+P_kna*Na_i))
    ENa = RToF*np.log(Nao/Na_i)
    ECa = 0.5*RToF*np.log(Cao/Ca_i)

    beta_K1=((3.0*np.exp(0.0002*((V-EK)+100.0)))+np.exp(0.1*((V-EK)-10.0)))/(1.0+np.exp(-0.5*(V-EK)))
    alpha_K1=0.1/(1.0+np.exp(0.06*((V-EK)-200.0)))
    xK1_inf=alpha_K1/(alpha_K1+beta_K1)
    g_K1=5.405
    IK1=g_K1*xK1_inf*(V-EK)

    g_to = 0.294 if celltype!="ENDO" else 0.073
    Ito=g_to*r*s*(V-EK)

    g_Kr=0.153
    IKr=g_Kr*Xr1*Xr2*(V-EK)*np.sqrt(Ko/5.4)

    if celltype in ("EPI","ENDO"): g_Ks=0.245
    else: g_Ks=0.062
    IKs=g_Ks*(Xs**2)*(V-EKs)

    g_CaL=3.98e-05
    denom = np.where(np.abs(V-15.0)>1e-5, (np.exp((2.0*(V-15.0)*F)/(R*T))-1.0), 1.0)
    ICaL_reg = (((g_CaL*d*f*f2*fCass*4.0*(V-15.0)*(F**2))/(R*T))*
                ((0.25*Ca_ss*np.exp((2.0*(V-15.0)*F)/(R*T)))-Cao))/denom
    ICaL_sing = g_CaL*d*f*f2*fCass*2.0*F*(0.25*Ca_ss-Cao)
    ICaL = np.where(np.abs(V-15.0)>1e-5, ICaL_reg, ICaL_sing)

    INaK=((((P_NaK*Ko)/(Ko+K_mk))*Na_i)/(Na_i+K_mNa))/(1.0+0.1245*np.exp(-0.1*V*F/(R*T))+0.0353*np.exp(-V*F/(R*T)))

    g_Na=14.838
    INa=g_Na*(m**3)*h*j*(V-ENa)

    g_bna=0.00029
    IbNa=g_bna*(V-ENa)

    alpha=2.5; gamma=0.35; K_sat=0.1; Km_Ca=1.38; K_NaCa=1000.0
    INaCa=(K_NaCa*((np.exp(gamma*VFoRT)*(Na_i**3)*Cao)-(np.exp((gamma-1.0)*VFoRT)*(Nao**3)*Ca_i*alpha)))/((Km_Nai**3+Nao**3)*(Km_Ca+Cao)*(1.0+K_sat*np.exp((gamma-1.0)*VFoRT)))

    g_bca=0.000592
    IbCa=g_bca*(V-ECa)

    g_pK=0.0146
    IpK=(g_pK*(V-EK))/(1.0+np.exp((25.0-V)/5.98))

    K_pCa=0.0005; g_pCa=0.1238
    IpCa=(g_pCa*Ca_i)/(Ca_i+K_pCa)

    Istim = i_stim

    # gate steady-states / taus
    xr1_inf=1.0/(1.0+np.exp((-26.0-V)/7.0))
    alpha_xr1=450.0/(1.0+np.exp((-45.0-V)/10.0)); beta_xr1=6.0/(1.0+np.exp((V+30.0)/11.5))
    tau_xr1=alpha_xr1*beta_xr1
    xr2_inf=1.0/(1.0+np.exp((V+88.0)/24.0))
    alpha_xr2=3.0/(1.0+np.exp((-60.0-V)/20.0)); beta_xr2=1.12/(1.0+np.exp((V-60.0)/20.0))
    tau_xr2=alpha_xr2*beta_xr2
    xs_inf=1.0/(1.0+np.exp((-5.0-V)/14.0))
    alpha_xs=1400.0/np.sqrt(1.0+np.exp((5.0-V)/6.0)); beta_xs=1.0/(1.0+np.exp((V-35.0)/15.0))
    tau_xs=alpha_xs*beta_xs+80.0
    m_inf=1.0/(1.0+np.exp((-56.86-V)/9.03))**2
    alpha_m=1.0/(1.0+np.exp((-60.0-V)/5.0)); beta_m=(0.1/(1.0+np.exp((V+35.0)/5.)))+(0.1/(1.0+np.exp((V-50.0)/200.)))
    tau_m=alpha_m*beta_m
    h_inf=1.0/(1.0+np.exp((V+71.55)/7.43))**2
    alpha_h=np.where(V<-40.0, 0.057*np.exp(-(V+80.0)/6.8), 0.0)
    beta_h=np.where(V<-40.0, 2.7*np.exp(0.079*V)+310000.0*np.exp(0.3485*V), 0.77/(0.13*(1.0+np.exp((V+10.66)/(-11.1)))))
    tau_h=1.0/(alpha_h+beta_h)
    j_inf=1.0/(1.0+np.exp((V+71.55)/7.43))**2
    alpha_j=np.where(V<-40.0, (((-25428.0*np.exp(0.2444*V))-(6.948e-06*np.exp(-0.04391*V)))*(V+37.78))/(1.0+np.exp(0.311*(V+79.23))), 0.0)
    beta_j=np.where(V<-40.0, (0.02424*np.exp(-0.01052*V))/(1.0+np.exp(-0.1378*(V+40.14))), (0.6*np.exp(0.057*V))/(1.0+np.exp(-0.1*(V+32.0))))
    tau_j=1.0/(alpha_j+beta_j)
    d_inf=1.0/(1.0+np.exp((-8.0-V)/7.5))
    alpha_d=(1.4/(1.0+np.exp((-35.0-V)/13.0)))+0.25; beta_d=1.4/(1.0+np.exp((V+5.0)/5.0))
    gamma_d=1.0/(1.0+np.exp((50.0-V)/20.0)); tau_d=alpha_d*beta_d+gamma_d
    f_inf=1.0/(1.0+np.exp((V+20.0)/7.0))
    tau_f=(1102.5*np.exp(-(V+27.0)**2/225.0))+(200.0/(1.0+np.exp((13.0-V)/10.0)))+(180.0/(1.0+np.exp((V+30.0)/10.0)))+20.0
    f2_inf=(0.67/(1.0+np.exp((V+35.0)/7.0)))+0.33
    tau_f2=(562.0*np.exp(-(V+27.0)**2/240.0))+(31.0/(1.0+np.exp((25.0-V)/10.0)))+(80.0/(1.0+np.exp((V+30.0)/10.)))
    fCass_inf=(0.6/(1.0+(Ca_ss/0.05)**2))+0.4
    tau_fCass=(80.0/(1.0+(Ca_ss/0.05)**2))+2.0
    s_inf=1.0/(1.0+np.exp((V+20.0)/5.0))
    tau_s=(85.0*np.exp(-(V+45.0)**2/320.))+(5./(1.+np.exp((V-20.0)/5.0)))+3.0
    r_inf=1.0/(1.0+np.exp((20.0-V)/6.0))
    tau_r=(9.5*np.exp(-(V+40.0)**2/1800.0))+0.8

    kcasr=max_sr-((max_sr-min_sr)/(1.0+(EC/Ca_SR)**2))
    k1=k1_prime/kcasr
    O=(k1*(Ca_ss**2)*R_prime)/(k3+(k1*(Ca_ss**2)))
    Irel=V_rel*O*(Ca_SR-Ca_ss)
    Iup=Vmax_up/(1.0+((K_up**2)/(Ca_i**2)))
    Ileak=V_leak*(Ca_SR-Ca_i)
    Ixfer=V_xfer*(Ca_ss-Ca_i)
    k2=k2_prime2*kcasr
    Ca_i_bufc=1.0/(1.0+((Buf_c*K_buf_c)/(Ca_i+K_buf_c)**2))
    Ca_sr_bufsr=1.0/(1.0+((Buf_sr*K_buf_sr)/(Ca_SR+K_buf_sr)**2))
    Ca_ss_bufss=1.0/(1.0+((Buf_ss*K_buf_ss)/(Ca_ss+K_buf_ss)**2))

    dV=-(IK1+Ito+IKr+IKs+ICaL+INaK+INa+IbNa+INaCa+IbCa+IpK+IpCa+Istim)
    if use_sac:
        Gs=0.075; Es=-50; lam=1.05
        Isac=np.where(lam>1.0, Gs*(lam-1.0)*(V-Es), 0.0)
        dV=dV-Isac

    dR_prime=((-k2)*Ca_ss*R_prime)+(k4*(1.0-R_prime))
    dCa_i=Ca_i_bufc*(((((Ileak-Iup)*V_sr)/V_c)+Ixfer)-((((IbCa+IpCa)-(2*INaCa))*Cm)/(2*VcF)))
    dCa_SR=Ca_sr_bufsr*(Iup-(Irel+Ileak))
    dCa_ss=Ca_ss_bufss*((((-ICaL*Cm)/(2*V_ss*F))+((Irel*V_sr)/V_ss))-((Ixfer*V_c)/V_ss))
    dNa_i=((-(INa+IbNa+(3*INaK)+(3*INaCa)))/(VcF))*Cm
    dK_i=((-((IK1+Ito+IKr+IKs+IpK+Istim)-(2*INaK)))/(VcF))*Cm

    ginf={1:xr1_inf,2:xr2_inf,3:xs_inf,4:m_inf,5:h_inf,6:j_inf,
          7:d_inf,8:f_inf,9:f2_inf,10:fCass_inf,11:s_inf,12:r_inf}
    gtau={1:tau_xr1,2:tau_xr2,3:tau_xs,4:tau_m,5:tau_h,6:tau_j,
          7:tau_d,8:tau_f,9:tau_f2,10:tau_fCass,11:tau_s,12:tau_r}
    dconc={13:dCa_i,14:dCa_SR,15:dCa_ss,16:dR_prime,17:dNa_i,18:dK_i}
    return dV, ginf, gtau, dconc

def step(y, i_stim, dt, celltype="EPI", use_sac=False):
    """One explicit step: Euler for V and concentrations, Rush-Larsen for gates."""
    dV, ginf, gtau, dconc = rhs(y, i_stim, celltype, use_sac, dt_rl=dt)
    ynew = y.copy()
    ynew[0] = y[0] + dt*dV
    # Overflow guard only: clip V to a range WIDER than any physiological AP
    # (a real TNNP AP lives in [-90, +45]). This never affects a well-calibrated
    # run; it merely stops exp() overflow warnings if the gain is set absurdly
    # high, so such runs fail visibly (peakV pinned near +80) instead of NaN-ing.
    np.clip(ynew[0], -120.0, 80.0, out=ynew[0])
    for i in GATE_IDX:
        ynew[i] = ginf[i] + (y[i]-ginf[i])*np.exp(-dt/gtau[i])
    for i in CONC_IDX:
        ynew[i] = y[i] + dt*dconc[i]
    return ynew

if __name__ == "__main__":
    dt=0.02; T=400.0
    nsteps=int(T/dt)
    y=init_state(1,"EPI")
    stim_start=1.0; stim_dur=1.0; stim_amp=-52.0
    Vtrace=np.empty(nsteps); tt=np.empty(nsteps)
    for k in range(nsteps):
        t=k*dt
        istim = stim_amp if (stim_start<=t<=stim_start+stim_dur) else 0.0
        y=step(y, istim, dt, "EPI", use_sac=False)
        Vtrace[k]=y[0,0]; tt[k]=t
    Vpeak=Vtrace.max()
    Vrest=Vtrace[0]
    # APD90
    Vmin=Vtrace.min(); amp=Vpeak-Vrest
    thr=Vpeak-0.9*amp
    above=np.where(Vtrace>=thr)[0]
    if len(above)>0:
        apd90=(above[-1]-above[0])*dt
    else:
        apd90=float('nan')
    print(f"Resting V   = {Vrest:.2f} mV")
    print(f"Peak V      = {Vpeak:.2f} mV")
    print(f"APD90       = {apd90:.1f} ms")
    print("Expected: rest ~ -85 mV, peak ~ +35..+45 mV, APD90 ~ 260..320 ms (EPI)")
