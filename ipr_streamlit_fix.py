import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.stats import linregress
from shapely.geometry import LineString
import plotly.graph_objects as go

#Functions 

## Fetkovich Method
def fetkovich(data, Pres):
    """
    Function to calculate the IPR data points using Fetkovich Method

    Input:
    - data --> production file input with pwf and q column. The data table is inputted as dataframe.
    - Pres --> reservoir pressure (psi)

    Output:
    - pwf and q data table which calculated with Fetkovich method    
    
    Reference: https://petrowiki.spe.org/Oil_well_performance
    """

    data['dpsqr'] = Pres**2 - data['pwf']**2
    data['log_q'] = np.log(data['q'])
    data['log_dpsqr'] = np.log(data['dpsqr'])

    slope, intercept, r_value, p_value, std_err = linregress(np.array(data['log_dpsqr']),np.array(data['log_q']))
    n = slope
    C = np.exp(intercept)
    
    pwf_fet = np.linspace(Pres, 0, 20).tolist()
    q_fet= C*pow((Pres**2 - np.asarray(pwf_fet)**2),n)

    return (q_fet,pwf_fet)


def straightline (qtest,pwftest,Pres):
    """
    Function to calculate the IPR data points using Straight Line Method

    Inputs:
    - qtest --> single liquid test rate (stb/d)
    - pwftest --> single bottomhole pressure test rate (psi)
    - Pres --> reservoir pressure (psi)

    Output:
    - pwf and q which data table which calculated with Straight Line method    
    
    """
   
    if Pres == pwftest:
        J = 0
    else:
        J = qtest/(Pres - pwftest)
    pwf_sl = np.linspace(Pres, 0, 20).tolist()
    q_sl = J*(Pres - np.asarray(pwf_sl))

    return (q_sl,pwf_sl)


def vogel(qtest,pwftest,Pres,pb): 
    """
    Function to calculate the IPR data points using Vogel Method 

    Inputs:
    - qtest --> single liquid test rate (stb/d)
    - pwftest --> single bottomhole pressure test rate (psi)
    - Pres --> reservoir pressure (psi)
    - pb --> bubble point pressure (psi)

    Output:
    - pwf and q data table which calculated with Vogel method    
    
    Reference: https://petrowiki.spe.org/Oil_well_performance
    """
    if Pres == pwftest:
        J = 0
    else:  
        if pwftest >= pb:
            J = qtest / (Pres - pwftest)
        elif pwftest < pb:
            J = qtest / (Pres - pb + pb/1.8*(1-0.2*pwftest/Pres-0.8*(pwftest/Pres)**2))

    qob = J * (Pres - pb)
    qmax = qob + J*pb/1.8
    pwf_vogel = []
    q_vogel = []
    pwf_vogel = np.linspace(Pres, 0, 20).tolist()
    
    for i in range(len(pwf_vogel)):
        if pwf_vogel[i] < pb:
            q_vogel.append(qob + (qmax-qob)*(1-0.2*pwf_vogel[i]/pb-0.8*(pwf_vogel[i]/pb)**2))
        elif pwf_vogel[i] >= pb:
            q_vogel.append(J * (Pres - pwf_vogel[i]))      
    return (q_vogel,pwf_vogel)


def plotting (q,pwf,title):
    """
    Function to plot the q and pwf data table

    Input:
    - q --> liquid rate data table
    - pwf --> bottom hole pressure data table
    - title --> title of the plot

    Output:
    - pwf and q data table which calculated with Vogel method    
    
    Reference: https://petrowiki.spe.org/Oil_well_performance
    """
    plot_df = pd.DataFrame({'pwf':pwf,'q':q})
    chart = px.line(plot_df, x='q',y='pwf',labels=dict(q="Liquid Rate, STB/D", pwf="Bottom Hole Pressure, psi"),title=title,width=800,height=600)
    
    st.plotly_chart(chart)



def ipr_df (ipr,q,wc,gor):
    """
    Function to calculate the water cut and GOR of each data points

    Input:
    - ipr --> bottom hole pressure data table for IPR
    - q --> liquid rate data table
    - wc --> water cut (%)
    - gor --> gas oil ratio (scf/stb)

    Output:
    - dataframe with qliquid, qwater, qoil, qgas, and IPR 

    """

    df_new = pd.DataFrame({'IPR':ipr,'Qliquid':q})
    df_new['Qwater'] = df_new['Qliquid'] * wc/100
    df_new['Qoil'] = df_new['Qliquid'] * (1-wc/100)
    df_new['Qgas'] = df_new['Qoil'] * gor
    df_new = df_new[['Qliquid','Qwater','Qoil','Qgas','IPR']]

    return(df_new)


def calc_tpr (qo,qg,z,t_avg,p_avg,d,rho_o,ift,sg_g,mo,mg,depth):
    """
    Function to calculate TPR with various inputs using Beggs & Brill Correlation 

    Input:
    - qo --> bottom hole pressure data table for IPR
    - qg --> liquid rate data table
    - z --> water cut (%)
    - t_avg --> average temperature at tubing (F)
    - p_avg --> average pressure at tubing (psi)
    - d --> well ID (ft)
    - rho_o --> oil density (lbm/ft3)
    - ift --> surface tension (dynes/cm)
    - sg_g --> gas specific gravity 
    - mo --> oil viscosity (cp)
    - mg --> gas viscosity (cp)
    - depth --> well depth (ft)

    Output:
    - calculated pwf data for TPR plot

    Reference: Beggs, H. D. (2003). Production Optimization Using Nodal Analysis.

    """

    A = math.pi * d**2 / 4

    #calculate qo and qg at well condition
    Qo = qo*5.615 / (24*60*60)
    Qg = qg*0.0283 * z * (t_avg + 460) / p_avg / (24*60*60)

    usl = Qo/A
    usg = Qg/A
    um = usl + usg

    lamb_l = usl / um
    lamb_g = usg / um

    nfr = (um ** 2) / (32.17 * d)
    nvl = 1.938 * usl * (rho_o / ift) ** 0.25

    L1 = 316 * lamb_l ** 0.302
    L2 = 0.0009252 * lamb_l ** (-2.4684)
    L3 = 0.1 * lamb_l ** (-1.4516)
    L4 = 0.5 * lamb_l ** (-6.738)

    if (lamb_l < 0.01 and nfr < L1) or (lamb_l >= 0.01 and nfr < L2):
        #pattern = "Segregated"
        ylo = (0.98 * lamb_l ** 0.4846) / (nfr ** 0.0868)
        C = (1 - lamb_l) * math.log(0.011 * (lamb_l ** -3.768) * (nvl ** 3.539) * (nfr ** -1.614))
        chi = 1 + C * (math.sin(1.8 * rad) - 0.333 * (math.sin(1.8 * rad)) ** 3)
        yl = ylo * chi
        
    else:
        if (0.01 <= lamb_l < 0.4 and L3 < nfr <= L1) or (lamb_l >= 0.4 and L3 < nfr <= L4):
            #pattern = "Intermittent"
            ylo = (0.845 * lamb_l ** 0.5351) / (nfr ** 0.0173)
            C = (1 - lamb_l) * math.log(2.96 * (lamb_l ** 0.305) * (nvl ** -0.4473) * (nfr ** 0.0978))
            chi = 1 + C * (math.sin(1.8 * rad) - 0.333 * (math.sin(1.8 * rad)) ** 3)
            yl = ylo * chi
            
        else:
            if (lamb_l < 0.4 and nfr >= L1) or (lamb_l >= 0.4 and nfr > L4):
                #pattern = "Distributed"
                ylo = (1.065 * lamb_l ** 0.5824) / (nfr ** 0.0609)
                C = 0
                chi = 1
                yl = ylo * chi
                    
            else:
                #pattern = "Transition"
                A1 = (L3 - nfr) / (L3 - L2)
                B1 = 1 - A1
                yloSeg = (0.98 * lamb_l ** 0.4846) / (nfr ** 0.0868)
                yloInt = (0.845 * lamb_l ** 0.5351) / (nfr ** 0.0173)
                CSeg = (1 - lamb_l) * math.log(0.011 * (lamb_l ** -3.768) * (nvl ** 3.539) * (nfr ** -1.614))
                CInter = (1 - lamb_l) * math.log(2.96 * (lamb_l ** 0.305) * (nvl ** -0.4473) * (nfr ** 0.0978))
                chiSeg = 1 + CSeg * (math.sin(1.8 * rad) - 0.333 * (math.sin(1.8 * rad)) ** 3)
                chiInt = 1 + CInter * (math.sin(1.8 * rad) - 0.333 * (math.sin(1.8 * rad)) ** 3)
                ylSeg = yloSeg * chiSeg
                ylInt = yloInt * chiInt
                yl = A1 * ylSeg + B1 * ylInt

    #mixture properties
    rho_g = 2.7 *sg_g*p_avg/(z*(t_avg+460))

    rhom = lamb_g * rho_g + lamb_l * rho_o
    mum = lamb_g * mg + lamb_l * mo
    NRe = 1488 * rhom * um * d / mum

    fn = 1/4 * (0.0056 + 0.5/NRe**0.32)
    x = lamb_l / yl**2
    S = math.log(x) / (-0.0523 + 3.182 * math.log(x) - 0.8725 * (math.log(x)) ** 2 + 0.01853 * math.log(x) ** 4)
    ftp = math.exp(S)*fn
    dpdz = (2 * ftp * rhom * um ** 2) / (32.17 * d * 144)

    return(dpdz*depth)  

#Streamlit Codes
st.title('IPR - TPR Program')
st.sidebar.write('# IPR Inputs')
ipr_type = st.sidebar.selectbox("Choose IPR Method: ",['Vogel','Fetkovich','Straight Line'])

df = pd.DataFrame({'pwf':[0],'q':[0]})

if ipr_type =='Vogel' or ipr_type=='Straight Line':
    qtest = st.sidebar.number_input("Test Rate (STB/d): ",step=10.0,min_value=0.0,value=1338.0)
    pwftest = st.sidebar.number_input("Test Buttom Hole Pressure (psi): ",step=10.0,min_value=0.0,value=768.0)

elif ipr_type == 'Fetkovich':
    uploaded_file = st.sidebar.file_uploader("Choose Production Test File (with 'pwf' column for Bottomhole Pressure (psi) and 'q' column for Liquid Rate (bopd))")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.DataFrame(columns = ['pwf','q'], data=[[1653,252],[1507,516],[1335,768]])
        
Pres = st.sidebar.number_input("Reservoir Pressure (psi): ",step=10.0,min_value=0.0,value=1734.0)
pb = st.sidebar.number_input("Bubble Point Pressure (psi) ",step=10.0,max_value=float(Pres),min_value=0.0,value=1680.0)
wc = st.sidebar.number_input("Water Cut (%): ",step=1.0,max_value=100.0,min_value=0.0,value=50.0)
gor = st.sidebar.number_input("Gas Oil Ratio (SCF/STB) ",step=10.0,min_value=0.0,value=1000.0)

if ipr_type =='Vogel':
    q, ipr = vogel(qtest,pwftest,Pres,pb)
    #plotting (q,ipr,'Vogel IPR Curve')

elif ipr_type =='Fetkovich':
    q, ipr = fetkovich(df,Pres)
    #plotting (q,ipr,'Fetkovich IPR Curve')

elif ipr_type =='Straight Line':
    q, ipr = straightline(qtest,pwftest,Pres)
    #plotting (q,ipr,'Straight Line IPR Curve')

output_df = ipr_df(ipr,q,wc,gor)

st.sidebar.write('\n')
st.sidebar.write('\n# TPR Inputs')
depth = st.sidebar.number_input("Well Depth (ft): ",step=10.0,min_value=0.0,value=5000.0)
d = st.sidebar.number_input("Well ID (ft): ",step=0.001,min_value=0.000,value=0.1181) #ft
teta = st.sidebar.number_input("Well Deviation (degrees): ",step=1.0,value=0.0)
z = st.sidebar.number_input("Gas Deviation Factor (Z): ",step=0.01,min_value=0.0,max_value = 1.0,value=0.935)
t_avg = st.sidebar.number_input("Average Temperature (F): ",step=1.0,min_value=0.0,value=175.0) #F
p_avg = st.sidebar.number_input("Average Pressure (psi): ",step=1.0,min_value=0.0,value=800.0)
mo = st.sidebar.number_input("Oil Viscosity (cp): ",step=0.01,min_value=0.0,value=2.0) #cp
mg = st.sidebar.number_input("Gas Viscosity (cp): ",step=0.01,min_value=0.0,value=0.1131) #cp 0.1131 #cp
rho_o = st.sidebar.number_input("Oil Density (lbm/ft3): ",step=0.01,min_value=0.0,value=49.9) #cp50
sg_g = st.sidebar.number_input("Gas Gravity : ",step=0.01,min_value=0.0,max_value = 1.0, value=0.709) #cp0.709
ift = st.sidebar.number_input("Surface Tension (dyne/cm): ",step=0.01,min_value=0.0,value=30.0) #cp30

rad = math.radians(teta)

#TPR
tpr = pd.DataFrame()

depth_list = np.linspace(0,depth,num=20).tolist()
tpr['depth'] = depth_list


tpr_list = []

for i in range(1,len(output_df)):
    index = i
    qo = output_df['Qoil'][index]
    qg = output_df['Qgas'][index]

    tpr_list.append(calc_tpr(qo,qg,z,t_avg,p_avg,d,rho_o,ift,sg_g,mo,mg,depth))

tpr_list.insert(0,0)
output_df['TPR'] = tpr_list
#st.write(output_df)


fig = go.Figure(data=go.Scatter(x=output_df['Qliquid'],y=output_df['IPR'], name = 'IPR', mode = 'lines',line_color='red'))
fig.add_traces(go.Scatter(x=output_df['Qliquid'],y=output_df['TPR'], name = 'TPR', mode = 'lines', line_color ='blue'))

line_1 = LineString(np.column_stack((output_df['Qliquid'],output_df['IPR'])))
line_2 = LineString(np.column_stack((output_df['Qliquid'],output_df['TPR'])))
intersection = line_1.intersection(line_2)

x, y = intersection.xy

fig.add_traces(go.Scatter(x=np.array(x[0]), y=np.array(y[0]),
                          mode = 'markers',
                          marker=dict(line=dict(color='black', width = 2),
                                      symbol = 'diamond',
                                      size = 14,
                                      color = 'rgba(255, 255, 0, 0.6)'),
                         name = 'Operating Point'),
)

fig.update_layout(
    xaxis_title="Liquid Rate (STB/D)",
    yaxis_title='Bottom Hole Pressure (psi)',
    width = 900,
    height = 600
)
    
st.plotly_chart(fig)

if ipr_type =='Fetkovich':

    st.write('\n\n## Production Test Data Input')

    df_display = df[['pwf','q']]
    df_display.columns = ['Bottom Hole Pressure (psi)','Liquid Rate (BOPD)']
    st.write(df_display)

st.write('## Operating Point')

st.write('Q = ', x[0], ' STB/D')
st.write('PWF = ',y[0], ' psi')


st.write('\n\n## Table Data')
st.write(output_df)
