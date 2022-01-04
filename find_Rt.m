%
% ======================================================================
%
%    MATLAB code to compute REPRODUCTION NUMBERS of Covid-19
%    written by
%
%        ************************************************
%        *  Janaina P. Zingano and Paulo R. Zingano     *
%        *  Department of Pure and Applied Mathematics  * 
%        *  Institute of Mathematics and Statistics     *
%        *  Universidade Federal do Rio Grande do Sul   *
%        *  Porto Alegre, RS 91509-900, Brazil          *
%        ************************************************
%
%    DATE of last revision: June/25/2020
%
% ======================================================================
%
%    This program computes the daily 7D REPRODUCTION NUMBER of Covid-19
%    given by 
%             Rt = I0(t+3)/I0(t-3)
%    as discussed in the accompanying explanatory notes
%                           doi:10.20944/preprints202006.0370.v1
%    (see p. 7),
%    based on the SEIR ALGORITHM developed by the authors
%    (see the reference above for full details).
%    The program can also be used to generate other reproductive numbers
%    (including R_0 --- see p. 7 of the accompanying notes)
%    or the estimated size of the populations S (susceptible), E (exposed),
%    I (active infected), R (recovered) and D (deceased), by plotting or
%    printing the values of the vector variables S0, E0, I0, R0, D0,
%    as explained in the basic reference text 
%                           doi:10.20944/preprints202006.0370.v1
%
% ======================================================================
%
%    The user must supply the following data:
%    (1) the total population of the region under investigation
%    (2) the FULL name of the ASCII file containing 
%            the corresponding numerical data to be considered
%
%    The numerical data for the region concerned is a list of rows, 
%    each row informing the following 4 numbers 
%     DAY, MONTH no.,TOTAL no.of CASES reported,TOTAL no.of DEATHS reported
%    (separated by any number of blank spaces)
%    for a given collection of CONSECUTIVE days
%
%    Example:
%         29  03   10510   750
%         30  03   11070      950
%         31 03  12150        1200
%
%         01 04           
%                         13200  1500
%         02       04  14000  1900
%         03 04  15100    2700
%         04 04  17000  3500 
%         05 04 19515 4575
%    
%     A realistic example of such a datafile can be found on the site
% https://drive.google.com/drive/folders/16kLxlZyqH-QATOLQI6QWTx7qZnL3IoCP
%     (file "Covid19_data_Brazil.txt") for further illustration.
%     < the full name of this file is: Covid19_data_Brazil.txt >

% ======================================================================
%

% ======================================================================
%   The user may choose to change the values of the following parameters
%   (if desired):
% ======================================================================

T_incub = 5.2;   % <--- average INCUBATION TIME of Covid-19
T_transm = 14;   % <--- average TRANSMISSION TIME of Covid-19

delta = 1/T_incub;    % <--- rate of migration E --> I (SEIR model) 
gamma = 1/T_transm;   % <--- recovery rate (migration I --> R)

rE0 = 0;   % <--- rate of direct migration E --> R (SEIR model)

f_correction = 5;  % <--- estimated correction factor for underreporting

f_suscep = 0.75;   % <--- fraction of the total susceptible population
                   %      that is exposed to infection
                   
% ====================================================================
%    Definitions used in the PHASES 1 and 2 of the SEIR algorithm
%  (for the computation of Rt there is no need to execute PHASE 3):
% ====================================================================

                % <--- value initially used for the LETHALITY RATE
r0 = 0.00253;   %      to determine the values of betas_avg, rrs_avg 
                %      and INITIAL DATA arrays: S0, E0, I0, R0, D0  

beta_values = 1e-3:1e-3:2.50;

r_values = 1e-4:1e-5:0.20;

factor_R = 0.342;    % <--- used to obtain Rb (and also Eb, Ib)
                     %      in the PHASE 1 of the SEIR algorithm

% futuro_local = 10; % <--- use for more regular (C,D) data
% futuro_local = 05;
futuro_local = 02;   % <--- use for less regular (C,D) data 

iter_max = 5;   % <--- number of iterations to obtain beta, r in PHASE 1
                %     (i.e., vector arrays: betas_avg, rrs_avg)

nivel_minimo_C = 100;  % <--- used to define j0_C
nivel_minimo_D = 10;   % <--- used to define j0_D

format compact

% ================================================================
%   Specifying total population size and data values for C, D:
% ================================================================

disp('===============================================================')

population = input('Enter total population: ');

filename = input('Enter name of datafile: ', 's'); 

% ---------------------------------------------------
%    Reading data from input file:
% ---------------------------------------------------

fileID = fopen(filename,'r');     

A = fscanf(fileID,'%g',[1 inf]);

fclose(fileID);  

no_dados_passados = length(A)/4;   % <--- number of data rows
jF = no_dados_passados;

dados_passados = zeros(no_dados_passados, 4);  % <--- (jF x 4) data matrix

for i = 1:no_dados_passados
    dados_passados(i,1) = A(4*(i-1) + 1);      % <--- day
    dados_passados(i,2) = A(4*(i-1) + 2);      % <--- month
    dados_passados(i,3) = A(4*(i-1) + 3);      % <--- total no. of CASES reported
    dados_passados(i,4) = A(4*(i-1) + 4);      % <--- total no. of DEATHS reported
end

dados_passados_resumido = dados_passados;

dados_passados = zeros(jF, 6);

dados_passados(:,1) = dados_passados_resumido(:,1);
dados_passados(:,2) = dados_passados_resumido(:,2);
dados_passados(:,4) = dados_passados_resumido(:,3);
dados_passados(:,6) = dados_passados_resumido(:,4);

dados_passados(1,3) = dados_passados(1,4);
dados_passados(1,5) = dados_passados(1,6);
for i = 2:jF
    dados_passados(i,3) = dados_passados(i,4) - dados_passados(i-1,4);
    dados_passados(i,5) = dados_passados(i,6) - dados_passados(i-1,6);
end    

% ====================================================================
%  END of input data reading and initial processing 
% ====================================================================

% input data are saved onto the 6-column array: dados_passados

  month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', ...
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec' ];
  month_name = [ month_name, month_name, month_name ]; 
  
  last_day_0 = [ 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 ];
  last_day_1 = [ 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 ];
  last_day = [ last_day_0, last_day_1, last_day_1 ];

N = f_suscep*population;  % <--- N: total number of susceptible people

jF = no_dados_passados; 

j0_C = min( find( dados_passados(:,4) >= nivel_minimo_C ) );
distancia_nivel_C = dados_passados(j0_C,4) - nivel_minimo_C;
if j0_C > 1
   j0_a = j0_C - 1;  
   if abs(dados_passados(j0_a,4) - nivel_minimo_C) <= distancia_nivel_C
      j0_C = j0_C - 1;
   end
end   
   
j0_D = min( find( dados_passados(:,6) >= nivel_minimo_D ) );
distancia_nivel_D = dados_passados(j0_D,6) - nivel_minimo_D;
if j0_D > 1
   j0_a = j0_D - 1;  % <--- j0_D alternativo
   if abs(dados_passados(j0_a,4) - nivel_minimo_D) <= distancia_nivel_D
      j0_D = j0_D - 1;
   end
end   

% j0 = max( j0_C, j0_D );
j0 = j0_C;

% Regularizing data values:

C = dados_passados(1:jF,4);
Cr = zeros(jF,1);

Cr(1) = (3*C(1)+C(2)+C(3))/5;
Cr(2) = (C(1)+2*C(2)+C(3)+C(4))/5;
Cr(3:jF-2) = (C(1:jF-4)+C(2:jF-3)+C(3:jF-2)+C(4:jF-1)+C(5:jF))/5;
Cr(jF-1) = (C(jF-3)+C(jF-2)+2*C(jF-1)+C(jF))/5;
Cr(jF) = (C(jF-2)+C(jF-1)+3*C(jF))/5;

D = dados_passados(1:jF,6);
Dr = zeros(jF,1);

Dr(1) = (3*D(1)+D(2)+D(3))/5;
Dr(2) = (D(1)+2*D(2)+D(3)+D(4))/5;
Dr(3:jF-2) = (D(1:jF-4)+D(2:jF-3)+D(3:jF-2)+D(4:jF-1)+D(5:jF))/5;
Dr(jF-1) = (D(jF-3)+D(jF-2)+2*D(jF-1)+D(jF))/5;
Dr(jF) = (D(jF-2)+D(jF-1)+3*D(jF))/5;

Cnr = C;
Dnr = D;

% Specify data (whether regularized or not)
% to be used:

C_check = Cr;
% C_check = Cnr;

D_check = Dr;
% D_check = Dnr;

% Generating initial values Sb, Eb, Ib, Rb, Db:

factor_E = T_incub /(T_incub + T_transm);
factor_I = T_transm/(T_incub + T_transm);

  Cb = C_check(j0) - D_check(j0);
  Cb_corrected = f_correction * Cb;  % <--- Cb_corrected contains
                                     %  the classes E, I, R at database jb 
  Rb = factor_R * Cb_corrected;

  Eb = factor_E * (1 - factor_R)*Cb_corrected;
  Ib = factor_I * (1 - factor_R)*Cb_corrected;
% Observe: Eb + Ib + Rb = Cb_corrected;
  Db = D_check(j0);
  Sb = N - (Eb + Ib + Rb + Db);

% Vectors S0, E0, I0, R0, D0 contain the initial data
% to be used in subsequent runnings:

S0 = NaN*ones(jF,1);
E0 = NaN*ones(jF,1);
I0 = NaN*ones(jF,1);
R0 = NaN*ones(jF,1);
D0 = NaN*ones(jF,1);

C0 = NaN*ones(jF,1);

S0(j0) = Sb;
E0(j0) = Eb;
I0(j0) = Ib;
R0(j0) = Rb;
D0(j0) = Db;

C0(j0) = C_check(j0);

% Vectors betas, rrs contain the parameters  beta, r:

betas = NaN*ones(jF,1);
rrs = NaN*ones(jF,1);

% Generating beta, r at the initial day j0 = j0_C:

t0 = j0;
tF = min( t0+futuro_local, jF );

if t0 == tF
   disp('***** WARNING ***** WARNING ***** WARNING *****')
   disp('Not enough data points to determine parameters')
   disp('Execution will STOP')
   return
end

nh = 10;
h = 1/nh;
t = (t0:h:tF)';
nt = length(t);

S = zeros(nt,1);
E = zeros(nt,1);
I = zeros(nt,1);
R = zeros(nt,1);
D = zeros(nt,1);
  
%
% ================================================================
%
%    Estimating the values of beta, r at the initial day j0:
%
% ================================================================
%

disp('----------------------------------------------------------')
disp('Estimating missing data and parameters ...')
disp('Please wait as this operation may take a few seconds ... ')

r = r0;
rE = rE0;

delta_rE = delta + rE;
gamma_r = gamma + r;

tic
for iter = 1:iter_max
    
  % First, find beta: 
    
    F_sq_C = Inf;

    for beta = beta_values

        S(1) = S0(t0);
        E(1) = E0(t0);
        I(1) = I0(t0);
        R(1) = R0(t0);
        D(1) = D0(t0);

        for i = 1:nt-1
    
            Si = S(i);
            Ei = E(i);
            Ii = I(i);
            Ri = R(i);
            Di = D(i);
    
            k1S = - h*( beta*Si/N*Ii );
            k1E = h*( beta*Si/N*Ii - delta_rE*Ei );
            k1I = h*( delta*Ei - gamma_r*Ii );
            k1R = h*( gamma*Ii + rE*Ei );
            k1D = h*( r*Ii );
    
            k2S = - h*( beta*(Si+k1S)/N*(Ii+k1I) );
            k2E = h*( beta*(Si+k1S)/N*(Ii+k1I) - delta_rE*(Ei+k1E) );
            k2I = h*( delta*(Ei+k1E) - gamma_r*(Ii+k1I) );
            k2R = h*( gamma*(Ii+k1I) + rE*(Ei+k1E) );
            k2D = h*( r*(Ii+k1I) );
    
            S(i+1) = Si + (k1S + k2S)/2;
            E(i+1) = Ei + (k1E + k2E)/2;
            I(i+1) = Ii + (k1I + k2I)/2;
            R(i+1) = Ri + (k1R + k2R)/2;
            D(i+1) = Di + (k1D + k2D)/2;
    
        end
    
        C = (E + I + R)/f_correction + D;
    
        Sum_sq = 0;
        for count = 1:tF-t0
            i = 1 + count*nh;
            Sum_sq = Sum_sq + (C(i) - C_check(t0+count))^2;
        end
        
        if Sum_sq < F_sq_C
           beta_opt = beta;
           F_sq_C = Sum_sq;
        end  
    
    end    

    beta = beta_opt;
    
  % Second, find r:
    
    F_sq_D = Inf;

    for r = r_values

        gamma_r = gamma + r;
        
        for i = 1:nt-1
    
            Si = S(i);
            Ei = E(i);
            Ii = I(i);
            Ri = R(i);
            Di = D(i);
    
            k1S = - h*( beta*Si/N*Ii );
            k1E = h*( beta*Si/N*Ii - delta_rE*Ei );
            k1I = h*( delta*Ei - gamma_r*Ii );
            k1R = h*( gamma*Ii + rE*Ei );
            k1D = h*( r*Ii );
    
            k2S = - h*( beta*(Si+k1S)/N*(Ii+k1I) );
            k2E = h*( beta*(Si+k1S)/N*(Ii+k1I) - delta_rE*(Ei+k1E) );
            k2I = h*( delta*(Ei+k1E) - gamma_r*(Ii+k1I) );
            k2R = h*( gamma*(Ii+k1I) + rE*(Ei+k1E) );
            k2D = h*( r*(Ii+k1I) );
    
            S(i+1) = Si + (k1S + k2S)/2;
            E(i+1) = Ei + (k1E + k2E)/2;
            I(i+1) = Ii + (k1I + k2I)/2;
            R(i+1) = Ri + (k1R + k2R)/2;
            D(i+1) = Di + (k1D + k2D)/2;
    
        end
    
        Sum_sq = 0;
        for count = 1:tF-t0
            i = 1 + count*nh;
            Sum_sq = Sum_sq + (D(i) - D_check(t0+count))^2;
        end
        
        if Sum_sq < F_sq_D
           r_opt = r;
           F_sq_D = Sum_sq;
        end  
    
    end    

    r = r_opt;
  
    gamma_r = gamma + r;
      
end

betas(j0) = beta_opt;
rrs(j0) = r_opt;

% ----------------------------------------
%
%  With the (hopefully good) values of beta, r 
%  found above
%  the solution S,E,I,R,D is computed
%  to provide the new initial values
%  S0,E0,I0,R0,D0
%  at the next day t0+1:
%
% ----------------------------------------

   beta = beta_opt;
   r = r_opt;

   gamma_r = gamma + r;

   for i = 1:nt-1
    
       Si = S(i);
       Ei = E(i);
       Ii = I(i);
       Ri = R(i);
       Di = D(i);
    
       k1S = - h*( beta*Si/N*Ii );
       k1E = h*( beta*Si/N*Ii - delta_rE*Ei );
       k1I = h*( delta*Ei - gamma_r*Ii );
       k1R = h*( gamma*Ii + rE*Ei );
       k1D = h*( r*Ii );
    
       k2S = - h*( beta*(Si+k1S)/N*(Ii+k1I) );
       k2E = h*( beta*(Si+k1S)/N*(Ii+k1I) - delta_rE*(Ei+k1E) );
       k2I = h*( delta*(Ei+k1E) - gamma_r*(Ii+k1I) );
       k2R = h*( gamma*(Ii+k1I) + rE*(Ei+k1E) );
       k2D = h*( r*(Ii+k1I) );
    
       S(i+1) = Si + (k1S + k2S)/2;
       E(i+1) = Ei + (k1E + k2E)/2;
       I(i+1) = Ii + (k1I + k2I)/2;
       R(i+1) = Ri + (k1R + k2R)/2;
       D(i+1) = Di + (k1D + k2D)/2;
    
   end
    
   C = (E + I + R)/f_correction + D;  % <--- C(t) is the TOTAL number of
                                      %      REPORTED CASES up to time t
%
% ================================================================
%
%    Estimating the values of beta, r (and initial data)
%    at the other points t0 = j0+1, j0+2, ..., jF-1, jF:
%
% ================================================================
%

for t0 = j0+1:jF-1

    S0(t0) = S(1+nh);   % <--- the estimated initial data
    E0(t0) = E(1+nh);   %      of S,E,I,R,D
    I0(t0) = I(1+nh);   %      at the present day t0
    R0(t0) = R(1+nh);   %      is obtained from the last
    D0(t0) = D(1+nh);   %      good solution S,E,I,R,D
                        %      found at the previous day t0-1
    C0(t0) = C(1+nh);   % <--- The same is done for the official 
                        % <--- reported number of cases (variable C)
   
    tF = min( t0+futuro_local, jF );

    t = (t0:h:tF)';
    nt = length(t);

    sol_size = nt;

    S = zeros(nt,1);
    E = zeros(nt,1);
    I = zeros(nt,1);
    R = zeros(nt,1);
    D = zeros(nt,1);
    
    S(1) = S0(t0);
    E(1) = E0(t0);
    I(1) = I0(t0);
    R(1) = R0(t0);
    D(1) = D0(t0);
    
%
% ================================================================
%
%    Estimating the values of beta, r at the current point t0:
%
% ================================================================
%

    r = rrs(t0-1);
    rE = rE0;

    delta_rE = delta + rE;
    gamma_r = gamma + r;

    for iter = 1:iter_max
    
    % First, find beta: 
    
        F_sq_C = Inf;

        for beta = beta_values

            for i = 1:nt-1
    
                Si = S(i);
                Ei = E(i);
                Ii = I(i);
                Ri = R(i);
                Di = D(i);
    
                k1S = - h*( beta*Si/N*Ii );
                k1E = h*( beta*Si/N*Ii - delta_rE*Ei );
                k1I = h*( delta*Ei - gamma_r*Ii );
                k1R = h*( gamma*Ii + rE*Ei );
                k1D = h*( r*Ii );
    
                k2S = - h*( beta*(Si+k1S)/N*(Ii+k1I) );
                k2E = h*( beta*(Si+k1S)/N*(Ii+k1I) - delta_rE*(Ei+k1E) );
                k2I = h*( delta*(Ei+k1E) - gamma_r*(Ii+k1I) );
                k2R = h*( gamma*(Ii+k1I) + rE*(Ei+k1E) );
                k2D = h*( r*(Ii+k1I) );
    
                S(i+1) = Si + (k1S + k2S)/2;
                E(i+1) = Ei + (k1E + k2E)/2;
                I(i+1) = Ii + (k1I + k2I)/2;
                R(i+1) = Ri + (k1R + k2R)/2;
                D(i+1) = Di + (k1D + k2D)/2;
    
            end
    
            C = (E + I + R)/f_correction + D;
    
            Sum_sq = 0;
            for count = 1:tF-t0
                i = 1 + count*nh;
                Sum_sq = Sum_sq + (C(i) - C_check(t0+count))^2;
            end
        
            if Sum_sq < F_sq_C
               beta_opt = beta;
               F_sq_C = Sum_sq;
            end  
    
        end    

        beta = beta_opt;
    
    % Second, find r:
    
        F_sq_D = Inf;

        for r = r_values

            gamma_r = gamma + r;
        
            for i = 1:nt-1
    
                Si = S(i);
                Ei = E(i);
                Ii = I(i);
                Ri = R(i);
                Di = D(i);
    
                k1S = - h*( beta*Si/N*Ii );
                k1E = h*( beta*Si/N*Ii - delta_rE*Ei );
                k1I = h*( delta*Ei - gamma_r*Ii );
                k1R = h*( gamma*Ii + rE*Ei );
                k1D = h*( r*Ii );
    
                k2S = - h*( beta*(Si+k1S)/N*(Ii+k1I) );
                k2E = h*( beta*(Si+k1S)/N*(Ii+k1I) - delta_rE*(Ei+k1E) );
                k2I = h*( delta*(Ei+k1E) - gamma_r*(Ii+k1I) );
                k2R = h*( gamma*(Ii+k1I) + rE*(Ei+k1E) );
                k2D = h*( r*(Ii+k1I) );
    
                S(i+1) = Si + (k1S + k2S)/2;
                E(i+1) = Ei + (k1E + k2E)/2;
                I(i+1) = Ii + (k1I + k2I)/2;
                R(i+1) = Ri + (k1R + k2R)/2;
                D(i+1) = Di + (k1D + k2D)/2;
    
            end
    
            Sum_sq = 0;
            for count = 1:tF-t0
                i = 1 + count*nh;
                Sum_sq = Sum_sq + (D(i) - D_check(t0+count))^2;
            end
        
            if Sum_sq < F_sq_D
               r_opt = r;
               F_sq_D = Sum_sq;
            end  
    
        end    

        r = r_opt;
  
        gamma_r = gamma + r;
      
    end

  % Saving parameter values at current t0:
  
    betas(t0) = beta_opt;
    rrs(t0) = r_opt;

  % ----------------------------------------
  %
  %  With the (hopefully good) values of beta, r 
  %  just found (iterations above)
  %  the solution S,E,I,R,D is computed
  %  to provide the new initial values
  %  S0,E0,I0,R0,D0
  %  at the next day t0+1:
  %
  % ----------------------------------------
  
    beta = beta_opt;
    r = r_opt;
    
    gamma_r = gamma + r;

    for i = 1:nt-1
    
        Si = S(i);
        Ei = E(i);
        Ii = I(i);
        Ri = R(i);
        Di = D(i);
    
        k1S = - h*( beta*Si/N*Ii );
        k1E = h*( beta*Si/N*Ii - delta_rE*Ei );
        k1I = h*( delta*Ei - gamma_r*Ii );
        k1R = h*( gamma*Ii + rE*Ei );
        k1D = h*( r*Ii );
    
        k2S = - h*( beta*(Si+k1S)/N*(Ii+k1I) );
        k2E = h*( beta*(Si+k1S)/N*(Ii+k1I) - delta_rE*(Ei+k1E) );
        k2I = h*( delta*(Ei+k1E) - gamma_r*(Ii+k1I) );
        k2R = h*( gamma*(Ii+k1I) + rE*(Ei+k1E) );
        k2D = h*( r*(Ii+k1I) );
    
        S(i+1) = Si + (k1S + k2S)/2;
        E(i+1) = Ei + (k1E + k2E)/2;
        I(i+1) = Ii + (k1I + k2I)/2;
        R(i+1) = Ri + (k1R + k2R)/2;
        D(i+1) = Di + (k1D + k2D)/2;
    
    end
    
    C = (E + I + R)/f_correction + D;

end

% Estimating data and parameter values
% at the last day j = jF:

betas(jF) = betas(jF-1);
rrs(jF) = rrs(jF-1);

S0(jF) = S(1+nh);
E0(jF) = E(1+nh);
I0(jF) = I(1+nh);
R0(jF) = R(1+nh);
D0(jF) = D(1+nh);

C0(jF) = C(1+nh);

% ====================================================================
%
%    END OF PARAMETER/INITIAL DATA determination
%
% ====================================================================

disp('Missing data and parameter determination CONCLUDED!!!!')

% =================================================================
%
%     Plotting the results obtained for the initial data
%     at every initial day t0 from j0 through jF:
%
% =================================================================

for i = j0-10:j0-1
    resto_i = mod(i,10);
    if resto_i == 0
       j0_plot = i;
    end
end

for i = jF+1:jF+10
    resto_i = mod(i,10);
    if resto_i == 0
       jF_plot = i;
    end
end

betas_avg = NaN*ones(jF,1);   % <--- 5-point local averages
for j = j0:jF
    t0 = max( j-2, j0 );
    tF = min( j+2, jF );
    betas_avg(j) = sum(betas(t0:tF))/(tF-t0+1);
end

rrs_avg= NaN*ones(jF,1);   % <--- 5-point local averages
for j = j0:jF
    t0 = max( j-2, j0 );
    tF = min( j+2, jF );
    rrs_avg(j) = sum(rrs(t0:tF))/(tF-t0+1);
end

% ======================================================
%   Computation of reproduction numbers Rtb:
% ======================================================

Rtb = NaN*ones(jF,1);
Rtb(j0:jF) = delta*E0(j0:jF)./I0(j0:jF)./ (rrs_avg(j0:jF) + gamma);
Rtb_max_plot = 1.10*max(Rtb);  

% ======================================================
%   Computation of reproduction numbers Rt:
% ======================================================

Rt = NaN*ones(jF,1);   % <--- reproduction in the 7D-period [ t-3, t+3 ]
Rt(4:jF-3) = I0(7:jF)./I0(1:jF-6);

Rt_max_plot = 1.10*max(Rt);
y_min = 0;
y_max = max(2, Rt_max_plot);

% ======================================================
%   Plotting of Rt values found:
% ======================================================

x0 = j0_plot;
x1 = jF_plot;

% rgb color codes:
light_blue = [220 245 255]/255;
light_yellow = [255 255 220]/255;
light_orange = [255 230 204]/255;

figure(100)
hold off
plot((1:jF), Rt,'-r')
hold on
% legend(' R_t')
%
fill([x0 x1 x1 x0],[y_min y_min 1 1],light_blue)
fill([x0 x1 x1 x0],[1 1 1.5 1.5],light_yellow)
fill([x0 x1 x1 x0],[1.5 1.5 y_max y_max],light_orange)
plot([x0 x1],[1.0 1.0],'-','Color',light_blue)
plot([x0 x1],[1.5 1.5],'-','Color',light_yellow)
if y_max > 2
   plot([x0 x1],[2.0 2.0],'-','Color',light_orange)
   plot([0 jF_plot],[2 2],':r')
end   
%
plot((1:jF), Rt,'-r')
plot([0 jF_plot],[3 3],':k')
plot([0 jF_plot],[1 1],':b')
plot([0 jF_plot],[1.5 1.5],':k')
xlabel('t (days)')
ylabel('R_t')
axis([j0_plot jF_plot y_min y_max])
%
for i = j0_plot:10:jF_plot
    plot([i i],[y_min y_min+0.035],'-k')
end
%
title('Time evolution of 7D reproduction number R_t')

% =======================================================
%
%    Printing out Rt values on screen:
%
% =======================================================

fprintf(' |===========|==========| \n')

format_1 = ' | day/month |    R_t   | \n';

fprintf(format_1)

fprintf(' |-----------|----------| \n')

format_2 = ' |  %02d/%3s   |  %6.4f  | \n';

for j = j0+3:jF-3
    day = dados_passados(j,1);
    month = dados_passados(j,2);
    name_month = month_name(1+(month-1)*3:month*3);
    fprintf(format_2,day,name_month,Rt(j))
end

fprintf(' |===========|==========| \n')

% =======================================================




