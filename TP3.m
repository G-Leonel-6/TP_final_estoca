clear all; close all

%==============EJERCICIO 1==============================%

%(a) Genere un proceso Bernoulli de largo N = 1000 que emule una secuencia de bits equiprobables. Con esta secuencia, 
%defina un proceso de sımbolos antipodales x(n) con el mapeo de la Ec. (1). 
%Grafique las primeras 100 muestras de este proceso.

%defino el proceso de bernoulli con N=1000

N=1000;
p = 1/2;

b = binornd(1, p, 1, N);

x1 = 2*b - ones(1,N);



%grafico del proceso x
figure();
stem(0:N-1, x1, 'filled');
title("Proceso $x(n)$", 'Interpreter', 'Latex');
ylabel("$x(n)$", 'Interpreter', 'Latex');
ylim([-1.2 1.2]);
xlabel("$n$",'Interpreter', 'Latex');
xlim([-1 101]);
grid on;


% (b) Siguiendo el modelo simplificado de la Figura 2, suponga un canal equivalente en el dominio
% discreto h(n) = {0.5, 1, 0.2, 0.1, 0.05, 0.01} y ruido v(n) ∼ N (0, σ2
% v = 0.002). Aplique la
% secuencia de s´ımbolos x(n) al canal equivalente y obtenga el proceso discreto y(n) la salida
% del mismo. Grafique las primeras 100 muestras de este proceso.

%defino h(n)
h = [0.5 1 0.2 0.1 0.05 0.01];

%defino el ruido gaussiano
var_v = 0.002; %varianza del ruido
v1 = normrnd(0, sqrt(var_v), 1, N);
y1 = filter(h, 1, x1) + v1;


figure();
stem(0:N-1, y1, 'filled');
title("Proceso $y(n)$", 'Interpreter', 'Latex');
ylabel("$y(n)$", 'Interpreter', 'Latex');
xlabel("$n$",'Interpreter', 'Latex');
xlim([-1 101]);
grid on;

% (c) Estime la funci´on de autocorrelaci´on y la PSD (implementando el estimador de Welch con
% los par´ametros que considere apropiados) de los procesos de entrada x(n) y salida del canal
% y(n). Grafique cada caso (para las PSDs considere adem´as las respuestas te´oricas).


% Parámetros del método de Welch
M = 80; % Ancho del segmento
overlap = M/2; % Solapamiento del 50%
NFFT = 1200;
%estimo la psd de la señal x(n) utilizando el metodo welch

S_X = metodo_Welch(x1, M,overlap, NFFT);
ws = linspace(0,2*pi,NFFT);


% Graficar la PSD
figure();
hold on
plot(ws,10*log10(S_X), "red", LineWidth = 1.5); 
title("PSD del proceso $x(n)$", 'Interpreter', 'Latex');
xlabel('$\omega$', 'Interpreter', 'latex');
ylabel('$S_X(\omega)$', 'Interpreter', 'latex');
grid on;
legend('PSD metodo Welch', 'Location', 'south');
xlim([0 pi]);

%grafico de la PSD de y
S_Y = metodo_Welch(y1, M,overlap, NFFT);
ws = linspace(0,2*pi,NFFT);


% Graficar la PSD
figure();
hold on
plot(ws,10*log10(S_Y), "red", LineWidth=1.5); 
title("PSD del proceso $y(n)$", 'Interpreter', 'Latex');
xlabel('$\omega$', 'Interpreter', 'latex');
ylabel('$S_Y(\omega)$', 'Interpreter', 'latex');
legend('PSD metodo Welch', 'Location', 'south');
grid on;
xlim([0 pi]);


%calculo la autocorrelacion de x e y

Rx = ifft(S_X, 'symmetric');
Ry = ifft(S_Y, 'symmetric');

figure();
plot(Rx, LineWidth = 1.5);
xlim([0 100]);
title("Autocorrelacion de $y(n)$", 'Interpreter', 'latex');
xlabel("$k$", 'Interpreter', 'latex');
ylabel("$R_y(k)$", 'Interpreter', 'latex');
grid on;


figure();
plot(Ry, LineWidth = 1.5);
title("Autocorrelacion de $y(n)$", 'Interpreter', 'latex');
xlabel("$k$", 'Interpreter', 'latex');
ylabel("$R_y(k)$", 'Interpreter', 'latex');
xlim([0 100]);
grid on;


%===========EJERCICIO 2============================================%
% (b) Utilice el algoritmo implementado para ecualizar el sistema, generando los datos con la
% estad´ıstica del Ejercicio 1. Estime la curva de aprendizaje J(n) (con 500 realizaciones)
% en funci´on de diferentes retardos, suponga D = 1, 2, 3, ..., 9. Considere un largo M = 8
% para el filtro, un paso µ = 0.05 y condiciones iniciales nulas w(0) = 0. Grafique las J(n)
% resultantes de cada retardo superpuestas en un mismo gr´afico. Tambi´en haga un gr´afico
% de J(∞) (puede tomar el promedio en la zona estacionaria de la curva J(n)) en funci´on
% del retardo. Determine el retardo ´optimo que produce el menor error.

%defino los parametros del filtro LMS
mu = 0.05;
D = 1:9;
L = 500;
M = 8;

%genero el proceso x 
N=1000;
b = binornd(1, 1/2, L, N);
x = 2*b - ones(1,N);

%genero el canal
h = [0.5 1 0.2 0.1 0.05 0.01]; %es un filtro de tipo MA, estos coef son el numerador del filtro

%defino el ruido gaussiano
var_v = 0.002;
v = normrnd(0, sqrt(var_v), L, N);

y = (filter(h, 1, x'))' + v;

J = zeros(9, N-M+1);

J_inf = zeros(1, 9);

for i = 1:9
   
    %genero la funcion deseada d(n) = x(n-D)
    d = [zeros(L, D(i)) x];
    
    [z, J(i,:), w] = LMS(y, mu, M, d);
    
    %calculo el error estacionario calculando el promedio de la parte
    %estacionaria de J
    J_inf(i) = mean(J(i, 250:end));
end

figure();
hold on;

for i = 1:length(D)
   plot(J(i,:), 'DisplayName', ['D = ', num2str(D(i))]); 
end
title("Funcion de costos $J(n)$", 'Interpreter', 'latex');
xlabel("$n$", 'Interpreter', 'latex');
ylabel("$J(n)$", 'Interpreter', 'latex');
legend('Location', 'best');
grid on;


figure();
plot(D, J_inf, LineWidth = 1.5);
title("Error estacionario $J(\infty)$ en funcion de D", 'Interpreter', 'latex');
xlabel("$D$", 'Interpreter', 'latex');
grid on;

%el retardo optimo es D=6 segun la observacion de la fig anterior
%grafico d(n) con z(n) superpuestas (con d perteneciente al ej 1)

mu = 0.05;
D = 6;
L = 1;
M = 8;

d1 = [zeros(1, D) x1];

[z1, J1, w] = LMS(y1, mu, M, d1);

figure();
stem(d1(M:end-D));
hold on;
stem(z1);
title("$z(n)$ y $x(n-D)$ con $D=6$", 'Interpreter', 'latex');
xlabel("$n$", 'Interpreter', 'latex')
legend("$d(n)=x(n-D)$", "$z(n)$", 'Interpreter', 'latex');
xlim([0 200]);
grid on;


%calculo la respuesta impulsiva de la cascada h*w
h_c = conv(h,w);

figure();
stem(h_c, 'filled');
title("Respuesta impulsiva de la cascada $h(n)*w(n)$", 'Interpreter', 'latex');
xlabel("$n$", 'Interpreter', 'latex');
ylabel("$h_c(n)$", 'Interpreter', 'latex');
grid on;

H_c = fft(h_c, 1024);
w = linspace(0, 2*pi, 1024);

figure();
plot(w, abs(H_c), LineWidth = 1.5);
title("Respuesta en frecuencia de la cascada");
xlabel("$\omega$", 'Interpreter', 'latex');
ylabel("$H_c(\omega)$", 'Interpreter', 'latex');
xlim([0 pi]);
grid on;



%===========EJERCICIO 3============================================%
%a)
%generamos una secuencia de N=10000 de simbolos antipodales xO=-1, x1=1
N=10000;

b = binornd(1, p, 1, N);

x = 2*b - ones(1,N);

%generamos ruido
var_v = [0.1 0.2 0.3]; %varianza del ruido

u = zeros(3, N);

for i = 1:3
    %ruido
    v = normrnd(0, sqrt(var_v(i)), 1, N);
    
    %señal con ruido
    u(i,:) = x + v;
    
    %obtenemos la densidad de probabilidad
    n = linspace(-3, 3, 1000);
    
    pd1 = makedist('Normal','mu',1,'sigma',sqrt(var_v(i)));
    pd2 = makedist('Normal','mu',-1,'sigma',sqrt(var_v(i)));
    
    n1 = pdf(pd1, n);
    n2 = pdf(pd2, n);
    
    figure();
    hold on;
    histogram(u(i,:), 50,'Normalization', 'pdf', 'DisplayName', "Histograma de u");
    title(["Histograma de $u(n)$ con ruido de varianza $\sigma_v^2 = $", num2str(var_v(i))], 'Interpreter', 'latex');
    plot(n, 0.5*n1, 'DisplayName',['$\mathcal{N}(\mu=1, \sigma_v^2 =$', num2str(var_v(i)), '$)$'], LineWidth=1.5);
    plot(n, 0.5*n2,'DisplayName', ['$\mathcal{N}(\mu=-1, \sigma_v^2 =$', num2str(var_v(i)), '$)$'], LineWidth=1.5);
    xline(0, 'HandleVisibility','off');
    legend('Location', 'North', 'Interpreter', 'latex');
    grid on;
end

%b)
%generamos una secuencia de N=100000
N=100000;
%bernoulli equiprobable
b = binornd(1, p, 1, N);

%simbolos antipodales x0 = -1, x1=1 
s1 = 2*b - ones(1,N);

%simbolos antipodales x0 = -3, x1=3
s2 = s1*3;

%generamos ruido de varianza 0.5
var_v = 0.5;
v = normrnd(0, sqrt(var_v), 1, N);

%señales de entrada al clasificador 
u1 = s1+v;
u2 = s2+v;

%grafico el histograma para la señal de simbolos antipodales -1, 1
n = linspace(-3, 3, 1000);
pd1 = makedist('Normal','mu',1,'sigma',sqrt(var_v));
pd2 = makedist('Normal','mu',-1,'sigma',sqrt(var_v));
n1 = pdf(pd1, n);
n2 = pdf(pd2, n);


figure();
hold on;
histogram(u1, 50,'Normalization', 'pdf', 'DisplayName', "Histograma de u");
title("Histograma de $u(n)$ con ruido de varianza $\sigma_v^2 = 0.5$", 'Interpreter', 'latex');
plot(n, 0.5*n1, 'DisplayName','$\mathcal{N}(\mu=1, \sigma_v^2 = 0.5)$', LineWidth=1.5);
plot(n, 0.5*n2,'DisplayName', '$\mathcal{N}(\mu=-1, \sigma_v^2 = 0.5)$', LineWidth=1.5);
xline(0, 'HandleVisibility','off');
legend('Location', 'North', 'Interpreter', 'latex');
grid on;


%grafico el histograma para la señal de simbolos antipodales -3, 3
n = linspace(-6, 6, 1000);
pd1 = makedist('Normal','mu',3,'sigma',sqrt(var_v));
pd2 = makedist('Normal','mu',-3,'sigma',sqrt(var_v));
n1 = pdf(pd1, n);
n2 = pdf(pd2, n);


figure();
hold on;
histogram(u2, 50,'Normalization', 'pdf', 'DisplayName', "Histograma de u");
title("Histograma de $u(n)$ con ruido de varianza $\sigma_v^2 = 0.5$", 'Interpreter', 'latex');
plot(n, 0.5*n1, 'DisplayName','$\mathcal{N}(\mu=3, \sigma_v^2 = 0.5)$', LineWidth=1.5);
plot(n, 0.5*n2,'DisplayName', '$\mathcal{N}(\mu=-3, \sigma_v^2 = 0.5)$', LineWidth=1.5);
xline(0, 'HandleVisibility','off');
legend('Location', 'North', 'Interpreter', 'latex');
grid on;


%calculamos la probabilidad de error en cada caso
%como se trata de simbolos antipodales, la probabilidad de error es
%Pe=Q(a/sigma_v)

%para el primer caso
Pe1 = qfunc(1/sqrt(var_v));
Pe2 = qfunc(3/sqrt(var_v));

%ahora calculamos la tasa de error para cada caso
%la tasa de error SER se define como la relacion entre el numero de
%simbolos con error sobre la cantidad de simbolos totales

[~, SER1] = clasificador_ML(u1, s1, 1, var_v);
[~, SER2] = clasificador_ML(u2, s2, 3, var_v);

%c)

%genero una secuencia de u(n) de largo 10000
N = 10000;

b = binornd(1, p, 1, N);

%simbolos antipodales x0 = -1, x1=1 
s1 = 2*b - ones(1,N);

%genero un vector de varianzas para el ruido
var_v = 5:-0.2:0.2;

SER = zeros(1, length(var_v));%TASA DE ERROR
Pe = zeros(1, length(var_v)); %PROBABILIDAD DE ERROR
SNR = zeros(1, length(var_v)); %RELACION SEÑAL RUIDO
for i = 1:length(var_v)
    v = normrnd(0, sqrt(var_v(i)), 1, N);
    u = s1+v;
    [~, SER(i)] = clasificador_ML(u, s1, 1, var_v(i));
    Pe(i) = qfunc(1/sqrt(var_v(i)));
    SNR(i) = 1/var_v(i);
end

%grafico en escala logaritmica
figure();
loglog(SNR, SER, 'DisplayName', "SER", LineWidth=1.5);
hold on;
loglog(SNR, Pe, 'DisplayName', "Probabilidad de error(teorico)", LineWidth=1.5);
title("Probabilidad de error vs SNR");
xlabel("$log_{10}(SNR)$", 'Interpreter', 'latex');
ylabel("$log_{10}(SER)$", 'Interpreter', 'latex');
legend;
grid on;


%=========================Funciones======================================%

%estima la PSD de una señal mediante el metodo Welch. Recibe como
%parametros, ademas de la señal, el largo de la señal,
%el numero de segmentos M y el solapamiento
function Sx = metodo_Welch(x, M, overlap, nfft) 

    N = length(x); % Longitud de la señal
    ventana = hamming(M);
    V = sum(abs(ventana).^2)/M; 
    n_segmentos = floor(N/(overlap)); % Corregido el cálculo del número de segmentos

    Sx = zeros(1, nfft);

    for i = 1:n_segmentos-1
        inicio = (i-1)*floor(overlap)+1;
        fin = inicio + M - 1;

        seg = x(inicio:fin);
        x_i = ventana' .* seg;
        X_i = fft(x_i, nfft);
        Sx =  Sx + ((abs(X_i)).^2);
    
    end

    Sx = Sx/(M*V*n_segmentos);

end


%Ecualizador LMS
%M: orden del filtro, x: señal de entrada,
%mu:tamaño del paso, D:retardo, 
%nuestra señal esperada es d(n) = x(n-D)
function [z, J, w] = LMS(x, mu, M, d)
    %obtengo el tamaño del proceso de entrada x
    s = size(x);
    N = s(2); %largo de cada realizacion
    L = s(1); %cantidad de realizaciones
    
    %genero una matriz de coeficientes de tamaño LxM
    %inicializo los coeficientes en 0 (condiciones iniciales nulas)
    w = zeros(L, M); 
    
    %genero un vector de salidas
    z = zeros(L, N-M+1);
    
    %funcion de costos
    J = zeros(1, length(z));
    
    for i = 1:N-M+1
        
       n = i+M-1; %instante actual, tomo como el primer instante n=M
       
       %calculo la salida
       aux = w .* x(:,n:-1:i);
       z(:, i) = sum(aux');
       
       %calculo el error de estimacion
       e = d(:, n) - z(:, i);
       
       %adapto los coeficientes del filtro
       w = w + mu .* x(:,n:-1:i) .* e;
       
       %calculo la funcion de costos en el instante i
       J(i) = (1/L) * sum(abs(e).^2); 
    end
    
end

%clasificador ML para señales con simbolos antipodales
%u: señal de entrada, s: señal original
%a: valor del simbolo, sigma_v: varianza del ruido de entrada
%y: señal de salida del clasificador
%SER = tasa de error
function [y, SER] = clasificador_ML(u, s, a, var_v) 
    N = length(u);
    y = zeros(1, N);
    Nse = 0; %numero de simbolos con error
    for i=1:N
        T = (1/var_v) * (2*a*u(i)); %estadistico para el caso de simbolos antipodales
        
        if((T<=0 && s(i)==a) || T>0 && s(i)==-a) %verifico los simbolos incorrectos
            Nse = Nse+1;
        end
        
        if(T<=0) %si T<0, el clasificador se decide por H0
            y(i) = -a;
        end
        
        if(T>0) %si T>0, el clasificador se decide por H1
            y(i) = a;
        end

    end

    SER = Nse / N;

end
