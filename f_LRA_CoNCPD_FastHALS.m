%% ## before you run this code, please download the tensor toolbox via https://www.tensortoolbox.org/
function out = f_LRA_CoNCPD_FastHALS(T)
%% LRA_CoNCPD-FastHALS
% LRA_CoNCPD based on FastHALS
% input: T- data information (struct)
% required
%        T.subject(cell) - data of tensors or matrices
%        T.R (vector)    - component number or rank of each data
%        T.C (vector)    - coupled component number of data
% default
%        T.tol           - stopping tolerance
%        T.eps           - nonnegative papameter
%        T.maxT          - maximum of execution time-10e10;
%        T.maxIter       - maximum of Iteration Number-1000;
%        T.iscore        - core tensor exists or not (1)
%        T.islargescale  - largescale or not (1)
%        T.isnorm        - factor matrix normalize or not(0)
%output: out.U---------factor matrices of tensor and matrix
%        out.G---------core tensors or matrices/features
%        out.obj-------object function value
%        out.relerr----relative error
%        out.tensorfit-tensorfit
%        out.iter------number of iteration
%        out.Executime-Execution time of current algorithm
% author: Wang xiulin
% date  : June 22th,2020
% address : jyu-agora
% note: this code is used for joint analysis of tensor/matrix data
% The time comsumption of computing G is very large, we don't compute G
% directly when not all modes are coupled. We absorb G into the last
% mode, then we get the G from the last mode
% But if all modes are coupled, we need to compute G directly
if nargin < 1
    demo_LRA_CoNCPD_FastHALS;
    return;
end
t1 = clock;
out = LRA_CoNCPD_FastHALS(T);
t2 = clock;
out.RunningTime = etime(t2,t1);
end

function demo_LRA_CoNCPD_FastHALS
%% generate data tensor
clear;
clc;
close all;
startup; % add the path of toolbox
%% generation of tensor
for iter = 1:1
    % Params
    S = 1;                 % the number of tensors
    R = zeros(S,1);
    for s = 1:S
        R(s) = 4;          % the number of components
    end
    size_tens = [100 100 100]; % the size of tensors
    C = [4 0 0];            % the coupled number on each mode
    %- coupled tensor generation
    N = numel(size_tens);
    UC = cell(N,1);
    for n = 1:N
        UC{n} = max(0, rand(size_tens(n), C(n))); % couple information
    end
    U0 = cell(S,1);
    for p = 1:S
        for n = 1:N
            U0{p}{n} = max(0, rand(size_tens(n), R(p))); % factor matrix-randn
            U0{p}{n}(:,1:C(n)) = UC{n}(:,1:C(n));
        end
        if any(C == 0)
            T = full(ktensor(U0{p}));
            Z.iscore = 0;
        else
            lambda = 10*rand(R(p),1);
            T = full(ktensor(lambda, U0{p}));
            Z.iscore = 1;
        end
        Z.object{p} = T;
    end
    Z.R      = R;
    Z.C      = C;
    Z.tol  = 1e-8;
    Z.eps  = 1e-16;
    Z.maxT   =  10e10;
    Z.maxIter = 1000;
    Z.U0 = U0;
    %%
    tic
    out = LRA_CoNCPD_FastHALS(Z);
    out.iter
    out.obj(end)
    out.relerr(end)
    out.tensorfit(end)
    %     out.RunningTime
end
end
function out = LRA_CoNCPD_FastHALS(Z)
% LRA_CoNCPD based on FastHALS
% author: Wang xiulin
%% parse parameters
iter = 0;
T         = Z.object; % tensor
for p = 1:numel(T)
    size_tens{p} = size(T{p}); % size of tensors
end
N         = numel(size_tens{p}); % order of tensors
R         = Z.R;
C         = Z.C;
if isfield(Z,'tol');     tol = Z.tol;         else tol = 1e-8;         end % stopping tolerance
if isfield(Z,'eps');     eps = Z.eps;         else eps = 1e-16;        end % nonnegative papameter
if isfield(Z,'maxIter'); maxIter = Z.maxIter; else maxIter = 1000;     end % max # of iterations
if isfield(Z,'maxT');    maxT = Z.maxT;       else maxT = 10e10;       end % max time in seconds
if isfield(Z,'iscore');  iscore = Z.iscore;   else iscore = 0;         end % core tensor exist or not
if isfield(Z,'islargescale') islargescale = Z.islargescale; else islargescale = 1; end; % largescale or not
if isfield(Z,'isnorm'); isnorm = Z.isnorm;    else isnorm = 0;         end % factor matrix normalize or not
%% LRA
t1 = clock;
for p = 1:numel(T)   
    [U0{p}, outlra{p}] = CP_ALS(T{p},R(p)); 
end
t2 = clock;
RunningTimelra = etime(t2,t1);
%% initialization
obj0 = 0;
KrrU  = cell(numel(T),1);
KrrU0 = cell(numel(T),1);
Mnrm  = cell(numel(T),1);
MaTN  = cell(numel(T),1);
U     = cell(numel(T),1);
G     = cell(numel(T),1);
for p = 1:numel(T)  
    MaTN{p} = U0{p}{N}*khatrirao(U0{p}(N-1:-1:1))';
    Mnrm{p} = norm(MaTN{p},'fro');
    obj0     = obj0 + 0.5*Mnrm{p}.^2;
    KrrU{p}  = ones(R(p)); % U-kr'*U-kr
    KrrU0{p} = ones(R(p)); % U-kr'*U0-kr
    for n = 1:N  
        U{p}{n}   = max(0, rand(size_tens{p}(n), R(p))); % factor matrix-randn
        if iscore
            U{p}{n}  = U{p}{n}/norm(U{p}{n},'fro')*Mnrm{p}^(1/(N+1)); % normalize and average
        else
            U{p}{n}  = U{p}{n}/norm(U{p}{n},'fro')*Mnrm{p}^(1/(N));
        end 
        KrrU{p}   = KrrU{p}.*(U{p}{n}'*U{p}{n});
        KrrU0{p}  = KrrU0{p}.*(U{p}{n}'*U0{p}{n});
    end 
    if iscore
        G{p} = diag(rand(R(p),1));
        G{p} = G{p}*Mnrm{p}^(1/(N+1));
    else
        G{p} = eye(R(p));
    end
end
relerr0 = 1;
%%
krrn   = cell(numel(T),1);
mttkr  = cell(numel(T),1);
start_time = tic;
fprintf('Iteration:     ');
while iter<maxIter
    iter = iter + 1;
    fprintf('\b\b\b\b\b%5d', iter); 
    %% updating core tensors
    if iscore
        for p = 1:numel(T)
            if islargescale == 1
                Gup   = KrrU0{p}*ones(R(p),1);
                Gdown = KrrU{p};
            else
                %-
            end
            G{p} = pinv(Gdown + eps)*(Gup-beta(N+1)); 
            G{p} = diag(max(eps, G{p}));
        end
    end
    %% Updating factor matrices
    for n = 1:N
        % tensor/matrix
        for p = 1:numel(T)            
            KrrU{p}   = KrrU{p}./(U{p}{n}'*U{p}{n});
            KrrU0{p}  = KrrU0{p}./(U{p}{n}'*U0{p}{n});            
            krrn{p}   = G{p}*KrrU{p}*G{p}';
            mttkr{p}  = U0{p}{n}*KrrU0{p}'*G{p}';
        end
        for j = 1:max(R)
            if C(n) && j <= C(n)
                for p = 1:numel(T)
                    if p == 1
                        E1 = U{1}{n}(:,j);
                        E2 = mttkr{1}(:,j);
                        E3 = U{1}{n}*krrn{1}(:,j);
                        Gt = krrn{1}(j,j) + eps;
                    else
                        E2 = E2 + mttkr{p}(:,j);
                        E3 = E3 + U{p}{n}*krrn{p}(:,j);
                        Gt = Gt + krrn{p}(j,j) + eps;
                    end
                end
                UU = E1 + (E2 - E3)./Gt;
                UU = max(eps,UU);
                if (iscore && isnorm) || (n ~= N && isnorm && ~iscore) 
                   UU = UU./norm(UU); 
                end
                for p = 1:numel(T)
                    U{p}{n}(:,j) = UU;
                end
            else  %% individual
                for p = 1:numel(T)
                    if j <= R(p)
                        U{p}{n}(:,j) = U{p}{n}(:,j) + (mttkr{p}(:,j) - U{p}{n}*krrn{p}(:,j))./(krrn{p}(j,j) + eps);
                        U{p}{n}(:,j) = max(eps,U{p}{n}(:,j));
                        if (iscore && isnorm) || (n ~= N && isnorm && ~iscore) 
                            U{p}{n}(:,j) = U{p}{n}(:,j)./norm(U{p}{n}(:,j));
                        end
                    end
                end
            end
        end
        for p = 1:numel(T)
            if n <= N
                KrrU0{p} = KrrU0{p}.*(U{p}{n}'*U0{p}{n});
                KrrU{p}  = KrrU{p}.*(U{p}{n}'*U{p}{n});
            end
        end
    end    
    %% compute the target function value-relative error-tensorfit-
    obj = 0;
    relerr = 0;
    tensorfit = 0;
    for p = 1:numel(T)
        % Computation speed of target function value 1 is faster than 2 for
        % larger-scale problem
        if islargescale == 1
            obj1 = 0.5*(sum(sum(G{p}*KrrU{p}*G{p}'))-2*sum(sum(U{p}{N}.*mttkr{p}))+Mnrm{p}^2);% target function value 1
        else
            Mn = MaTN{p} - U{p}{N}*G{p}*khatrirao(U{p}(end-1:-1:1))';
            obj1 = 0.5*(norm(Mn,'fro').^2); % target function value 2
        end
        relerr1 = (2*obj1)^.5/Mnrm{p}; % relative error
        tensorfit1 = 1 - relerr1;
        obj = obj + obj1;
        relerr = relerr + relerr1;
        tensorfit = tensorfit + tensorfit1;
    end
    out.Executime(iter) = toc(start_time);
    out.lraobj(iter) = obj;
    out.lrarelerr(iter) = relerr;
    out.lratensorfit(iter) = tensorfit./numel(T);
    %-stopping criterion-
    relerr2 = abs(relerr-relerr0);  % retative error chage
    if relerr2 < tol || out.Executime(iter) > maxT; break; end
    relerr0 = relerr;
end
out.iter = iter;
out.U = U;
out.G = G;
out.alsout = outlra;
out.alsU0  = U0;
out.alsRunningTime = RunningTimelra;
obj = 0;
relerr = 0;
tensorfit = 0;
for p = 1:numel(T)
    Mnrm{p}  = norm(T{p});
    Mat = tenmat(T{p}, N);
    Mn = Mat.data - U{p}{N}*G{p}*khatrirao(U{p}(end-1:-1:1))';
    obj1 = 0.5*(norm(Mn,'fro').^2); % target function value 2
    relerr1 = (2*obj1)^.5/Mnrm{p}; % relative error
    tensorfit1 = 1 - relerr1;
    obj = obj + obj1;
    relerr = relerr + relerr1;
    tensorfit = tensorfit + tensorfit1;
end
out.obj = obj;
out.relerr = relerr;
out.tensorfit = tensorfit./numel(T);
end

function [U, out] = CP_ALS(T, J)
% input: 
%        T. - data of tensors or matrices
%        J. - component number or rank of each data
%output: U-factor matrices of tensor and matrix
%         out.obj-object function value
%         out.relerr-relative error
%         out.tensorfit-tensorfit
%         out.iter-number of iteration
% Author: Xiulin Wang
% Date: 14/11/2017
%% --
iter = 0;
relerr0 = 1;     % old relative error
islargescale = 1;
iter_max = 200; % maximum of iterations
tol  = 1e-4;     % stopping tolerance
size_tens = size(T);
N  = size(size_tens,2);
Mnrm = norm(T);
for n = 1:N
    Tnn   = tenmat(T,n);
    Tn{n} = Tnn.data;
end
for n = 1:N
    U{n} = randn(size_tens(n), J);
end
Ukrr = ones(J);
for n = 1:N
    Ukrr = Ukrr.*(U{n}'*U{n});
end
while iter<iter_max
    iter = iter + 1;
    for n = 1:N
        Ukrr = Ukrr./(U{n}'*U{n});
        krr  = khatrirao(U([end:-1:n+1 n-1:-1:1]));
        TB   = Tn{n}*krr;
        U{n} = TB*pinv(Ukrr);
%         U{n} = max(eps,U{n});
        Ukrr = Ukrr.*(U{n}'*U{n});
    end
    %- object funtion value-relative error-tensorfit-
    if islargescale == 1
        obj1 = 0.5*(sum(sum(Ukrr))-2*sum(sum(U{N}.*TB))+Mnrm^2); %%
    else
        Mn = Tn{N} - U{N}*krr';
        obj1 = 0.5*(norm(Mn,'fro').^2);
    end 
    relerr1 = (2*obj1)^.5/Mnrm; % relative error    
    tensorfit = 1 - relerr1;   % tensorfit
    out.obj(iter) = obj1;
    out.relerr(iter) = relerr1;
    out.tensorfit(iter) = tensorfit;
    %-stopping criterion-
    relerr2 = abs(relerr1 -relerr0);  % retative error chage 
    if relerr2 < tol; break; end
    relerr0 = relerr1;
end
out.iter = iter;
end





