EEGsynthetic2;
% Create DMD data matrices
X1 = y(1:end -1,:)';
X2 = y(2:end,:)';
% SVD and rank -2 truncation
r = 2; % rank truncation
[U, S, V] = svd(X1);
Ur = U(:, 1:r);
Sr = S(1:r, 1:r);
Vr = V(:, 1:r);
% Build Atilde and DMD Modes
Splus=S;
Splus(1,1)=1/S(1,1);
Splus(2,2)=1/S(2,2);
Splus(3,3)=1/S(3,3);
Splus(4,4)=1/S(4,4);
Splus(5,5)=1/S(5,5);
A=X2*V*Splus'*U'
Atilde = Ur'*X2*Vr/Sr
%pause
[W, D] = eig(Atilde);
Phi = X2*Vr/Sr*W; % DMD Modes
% DMD Spectra
lambda = diag(D);
omega = lambda;
% Compute DMD Solution
x1 = X1(:, 1);
b = Phi\x1;
time_dynamics = zeros(r,length(t));
for iter = 1:length(t)
%time_dynamics (:,iter) = (b.*exp(omega*t(iter)));
time_dynamics (:,iter) = b.*omega*t(iter);
end
X_dmd = Phi*time_dynamics ;
subplot(2,2,4);
surfl(real(X_dmd'));
shading interp; colormap(gray); view (-20,60);
set(gca , 'YTick', numel(t)/4 * (0:4)),
set(gca , 'Yticklabel',{'0','\pi','2\pi','3\pi','4\pi'});
set(gca , 'XTick', linspace(1,numel(xi),3)),
set(gca , 'Xticklabel',{'-10', '0', '10'});