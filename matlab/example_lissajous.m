%Created on Wed Jan 23 00:22:09 2019
% Generating simple calligraphic like glyphs with Lissajous curves
% author: Daniel Berio

S = {};
n = 200;

t = linspace(0, pi*3.8, n);

delta = randu(-pi/2, pi/2);
da = randu(-pi/2, pi/2);
db = randu(-pi/2, pi/2);
omega = 2.;

m = 5;
o = linspace(0, 0.2, m);
for i=1:m
    a = sin(linspace(0, pi*2, n) + da + o(i)*0.5)*200;
    b = cos(linspace(0, pi*2, n) + db + o(i)*1.0)*200;
    P = lissajous(t, a, b, omega, delta);
    S{i} = P;
end

title = sprintf('d=%.2f da=%.2f db=%.2f',delta, da, db)
axi('title', title);
axi('draw', S);

figure; hold on;
for i=1:m
    P = S{i};
    plot(P(1,:), P(2,:), 'k');
end

ax = gca;
ax.YDir = 'reverse';


function x = randu(a, b)
    x = a + rand()*(b-a);
end

function P = lissajous(t, a, b, omega, delta)
    P = [a.*cos(omega.*t + delta); b.*sin(t)];
end