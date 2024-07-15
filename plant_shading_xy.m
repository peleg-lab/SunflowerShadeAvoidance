function [xypos, shaded_area, leaf_radii] = plant_shading_xy(leaf_radius, N_plants, dt, Tmax, GF, plant_x, plant_y, F_factor, k, noise_pref, F_power)
% 2D plant shading interaction model.
% If crowns from two plants overlap, plants experience symmetrical
% repulsion.
% Random fluctuations are added to each step by sampling from distribution
% specified by noise_pref
% Plant movements are bounded by the area specified by k. If the plant
% makes a step that would take it outside this boundary, it will reflect
% back inside
% Inputs:
%   leaf_radius: initial radius of plants
%   N_plants: number of plants
%   dt: time step
%   Tmax: final time
%   GF: radius growth rate
%   plant_x: N_plants x 1 vector of initial x positions
%   plant_y: N_plants x 1 vector of initial y positions
%   F_factor: scaling factor on repulsions
%   k: Tmax/dt x 1 vector of the radius of the bounding area at each time
%       step
%   noise_pref: either the variance of the Gaussian from which to sample
%       random steps, or a specified distribution
%   F_power: the power of the shade avoidance force (1,2,3, or 4)

dbstop if error;

% If noise_pref is not specified, set to 1 to sample Gaussian of mean 0 & variance 1
if nargin < 11
    noise_pref = 1;
end


steps = floor(Tmax/dt);
x = zeros(N_plants,2);
x(:,1) = plant_x;
x(:,2) = plant_y;
xypos = zeros(N_plants,2,steps);
shaded_area = zeros(steps,N_plants);
leaf_radii = zeros(N_plants, steps);
A = (pi*leaf_radius(1).^2)*ones(steps,1); % assuming all plants are equal size
A = A + GF*(1:steps)';

for step = 1:steps
    x = steepest_descent(dt, N_plants, x, leaf_radius, F_factor, plant_x, plant_y, k(step), noise_pref, F_power);
    xypos(:,:,step) = x;
    % calculate shaded area of each plant
    shade_temp = circle_intersection(x,leaf_radius);
    shaded_area(step,:) = shaded_area(step,:)+shade_temp;
    % leaf diameter grows at rate GF
    leaf_radius = sqrt(A(step)/pi)*ones(N_plants,1);
    leaf_radii(:,step) = leaf_radius;
end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% update plant positions
function x = steepest_descent(dt, N_plants, x, leaf_radius, ...
    F_factor, plant_x, plant_y, k, noise_pref, F_power)

% calculate forces on plants
F = forces(N_plants, x, leaf_radius, F_power);
F = F * F_factor;        % multiply by prefactor
F(abs(F)<1e-5) = 0;      % neglecting very small forces. Without this, even in zero-noise we can get patterns
F = round(F,3);
x = x + F*dt;

% add noise in crown movement
if isnumeric(noise_pref) % if noise is a number, draw from random distrib where noise is the variance
    x(:,1) = x(:,1) + noise_pref*sqrt(dt)*(randn(size(x(:,1))));
    x(:,2) = x(:,2) + noise_pref*sqrt(dt)*(randn(size(x(:,2))));

else
    step = arrayfun(noise_pref,rand(N_plants,1));
    angle = 2*pi*rand(N_plants,1);
    dx = step.*cos(angle);
    dy = step.*sin(angle);

    x(:,1) = x(:,1) + dx;
    x(:,2) = x(:,2) + dy;
end


% compute reflecting boundary conditions defined by circle of radius k
centers = [plant_x, plant_y];
for plant = 1:N_plants
    if sqrt(sum((x(plant,:) - centers(plant,:)).^2)) > 2*k % if step goes 2x the size of the boundary, reflect to the boundary edge
        r = x(plant,:)-centers(plant,:);
        angle = atan2(r(2),r(1));
        new_pos = k*[cos(angle),sin(angle)]+centers(plant,:);
        x(plant,:) = new_pos;
    elseif sqrt(sum((x(plant,:) - centers(plant,:)).^2)) > k % if plant is outside bound, reflect back to within boundary
        r = x(plant,:)-centers(plant,:);
        angle = atan2(r(2),r(1));
        new_pos = k*[cos(angle),sin(angle)]+centers(plant,:) - (r-k*[cos(angle),sin(angle)]);
        x(plant,:) = new_pos;
    end
    
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% calculate forces
function F = forces (N_plants, x, leaf_radius, F_power)
F = zeros(N_plants, 2);
[no_ints, pair, connector] = all_interactions (N_plants, x, leaf_radius); % determine all interacting pairs
for i = 1:no_ints
    FORCE = force_LJ_rep(connector(i,:), sum(leaf_radius(pair(no_ints,:))), F_power); % interaction range is sum of leaf radii
    FORCE(abs(FORCE)<5e-6)=0;    % neglecting very small forces. Without this, even in zero-noise we can get patterns
    
    F(pair(i,1),:)=F(pair(i,1),:)-FORCE;
    F(pair(i,2),:)=F(pair(i,2),:)+FORCE; %action=reaction
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ip, pair, connector]= all_interactions(N_plants, x, leaf_radius) % obtain interacting pairs
ip=0;
connector = zeros(1,2);
pair=zeros(1,2);

for i=1:N_plants-1
    for j=i+1:N_plants
        distance = (x(j,:)-x(i,:));
        if norm(distance) < leaf_radius(i)+leaf_radius(j) % plants interact if they are closer than the sum of their radii
            ip = ip + 1; % interaction pair counter
            pair(ip,:) = [i j]; % particle numbers (i,j) belonging to pair (ip)
            connector(ip,:) = distance;
        end
    end
end
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% force equation
function force_LJ = force_LJ_rep (r_vector,leaf_radius, F_power)
% here what is defined as "leaf_radius" is actually 2* the radius
% (equililbrium separation)
% the argument F_power specifies the exponent of the force
r=norm(r_vector); %two-body force
r_leaf_radius = leaf_radius;
if r<=r_leaf_radius
    
    r = max(.5*r_leaf_radius, r); % r is no less than .5*radius to prevent forces blowing up
    r_vector = r_vector*r/norm(r_vector); % rescale r_vector if necessary
    
    if F_power == 1
        force_LJ = r_leaf_radius.*r_vector/(r.^2) - 1.*r_vector/(norm(r_vector));
    elseif F_power == 2
        force_LJ = r_leaf_radius.^2.*r_vector./(r.^3) - 1.*r_vector/(norm(r_vector));
    elseif F_power == 3
        force_LJ = r_leaf_radius.^3.*r_vector./(r.^4) - 1.*r_vector/(norm(r_vector));
    elseif F_power == 4
        force_LJ = r_leaf_radius.^4.*r_vector./(r.^5) - 1.*r_vector/(norm(r_vector));
    else
        error('F_power not specified')
    end
    
else
    force_LJ = 0;
end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% determine shaded areas
function shaded_ar = circle_intersection(x,leaf_radius)
N=numel(x(:,1));
[ip,pair,connector]= all_interactions (N,x,leaf_radius);
shaded_ar=zeros(1,N);
for i=1:ip
    d=norm(connector(i,:));
    
    
    
    R1=leaf_radius(pair(i,1));
    R2=leaf_radius(pair(i,2));
    if R1<R2  
        TempR=R2;
        R2=R1;
        R1=TempR;
    end
    
    if d<(R1-R2)
        area12=pi*R2^2;
    elseif d>R1+R2
        area12=0;
    else
        area12=R1^2*acos((d^2+R1^2-R2^2)/(2*d*R1))+R2^2*acos((d^2+R2^2-R1^2)/(2*d*R2))-(1/2)*sqrt((-d+R1-R2)*(-d-R1+R2)*(-d+R1+R2)*(d+R1+R2));
    end
    
    
    if rand>.5 % only one of the plants is being shaded, randomly determine which
        shaded_ar(1,pair(i,1))=shaded_ar(1,pair(i,1))+(area12);
    else
        shaded_ar(1,pair(i,2))=shaded_ar(1,pair(i,2))+(area12);
    end
end


end
