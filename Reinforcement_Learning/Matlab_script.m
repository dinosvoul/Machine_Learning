%This script simulates a pendulum, by using Q learning, we train the Q
%matrix in an scenario when the pendulum is started randomly 
% between -90º to 90º. Note that the sampling time Ts is very
%important as we should actuate fast enough, if it was too slow, the
%pendulum would always fall down.

clearvars -except Q
close all
clc


%Model parameters
a1 = 9.81/0.305;
b = 5;
c = 107.5;

%sample time
Ts = 0.01;

%Initial conditions
%we will put pendulum randomly between [-90 to 90 degrees]
x1(1)= pi*rand(1)-pi/2;
x2(1) = 0;

%Define posible inputs
a=[-1 -0.1 -0.05 0 0.05 0.1 1];

%Define the range of position and velocity
div_pos = 4;
min_pos = -pi;
max_pos = pi;
div_vel=10;
min_vel=-10;
max_vel = 10;

%Initialize Q matrix from -1 to 0
Q=-rand(div_pos*div_vel,length(a));
%Load trained Q matrix
%load Q_trained2

%Simulation time
tend = 30;%s
%time vector
t = 0:Ts:tend;

%Q learning parameters from algorithm
gamma=0.5;
alpha=0.65;

%Compute the actual state from angular position and velocity
state = StateLabel(x1(1),x2(1),div_pos,min_pos,max_pos,div_vel,min_vel,max_vel);

%Variable to save the time that simulation is restarted (when pendulum goes
%below the critical points -90º and 90.
reset_time = [];

%Simulation: Simple forward euler method
for i = 1:length(t)-1    
    %Choose the position of the action with highest value in Q matrix
    a_chosen = find(Q(state,:)==max(Q(state,:)));
    %Apply the chosen a action
    u(i) = a(a_chosen);

    %system simulation: theta and theta_dot
    x1(i+1) = x1(i)+Ts*x2(i);     %theta state x1
    x2(i+1) = x2(i)+Ts*(-a1*sin(x1(i)+pi)-b*x2(i)+c*u(i)); %ang velocity x2
    
    %Update Q matrix: 
    %First compute the reward using the position, far from 0 is bad reward
    r=-x1(i+1)^2;
    %Compute the actual state 
    [state_new state_pos state_vel] = StateLabel(x1(i+1),x2(i+1),div_pos,min_pos,max_pos,div_vel,min_vel,max_vel);
    %Update Q matrix by using new state
    Q(state,a_chosen)=Q(state,a_chosen)+alpha*(r+gamma*max(Q(state_new,:))...
        -Q(state,a_chosen));
    state = state_new;
    
    %If angle goes lower than -90º or higher than 90º, restart simulation
    if x1(i+1)<-pi/2 || x1(i+1)>pi/2
       reset_time = [reset_time (i+1)];
       x1(i+1)= pi*rand(1)-pi/2;
       x2(i+1)= 0;
    end
end

%Plot results
figure(2);
subplot(2,1,1), plot(t,x1*180/pi,'Linewidth',1),grid,xlabel('t (s)'),ylabel('\theta (º)'), title('Angular position','fontweight','bold','fontsize',12)
hold on, plot(t(reset_time),x1(reset_time),'r.')
subplot(2,1,2), plot(t,x2,'Linewidth',1),grid,xlabel('t (s)'),ylabel('\omega (rad/s)'), title('Angular velocity','fontweight','bold','fontsize',12)