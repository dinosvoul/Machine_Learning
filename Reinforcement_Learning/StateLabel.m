function [s state_pos state_vel] = StateLabel(pos,vel,div_pos,min_pos,max_pos,div_vel,min_vel,max_vel)
    %Divide pos [min_pos, max_pos] into div_pos parts. The same with 
    % vel[min_vel, max_vel] into div_vel parts.

    range_pos=abs(max_pos-min_pos);
    %Divide 'pos' data into 'div_pos' parts
    if pos<=min_pos
    state_pos = 1;
    elseif vel>=max_vel
    state_pos = div_pos;    
    else 
    state_pos=ceil((pos+max_pos)/range_pos*div_pos);
    end
    
    range_vel = abs(max_vel-min_vel);
    %Divide 'vel' data into 'div_vel' parts
    if vel<=min_vel
    state_vel = 1;
    elseif vel>=max_vel
    state_vel = div_vel;    
    else 
    state_vel=ceil((vel+max_vel)/range_vel*div_vel);
    end
    
    s=(state_vel-1)*4+state_pos;

end