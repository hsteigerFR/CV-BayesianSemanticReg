function [E] = systemP(tx,ty,s,U,V,U2,V2,Z,Z2,Ic,Jc)
"""System of equation for fsolve. The variables are the scale factor s and x-y translations T.""";
"""THE SYSTEM IS NON LINEAR AND CANNOT BE SOLVED EXPLICITLY as s*T appears in the GMMs definition.""";
    E1 = 0;
    E2 = 0;
    E3 = 0;
    for k = 1 : size(U,2)
        sig = det(V(:,:,k));
        ux = U(1,k);
        uy = U(2,k);
        Vx = V(1,1,k);
        Vy = V(2,2,k);
        Cov = V(1,2,k);
        for i = 1 : length(Ic)
            x = Jc(i)-1;
            y = Ic(i)-1;
            E1 = E1 + Z(Ic(i),Jc(i),k)*(Vy*(x-s*(ux + tx)) + Cov*(y-s*(uy+ty)));
            E2 = E2 + Z(Ic(i),Jc(i),k)*(Vx*(y-s*(uy + ty)) + Cov*(x-s*(ux+tx)));
            E3 = E3 + Z(Ic(i),Jc(i),k)*(-2*s^2 + (1/sig)*(Vy*(x-s*(ux + tx))^2 + Vx*(y-s*(uy + ty))^2 - 2*Cov*(x-s*(ux + tx))*(y-s*(uy + ty)) + s*(Vy*(ux+tx)*(x-s*(ux + tx)) + Vx*(uy+ty)*(y-s*(uy + ty)) -Cov*( (ux+tx)*(y-s*(uy + ty)) + (uy+ty)*(x-s*(ux + tx)) ))));
        end
    end
    
    for k = 1 : size(U2,2)
        sig = det(V2(:,:,k));
        ux = U2(1,k);
        uy = U2(2,k);
        Vx = V2(1,1,k);
        Vy = V2(2,2,k);
        Cov = V2(1,2,k);
        for i = 1 : length(Ic)
            x = Jc(i)-1;
            y = Ic(i)-1;
            E1 = E1 + Z2(Ic(i),Jc(i),k)*(Vy*(x-s*(ux + tx)) + Cov*(y-s*(uy+ty)));
            E2 = E2 + Z2(Ic(i),Jc(i),k)*(Vx*(y-s*(uy + ty)) + Cov*(x-s*(ux+tx)));
            E3 = E3 + Z2(Ic(i),Jc(i),k)*(-2*s^2 + (1/sig)*(Vy*(x-s*(ux + tx))^2 + Vx*(y-s*(uy + ty))^2 - 2*Cov*(x-s*(ux + tx))*(y-s*(uy + ty)) + s*(Vy*(ux+tx)*(x-s*(ux + tx)) + Vx*(uy+ty)*(y-s*(uy + ty)) -Cov*( (ux+tx)*(y-s*(uy + ty)) + (uy+ty)*(x-s*(ux + tx)) ))));
        end
    end
    E = [E1;E2;E3];
    
end

