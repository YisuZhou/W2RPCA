function [ x ] = rpca_soft( b,T )  
    x = sign(b).*max(abs(b) - T,0);  
end  