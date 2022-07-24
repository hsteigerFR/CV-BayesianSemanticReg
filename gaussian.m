function g = gaussian(T,s,U,V,X)
"""2D gaussian distribution probability density""";
"""T and s appears within this distribution to scale and translate it. s """;
factor = 1/(2*pi*(s^2)*sqrt(det(V)));
g = factor*exp(-(0.5/s^2)*(X-s*(U+T))'*inv(V)*(X-s*(U+T)));
end

