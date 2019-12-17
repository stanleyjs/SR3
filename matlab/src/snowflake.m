function y = snowflake(theta,epsilon)
  y = (2.*sqrt(theta) - 2.*epsilon.*log(epsilon + sqrt(theta)));
end