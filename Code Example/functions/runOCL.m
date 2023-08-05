function [model, hat_y_t,l_t] = runOCL(x_t, y_t, t, model, opt)
x_t = NormalizeData(x_t,1);

f_t = model.w*x_t';


if (f_t>=0)
    hat_y_t = 1;
else
    hat_y_t = -1;
end

l_t = max(0,1-y_t*f_t);

if (l_t > 0)
    model.w  = model.w + opt.eta*y_t*x_t;
end