def get_average_gradient(named_parameters):
    
        avg_grads = {}

        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                
                avg_grads[n]=p.grad.abs().mean().item()
        
        return avg_grads