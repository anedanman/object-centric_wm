import math


def create_cosine_scheduler(min_coef: float = 0., max_coef: float = 1., warmup: float = 0.25, skip: float = 0.):
    
    assert skip < warmup
    
    def scheduler(current_steps: int, total_steps: int):
        skip_steps = round(total_steps * skip)
        if current_steps < skip_steps:
            return min_coef
        
        warmup_steps = round(total_steps * warmup)
        if current_steps > warmup_steps:
            return max_coef
        a = 0.5 * (min_coef - max_coef)
        b = 0.5 * (min_coef + max_coef)
        progress = (current_steps - skip_steps) / (warmup_steps - skip_steps)
        return a * math.cos(math.pi * progress) + b
    
    return scheduler
