import numpy as np
import os

def write_cube(filename, title, size, func):
    with open(filename, 'w') as f:
        f.write(f'TITLE "{title}"\n')
        f.write(f'LUT_3D_SIZE {size}\n')
        
        # Domain min/max are optional, usually 0 1
        f.write('DOMAIN_MIN 0.0 0.0 0.0\n')
        f.write('DOMAIN_MAX 1.0 1.0 1.0\n')

        # Loop order: B outer, G middle, R inner (standard for .cube??)
        # Actually standard is:
        # for b in 0..size-1:
        #   for g in 0..size-1:
        #     for r in 0..size-1:
        #       r,g,b = ...
        
        # Wait, let's double check standard iteration order.
        # "The lines of table data shall be in the order of the first dimension (Red) changing fastest, then the second (Green), then the third (Blue)."
        
        step = 1.0 / (size - 1)
        
        for b_idx in range(size):
            for g_idx in range(size):
                for r_idx in range(size):
                    r = r_idx * step
                    g = g_idx * step
                    b = b_idx * step
                    
                    # Apply function
                    ro, go, bo = func(r, g, b)
                    
                    # Clamp
                    ro = max(0.0, min(1.0, ro))
                    go = max(0.0, min(1.0, go))
                    bo = max(0.0, min(1.0, bo))
                    
                    f.write(f'{ro:.6f} {go:.6f} {bo:.6f}\n')

def identity(r, g, b):
    return r, g, b

def warm(r, g, b):
    # Boost Red, slight Green, reduce Blue
    return r * 1.1 + 0.05, g * 1.05 + 0.02, b * 0.9

def cool(r, g, b):
    # Boost Blue, slight Cyan
    return r * 0.9, g * 0.95, b * 1.1 + 0.05

def sepia(r, g, b):
    # Standard sepia matrix
    # tr = 0.393*r + 0.769*g + 0.189*b
    # tg = 0.349*r + 0.686*g + 0.168*b
    # tb = 0.272*r + 0.534*g + 0.131*b
    # But mixed 50% with identity to not be too strong/monochrome
    
    tr = 0.393*r + 0.769*g + 0.189*b
    tg = 0.349*r + 0.686*g + 0.168*b
    tb = 0.272*r + 0.534*g + 0.131*b
    
    mix = 0.7
    return r*(1-mix) + tr*mix, g*(1-mix) + tg*mix, b*(1-mix) + tb*mix

os.makedirs('luts', exist_ok=True)

# Size 17 is decent small size, 33 is standard high quality
size = 33 
write_cube('luts/Identity.cube', 'Identity', size, identity)
write_cube('luts/Warm.cube', 'Warm Look', size, warm)
write_cube('luts/Cool.cube', 'Cool Look', size, cool)
write_cube('luts/Sepia.cube', 'Sepia', size, sepia)

print("Generated luts/Identity.cube")
print("Generated luts/Warm.cube")
print("Generated luts/Cool.cube")
print("Generated luts/Sepia.cube")
