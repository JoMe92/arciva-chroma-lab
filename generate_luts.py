import numpy as np
import os

def generate_lut(size, func):
    """
    Generates a 3D LUT (size x size x size x 3).
    func(r, g, b) -> (r, g, b)
    """
    lut = np.zeros((size, size, size, 3), dtype=np.float32)
    step = 1.0 / (size - 1)
    
    for b in range(size):
        for g in range(size):
            for r in range(size):
                rv = r * step
                gv = g * step
                bv = b * step
                
                rgb_out = func(rv, gv, bv)
                lut[r, g, b] = rgb_out
                
    return lut

def save_cube(filename, lut, size, title):
    with open(filename, 'w') as f:
        f.write(f'TITLE "{title}"\n')
        f.write(f'LUT_3D_SIZE {size}\n')
        
        # Cube format loops: B fast, G medium, R slow? Or R fast?
        # Standard .cube:
        # "The first line of data is (0,0,0), then (1,0,0) ... (N,0,0), then (0,1,0)..."
        # So R varies fastest.
        # lut array is [r, g, b] in my code above? 
        # yes, lut[r, g, b].
        # So nested loops: B outer, G middle, R inner.
        
        # Domain min/max are optional, usually 0 1
        f.write('DOMAIN_MIN 0.0 0.0 0.0\n')
        f.write('DOMAIN_MAX 1.0 1.0 1.0\n')

        for b in range(size):
            for g in range(size):
                for r in range(size):
                    val = lut[r, g, b]
                    # Clamp values to [0, 1] before writing
                    ro = max(0.0, min(1.0, val[0]))
                    go = max(0.0, min(1.0, val[1]))
                    bo = max(0.0, min(1.0, val[2]))
                    f.write(f'{ro:.6f} {go:.6f} {bo:.6f}\n')

def save_3dl(filename, lut, size):
    # Autodesk 3DL format
    # Header: Grid points
    # 0 64 128 ... (integers based on bit depth, usually 10-bit 1023)
    # Then data lines
    
    with open(filename, 'w') as f:
        # Using 10 bit scale
        scale = 1023.0
        
        # Grid definition
        # If uniform:
        f.write(' '.join(str(int(i * scale / (size-1))) for i in range(size)) + '\n')
        
        # 3DL loops: R, G, B order (R fastest) same as cube usually?
        # actually 3DL is usually R, G, B.
        for r in range(size):
            for g in range(size):
                for b in range(size):
                    val = lut[r, g, b]
                    # Output is usually integer triplets
                    # Clamp values to [0, 1] before scaling
                    vr = int(max(0.0, min(1.0, val[0])) * scale)
                    vg = int(max(0.0, min(1.0, val[1])) * scale)
                    vb = int(max(0.0, min(1.0, val[2])) * scale)
                    f.write(f'{vr} {vg} {vb}\n')

def save_xmp(filename, lut, size):
    # Simple XMP LookTable wrapping
    # We will write space-separated floats inside <crs:LookTable>
    with open(filename, 'w') as f:
        f.write('<x:xmpmeta xmlns:x="adobe:ns:meta/">\n')
        f.write(' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n')
        f.write('  <rdf:Description rdf:about="" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">\n')
        f.write('   <crs:LookTable>\n')
        
        # Data
        # XMP LookTable expects R, G, B order (R fastest)
        # It's a flat list of R G B values
        for r in range(size):
            for g in range(size):
                for b in range(size):
                    val = lut[r, g, b]
                    # Clamp values to [0, 1] before writing
                    ro = max(0.0, min(1.0, val[0]))
                    go = max(0.0, min(1.0, val[1]))
                    bo = max(0.0, min(1.0, val[2]))
                    f.write(f'{ro:.6f} {go:.6f} {bo:.6f} ')
        
        f.write('\n   </crs:LookTable>\n')
        f.write('  </rdf:Description>\n')
        f.write(' </rdf:RDF>\n')
        f.write('</x:xmpmeta>\n')

# LUT Functions
def identity(r, g, b):
    return (r, g, b)

def warm(r, g, b):
    # Boost Red, reduce Blue slightly
    return (min(1.0, r * 1.1), g, max(0.0, b * 0.9))

def cool(r, g, b):
    # Boost Blue, reduce Red
    return (max(0.0, r * 0.9), g, min(1.0, b * 1.1))

def sepia(r, g, b):
    # Simple Sepia matrix
    tr = 0.393*r + 0.769*g + 0.189*b
    tg = 0.349*r + 0.686*g + 0.168*b
    tb = 0.272*r + 0.534*g + 0.131*b
    return (min(1.0, tr), min(1.0, tg), min(1.0, tb))

def kodachrome(r, g, b):
    # Simulation of Kodachrome style:
    # High contrast, rich saturation, specific color shifts
    
    # 1. Contrast curve (S-curve)
    def s_curve(x):
        return x * x * (3.0 - 2.0 * x)
    
    # Apply contrast
    r, g, b = s_curve(r), s_curve(g), s_curve(b)
    
    # 2. Saturation boost
    gray = 0.3 * r + 0.59 * g + 0.11 * b
    sat = 1.3 # boost
    
    r = gray + (r - gray) * sat
    g = gray + (g - gray) * sat
    b = gray + (b - gray) * sat
    
    # 3. Tint helper (Warmer highlights, cooler shadows potentially)
    r *= 1.1 # Warmer
    b *= 0.9
    
    return (min(1.0, max(0.0, r)), min(1.0, max(0.0, g)), min(1.0, max(0.0, b)))

if __name__ == "__main__":
    size = 33
    output_dir = 'luts'
    os.makedirs(output_dir, exist_ok=True)
    
    luts = {
        'Identity': identity,
        'Warm': warm,
        'Cool': cool,
        'Sepia': sepia,
        'Kodachrome': kodachrome
    }
    
    for name, func in luts.items():
        print(f"Generating {name}...")
        data = generate_lut(size, func)
        save_cube(os.path.join(output_dir, f'{name}.cube'), data, size, name)
        
        # Generate .3dl and .xmp for Kodachrome to test
        if name == 'Kodachrome':
            save_3dl(os.path.join(output_dir, f'{name}.3dl'), data, size)
            save_xmp(os.path.join(output_dir, f'{name}.xmp'), data, size)
            print(f"Generated {name}.3dl and {name}.xmp")
