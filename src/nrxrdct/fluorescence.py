import xraylib

DEFAULT_LINES = ['Ka1', 'Ka2', 'Kb1', 'Kb2', 'La1', 'Lb1', 'Lg1']

def get_fluo_lines(element, energy_range, lines:list[str]=DEFAULT_LINES, verbose = False ):

    """
    
    """

    emin, emax = (energy_range)
    Z = xraylib.SymbolToAtomicNumber(element)

    line_name = lines
    line_poss = [xraylib.KA1_LINE, 
                 xraylib.KA2_LINE,
                 xraylib.KB1_LINE,
                 xraylib.KB2_LINE,
                 xraylib.LA1_LINE,
                 xraylib.LB1_LINE,
                 xraylib.LG1_LINE]
    lines = {}
    for name, line in zip(line_name, line_poss):
        try:
            en = xraylib.LineEnergy(Z, line)
            if en > emax:
                continue
            elif en < emin:
                continue
            else:
                lines[name] = en
        except Exception:
            print(f'Line {name} not available for {element}.')
            continue

    if verbose:        
        for name, line in lines.items():
            print(f"{element} {name}: {line:.4f} keV")
    return lines